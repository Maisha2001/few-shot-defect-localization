"""Freeze the cross-product protocol, generate predictions, then score frozen artifacts."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .adapt import calibration_metrics
from .data import Archive, RESERVED, SPLIT_SHA, Sample, file_hash, protocol
from .evaluate import ARCHIVE_SHA, fingerprint, near_reference, write_json
from .experiment import split_roles
from .metrics import native_counts, summarize
from .model import anomaly_scores, embeddings, load_encoder, pixels

SPEC = Path(__file__).resolve().parents[1] / "protocols/cross-product.json"
ARMS = ("random-32", "active-32", "reference-bank")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lock_files():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            if p.stem in {"final", "adapt", "data", "evaluate", "experiment", "metrics", "model", "export", "verify"}}


def freeze(inputs, exported, bank, output):
    spec = read(SPEC)
    manifest = read(exported / "manifest.json")
    if (file_hash(inputs / "VisA.tar") != ARCHIVE_SHA or file_hash(inputs / "1cls.csv") != SPLIT_SHA
            or file_hash(bank) != spec["bank_sha256"] or manifest["heads"] != spec["heads"]
            or manifest["graph_sha256"] != spec["graph_sha256"] or tuple(manifest["arms"]) != ARMS[:2]
            or file_hash(exported / "inspection.onnx") != spec["graph_sha256"]):
        raise ValueError("Inputs differ from the frozen experiment")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"frozen_at_utc": utc(), "spec": spec, "spec_sha256": file_hash(SPEC),
               "implementation": lock_files(), "archive_sha256": ARCHIVE_SHA, "split_sha256": SPLIT_SHA})


def selected_images(rows):
    result = [Sample(r["object"], r["image"], "", False) for r in rows
              if r["object"] in RESERVED and r["split"] == "test"]
    if {s.product for s in result} != set(RESERVED) or len(result) != len({s.image for s in result}):
        raise ValueError("Missing product or duplicate final image")
    return sorted(result, key=lambda s: sha256(s.image.encode()).digest())


def read_split(path):
    if file_hash(path) != SPLIT_SHA:
        raise ValueError("Split digest mismatch")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify_lock(output):
    lock = read(output / "protocol.json")
    if lock["spec_sha256"] != file_hash(SPEC) or lock["spec"] != read(SPEC) or lock["implementation"] != lock_files():
        raise ValueError("Final protocol or evaluator changed")
    return lock["spec"]


def predict(inputs, exported, bank_path, output):
    from .export import session
    spec = verify_lock(output)
    if (file_hash(inputs/"VisA.tar") != ARCHIVE_SHA or file_hash(bank_path) != spec["bank_sha256"]
            or file_hash(exported/"inspection.onnx") != spec["graph_sha256"]):
        raise ValueError("Final inference input changed")
    folder = output / "predictions"
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    start = perf_counter()
    split = inputs / "1cls.csv"
    samples = selected_images(read_split(split))
    references, _ = protocol(split)
    pool, calibration, dev = split_roles(split)
    earlier = {s.image: Sample(s.product,s.image,"",False) for s in references + pool + calibration + dev}
    graph = session(exported / "inspection.onnx")
    encoder = load_encoder(inputs)
    bank = torch.from_numpy(np.load(bank_path, allow_pickle=False))
    scores, maps, order, timings = [], [], [], []
    with torch.inference_mode(), Archive(inputs/"VisA.tar", list(earlier.values()) + samples) as archive:
        prints = [fingerprint(archive.load(path, "RGB")) for path in earlier]
        for sample in samples:
            im = archive.load(sample.image, "RGB")
            if near_reference(fingerprint(im), prints):
                raise ValueError("Possible duplicate across final and earlier products")
            x = pixels(im)
            t = perf_counter()
            score, heat = graph.run(["score","probability"], {"image": x[None].numpy()})
            timings.append(1000 * (perf_counter()-t))
            base_score, base_heat = anomaly_scores(embeddings(encoder, [x]), bank)
            scores.append(np.r_[score,base_score])
            maps.append(np.concatenate((heat,base_heat)))
            order.append({"image": sample.image, "product": sample.product, "size": list(im.size)})
    values = np.stack(maps)
    if not np.isfinite(values).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid final prediction")
    np.savez_compressed(folder/"outputs.npz", scores=np.stack(scores), maps=values, latency_ms=timings)
    write_json(folder/"order.json", order)
    write_json(folder/"frozen.json", {"frozen_at_utc": utc(), "protocol_sha256": file_hash(output/"protocol.json"),
               "files": {n:file_hash(folder/n) for n in ("outputs.npz","order.json")},
               "earlier_images_checked":len(earlier),"near_matches":0,"images":len(samples),
               "prediction_wall_seconds":perf_counter()-start})


def threshold_metrics(maps, totals, positives, threshold):
    selected = maps >= threshold
    tp_by_image = (positives * selected).sum(axis=(1,2))
    positives_by_image = positives.sum(axis=(1,2))
    selected_pixels = int(totals[selected].sum())
    tp = int(tp_by_image.sum())
    result = calibration_metrics(maps, totals, positives, threshold)
    result.update(true_positive_pixels=tp, pixel_precision=tp/selected_pixels if selected_pixels else None,
                  pixel_recall=tp/int(positives.sum()) if positives.sum() else None,
                  normal_images_with_foreground=int((selected.any(axis=(1,2)) & (positives_by_image == 0)).sum()),
                  defective_images_without_overlap=int(((tp_by_image == 0) & (positives_by_image > 0)).sum()))
    return result


def score(inputs, output):
    spec = verify_lock(output)
    folder = output / "predictions"
    frozen = read(folder/"frozen.json")
    if (file_hash(output/"protocol.json") != frozen["protocol_sha256"] or
            any(file_hash(folder/n) != h for n,h in frozen["files"].items()) or file_hash(inputs/"VisA.tar") != ARCHIVE_SHA):
        raise ValueError("Frozen inputs differ")
    dest = output / "scoring"
    dest.mkdir(exist_ok=False)
    opened = utc()
    rows = read_split(inputs/"1cls.csv")
    samples = selected_images(rows)
    order = read(folder/"order.json")
    if [(s.image,s.product) for s in samples] != [(r["image"],r["product"]) for r in order]:
        raise ValueError("Frozen prediction order differs from split")
    by_path = {r["image"]:r for r in rows if r["object"] in RESERVED and r["split"]=="test"}
    samples = [Sample(s.product,s.image,by_path[s.image]["mask"],by_path[s.image]["label"]=="anomaly") for s in samples]
    if any(bool(s.mask)!=s.anomaly or by_path[s.image]["label"] not in {"normal","anomaly"} for s in samples):
        raise ValueError("Invalid final labels")
    totals, positives = [], []
    with Archive(inputs/"VisA.tar", samples) as archive:
        for sample, row in zip(samples,order):
            mask = archive.load(sample.mask,"mask") if sample.mask else None
            t,p = native_counts(tuple(row["size"]), mask)
            totals.append(t)
            positives.append(p)
    totals, positives = np.stack(totals), np.stack(positives)
    labels = np.array([s.anomaly for s in samples])
    product = np.array([s.product for s in samples])
    np.savez_compressed(dest/"targets.npz", totals=totals, positives=positives, labels=labels, products=product)
    with np.load(folder/"outputs.npz", allow_pickle=False) as saved:
        scores, maps, ms = saved["scores"], saved["maps"], saved["latency_ms"]
    arms = {}
    for j,arm in enumerate(ARMS):
        arms[arm] = {}
        for group in ("pooled",*RESERVED):
            keep = np.ones(len(labels),dtype=bool) if group=="pooled" else product==group
            y, t, p, heat = labels[keep], totals[keep], positives[keep], maps[keep,j]
            result = summarize(y,scores[keep,j],heat,t,p)
            result["pixel_prevalence"] = float(p.sum()/t.sum())
            if arm in spec["heads"]:
                result.update(threshold_metrics(heat,t,p,spec["heads"][arm]["threshold"]))
            arms[arm][group] = result
    write_json(dest/"report.json", {"protocol":spec["protocol"], "primary":spec["primary"],
               "protocol_sha256":frozen["protocol_sha256"], "spec_sha256":file_hash(SPEC),
               "source_graph_sha256":spec["graph_sha256"], "predictions":frozen,
               "targets_sha256":file_hash(dest/"targets.npz"), "targets_opened_at_utc":opened,
               "onnx_cpu_ms":{"images":len(ms),"threads":4,"median":float(np.median(ms)),"p95":float(np.percentile(ms,95))},
               "heads":spec["heads"], "arms":arms,"completed_at_utc":utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","predict","score"))
    for name in ("inputs","output","exported","bank"):
        parser.add_argument("--"+name,type=Path,required=name in {"inputs","output"})
    a = parser.parse_args()
    if a.stage=="score":
        score(a.inputs,a.output)
    else:
        if a.exported is None or a.bank is None:
            parser.error("--exported and --bank are required for this stage")
        (freeze if a.stage=="freeze" else predict)(a.inputs,a.exported,a.bank,a.output)

if __name__=="__main__":
    main()
