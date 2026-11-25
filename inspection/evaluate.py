import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from .data import Archive, DEVELOPMENT, FIT, RESERVED, SPLIT_SHA, file_hash, protocol
from .metrics import native_counts, summarize
from .model import MODEL_REVISION, WEIGHTS_SHA, anomaly_scores, embeddings, load_encoder, pixels, reference_indices

ARCHIVE_SHA = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def chunks(seq, n=8):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def fingerprint(image):
    gray = image.convert("L")
    thumb = np.asarray(gray.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    small = np.asarray(gray.resize((9, 8), Image.Resampling.BOX))
    bits = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
    return bits, thumb

def near_reference(query, references):
    bits, thumb = query
    for ref_bits, ref_thumb in references:
        if (ref_bits ^ bits).bit_count() <= 4 and np.mean((thumb - ref_thumb) ** 2) <= 9:
            return True
    return False

def run(archive_path, split_path, model_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    torch.set_num_threads(4)
    if file_hash(archive_path) != ARCHIVE_SHA:
        raise ValueError("Unexpected VisA archive")
    fit, dev = protocol(split_path)
    write_json(output / "samples.json", {"reference": [asdict(s) for s in fit], "development": [asdict(s) for s in dev]})
    encoder = load_encoder(model_dir)
    refs, reference_prints = [], []
    scores, maps, totals, counts = [], [], [], []
    with Archive(archive_path, fit + dev) as archive:
        for group in chunks(fit):
            images = [archive.load(s.image, "RGB") for s in group]
            reference_prints.extend(fingerprint(im) for im in images)
            features = embeddings(encoder, [pixels(im) for im in images])
            refs.extend(f[reference_indices(s.image)] for s, f in zip(group, features))
        bank = torch.cat(refs)
        np.save(output / "reference-bank.npy", bank.numpy(), allow_pickle=False)
        fit_finished = perf_counter()
        for group in chunks(dev):
            images = [archive.load(s.image, "RGB") for s in group]
            for im, s in zip(images, group):
                if near_reference(fingerprint(im), reference_prints):
                    raise ValueError(f"Possible fit/development duplicate: {s.image}")
            features = embeddings(encoder, [pixels(im) for im in images])
            sc, heat = anomaly_scores(features, bank)
            scores.extend(sc)
            maps.extend(heat)
            for im, s in zip(images, group):
                mask = archive.load(s.mask, "mask") if s.anomaly else None
                total, pos = native_counts(im.size, mask)
                totals.append(total)
                counts.append(pos)
        scored = perf_counter()
    arrays = dict(labels=np.array([s.anomaly for s in dev]), image_scores=np.array(scores),
                  maps=np.stack(maps), totals=np.stack(totals), positives=np.stack(counts))
    np.savez_compressed(output / "development.npz", **arrays)
    products = np.array([s.product for s in dev])
    groups = {p: summarize(**{k:v[products == p] for k,v in arrays.items()}) for p in DEVELOPMENT}
    report = {"protocol":"visa-grouped-patch-v1", "archive_sha256":ARCHIVE_SHA, "split_sha256":SPLIT_SHA,
              "encoder":"facebook/dinov2-small", "model_revision":MODEL_REVISION, "weight_sha256":WEIGHTS_SHA,
              "input_size":224, "grid":[56,56], "reference_products":list(FIT),
              "reference_images":dict(Counter(s.product for s in fit)), "reference_patches":len(bank),
              "reserved_products":list(RESERVED), "development_products":list(DEVELOPMENT),
              "cross_split_near_matches":0, "pooled":summarize(**arrays), "by_product":groups,
              "artifacts":{n:file_hash(output / n) for n in ["samples.json","reference-bank.npy","development.npz"]},
              "runtime":{"device":"cpu","threads":torch.get_num_threads(),"torch":str(torch.__version__),
                         "python":platform.python_version(),"platform":platform.system(),
                         "load_and_fit_seconds":fit_finished-start,"development_seconds":scored-fit_finished,
                         "total_seconds":perf_counter()-start}}
    write_json(output / "report.json", report)
    return report

def main():
    p = argparse.ArgumentParser(description="Evaluate frozen DINOv2 patch retrieval on product-disjoint VisA images.")
    for arg in ("archive","split","model","output"):
        p.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    result = run(args.archive, args.split, args.model, args.output)
    print(json.dumps(result["pooled"], indent=2))

if __name__ == "__main__":
    main()
