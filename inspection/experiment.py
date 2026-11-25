"""Evaluate annotation budgets with disjoint source, calibration and evaluation products."""
import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from .adapt import calibrate, calibration_metrics, fit_head, next_image, probabilities
from .data import Archive, DEVELOPMENT, FIT, Sample, file_hash, protocol
from .evaluate import ARCHIVE_SHA, chunks, fingerprint, near_reference, write_json
from .metrics import native_counts, summarize
from .model import MODEL_REVISION, embeddings, load_encoder, pixels
from .verify import verify

BUDGETS = (4,8,16,32)
CALIBRATION = "cashew"

def order_key(image):
    return sha256(("visa-adaptation-v1/" + image).encode()).digest()

def split_roles(split_path):
    _, dev = protocol(split_path)
    with split_path.open(newline="",encoding="utf-8") as f:
        grouped = {p:[] for p in FIT}
        for row in csv.DictReader(f):
            if row["object"] in grouped and row["split"]=="test":
                grouped[row["object"]].append(Sample(row["object"],row["image"],row["mask"],row["label"]=="anomaly"))
    for samples in grouped.values():
        samples.sort(key=lambda s:order_key(s.image))
    cal = grouped.pop(CALIBRATION)[:8]
    pool = sorted([s for samples in grouped.values() for s in samples[:48]],key=lambda s:order_key(s.image))
    if len(cal)!=8 or len(pool)!=240:
        raise ValueError("Incomplete acquisition or calibration pool")
    return pool,cal,dev

def extract(archive, model, samples, previous):
    vectors, prints, sizes = [],[],[]
    for batch in chunks(samples):
        images=[archive.load(s.image,"RGB") for s in batch]
        for im in images:
            fp=fingerprint(im)
            if near_reference(fp,previous):
                raise ValueError("Possible near duplicate across product roles")
            prints.append(fp)
            sizes.append(im.size)
        vectors.append(embeddings(model,[pixels(im) for im in images]).numpy())
    return np.concatenate(vectors),prints,sizes

def mask_counts(archive, sample, size, grid):
    mask=archive.load(sample.mask,"mask") if sample.mask else None
    return native_counts(size,mask,cells=grid)

def prepare(inputs,baseline,out):
    out.mkdir(parents=True,exist_ok=False)
    report=json.loads((baseline/"report.json").read_text())
    verify(report,baseline)
    if file_hash(inputs/"VisA.tar")!=ARCHIVE_SHA:
        raise ValueError("VisA checksum mismatch")
    pool,cal,dev=split_roles(inputs/"1cls.csv")
    baseline_manifest=json.loads((baseline/"samples.json").read_text())
    if [asdict(s) for s in dev]!=baseline_manifest["development"]:
        raise ValueError("Development order differs from baseline")
    model=load_encoder(inputs)
    prior=[]
    meta={}
    with Archive(inputs/"VisA.tar",pool+cal+dev) as archive:
        for role,samples in (("pool",pool),("calibration",cal),("development",dev)):
            x,fp,sizes=extract(archive,model,samples,prior)
            np.save(out/(role+".npy"),x,allow_pickle=False)
            prior.extend(fp)
            meta[role]=[{"image":s.image,"size":sz} for s,sz in zip(samples,sizes)]
    write_json(out/"roles.json",meta)
    write_json(out/"manifest.json", {p.name:file_hash(p) for p in sorted(out.iterdir())})

def train(inputs,baseline,cache,out):
    out.mkdir(parents=True,exist_ok=False)
    for name,digest in json.loads((cache/"manifest.json").read_text()).items():
        if file_hash(cache/name)!=digest:raise ValueError("Feature cache changed")
    pool,cal,dev=split_roles(inputs/"1cls.csv")
    meta=json.loads((cache/"roles.json").read_text())
    for role,samples in (("pool",pool),("calibration",cal),("development",dev)):
        if [s.image for s in samples]!=[s["image"] for s in meta[role]]:raise ValueError("Sample order changed")
    x=np.load(cache/"pool.npy",mmap_mode="r")
    xc=np.load(cache/"calibration.npy",mmap_mode="r")
    xd=np.load(cache/"development.npy",mmap_mode="r")
    desc=x.mean(1)
    desc/=np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-12)
    trace={}
    with Archive(inputs/"VisA.tar",pool+cal) as archive:
        tc,pc=map(np.stack,zip(*(mask_counts(archive,s,tuple(m["size"]),56) for s,m in zip(cal,meta["calibration"]))))
        for policy in ("random","active"):
            selected=[];totals=[];positives=[];head=None
            for budget in BUDGETS:
                while len(selected)<budget:
                    j=next_image(policy,selected,x,head,desc)
                    total,pos=mask_counts(archive,pool[j],tuple(meta["pool"][j]["size"]),16)
                    selected.append(j);totals.append(total.ravel());positives.append(pos.ravel())
                head=fit_head(x[selected],np.stack(totals),np.stack(positives))
                calibrate(head,xc,tc,pc)
                key=f"{policy}-{budget}"
                scores,heat=probabilities(head,xd)
                np.savez_compressed(out/(key+".npz"),scores=scores,prob=heat)
                write_json(out/(key+".json"),head.to_dict())
                trace[key]={"acquisition_indices":list(selected),"masks_with_defects":int(sum(a.sum()>0 for a in positives)),
                            "head_sha256":file_hash(out/(key+".json")),"predictions_sha256":file_hash(out/(key+".npz"))}
    write_json(out/"trace.json",trace)
    # All choices and predictions exist before loading development targets.
    base=json.loads((baseline/"report.json").read_text())
    verify(base,baseline)
    products=np.array([s.product for s in dev])
    arms={}
    with np.load(baseline/"development.npz") as f:
        y,t,p=f["labels"],f["totals"],f["positives"]
    for key,rec in trace.items():
        with np.load(out/(key+".npz")) as f:score,prob=f["scores"],f["prob"]
        head=json.loads((out/(key+".json")).read_text())
        groups={}
        for name in ("pooled",*DEVELOPMENT):
            sel=np.ones(len(y),dtype=bool) if name=="pooled" else products==name
            group=summarize(y[sel],score[sel],prob[sel],t[sel],p[sel])
            group.update(calibration_metrics(prob[sel],t[sel],p[sel],head["threshold"]))
            groups[name]=group
        arms[key]={**rec,"threshold":head["threshold"],"groups":groups}
    result={"protocol":"visa-budget-v1","model_revision":MODEL_REVISION,"cache_manifest_sha256":file_hash(cache/"manifest.json"),
            "source_products":[p for p in FIT if p!=CALIBRATION],"pool_images":len(pool),"calibration_product":CALIBRATION,
            "calibration_images":len(cal),"calibration_anomalies":int((pc.sum((1,2))>0).sum()),
            "development_products":list(DEVELOPMENT),"reference_baseline":base["pooled"],"arms":arms,
            "deployment_arm":"active-32"}
    write_json(out/"report.json",result)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step",choices=("prepare","train"))
    for key in ("inputs","baseline","cache"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    start=time.perf_counter()
    if args.step=="prepare":prepare(args.inputs,args.baseline,args.cache)
    elif args.out is None:parser.error("--out is required for training")
    else:train(args.inputs,args.baseline,args.cache,args.out)
    print(f"{args.step} completed in {time.perf_counter()-start:.2f} seconds")

if __name__=="__main__":main()
