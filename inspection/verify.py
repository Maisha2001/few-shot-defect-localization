"""Check a report's structure and optionally verify the local evidence files."""
import argparse
import json
import math
from pathlib import Path

from .data import DEVELOPMENT, FIT, RESERVED, SPLIT_SHA, file_hash
from .evaluate import ARCHIVE_SHA
from .model import MODEL_REVISION, WEIGHTS_SHA

def verify(report, artifacts=None):
    if (report["protocol"],report["archive_sha256"],report["split_sha256"],report["weight_sha256"],report["model_revision"]) != (
        "visa-grouped-patch-v1",ARCHIVE_SHA,SPLIT_SHA,WEIGHTS_SHA,MODEL_REVISION):
        raise ValueError("Unexpected protocol or source revision")
    if tuple(report["reference_products"]) != FIT or tuple(report["development_products"]) != DEVELOPMENT or tuple(report["reserved_products"]) != RESERVED:
        raise ValueError("Product partitions differ")
    if report["reference_images"] != dict.fromkeys(FIT,32) or report["reference_patches"] != 3072:
        raise ValueError("Unexpected memory bank")
    if set(report["by_product"]) != set(DEVELOPMENT):
        raise ValueError("Incomplete evaluation")
    groups = list(report["by_product"].values())
    for key in ("images","anomalous_images","native_pixels","defect_pixels"):
        if sum(group[key] for group in groups) != report["pooled"][key]:
            raise ValueError("Inconsistent pooled counts")
    for group in [report["pooled"], *groups]:
        if not (0 < group["anomalous_images"] < group["images"] and 0 < group["defect_pixels"] < group["native_pixels"]):
            raise ValueError("Missing class")
        for level in ("image","pixel"):
            for metric in ("auroc","average_precision"):
                v = group[level][metric]
                if not math.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError("Invalid ranking metric")
    names = {"samples.json","reference-bank.npy","development.npz"}
    if set(report["artifacts"]) != names:
        raise ValueError("Incomplete evidence manifest")
    if artifacts:
        for name in names:
            if file_hash(artifacts / name) != report["artifacts"][name]:
                raise ValueError(f"Evidence differs: {name}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--artifacts", type=Path)
    args = p.parse_args()
    verify(json.loads(args.report.read_text()), args.artifacts)
    print("Baseline evidence verified")
