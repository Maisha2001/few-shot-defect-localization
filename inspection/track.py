"""Save a local, offline W&B record of the published experiment results."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    import wandb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    final = json.loads((root/"results/final.json").read_text())
    dev = json.loads((root/"results/adaptation.json").read_text())
    values = {}
    for arm,groups in final["arms"].items():
        for group,metrics in groups.items():
            for k in ("image","pixel"):
                for key,value in metrics[k].items():
                    values[f"final/{arm}/{group}/{k}_{key}"] = value
            for k in ("dice","pixel_recall","pixel_precision","pixel_brier","pixel_ece10"):
                if k in metrics:
                    values[f"final/{arm}/{group}/{k}"] = metrics[k]
    for arm,result in dev["arms"].items():
        values[f"development/{arm}/pixel_ap"] = result["groups"]["pooled"]["pixel"]["average_precision"]
    with wandb.init(project="few-shot-defect-localization", mode="offline", dir=str(args.output),
                    config={"protocol":final["protocol"],"primary":final["primary"],
                            "protocol_sha256":final["protocol_sha256"],"graph_sha256":final["source_graph_sha256"]}) as run:
        run.log(values)
        for rel in ("protocols/cross-product.json","results/final.json","results/final-audit.json"):
            dest = Path(run.dir)/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/rel,dest)
        print(f"Offline experiment {run.id}: {len(values)} metrics", flush=True)

if __name__=="__main__":
    main()
