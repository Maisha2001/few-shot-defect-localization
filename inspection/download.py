"""Download and verify the original VisA archive, split, license and model."""
import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from .data import SPLIT_SHA, file_hash
from .model import CONFIG_SHA, MODEL_REVISION, WEIGHTS_SHA
from .evaluate import ARCHIVE_SHA

SPLIT_BASE = "https://raw.githubusercontent.com/amazon-science/spot-diff/2a692ab575001cbde74d402d897a7286086c6199/"
MODEL_BASE = f"https://huggingface.co/facebook/dinov2-small/resolve/{MODEL_REVISION}/"
INPUTS = {
    "LICENSE-DATASET": (SPLIT_BASE + "LICENSE-DATASET", "9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94"),
    "1cls.csv": (SPLIT_BASE + "split_csv/1cls.csv", SPLIT_SHA),
    "config.json": (MODEL_BASE + "config.json", CONFIG_SHA),
    "model.safetensors": (MODEL_BASE + "model.safetensors", WEIGHTS_SHA),
    "VisA.tar": ("https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar", ARCHIVE_SHA),
}

def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in INPUTS.items():
        dest = directory / name
        if dest.exists():
            if file_hash(dest) != expected:
                raise ValueError(f"Unexpected contents of {dest}")
            continue
        digest = sha256()
        partial = dest.with_suffix(dest.suffix + ".part")
        with partial.open("xb") as out:
            try:
                with urlopen(url, timeout=120) as src:
                    for block in iter(lambda: src.read(4 << 20), b""):
                        out.write(block)
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Unexpected checksum for {name}")
            except BaseException:
                out.close()
                partial.unlink(missing_ok=True)
                raise
        partial.rename(dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    fetch(parser.parse_args().directory)
