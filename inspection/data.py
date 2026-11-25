"""Product-disjoint development protocol for the original VisA archive."""
from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tarfile

from PIL import Image

FIT = ("candle", "cashew", "pcb1", "pcb2", "pcb3", "pcb4")
DEVELOPMENT = ("capsules", "macaroni1", "macaroni2")
RESERVED = ("chewinggum", "fryum", "pipe_fryum")
SPLIT_SHA = "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"

@dataclass(frozen=True)
class Sample:
    product: str
    image: str
    mask: str
    anomaly: bool

def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()

def protocol(split_path):
    if file_hash(split_path) != SPLIT_SHA:
        raise ValueError("Expected the pinned VisA one-class split CSV")
    with Path(split_path).open(newline="", encoding="utf-8") as source:
        return select_samples(csv.DictReader(source))

def select_samples(rows, per_product=32):
    fit, dev = [], []
    paths = set()
    for row in rows:
        product = row["object"]
        if product not in FIT + DEVELOPMENT + RESERVED:
            raise ValueError(f"Unknown product {product}")
        if product in RESERVED:
            continue
        image = row["image"]
        if image in paths:
            raise ValueError("Repeated image path")
        paths.add(image)
        if row["split"] not in {"train", "test"} or row["label"] not in {"anomaly", "normal"}:
            raise ValueError("Unrecognized split or label")
        s = Sample(product, image, row["mask"], row["label"] == "anomaly")
        if product in FIT and row["split"] == "train":
            if s.anomaly or s.mask:
                raise ValueError("Reference images must be normal")
            fit.append(s)
        elif product in DEVELOPMENT and row["split"] == "test":
            if bool(s.mask) != s.anomaly:
                raise ValueError("Anomaly and mask disagree")
            dev.append(s)
    selected = []
    for product in FIT:
        rows = [s for s in fit if s.product == product]
        if len(rows) < per_product:
            raise ValueError(f"Missing normal references for {product}")
        selected.extend(sorted(rows, key=lambda s: sha256(s.image.encode()).digest())[:per_product])
    if set(Counter(s.product for s in dev)) != set(DEVELOPMENT):
        raise ValueError("Missing development product")
    return selected, sorted(dev, key=lambda s: s.image)

class Archive:
    """Decode only named, regular members. Archive paths never reach the filesystem."""
    def __init__(self, path, samples):
        wanted = {name for s in samples for name in (s.image, s.mask) if name}
        self.tar = tarfile.open(path, "r:")
        self.members = {}
        try:
            for m in self.tar:
                if m.name in wanted:
                    if m.name in self.members or not m.isfile() or not 0 < m.size <= 32 << 20:
                        raise ValueError(f"Invalid archive member {m.name}")
                    self.members[m.name] = m
            if self.members.keys() != wanted:
                raise ValueError("Missing selected images or masks")
        except BaseException:
            self.tar.close()
            raise

    def load(self, name, mode):
        with self.tar.extractfile(self.members[name]) as raw, Image.open(raw) as im:
            if mode == "mask":
                if im.mode not in ("L", "P", "1"):
                    raise ValueError(f"Unexpected mask encoding {im.mode}")
                return im.copy()
            return im.convert(mode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.tar.close()
