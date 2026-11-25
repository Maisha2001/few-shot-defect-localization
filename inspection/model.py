from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .data import file_hash

MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
WEIGHTS_SHA = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
CONFIG_SHA = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def pixels(image):
    square = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    return (torch.from_numpy(np.array(square)).permute(2, 0, 1).float().div(255) - MEAN) / STD

def load_encoder(directory):
    from transformers import Dinov2Model
    directory = Path(directory)
    for name, expected in [("config.json", CONFIG_SHA), ("model.safetensors", WEIGHTS_SHA)]:
        if file_hash(directory / name) != expected:
            raise ValueError(f"Unexpected model input {name}")
    return Dinov2Model.from_pretrained(directory, local_files_only=True, use_safetensors=True).eval()

@torch.inference_mode()
def embeddings(model, batch):
    patches = model(pixel_values=torch.stack(batch)).last_hidden_state[:, 1:]
    if patches.shape[1:] != (256, 384):
        raise ValueError("Unexpected DINOv2 patch dimensions")
    return F.normalize(patches, dim=-1)

def reference_indices(image_path, patches=256, count=16):
    keys = [sha256(f"{image_path}:{i}".encode()).digest() for i in range(patches)]
    return sorted(range(patches), key=keys.__getitem__)[:count]

def anomaly_scores(patches, bank):
    distance = (1 - (patches @ bank.T).amax(dim=-1)).clamp(0, 2)
    image_scores = distance.topk(4, dim=-1).values.mean(dim=-1)
    maps = F.interpolate(distance.reshape(-1, 1, 16, 16), size=(56, 56), mode="bilinear", align_corners=False)
    return image_scores.numpy(), maps[:, 0].numpy()
