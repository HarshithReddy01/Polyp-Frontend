import os
from pathlib import Path

import torch as _torch

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR

CHECKPOINT_MAP = {
    "Kvasir-Seg": MODEL_DIR / "checkpoint-Kvasir-Seg.pth",
    "BKAI-IGH":   MODEL_DIR / "checkpoint-BKAI-IGH.pth",
}

IMAGE_SIZE = (256, 256)
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
