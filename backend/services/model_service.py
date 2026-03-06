import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model import RUPNet
from core.config import CHECKPOINT_MAP, DEVICE

_models: dict[str, RUPNet] = {}


def get_model(name: str) -> RUPNet:
    if name not in _models:
        path = CHECKPOINT_MAP.get(name)
        if path is None:
            raise ValueError(f"Unknown model: {name}")
        m = RUPNet()
        state = torch.load(path, map_location=DEVICE)
        m.load_state_dict(state, strict=True)
        m.to(DEVICE)
        m.eval()
        _models[name] = m
    return _models[name]


def predict(tensor, name: str):
    model = get_model(name)
    with torch.no_grad():
        return model(tensor, heatmap=None)
