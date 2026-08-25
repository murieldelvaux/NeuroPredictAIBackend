from __future__ import annotations

from pathlib import Path
import yaml
import torch


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device_name: str) -> torch.device:
    name = device_name.lower()
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
