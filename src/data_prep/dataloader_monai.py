from __future__ import annotations

import pandas as pd
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset

from src.data_prep.transforms_monai import get_transforms

def load_manifest(manifest_path: str, split: str):
    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split].copy()

    records = []
    for _, row in df.iterrows():
        records.append({
            "image": row["image"],
            "label": int(row["label_id"]),
        })
    return records

def make_loader(
    manifest_path: str,
    split: str,
    batch_size: int = 1,
    num_workers: int = 0,
    cache: bool = True,
):
    records = load_manifest(manifest_path, split)
    xforms = get_transforms(train=(split == "train"))

    if cache:
        ds = CacheDataset(data=records, transform=xforms, cache_rate=1.0)
    else:
        ds = Dataset(data=records, transform=xforms)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=False,
    )