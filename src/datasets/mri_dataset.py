from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, metadata_csv: str, split: str, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = Path(row["image"])
        label = int(row["label_id"])
        sample = {"image": str(image_path), "label": label}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
