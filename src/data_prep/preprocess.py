from __future__ import annotations

from pathlib import Path
import yaml

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    ResizeWithPadOrCropd,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandAffined,
)


def build_preprocessing_pipeline(config: dict, train: bool = False):
    base = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=tuple(config["preprocessing"]["spacing"]), mode=("bilinear",)),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config["preprocessing"]["intensity_range"][0],
            a_max=config["preprocessing"]["intensity_range"][1],
            b_min=config["preprocessing"]["normalized_range"][0],
            b_max=config["preprocessing"]["normalized_range"][1],
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=tuple(config["preprocessing"]["spatial_size"])),
    ]

    if train:
        augmentations = [
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.3, max_k=3),
            RandZoomd(keys=["image"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandAffined(
                keys=["image"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                scale_range=(0.1, 0.1, 0.1),
                mode="bilinear",
                padding_mode="zeros",
            ),
        ]
        base.extend(augmentations)

    base.append(ToTensord(keys=["image"]))
    return Compose(base)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
