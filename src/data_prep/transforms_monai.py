from __future__ import annotations

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
    RandAffined,
)

def get_transforms(train: bool = False):
    """
    Pipeline MVP de preprocess para MRI T1 3D.
    - Padroniza orientação e spacing
    - Normaliza intensidade
    - Recorta o foreground e padroniza shape
    - (Opcional) augment leve no treino
    """
    xforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),  # -> [C, H, W, D]
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear",)),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0, a_max=3000,   # faixa ampla (ajustável)
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["image"]),
    ]

    if train:
        xforms = xforms[:-1] + [  # insere augment antes do ToTensor
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandAffined(
                keys=["image"],
                prob=0.25,
                rotate_range=(0.05, 0.05, 0.05),
                translate_range=(5, 5, 5),
                scale_range=(0.05, 0.05, 0.05),
                mode=("bilinear",),
                padding_mode="border",
            ),
            ToTensord(keys=["image"]),
        ]

    return Compose(xforms)