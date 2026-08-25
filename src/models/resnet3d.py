"""ResNet3D variants for volumetric MRI classification.

Provides three model sizes to trade-off accuracy vs. compute:

- ``ResNet3DTiny``  — 3 blocks, lightweight, good for prototyping / CPU
- ``ResNet3DSmall`` — 4 blocks, balanced (recommended for training on GPU)
- ``ResNet3DBase``  — 5 blocks, highest capacity

All variants share the same ``ResidualBlock3D`` building block and are
compatible with GradCAM3D (hook on the last conv layer of the final block).

Usage::

    from src.models.resnet3d import ResNet3DSmall

    model = ResNet3DSmall(in_channels=1, num_classes=3)

    # For Grad-CAM, target the last Conv3d of the deepest block:
    from src.explainability import GradCAM3D
    cam = GradCAM3D(model, target_layer=model.layer3[-1].conv2)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """Basic 3D residual block: two Conv3d layers with a skip connection.

    If ``stride > 1`` or ``in_ch != out_ch``, a 1×1×1 projection shortcut
    is applied to match dimensions before the addition.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        stride: Stride for the first Conv3d (downsamples spatial dims).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet3DBase(nn.Module):
    """Internal base class — use ResNet3DTiny / ResNet3DSmall / ResNet3DBase."""

    def __init__(
        self,
        channels: list[int],
        in_channels: int = 1,
        num_classes: int = 3,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        # Stem: initial conv to reduce spatial size quickly
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages — each stage halves spatial dims via stride=2
        layers: list[nn.Module] = []
        for i in range(1, len(channels)):
            stride = 1 if i == 1 else 2
            layers.append(ResidualBlock3D(channels[i - 1], channels[i], stride=stride))
        self.layers = nn.Sequential(*layers)

        # Store individual stages for Grad-CAM targeting
        for i, layer in enumerate(layers):
            setattr(self, f"layer{i + 1}", layer)

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        return self.head(x)


class ResNet3DTiny(_ResNet3DBase):
    """3-stage ResNet3D — fastest, lowest memory, suitable for CPU inference.

    Channels: 32 → 64 → 128.
    ~1.2M parameters.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, dropout: float = 0.4) -> None:
        super().__init__(
            channels=[32, 64, 128],
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )


class ResNet3DSmall(_ResNet3DBase):
    """4-stage ResNet3D — recommended for GPU training on ADNI/OASIS.

    Channels: 32 → 64 → 128 → 256.
    ~4.8M parameters.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, dropout: float = 0.4) -> None:
        super().__init__(
            channels=[32, 64, 128, 256],
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )


class ResNet3DBase(_ResNet3DBase):
    """5-stage ResNet3D — highest capacity, requires GPU with >= 12 GB VRAM.

    Channels: 32 → 64 → 128 → 256 → 512.
    ~18M parameters.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, dropout: float = 0.4) -> None:
        super().__init__(
            channels=[32, 64, 128, 256, 512],
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
