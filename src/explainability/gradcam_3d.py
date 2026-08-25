"""Grad-CAM 3D — volumetric gradient-weighted class activation mapping.

Produces a 3D heatmap (same spatial resolution as the input volume after
upsampling) highlighting which voxel regions most influenced the model's
prediction. This satisfies the XAI requirement from the clinical discovery:
"Doctors need explainable outputs, evidence-based correlations, and clear
interpretation of the prediction."

References:
    Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization. https://arxiv.org/abs/1610.02391

Usage::

    import torch
    from src.models.cnn_3d import SimpleResNet3D
    from src.explainability.gradcam_3d import GradCAM3D

    model = SimpleResNet3D(in_channels=1, num_classes=3)
    model.eval()

    # Target the last Conv3d layer inside the encoder
    cam = GradCAM3D(model, target_layer=model.encoder[6])  # Conv3d at index 6

    image = torch.randn(1, 1, 128, 128, 128)  # (B, C, D, H, W)
    heatmap = cam.generate(image, class_idx=2)  # class_idx 2 = AD
    # heatmap.shape == (128, 128, 128), values in [0, 1]
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM3D:
    """Grad-CAM for 3D CNN models (e.g. SimpleResNet3D on MRI volumes).

    Registers forward and backward hooks on a target Conv3d layer to capture
    activations and gradients, then computes a weighted sum to produce a
    class-discriminative volumetric heatmap.

    Args:
        model: A PyTorch model in eval() mode.
        target_layer: The Conv3d (or similar) layer to hook. Should be the
            last convolutional layer before the classifier for best results.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activation(
        self,
        module: nn.Module,  # noqa: ARG002
        input: tuple,  # noqa: ARG002
        output: torch.Tensor,
    ) -> None:
        """Forward hook: cache the layer output (activations)."""
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,  # noqa: ARG002
        grad_input: tuple,  # noqa: ARG002
        grad_output: tuple,
    ) -> None:
        """Backward hook: cache the gradient flowing through the layer."""
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int,
        upsample_to_input: bool = True,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a given class.

        Args:
            image_tensor: Input volume of shape (1, C, D, H, W). Must be on
                the same device as the model and require_grad is handled
                internally.
            class_idx: Target class index (0=CN, 1=MCI, 2=AD).
            upsample_to_input: If True, bilinearly upsample the CAM back to
                the spatial size of the input volume. Set to False to keep
                the raw CAM at the feature-map resolution.

        Returns:
            numpy.ndarray of shape (D, H, W) with values in [0.0, 1.0].
            Higher values indicate regions more relevant to the prediction.
        """
        # Ensure the model can propagate gradients
        self.model.zero_grad()

        # Forward pass — hooks capture activations
        output: torch.Tensor = self.model(image_tensor)  # (1, num_classes)

        # Backward pass for the target class — hooks capture gradients
        score = output[0, class_idx]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "GradCAM3D: activations or gradients were not captured. "
                "Check that target_layer is a valid layer of the model."
            )

        # Global Average Pooling over spatial dims → importance weights
        # Shape: (1, num_channels, D, H, W) → (1, num_channels, 1, 1, 1)
        weights = self._gradients.mean(dim=[2, 3, 4], keepdim=True)

        # Weighted combination of activation maps
        # Shape: (1, 1, D_feat, H_feat, W_feat)
        cam: torch.Tensor = (weights * self._activations).sum(dim=1, keepdim=True)

        # ReLU: keep only regions with positive influence
        cam = F.relu(cam)

        # Upsample to match input spatial resolution
        if upsample_to_input:
            target_size = image_tensor.shape[2:]  # (D, H, W)
            cam = F.interpolate(
                cam,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

        # Normalise to [0, 1] and convert to numpy
        cam_np: np.ndarray = cam.squeeze().cpu().numpy()  # (D, H, W)
        cam_min, cam_max = cam_np.min(), cam_np.max() #
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def generate_overlay(
        self,
        image_tensor: torch.Tensor,
        class_idx: int,
        threshold: float = 0.5,
    ) -> dict:
        """Generate heatmap + a binary mask of high-activation regions.

        Useful for the clinical dashboard to highlight suspicious voxels.

        Args:
            image_tensor: Input volume (1, C, D, H, W).
            class_idx: Target class index.
            threshold: Values above this are considered high-activation.

        Returns:
            dict with:
                - ``heatmap`` (np.ndarray): Normalised CAM, shape (D, H, W).
                - ``mask`` (np.ndarray): Boolean mask, shape (D, H, W).
                - ``activation_volume_pct`` (float): % of brain voxels highlighted.
        """
        heatmap = self.generate(image_tensor, class_idx, upsample_to_input=True)
        mask = heatmap >= threshold
        pct = float(mask.sum()) / float(mask.size) * 100.0

        return {
            "heatmap": heatmap,
            "mask": mask,
            "activation_volume_pct": round(pct, 2),
        }

    def remove_hooks(self) -> None:
        """Remove registered hooks (call when done to avoid memory leaks)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:  # noqa: BLE001
            pass
