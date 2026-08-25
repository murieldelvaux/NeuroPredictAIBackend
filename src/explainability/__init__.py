"""Explainability module — Grad-CAM 3D for volumetric MRI.

Provides visual explanations for CNN predictions on 3D NIfTI scans,
fulfilling the XAI requirement from the NeuroPredict AI clinical discovery.

Usage::

    from src.explainability import GradCAM3D
    from src.inference import load_model
    from src.models.cnn_3d import SimpleResNet3D

    model, cfg, device, _ = load_model("checkpoints/best_model.pth")
    cam = GradCAM3D(model, target_layer=model.encoder[-3])  # last Conv3d block
    heatmap = cam.generate(image_tensor, class_idx=2)  # 2 = DEM
"""
from src.explainability.gradcam_3d import GradCAM3D

__all__ = ["GradCAM3D"]
