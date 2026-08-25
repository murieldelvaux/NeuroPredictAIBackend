"""Inference pipeline — loads a trained checkpoint and runs prediction on a NIfTI file.

This module is the bridge between the trained model and the FastAPI backend.
Usage::

    from src.inference import load_model, predict_from_nifti

    model, cfg, device, runtime_calibration = load_model("checkpoints/best_model.pth")
    result = predict_from_nifti("path/to/scan.nii.gz", model, cfg, device, runtime_calibration)
    # result = {
    #   "classification": "MCI",
    #   "probabilities": {"CN": 0.12, "MCI": 0.71, "DEM": 0.17},
    #   "confidence": 0.71
    # }
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.data_prep.preprocess import build_preprocessing_pipeline
from src.models.cnn_3d import SimpleResNet3D
from src.utils.helpers import get_device, load_config
from src.utils.brain_ood import brain_ood_score

if TYPE_CHECKING:
    pass

CLASS_NAMES: list[str] = ["CN", "MCI", "DEM"]
CLASS_LABELS: dict[str, str] = {
    "CN": "Cognitive Normal",
    "MCI": "Mild Cognitive Impairment",
    "DEM": "Dementia due to Alzheimer's disease",
}


def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path = "config.yaml",
    force_cpu: bool = True,
) -> tuple[SimpleResNet3D, dict, torch.device, dict]:
    """Load a trained SimpleResNet3D from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint saved during training.
        config_path: Path to config.yaml (same one used during training).
        force_cpu: When True, always loads on CPU (recommended for API serving).

    Returns:
        Tuple of (model, config_dict, device, runtime_calibration).
    """
    cfg = load_config(config_path)
    device = torch.device("cpu") if force_cpu else get_device(cfg["training"]["device"])

    model = SimpleResNet3D(
        in_channels=cfg["model"]["input_channels"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and full checkpoint dict
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    runtime_calibration = {
        "confidence_threshold": float(checkpoint.get("confidence_threshold", 0.0)),
        "brain_ood_reference": checkpoint.get("brain_ood_reference"),
    }

    return model, cfg, device, runtime_calibration


def predict_from_nifti(
    nifti_path: str | Path,
    model: SimpleResNet3D,
    cfg: dict,
    device: torch.device,
    runtime_calibration: dict | None = None,
) -> dict:
    """Run inference on a single NIfTI MRI file.

    Args:
        nifti_path: Path to the .nii or .nii.gz file.
        model: Loaded and eval()-mode SimpleResNet3D.
        cfg: Config dict from load_config().
        device: torch.device to run inference on.

    Returns:
        dict with keys:
            - classification (str): Predicted class label ("CN", "MCI", or "DEM").
            - label_full (str): Human-readable label.
            - probabilities (dict): Per-class softmax probabilities.
            - confidence (float): Probability of the predicted class.
    """
    transform = build_preprocessing_pipeline(cfg, train=False)
    data = transform({"image": str(nifti_path)})
    image: torch.Tensor = data["image"].unsqueeze(0).to(device)  # (1, C, D, H, W)

    calib = runtime_calibration or {}
    brain_ref = calib.get("brain_ood_reference")
    if brain_ref is not None:
        ood = brain_ood_score(data["image"].cpu().numpy(), brain_ref)
        if ood["is_out_of_domain"]:
            return {
                "classification": "OUT_OF_DOMAIN",
                "label_full": "Scan appears out of training domain (possible non-brain MRI)",
                "confidence": 0.0,
                "probabilities": {},
                "ood": {
                    "score": round(float(ood["score"]), 4),
                    "threshold": round(float(ood["threshold"]), 4),
                    "hard_violations": {k: round(float(v), 4) for k, v in ood.get("hard_violations", {}).items()},
                    "metrics": {k: round(float(v), 4) for k, v in ood["metrics"].items()},
                    "z_by_metric": {k: round(float(v), 4) for k, v in ood["z_by_metric"].items()},
                },
            }

    with torch.no_grad():
        logits = model(image)  # (1, num_classes)
        probs: np.ndarray = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    conf_threshold = float(calib.get("confidence_threshold", 0.0))

    if conf_threshold > 0.0 and confidence < conf_threshold:
        return {
            "classification": "UNCERTAIN",
            "label_full": "Prediction uncertainty above calibrated limit",
            "probabilities": {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(CLASS_NAMES)
            },
            "confidence": round(confidence, 4),
            "confidence_threshold": round(conf_threshold, 4),
        }

    return {
        "classification": pred_class,
        "label_full": CLASS_LABELS[pred_class],
        "probabilities": {
            name: round(float(probs[i]), 4)
            for i, name in enumerate(CLASS_NAMES)
        },
        "confidence": round(confidence, 4),
        "confidence_threshold": round(conf_threshold, 4) if conf_threshold > 0.0 else None,
    }


def predict_batch(
    nifti_paths: list[str | Path],
    model: SimpleResNet3D,
    cfg: dict,
    device: torch.device,
    runtime_calibration: dict | None = None,
) -> list[dict]:
    """Convenience wrapper to run predict_from_nifti on multiple files."""
    return [predict_from_nifti(p, model, cfg, device, runtime_calibration=runtime_calibration) for p in nifti_paths]
