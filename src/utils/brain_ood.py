from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


DEFAULT_HARD_LIMITS = {
    "foreground_ratio": [0.02, 0.70],
    "largest_component_ratio": [0.60, 1.00],
    "lr_symmetry": [0.05, 1.00],
    "com_offset": [0.00, 0.45],
    "intensity_std": [0.01, 1.00],
    "p99_intensity": [0.05, 1.00],
}


def _safe_std(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    out = np.where(out < 1e-6, 1e-6, out)
    return out


def _extract_volume_metrics(volume: np.ndarray) -> dict[str, float]:
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape}")

    vmax = float(np.max(arr))
    if vmax <= 0:
        return {
            "foreground_ratio": 0.0,
            "largest_component_ratio": 0.0,
            "lr_symmetry": 0.0,
            "com_offset": 1.0,
            "intensity_std": 0.0,
            "p99_intensity": 0.0,
        }

    mask = arr > max(0.05 * vmax, 1e-6)
    foreground_ratio = float(mask.mean())

    labeled, ncomp = ndimage.label(mask)
    if ncomp > 0:
        comp_sizes = np.bincount(labeled.ravel())
        largest = int(comp_sizes[1:].max()) if len(comp_sizes) > 1 else 0
        largest_component_ratio = float(largest / max(mask.sum(), 1))
    else:
        largest_component_ratio = 0.0

    mid = arr.shape[0] // 2
    left = arr[:mid]
    right = np.flip(arr[-mid:], axis=0)
    if left.size == 0 or right.size == 0:
        lr_symmetry = 0.0
    else:
        l = left.reshape(-1)
        r = right.reshape(-1)
        l = (l - l.mean()) / (l.std() + 1e-6)
        r = (r - r.mean()) / (r.std() + 1e-6)
        lr_symmetry = float(np.clip(np.mean(l * r), -1.0, 1.0))

    if mask.any():
        com = np.array(ndimage.center_of_mass(mask), dtype=np.float32)
        center = (np.array(mask.shape, dtype=np.float32) - 1.0) / 2.0
        com_offset = float(np.linalg.norm(com - center) / (np.linalg.norm(center) + 1e-6))
    else:
        com_offset = 1.0

    return {
        "foreground_ratio": foreground_ratio,
        "largest_component_ratio": largest_component_ratio,
        "lr_symmetry": lr_symmetry,
        "com_offset": com_offset,
        "intensity_std": float(arr.std()),
        "p99_intensity": float(np.quantile(arr, 0.99)),
    }


def fit_brain_reference(
    volumes: list[np.ndarray],
    z_threshold: float = 2.5,
    percentile_threshold: float = 95.0,
) -> dict[str, Any]:
    if not volumes:
        raise ValueError("fit_brain_reference requires at least one volume")

    metrics = [_extract_volume_metrics(v) for v in volumes]
    keys = ["foreground_ratio", "largest_component_ratio", "lr_symmetry", "com_offset"]
    mat = np.array([[m[k] for k in keys] for m in metrics], dtype=np.float64)

    mean = mat.mean(axis=0)
    std = _safe_std(mat.std(axis=0))

    z = np.abs((mat - mean) / std)
    scores = z.mean(axis=1)

    # Threshold is robustly derived from in-domain training volumes.
    score_threshold = float(np.percentile(scores, percentile_threshold))

    hard_limits = DEFAULT_HARD_LIMITS

    return {
        "metric_keys": keys,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "score_threshold": max(score_threshold, z_threshold),
        "z_threshold_floor": float(z_threshold),
        "percentile_threshold": float(percentile_threshold),
        "hard_limits": hard_limits,
        "fit_samples": int(len(volumes)),
    }


def brain_ood_score(volume: np.ndarray, reference: dict[str, Any]) -> dict[str, Any]:
    keys = reference["metric_keys"]
    m = _extract_volume_metrics(volume)
    vec = np.array([m[k] for k in keys], dtype=np.float64)
    mean = np.array(reference["mean"], dtype=np.float64)
    std = _safe_std(np.array(reference["std"], dtype=np.float64))

    z = np.abs((vec - mean) / std)
    score = float(z.mean())
    threshold = float(reference.get("score_threshold", reference.get("z_threshold_floor", 3.0)))
    hard_limits = reference.get("hard_limits", DEFAULT_HARD_LIMITS)
    hard_violations: dict[str, float] = {}
    for k, lim in hard_limits.items():
        lo, hi = float(lim[0]), float(lim[1])
        v = float(m[k])
        if v < lo or v > hi:
            hard_violations[k] = v

    is_out_of_domain = bool((score > threshold) or (len(hard_violations) > 0))

    return {
        "is_out_of_domain": is_out_of_domain,
        "score": score,
        "threshold": threshold,
        "hard_violations": hard_violations,
        "z_by_metric": {k: float(v) for k, v in zip(keys, z)},
        "metrics": m,
    }