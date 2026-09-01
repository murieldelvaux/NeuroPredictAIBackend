from __future__ import annotations
import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
# pyrefly: ignore [missing-import]
import numpy as np

from src.datasets.mri_dataset import MRIDataset
from src.models.cnn_3d import SimpleResNet3D
from src.models.hybrid_cnn_resnet_transformer import HybridCNNResNetTransformer3D
from src.utils.helpers import load_config, get_device
from src.data_prep.preprocess import build_preprocessing_pipeline
from src.utils.focal_loss import FocalLoss
from src.utils.brain_ood import fit_brain_reference



def _effective_num_class_weights(labels: np.ndarray, num_classes: int, beta: float = 0.999) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    effective_num = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.5, 2.0)
    return weights


def _build_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_inv = 1.0 / np.maximum(counts, 1.0)
    sample_weights = class_inv[labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _fit_brain_reference_from_dataset(dataset: MRIDataset, max_samples: int = 256) -> dict:
    n = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int) if n > 0 else []
    vols: list[np.ndarray] = []
    for idx in indices:
        sample = dataset[int(idx)]
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        vols.append(arr)
    return fit_brain_reference(vols)


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    device = get_device(cfg["training"]["device"])

    train_transform = build_preprocessing_pipeline(cfg, train=True)
    val_transform   = build_preprocessing_pipeline(cfg, train=False)

    train_ds = MRIDataset(cfg["data"]["unified_metadata"], split="train", transform=train_transform)
    val_ds   = MRIDataset(cfg["data"]["unified_metadata"], split="val",   transform=val_transform)

    num_classes = int(cfg["model"]["num_classes"])
    train_labels = train_ds.metadata["label_id"].astype(int).to_numpy()

    use_weighted_sampler = bool(cfg["training"].get("use_weighted_sampler", True))
    sampler = _build_weighted_sampler(train_labels, num_classes=num_classes) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader   = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=cfg["training"]["num_workers"])

    arch = cfg["model"].get("arch", "hybrid").lower()
    if arch in ["hybrid", "hybrid_cnn_resnet_transformer", "transformer"]:
        print("==> Initializing HybridCNNResNetTransformer3D model")
        model = HybridCNNResNetTransformer3D(
            in_channels=cfg["model"]["input_channels"],
            num_classes=cfg["model"]["num_classes"],
            dropout=float(cfg["model"].get("dropout", 0.4)),
            embed_dim=int(cfg["model"].get("embed_dim", 256)),
            num_transformer_layers=int(cfg["model"].get("num_transformer_layers", 4)),
            num_heads=int(cfg["model"].get("num_heads", 8)),
        ).to(device)
    else:
        print("==> Initializing SimpleResNet3D model")
        model = SimpleResNet3D(
            in_channels=cfg["model"]["input_channels"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)

    beta = float(cfg["training"].get("class_weight_beta", 0.999))
    class_weights_np = _effective_num_class_weights(train_labels, num_classes=num_classes, beta=beta)
    use_loss_weights_when_sampler = bool(cfg["training"].get("use_loss_weights_with_sampler", False))
    if use_weighted_sampler and not use_loss_weights_when_sampler:
        class_weights_np = np.ones_like(class_weights_np)

    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    print("Train class counts:", np.bincount(train_labels, minlength=num_classes).tolist())
    print("Class weights:", [round(float(x), 4) for x in class_weights_np])
    print(f"Weighted sampler enabled: {use_weighted_sampler}")

    gamma = float(cfg["training"].get("focal_gamma", 2.0))
    criterion = FocalLoss(weight=class_weights, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"].get("learning_rate", 1e-4)), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    best_val_f1 = -1.0
    class_names = ["CN", "MCI", "DEM"]

    # Fit an in-domain brain reference with non-augmented train scans.
    brain_ref_ds = MRIDataset(cfg["data"]["unified_metadata"], split="train", transform=val_transform)
    brain_ood_reference = _fit_brain_reference_from_dataset(
        brain_ref_ds,
        max_samples=int(cfg["training"].get("ood_fit_samples", 256)),
    )
    print(f"Brain OOD reference fitted with {brain_ood_reference['fit_samples']} samples.")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            clinical = batch["clinical"].to(device) if "clinical" in batch else None
            optimizer.zero_grad()
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels_b = batch["label"].to(device)
                clinical_b = batch["clinical"].to(device) if "clinical" in batch else None
                outputs = model(images, clinical_b)
                val_loss += criterion(outputs, labels_b).item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels_b.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        y_true = np.asarray(all_labels, dtype=np.int64)
        y_pred = np.asarray(all_preds, dtype=np.int64)
        y_prob = np.asarray(all_probs, dtype=np.float32)

        macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        pred_conf = y_prob[np.arange(len(y_pred)), y_pred]
        correct_conf = pred_conf[y_pred == y_true]
        if len(correct_conf) > 0:
            confidence_threshold = float(np.quantile(correct_conf, float(cfg["training"].get("confidence_quantile", 0.15))))
        else:
            confidence_threshold = 0.5

        print(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")
        print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Val macro-F1: {macro_f1:.4f}  |  Val balanced-acc: {bal_acc:.4f}")
        print(f"  Calibrated confidence threshold: {confidence_threshold:.4f}")
        print(classification_report(all_labels, all_preds, labels=list(range(num_classes)), target_names=class_names, zero_division=0))

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
                "macro_f1": macro_f1,
                "balanced_accuracy": bal_acc,
                "class_names": class_names,
                "confidence_threshold": confidence_threshold,
                "brain_ood_reference": brain_ood_reference,
            }, checkpoints_dir / "best_model.pth")
            print(f"  ✓ best model saved (macro_f1={macro_f1:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
