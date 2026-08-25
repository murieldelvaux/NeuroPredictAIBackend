from __future__ import annotations

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix
# pyrefly: ignore [missing-import]
import numpy as np

from src.datasets.mri_dataset import MRIDataset
from src.models.cnn_3d import SimpleResNet3D
from src.models.hybrid_cnn_resnet_transformer import HybridCNNResNetTransformer3D
from src.utils.helpers import load_config, get_device
from src.data_prep.preprocess import build_preprocessing_pipeline


def evaluate(config_path: str, split: str = "val") -> None:
    cfg = load_config(config_path)
    device = get_device(cfg["training"]["device"])

    transform = build_preprocessing_pipeline(cfg, train=False)
    dataset = MRIDataset(cfg["data"]["unified_metadata"], split=split, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"], pin_memory=False)

    arch = cfg["model"].get("arch", "hybrid").lower()
    if arch in ["hybrid", "hybrid_cnn_resnet_transformer", "transformer"]:
        model = HybridCNNResNetTransformer3D(
            in_channels=cfg["model"]["input_channels"],
            num_classes=cfg["model"]["num_classes"],
            dropout=float(cfg["model"].get("dropout", 0.4)),
            embed_dim=int(cfg["model"].get("embed_dim", 256)),
            num_transformer_layers=int(cfg["model"].get("num_transformer_layers", 4)),
            num_heads=int(cfg["model"].get("num_heads", 8)),
        ).to(device)
    else:
        model = SimpleResNet3D(
            in_channels=cfg["model"]["input_channels"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)

    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_preds, dtype=np.int64)
    num_classes = int(cfg["model"]["num_classes"])
    class_names = ["CN", "MCI", "DEM"]

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)) if len(y_true) else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0

    print(f"{split.title()} accuracy: {acc:.4f} ({int((y_true == y_pred).sum())}/{len(y_true)})")
    print(f"{split.title()} macro-F1: {macro_f1:.4f}")
    print(f"{split.title()} balanced-accuracy: {bal_acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=list(range(num_classes))))
    print(classification_report(y_true, y_pred, labels=list(range(num_classes)), target_names=class_names, zero_division=0))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a 3D CNN model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(args.config, split=args.split)
