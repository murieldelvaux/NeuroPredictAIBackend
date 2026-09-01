from __future__ import annotations
import sys
import logging
import shutil
from pathlib import Path
from typing import Callable, Optional, Dict, Any

# Garante que a raiz do projeto esteja no sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
from sklearn.metrics import f1_score, balanced_accuracy_score
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, Subset

from src.datasets.mri_dataset import MRIDataset
from src.models.hybrid_cnn_resnet_transformer import HybridCNNResNetTransformer3D
from src.models.cnn_3d import SimpleResNet3D
from src.data_prep.preprocess import build_preprocessing_pipeline
from src.utils.focal_loss import FocalLoss
from src.utils.helpers import get_device, load_config


logger = logging.getLogger(__name__)


def incremental_train(
    config_path: str | Path = "config.yaml",
    epochs: int = 3,
    lr: float = 1e-5,
    checkpoint_path: str | Path = "checkpoints/best_model.pth",
    max_train_samples: Optional[int] = None,
    on_epoch_end: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, Any]:
    """
    Executa ciclo adaptativo de re-treinamento incremental (fine-tuning) incorporando
    novas anotações clínicas e exames validados.
    
    Args:
        config_path: Caminho para o config.yaml.
        epochs: Número de épocas do ciclo curto de fine-tuning (padrão: 3).
        lr: Taxa de aprendizado reduzida (padrão: 1e-5).
        checkpoint_path: Caminho do checkpoint a ser atualizado.
        max_train_samples: Limite opcional de amostras para ciclos rápidos de teste.
        on_epoch_end: Callback opcional de progresso fn(epoch, total_epochs, train_loss, val_f1).
        
    Returns:
        Dicionário com métricas e resumo do re-treinamento.
    """
    cfg = load_config(config_path)
    device = get_device(cfg.get("training", {}).get("device", "cpu"))
    ckpt_file = Path(checkpoint_path)

    # 1. Pipelines de pré-processamento e Datasets
    train_transform = build_preprocessing_pipeline(cfg, train=True)
    val_transform = build_preprocessing_pipeline(cfg, train=False)

    metadata_path = cfg["data"]["unified_metadata"]
    full_train_ds = MRIDataset(metadata_path, split="train", transform=train_transform)
    val_ds = MRIDataset(metadata_path, split="val", transform=val_transform)

    # Identifica amostras prioritárias de feedback clínico recente para experience replay
    feedback_mask = full_train_ds.metadata.get("source", "") == "CLINICAL_FEEDBACK"
    feedback_indices = list(full_train_ds.metadata[feedback_mask].index)
    other_indices = list(full_train_ds.metadata[~feedback_mask].index)

    effective_limit = max_train_samples if max_train_samples is not None else 64
    remaining = effective_limit - len(feedback_indices)
    if remaining > 0:
        selected_indices = feedback_indices + other_indices[:remaining]
    else:
        selected_indices = feedback_indices[:effective_limit]

    train_ds = Subset(full_train_ds, selected_indices)

    if max_train_samples is not None and max_train_samples < len(val_ds):
        val_ds = Subset(val_ds, list(range(min(max_train_samples, len(val_ds)))))
    elif len(val_ds) > 32:
        val_ds = Subset(val_ds, list(range(32)))


    num_classes = int(cfg["model"]["num_classes"])
    batch_size = max(1, int(cfg["training"].get("batch_size", 2)))
    num_workers = int(cfg["training"].get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # 2. Inicialização do Modelo
    arch = cfg["model"].get("arch", "hybrid").lower()
    if arch in ["hybrid", "hybrid_cnn_resnet_transformer", "transformer"]:
        model = HybridCNNResNetTransformer3D(
            in_channels=cfg["model"]["input_channels"],
            num_classes=num_classes,
            dropout=float(cfg["model"].get("dropout", 0.4)),
            embed_dim=int(cfg["model"].get("embed_dim", 256)),
            num_transformer_layers=int(cfg["model"].get("num_transformer_layers", 4)),
            num_heads=int(cfg["model"].get("num_heads", 8)),
        ).to(device)
    else:
        model = SimpleResNet3D(
            in_channels=cfg["model"]["input_channels"],
            num_classes=num_classes,
        ).to(device)

    # 3. Carrega checkpoint base existente
    raw_checkpoint = {}
    if ckpt_file.exists():
        raw_checkpoint = torch.load(ckpt_file, map_location=device)
        state = raw_checkpoint.get("model_state_dict", raw_checkpoint)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            logger.warning(f"Carregando checkpoint base com strict=False: {e}")
            model.load_state_dict(state, strict=False)
        logger.info(f"Pesos base carregados de {ckpt_file}")
    else:
        logger.warning(f"Checkpoint {ckpt_file} não encontrado. Treinando a partir dos pesos inicializados.")

    # 4. Avaliação Inicial
    initial_f1 = float(raw_checkpoint.get("macro_f1", 0.0)) if isinstance(raw_checkpoint, dict) else 0.0

    # 5. Configuração do Otimizador e Loss
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    logger.info(f"Iniciando fine-tuning incremental por {epochs} épocas (lr={lr:.2e}) com {len(train_ds)} amostras.")

    history = []
    final_val_loss = 0.0
    final_macro_f1 = initial_f1
    final_bal_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            clinical = batch["clinical"].to(device) if "clinical" in batch else None

            optimizer.zero_grad()
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_train_loss += loss.item() * bs
            n_samples += bs

        train_loss = running_train_loss / max(n_samples, 1)

        # Validação da época
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        n_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels_b = batch["label"].to(device)
                clinical_b = batch["clinical"].to(device) if "clinical" in batch else None

                outputs = model(images, clinical_b)
                loss_v = criterion(outputs, labels_b)
                bs = images.size(0)
                val_loss += loss_v.item() * bs
                n_val_samples += bs

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_b.cpu().numpy())

        val_loss = val_loss / max(n_val_samples, 1)
        y_true = np.asarray(all_labels, dtype=np.int64)
        y_pred = np.asarray(all_preds, dtype=np.int64)

        macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0)) if len(y_true) else 0.0
        bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0

        final_val_loss = val_loss
        final_macro_f1 = macro_f1
        final_bal_acc = bal_acc

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_balanced_accuracy": round(bal_acc, 4),
        })

        logger.info(f"[Fine-tuning] Época {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {macro_f1:.4f}")

        if on_epoch_end is not None:
            try:
                on_epoch_end(epoch + 1, epochs, train_loss, macro_f1)
            except Exception as cb_err:
                logger.warning(f"Erro no callback de progresso: {cb_err}")

    # 6. Atualização atômica do Checkpoint com Backup
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file = ckpt_file.with_name("best_model.backup.pth")
    if ckpt_file.exists():
        try:
            shutil.copy2(ckpt_file, backup_file)
            logger.info(f"Backup do checkpoint criado em {backup_file}")
        except Exception as bkp_err:
            logger.warning(f"Não foi possível criar backup: {bkp_err}")

    torch.save({
        "epoch": int(raw_checkpoint.get("epoch", 0)) + epochs if isinstance(raw_checkpoint, dict) else epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": final_val_loss,
        "macro_f1": final_macro_f1,
        "balanced_accuracy": final_bal_acc,
        "class_names": ["CN", "MCI", "DEM"],
        "confidence_threshold": raw_checkpoint.get("confidence_threshold", 0.4) if isinstance(raw_checkpoint, dict) else 0.4,
        "brain_ood_reference": raw_checkpoint.get("brain_ood_reference") if isinstance(raw_checkpoint, dict) else None,
    }, ckpt_file)

    logger.info(f"✓ Checkpoint atualizado com sucesso em {ckpt_file} (F1: {final_macro_f1:.4f})")

    return {
        "status": "success",
        "epochs_trained": epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "initial_f1": round(initial_f1, 4),
        "final_f1": round(final_macro_f1, 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_balanced_accuracy": round(final_bal_acc, 4),
        "history": history,
        "checkpoint_path": str(ckpt_file),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Incremental fine-tuning for continuous learning.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    res = incremental_train(config_path=args.config, epochs=args.epochs, lr=args.lr)
    print(res)
