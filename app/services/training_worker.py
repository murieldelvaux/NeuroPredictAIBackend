from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from app.services.ai_service import model_service

logger = logging.getLogger(__name__)


class TrainingWorker:
    """
    Gerenciador assíncrono para execução segura de re-treinamento contínuo
    com lock de concorrência e hot-reload automático do modelo.
    """

    def __init__(self):
        self.lock = asyncio.Lock()
        self.status: str = "idle"  # "idle" | "training" | "completed" | "failed"
        self.task_id: Optional[str] = None
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.train_loss: float = 0.0
        self.val_f1: float = 0.0
        self.last_result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None

    def get_status(self) -> Dict[str, Any]:
        """Retorna o estado operacional e métricas da última execução."""
        return {
            "status": self.status,
            "task_id": self.task_id,
            "is_running": self.status == "training",
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": {
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "train_loss": self.train_loss,
                "val_f1": self.val_f1,
            },
            "last_result": self.last_result,
            "error": self.error_message,
            "model_reloaded": model_service.is_loaded,
        }

    def trigger_training(
        self,
        epochs: int = 3,
        lr: float = 1e-5,
        max_train_samples: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Dispara o processo de fine-tuning em background.
        
        Returns:
            (success: bool, message_or_task_id: str)
        """
        if self.lock.locked() or self.status == "training":
            return False, "Re-treinamento já está em andamento. Aguarde a finalização da tarefa atual."

        new_task_id = f"train-{uuid.uuid4().hex[:8]}"
        self.task_id = new_task_id
        self.status = "training"
        self.started_at = datetime.now().isoformat()
        self.completed_at = None
        self.error_message = None
        self.current_epoch = 0
        self.total_epochs = epochs

        # Dispara tarefa assíncrona não-bloqueante no event loop
        asyncio.create_task(self._run_training_job(epochs, lr, max_train_samples))
        logger.info(f"Tarefa de re-treinamento {new_task_id} iniciada com sucesso.")
        return True, new_task_id

    async def _run_training_job(
        self,
        epochs: int,
        lr: float,
        max_train_samples: Optional[int],
    ) -> None:
        async with self.lock:
            try:
                from src.train_incremental import incremental_train

                def _progress_cb(cur: int, tot: int, loss: float, f1: float) -> None:
                    self.current_epoch = cur
                    self.total_epochs = tot
                    self.train_loss = round(loss, 4)
                    self.val_f1 = round(f1, 4)

                # Executa o treino em thread pool separada para não bloquear requests do backend
                result = await asyncio.to_thread(
                    incremental_train,
                    config_path="config.yaml",
                    epochs=epochs,
                    lr=lr,
                    checkpoint_path="checkpoints/best_model.pth",
                    max_train_samples=max_train_samples,
                    on_epoch_end=_progress_cb,
                )

                self.last_result = result
                self.status = "completed"
                self.completed_at = datetime.now().isoformat()
                logger.info(f"Re-treinamento {self.task_id} concluído. Resultado: {result}")

                # ── Hot-Reload dos pesos no serviço de inferência ─────────────
                logger.info("Iniciando hot-reload do modelo no ModelService...")
                model_service.load()
                logger.info("✓ Hot-reload concluído com sucesso!")

            except Exception as exc:
                logger.error(f"Erro durante o re-treinamento: {exc}", exc_info=True)
                self.status = "failed"
                self.error_message = str(exc)
                self.completed_at = datetime.now().isoformat()


training_worker = TrainingWorker()
