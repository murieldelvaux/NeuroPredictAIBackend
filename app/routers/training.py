from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.training_worker import training_worker

router = APIRouter()


class TriggerTrainingRequest(BaseModel):
    epochs: int = Field(3, ge=1, le=50, description="Número de épocas do ciclo de fine-tuning")
    learning_rate: float = Field(1e-5, gt=0.0, le=1e-2, description="Taxa de aprendizado adaptativa")
    max_train_samples: Optional[int] = Field(None, ge=1, description="Limite de amostras para ciclos de teste rápido")


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_training(payload: Optional[TriggerTrainingRequest] = None):
    """
    Dispara manualmente o re-treinamento incremental contínuo do modelo em background.
    Atualiza checkpoints/best_model.pth e realiza o hot-reload automático dos pesos.
    """
    req = payload or TriggerTrainingRequest()
    success, result_msg = training_worker.trigger_training(
        epochs=req.epochs,
        lr=req.learning_rate,
        max_train_samples=req.max_train_samples,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=result_msg,
        )

    return {
        "message": "Re-treinamento incremental iniciado em background com sucesso.",
        "task_id": result_msg,
        "status": "training",
        "epochs": req.epochs,
        "learning_rate": req.learning_rate,
    }


@router.get("/status")
async def get_training_status():
    """
    Retorna o status operacional do worker de re-treinamento, progresso e métricas da última execução.
    """
    return training_worker.get_status()
