import pytest
import asyncio
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from src.train_incremental import incremental_train
from app.services.training_worker import training_worker
from app.services.ai_service import model_service
from app.main import app


def test_incremental_train_execution():
    """Valida a execução de 1 ciclo adaptativo de fine-tuning incremental e geração de backup."""
    ckpt_path = Path("checkpoints/best_model.pth")
    assert ckpt_path.exists()

    res = incremental_train(
        config_path="config.yaml",
        epochs=1,
        lr=1e-5,
        checkpoint_path=str(ckpt_path),
        max_train_samples=2,
    )

    assert res["status"] == "success"
    assert res["epochs_trained"] == 1
    assert "final_f1" in res
    assert "history" in res
    assert len(res["history"]) == 1

    # Verifica se o backup foi gerado
    backup = ckpt_path.with_name("best_model.backup.pth")
    assert backup.exists()


@pytest.mark.asyncio
async def test_training_worker_concurrency_lock():
    """Valida que o TrainingWorker impede múltiplas execuções simultâneas via lock assíncrono."""
    status = training_worker.get_status()
    assert isinstance(status, dict)
    assert "status" in status
    assert "is_running" in status

    # Dispara tarefa simulada
    success, msg = training_worker.trigger_training(epochs=1, lr=1e-5, max_train_samples=2)
    assert success is True
    assert msg.startswith("train-")

    # Segunda tentativa imediata deve ser rejeitada pelo lock/status
    second_success, second_msg = training_worker.trigger_training(epochs=1, lr=1e-5)
    assert second_success is False
    assert "já está em andamento" in second_msg

    # Aguarda a finalização da tarefa em background
    for _ in range(60):
        if not training_worker.get_status()["is_running"]:
            break
        await asyncio.sleep(0.5)

    final_status = training_worker.get_status()
    assert final_status["status"] == "completed"
    assert final_status["last_result"] is not None
    assert model_service.is_loaded is True


@pytest.mark.asyncio
async def test_train_api_endpoints():
    """Testa os endpoints administrativos GET /train/status e POST /train/trigger."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET /train/status
        resp = await client.get("/train/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "progress" in data

        # POST /train/trigger
        resp_trigger = await client.post("/train/trigger", json={"epochs": 1, "learning_rate": 0.00001, "max_train_samples": 2})
        assert resp_trigger.status_code in [202, 409]
        if resp_trigger.status_code == 202:
            body = resp_trigger.json()
            assert body["status"] == "training"
            assert "task_id" in body

        # Aguarda liberar
        for _ in range(60):
            st = await client.get("/train/status")
            if not st.json()["is_running"]:
                break
            await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_validate_diagnosis_e2e_integration():
    """Valida o fluxo clínico: validação médica -> dataset -> background retrain trigger."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/patients/pat-01/validate-diagnosis",
            json={"diagnosis": "MCI", "notes": "Validação com concordância especialista"},
        )
        assert resp.status_code == 200
        patient = resp.json()
        assert patient["validated_diagnosis"] == "MCI"
        assert patient["validated_at"]

        # Aguarda liberar qualquer tarefa em background
        for _ in range(60):
            st = await client.get("/train/status")
            if not st.json()["is_running"]:
                break
            await asyncio.sleep(0.5)
