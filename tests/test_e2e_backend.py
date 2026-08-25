import pytest
from httpx import AsyncClient, ASGITransport
from pathlib import Path
from app.main import app
from app.services.ai_service import model_service


@pytest.mark.asyncio
async def test_health_check():
    """Verifica se o backend inicia e o modelo híbrido está carregado."""
    model_service.load()
    assert model_service.is_loaded is True

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_predict_endpoint_with_mri():
    """Testa a rota de predição enviando o volume MRI 3D e dados clínicos."""
    sample_mri = Path("data/sample_mri.nii.gz")
    assert sample_mri.exists(), "Sample MRI not found in data/sample_mri.nii.gz"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with open(sample_mri, "rb") as f:
            files = {"mri_file": ("sample_mri.nii.gz", f, "application/gzip")}
            data = {
                "patient_id": "TEST_PATIENT_001",
                "prediction_date": "2026-08-24",
                "age": "72.5",
                "mmse": "26.0",
                "cdr": "0.5",
            }
            resp = await client.post("/predict", data=data, files=files)

        assert resp.status_code == 200
        result = resp.json()

        print("\nResultado da Predição E2E:")
        print(result)

        assert result["patient_id"] == "TEST_PATIENT_001"
        assert result["classification"] in ["CN", "MCI", "DEM"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["risk_score"] <= 1.0
        assert "CN" in result["probabilities"]
        assert "MCI" in result["probabilities"]
        assert "DEM" in result["probabilities"]
        assert len(result["explanation"]) > 0
