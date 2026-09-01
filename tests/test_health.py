# pyrefly: ignore [missing-import]
import pytest
import json
from datetime import date
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
# pyrefly: ignore [missing-import]
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.config import settings
from app.services.ai_service import model_service
from app.schemas.prediction import PredictionOutput


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_patients():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/patients")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 3  # seed patients


@pytest.mark.asyncio
async def test_get_patient_detail():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/patients/pat-01")
    assert resp.status_code == 200
    body = resp.json()
    assert "patient" in body
    assert body["patient"]["id"] == "pat-01"
    assert body["patient"]["sex"] == "Feminino"
    assert body["patient"]["date_of_birth"] == "14/03/1950"
    assert body["patient"]["created_at"]
    assert "/" in body["patient"]["created_at"]
    assert body["patient"]["clinical_data"]["mri_file"][0]["filename"] == "pat-01-mri.nii.gz"
    assert body["patient"]["mri_file"][0]["url"] == "/patients/pat-01/pat-01-mri.nii.gz"


@pytest.mark.asyncio
async def test_predict_forwards_prediction_date(monkeypatch):
    captured = {}

    def fake_predict(nii_path, clinical=None, prediction_date=None):
        captured["prediction_date"] = prediction_date
        return PredictionOutput(
            patient_id="",
            prediction_date=prediction_date or date.today(),
            risk_score=0.1,
            classification="CN",
            confidence=0.9,
            probabilities={"CN": 0.9, "MCI": 0.08, "AD": 0.02},
            explanation=[],
            model_version="mock",
        )

    monkeypatch.setattr(model_service, "is_loaded", True)
    monkeypatch.setattr(model_service, "predict", fake_predict)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            data={
                "patient_id": "pat-01",
                "prediction_date": "2026-07-15",
                "age": "70",
                "mmse": "19",
                "cdr": "2",
                "cdrtot": "0.5",
            },
            files={"mri_file": ("sample.nii.gz", b"fake-mri-bytes", "application/gzip")},
        )

    assert resp.status_code == 200
    assert captured["prediction_date"] == date(2026, 7, 15)
    assert resp.json()["prediction_date"] == "15/07/2026"


def test_create_patient_with_mri_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "mri_storage_dir", str(tmp_path / "mri"), raising=False)

    client = TestClient(app)
    resp = client.post(
        "/patients",
        data={
            "name": "Joaquim Silva",
            "age": "72",
            "sex": "M",
            "date_of_birth": "1956-06-20",
            "clinical_data": json.dumps(
                {
                    "mmse": 19,
                    "moca": 18,
                    "cdr": 2,
                    "cdrtot": 0.5,
                    "comorbidities": ["Hypertension", "Hypercholesterolemia"],
                    "biomarkers": ["ApoE4 positive (e3/e4)", "Family history of early onset AD"],
                    "symptoms": ["Mild word retrieval delays", "Subjective short term recall deficits", "perda de memória"],
                    "medications": ["Lisinopril 10mg daily", "sertralina"],
                    "family_history": True,
                    "education_years": 14,
                }
            ),
        },
        files={"mri_file": ("joaquim-silva-mri.nii.gz", b"fake-mri-bytes", "application/gzip")},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["clinical_data"]["mri_file"][0]["url"] == f"/patients/{body['id']}/joaquim-silva-mri.nii.gz"
    assert body["name"] == "Joaquim Silva"
    assert body["clinical_data"]["mmse"] == 19
    assert body["clinical_data"]["biomarkers"][0] == "ApoE4 positive (e3/e4)"
    assert body["clinical_data"]["mri_file"][0]["filename"] == "joaquim-silva-mri.nii.gz"
    assert body["clinical_data"]["mri_file"][0]["size"] == len(b"fake-mri-bytes")
    assert body["mri_file"][0]["url"] == f"/patients/{body['id']}/joaquim-silva-mri.nii.gz"

    download = client.get(body["mri_file"][0]["url"])
    assert download.status_code == 200
    assert download.content == b"fake-mri-bytes"
    assert download.headers["content-type"].startswith("application/octet-stream")
    assert download.headers["content-disposition"] == 'inline; filename="joaquim-silva-mri.nii.gz"'

    update = client.post(
        f"/patients/{body['id']}/mri-file",
        files={"mri_file": ("second.nii.gz", b"second-bytes", "application/gzip")},
    )
    assert update.status_code == 200
    update_body = update.json()
    assert update_body["filename"] == "second.nii.gz"
    assert update_body["url"] == f"/patients/{body['id']}/second.nii.gz"

    detail = client.get(f"/patients/{body['id']}")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert len(detail_body["patient"]["mri_file"]) == 2


def test_update_clinical_data_aggregates_cognitive_history(client):
    """Valida que novas avaliações cognitivas são agregadas ao histórico longitudinal para gerar os gráficos."""
    # 1. Consulta o histórico inicial do pat-01 (já possui 3 avaliações no seed)
    res_get = client.get("/patients/pat-01")
    assert res_get.status_code == 200
    initial_history = res_get.json()["patient"]["clinical_data"].get("cognitive_history", [])
    initial_count = len(initial_history)
    assert initial_count >= 3

    # 2. Envia nova avaliação da consulta atual (com data explícita)
    patch_payload_1 = {
        "mmse": 20.0,
        "moca": 17.0,
        "cdr": 1.0,
        "cdrtot": 4.0,
        "assessment_date": "15/08/2026",
        "notes": "Nova consulta semestral - progressão de sintomas",
    }
    resp_patch_1 = client.patch("/patients/pat-01/clinical-data", json=patch_payload_1)
    assert resp_patch_1.status_code == 200
    history_after_1 = resp_patch_1.json()["clinical_data"]["cognitive_history"]
    assert len(history_after_1) == initial_count + 1
    assert any(h["date"] == "15/08/2026" and h["mmse"] == 20.0 for h in history_after_1)

    # 3. Envia outra avaliação posterior (ex: sem data explícita, adotando a data de hoje)
    patch_payload_2 = {
        "mmse": 18.0,
        "moca": 15.0,
        "cdr": 2.0,
        "cdrtot": 6.5,
        "assessment_date": "01/09/2026",
        "notes": "Consulta de retorno",
    }
    resp_patch_2 = client.patch("/patients/pat-01/clinical-data", json=patch_payload_2)
    assert resp_patch_2.status_code == 200
    history_after_2 = resp_patch_2.json()["clinical_data"]["cognitive_history"]
    assert len(history_after_2) == initial_count + 2
    assert history_after_2[-1]["date"] == "01/09/2026"
    assert history_after_2[-1]["mmse"] == 18.0
    assert history_after_2[-1]["cdr"] == 2.0

