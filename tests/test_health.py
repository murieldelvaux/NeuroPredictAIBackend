import pytest
import json
from datetime import date
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from app.main import app
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
    assert body["patient"]["clinical_data"]["mri_file"]["filename"] == "pat-01-mri.nii.gz"


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


def test_create_patient_with_mri_upload():
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
    assert body["name"] == "Joaquim Silva"
    assert body["clinical_data"]["mmse"] == 19
    assert body["clinical_data"]["biomarkers"][0] == "ApoE4 positive (e3/e4)"
    assert body["clinical_data"]["mri_file"]["filename"] == "joaquim-silva-mri.nii.gz"
    assert body["clinical_data"]["mri_file"]["size"] == len(b"fake-mri-bytes")
