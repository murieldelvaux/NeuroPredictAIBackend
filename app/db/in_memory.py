from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional
import asyncio

from sqlalchemy import select, func, delete

from app.db.database import async_session_maker, engine
from app.db.models import Base, PatientRecord, PredictionRecord
from app.schemas.patient import Patient, PatientDetail, PatientResponse

# ---------------------------------------------------------------------------
# Seed data used for local development and tests
# ---------------------------------------------------------------------------
_seed_predictions: dict[str, list[dict]] = {
    "pat-01": [
        {
            "patient_id": "pat-01",
            "prediction_date": date(2026, 5, 10),
            "risk_score": 0.61,
            "classification": "MCI",
            "confidence": 0.74,
            "probabilities": {"CN": 0.18, "MCI": 0.74, "AD": 0.08},
            "explanation": [
                {"feature": "mmse", "impact": 0.42, "direction": "risk"},
                {"feature": "cdr", "impact": 0.31, "direction": "risk"},
                {"feature": "education_years", "impact": 0.15, "direction": "protective"},
                {"feature": "family_history", "impact": 0.12, "direction": "risk"},
            ],
            "model_version": "resnet3d-oasis3-v1",
        },
        {
            "patient_id": "pat-01",
            "prediction_date": date(2026, 7, 1),
            "risk_score": 0.67,
            "classification": "MCI",
            "confidence": 0.71,
            "probabilities": {"CN": 0.12, "MCI": 0.71, "AD": 0.17},
            "explanation": [
                {"feature": "mmse", "impact": 0.45, "direction": "risk"},
                {"feature": "cdr", "impact": 0.33, "direction": "risk"},
                {"feature": "education_years", "impact": 0.13, "direction": "protective"},
                {"feature": "family_history", "impact": 0.09, "direction": "risk"},
            ],
            "model_version": "resnet3d-oasis3-v1",
        },
    ],
    "pat-02": [
        {
            "patient_id": "pat-02",
            "prediction_date": date(2026, 6, 20),
            "risk_score": 0.09,
            "classification": "CN",
            "confidence": 0.91,
            "probabilities": {"CN": 0.91, "MCI": 0.07, "AD": 0.02},
            "explanation": [
                {"feature": "mmse", "impact": 0.50, "direction": "protective"},
                {"feature": "cdr", "impact": 0.28, "direction": "protective"},
                {"feature": "education_years", "impact": 0.22, "direction": "protective"},
            ],
            "model_version": "resnet3d-oasis3-v1",
        },
    ],
    "pat-03": [
        {
            "patient_id": "pat-03",
            "prediction_date": date(2026, 4, 15),
            "risk_score": 0.88,
            "classification": "AD",
            "confidence": 0.85,
            "probabilities": {"CN": 0.03, "MCI": 0.12, "AD": 0.85},
            "explanation": [
                {"feature": "mmse", "impact": 0.55, "direction": "risk"},
                {"feature": "cdrtot", "impact": 0.40, "direction": "risk"},
                {"feature": "comorbidities", "impact": 0.30, "direction": "risk"},
                {"feature": "family_history", "impact": 0.20, "direction": "risk"},
                {"feature": "education_years", "impact": 0.10, "direction": "protective"},
            ],
            "model_version": "resnet3d-oasis3-v1",
        },
        {
            "patient_id": "pat-03",
            "prediction_date": date(2026, 7, 5),
            "risk_score": 0.92,
            "classification": "AD",
            "confidence": 0.89,
            "probabilities": {"CN": 0.01, "MCI": 0.10, "AD": 0.89},
            "explanation": [
                {"feature": "mmse", "impact": 0.58, "direction": "risk"},
                {"feature": "cdrtot", "impact": 0.44, "direction": "risk"},
                {"feature": "comorbidities", "impact": 0.32, "direction": "risk"},
                {"feature": "family_history", "impact": 0.21, "direction": "risk"},
                {"feature": "education_years", "impact": 0.08, "direction": "protective"},
            ],
            "model_version": "resnet3d-oasis3-v1",
        },
    ],
}

_seed_patients = [
    Patient(
        id="pat-01",
        name="Eleanor Vance",
        age=74,
        sex="F",
        date_of_birth=date(1950, 3, 14),
        clinical_data={
            "mmse": 22,
            "moca": 19,
            "cdr": 0.5,
            "cdrtot": 2.5,
            "biomarkers": ["abeta42_low", "tau_high"],
            "symptoms": ["memory_loss", "disorientation"],
            "medications": ["donepezil", "losartan"],
            "mri_file": {
                "filename": "pat-01-mri.nii.gz",
                "content_type": "application/gzip",
                "size": 15,
            },
            "comorbidities": ["hypertension"],
            "family_history": True,
            "education_years": 12,
        },
    ),
    Patient(
        id="pat-02",
        name="Robert Chen",
        age=68,
        sex="M",
        date_of_birth=date(1956, 7, 22),
        clinical_data={
            "mmse": 28,
            "moca": 27,
            "cdr": 0.0,
            "cdrtot": 0.0,
            "biomarkers": ["abeta42_normal", "ptau_normal"],
            "symptoms": ["mild_forgetfulness"],
            "medications": ["atorvastatin"],
            "mri_file": {
                "filename": "pat-02-mri.nii.gz",
                "content_type": "application/gzip",
                "size": 15,
            },
            "comorbidities": [],
            "family_history": False,
            "education_years": 16,
        },
    ),
    Patient(
        id="pat-03",
        name="Maria Santos",
        age=81,
        sex="F",
        date_of_birth=date(1943, 11, 5),
        clinical_data={
            "mmse": 17,
            "moca": 13,
            "cdr": 1.0,
            "cdrtot": 5.0,
            "biomarkers": ["tau_very_high", "hippocampal_atrophy"],
            "symptoms": ["memory_loss", "language_difficulty", "apathy"],
            "medications": ["memantine", "sertraline", "metformin"],
            "mri_file": {
                "filename": "pat-03-mri.nii.gz",
                "content_type": "application/gzip",
                "size": 15,
            },
            "comorbidities": ["diabetes", "depression"],
            "family_history": True,
            "education_years": 8,
        },
    ),
]

_initialized = False
_init_lock = asyncio.Lock()


def _patient_to_schema(record: PatientRecord) -> Patient:
    return Patient(
        id=record.id,
        name=record.name,
        age=record.age,
        sex=record.sex,
        date_of_birth=record.date_of_birth,
        clinical_data=record.clinical_data,
        created_at=record.created_at.isoformat(),
        last_prediction=record.last_prediction,
    )


def _prediction_to_schema_value(prediction: dict) -> dict:
    data = dict(prediction)
    if isinstance(data.get("prediction_date"), (datetime, date)):
        data["prediction_date"] = data["prediction_date"].isoformat()
    return data


def _clinical_data_to_dict(clinical_data):
    if hasattr(clinical_data, "model_dump"):
        return clinical_data.model_dump(mode="json")
    return clinical_data


async def init_db() -> None:
    global _initialized
    if _initialized:
        return

    async with _init_lock:
        if _initialized:
            return

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with async_session_maker() as session:
            result = await session.execute(select(func.count()).select_from(PatientRecord))
            count = int(result.scalar_one() or 0)
            if count == 0:
                for patient in _seed_patients:
                    last_prediction = _seed_predictions.get(patient.id)
                    session.add(
                        PatientRecord(
                            id=patient.id,
                            name=patient.name,
                            age=patient.age,
                            sex=patient.sex,
                            date_of_birth=patient.date_of_birth,
                            clinical_data=_clinical_data_to_dict(patient.clinical_data),
                            created_at=datetime.now(),
                            last_prediction=_prediction_to_schema_value(last_prediction[-1]) if last_prediction else None,
                        )
                    )

                for patient_id, predictions in _seed_predictions.items():
                    for prediction in predictions:
                        session.add(
                            PredictionRecord(
                                patient_id=patient_id,
                                payload=_prediction_to_schema_value(prediction),
                                created_at=datetime.now(),
                            )
                        )

                await session.commit()

        _initialized = True


async def list_patients() -> List[Patient]:
    await init_db()
    async with async_session_maker() as session:
        result = await session.execute(select(PatientRecord).order_by(PatientRecord.created_at.asc()))
        return [_patient_to_schema(record) for record in result.scalars().all()]


async def get_patient(patient_id: str) -> Optional[PatientDetail]:
    await init_db()
    async with async_session_maker() as session:
        patient_result = await session.get(PatientRecord, patient_id)
        if not patient_result:
            return None

        prediction_result = await session.execute(
            select(PredictionRecord)
            .where(PredictionRecord.patient_id == patient_id)
            .order_by(PredictionRecord.created_at.asc(), PredictionRecord.id.asc())
        )
        predictions = [item.payload for item in prediction_result.scalars().all()]
        return PatientDetail(
            patient=PatientResponse(**_patient_to_schema(patient_result).model_dump()),
            predictions=predictions,
        )


async def create_patient(patient: Patient) -> Patient:
    await init_db()
    async with async_session_maker() as session:
        existing = await session.get(PatientRecord, patient.id)
        if existing:
            raise ValueError(f"Patient {patient.id} already exists")

        record = PatientRecord(
            id=patient.id,
            name=patient.name,
            age=patient.age,
            sex=patient.sex,
            date_of_birth=patient.date_of_birth,
            clinical_data=_clinical_data_to_dict(patient.clinical_data),
            created_at=datetime.now(),
            last_prediction=patient.last_prediction,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return _patient_to_schema(record)


async def save_prediction(patient_id: str, prediction: dict):
    await init_db()
    async with async_session_maker() as session:
        patient = await session.get(PatientRecord, patient_id)
        if not patient:
            return

        prediction_payload = _prediction_to_schema_value(prediction)
        session.add(
            PredictionRecord(
                patient_id=patient_id,
                payload=prediction_payload,
                created_at=datetime.now(),
            )
        )
        patient.last_prediction = prediction_payload
        await session.commit()