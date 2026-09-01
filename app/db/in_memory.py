from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
import asyncio
import logging
import pandas as pd

# pyrefly: ignore [missing-import]
from sqlalchemy import select, func, delete, text

from app.core.config import settings
from app.db.database import async_session_maker, engine
from app.db.models import Base, PatientRecord, PredictionRecord
from app.schemas.patient import MRIFile, Patient, PatientDetail, PatientResponse, ClinicalDataUpdate, CognitiveAssessment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed data used for local development and tests
# ---------------------------------------------------------------------------
_seed_predictions: dict[str, list[dict]] = {
    "pat-01": [
        {
            "patient_id": "pat-01",
            "prediction_date": date(2025, 11, 10),
            "risk_score": 0.45,
            "classification": "MCI",
            "confidence": 0.65,
            "probabilities": {"CN": 0.35, "MCI": 0.65, "DEM": 0.00},
            "explanation": [
                {"feature": "MMSE Score", "impact": 0.28, "direction": "risk"},
                {"feature": "MRI 3D Features (Hybrid CNN-ResNet-Transformer)", "impact": 0.65, "direction": "risk"},
            ],
            "model_version": "hybrid-transformer-v1",
        },
        {
            "patient_id": "pat-01",
            "prediction_date": date(2026, 7, 1),
            "risk_score": 0.67,
            "classification": "MCI",
            "confidence": 0.78,
            "probabilities": {"CN": 0.12, "MCI": 0.78, "DEM": 0.10},
            "explanation": [
                {"feature": "MMSE Score", "impact": 0.35, "direction": "risk"},
                {"feature": "CDR Rating", "impact": 0.15, "direction": "risk"},
                {"feature": "MRI 3D Features (Hybrid CNN-ResNet-Transformer)", "impact": 0.78, "direction": "risk"},
            ],
            "model_version": "hybrid-transformer-v1",
        },
    ],
    "pat-02": [
        {
            "patient_id": "pat-02",
            "prediction_date": date(2026, 6, 20),
            "risk_score": 0.04,
            "classification": "CN",
            "confidence": 0.96,
            "probabilities": {"CN": 0.96, "MCI": 0.03, "DEM": 0.01},
            "explanation": [
                {"feature": "MRI 3D Features (Hybrid CNN-ResNet-Transformer)", "impact": 0.96, "direction": "protective"},
                {"feature": "MMSE Score", "impact": 0.02, "direction": "protective"},
            ],
            "model_version": "hybrid-transformer-v1",
        },
    ],
    "pat-03": [
        {
            "patient_id": "pat-03",
            "prediction_date": date(2025, 8, 15),
            "risk_score": 0.68,
            "classification": "MCI",
            "confidence": 0.72,
            "probabilities": {"CN": 0.08, "MCI": 0.72, "DEM": 0.20},
            "explanation": [
                {"feature": "MMSE Score", "impact": 0.40, "direction": "risk"},
                {"feature": "CDR Rating", "impact": 0.30, "direction": "risk"},
            ],
            "model_version": "hybrid-transformer-v1",
        },
        {
            "patient_id": "pat-03",
            "prediction_date": date(2026, 7, 5),
            "risk_score": 0.92,
            "classification": "DEM",
            "confidence": 0.89,
            "probabilities": {"CN": 0.01, "MCI": 0.10, "DEM": 0.89},
            "explanation": [
                {"feature": "MRI 3D Features (Hybrid CNN-ResNet-Transformer)", "impact": 0.89, "direction": "risk"},
                {"feature": "MMSE Score", "impact": 0.43, "direction": "risk"},
                {"feature": "CDR Rating", "impact": 0.30, "direction": "risk"},
            ],
            "model_version": "hybrid-transformer-v1",
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
            "mmse": 22.0,
            "moca": 19.0,
            "cdr": 0.5,
            "cdrtot": 2.5,
            "biomarkers": ["ApoE4 positive (ε3/ε4)", "Hippocampal volume loss"],
            "symptoms": ["Mild word retrieval delays", "Subjective short term recall deficits"],
            "medications": ["Donepezil 5mg daily", "Losartan 50mg"],
            "mri_file": [
                {
                    "filename": "pat-01-mri.nii.gz",
                    "content_type": "application/gzip",
                    "size": 15,
                }
            ],
            "comorbidities": ["Hypertension"],
            "family_history": True,
            "education_years": 12,
            "cognitive_history": [
                {"date": "10/05/2024", "mmse": 26.0, "moca": 24.0, "cdr": 0.0, "cdrtot": 0.0, "notes": "Avaliação basal"},
                {"date": "15/11/2025", "mmse": 24.0, "moca": 21.0, "cdr": 0.5, "cdrtot": 1.5, "notes": "Início de queixas de memória"},
                {"date": "01/07/2026", "mmse": 22.0, "moca": 19.0, "cdr": 0.5, "cdrtot": 2.5, "notes": "Piora cognitiva leve"},
            ],
        },
    ),
    Patient(
        id="pat-02",
        name="Robert Chen",
        age=68,
        sex="M",
        date_of_birth=date(1956, 7, 22),
        clinical_data={
            "mmse": 28.0,
            "moca": 27.0,
            "cdr": 0.0,
            "cdrtot": 0.0,
            "biomarkers": ["ApoE4 negative"],
            "symptoms": ["Sem queixas cognitivas objetivas"],
            "medications": ["Atorvastatina 20mg"],
            "mri_file": [
                {
                    "filename": "pat-02-mri.nii.gz",
                    "content_type": "application/gzip",
                    "size": 15,
                }
            ],
            "comorbidities": ["Dislipidemia"],
            "family_history": False,
            "education_years": 16,
            "cognitive_history": [
                {"date": "12/06/2024", "mmse": 29.0, "moca": 28.0, "cdr": 0.0, "cdrtot": 0.0, "notes": "Checkup preventivo"},
                {"date": "20/06/2026", "mmse": 28.0, "moca": 27.0, "cdr": 0.0, "cdrtot": 0.0, "notes": "Cognitivamente estável"},
            ],
        },
    ),
    Patient(
        id="pat-03",
        name="Maria Santos",
        age=81,
        sex="F",
        date_of_birth=date(1943, 11, 5),
        clinical_data={
            "mmse": 14.0,
            "moca": 11.0,
            "cdr": 1.0,
            "cdrtot": 5.0,
            "biomarkers": ["Tau elevated", "Severe temporal atrophy"],
            "symptoms": ["Desorientação têmporo-espacial", "Perda progressiva de autonomia"],
            "medications": ["Memantina 20mg", "Donepezil 10mg"],
            "mri_file": [
                {
                    "filename": "pat-03-mri.nii.gz",
                    "content_type": "application/gzip",
                    "size": 15,
                }
            ],
            "comorbidities": ["Diabetes Tipo 2", "Hipertensão"],
            "family_history": True,
            "education_years": 8,
            "cognitive_history": [
                {"date": "10/01/2024", "mmse": 20.0, "moca": 16.0, "cdr": 0.5, "cdrtot": 2.0, "notes": "MCI diagnosticado"},
                {"date": "15/08/2025", "mmse": 17.0, "moca": 13.0, "cdr": 1.0, "cdrtot": 3.5, "notes": "Evolução para demência leve"},
                {"date": "05/07/2026", "mmse": 14.0, "moca": 11.0, "cdr": 1.0, "cdrtot": 5.0, "notes": "Demência moderada"},
            ],
        },
    ),
]

_initialized = False
_init_lock = asyncio.Lock()


def _mri_storage_root() -> Path:
    return Path(settings.mri_storage_dir)


def _mri_file_url(patient_id: str, filename: str) -> str:
    return f"/patients/{patient_id}/{filename}"


def _safe_filename(filename: Optional[str]) -> str:
    if not filename:
        return "mri.nii.gz"
    return Path(filename).name


def _patient_mri_file_path(patient_id: str, filename: Optional[str]) -> Path:
    return _mri_storage_root() / patient_id / _safe_filename(filename)


async def store_patient_mri_file(patient_id: str, uploaded_file) -> MRIFile:
    file_bytes = await uploaded_file.read()
    safe_name = _safe_filename(uploaded_file.filename)
    file_path = _patient_mri_file_path(patient_id, safe_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(file_bytes)

    return MRIFile(
        filename=safe_name,
        content_type=uploaded_file.content_type,
        size=len(file_bytes),
        url=_mri_file_url(patient_id, safe_name),
    )


def _normalize_mri_files(mri_value) -> List[dict]:
    if isinstance(mri_value, list):
        return [item for item in mri_value if isinstance(item, dict)]
    if isinstance(mri_value, dict):
        return [mri_value]
    return []


def _patient_to_schema(record: PatientRecord) -> Patient:
    clinical_data = dict(record.clinical_data or {})
    mri_files = _normalize_mri_files(clinical_data.get("mri_file"))
    for file_meta in mri_files:
        filename = file_meta.get("filename")
        if filename and not file_meta.get("url"):
            file_meta["url"] = _mri_file_url(record.id, filename)
    clinical_data["mri_file"] = mri_files

    return Patient(
        id=record.id,
        name=record.name,
        age=record.age,
        sex=record.sex,
        date_of_birth=record.date_of_birth,
        clinical_data=clinical_data,
        mri_file=[MRIFile(**item) for item in mri_files],
        created_at=record.created_at.isoformat() if record.created_at else "",
        last_prediction=record.last_prediction,
        validated_diagnosis=record.validated_diagnosis,
        validated_at=record.validated_at.isoformat() if record.validated_at else None,
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
            # Automatic schema migration for existing databases (Postgres / SQLite)
            try:
                await conn.execute(text("ALTER TABLE patients ADD COLUMN IF NOT EXISTS validated_diagnosis VARCHAR(16)"))
            except Exception:
                try:
                    await conn.execute(text("ALTER TABLE patients ADD COLUMN validated_diagnosis VARCHAR(16)"))
                except Exception:
                    pass
            try:
                await conn.execute(text("ALTER TABLE patients ADD COLUMN IF NOT EXISTS validated_at TIMESTAMP"))
            except Exception:
                try:
                    await conn.execute(text("ALTER TABLE patients ADD COLUMN validated_at DATETIME"))
                except Exception:
                    pass

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
                            validated_diagnosis=patient.validated_diagnosis,
                            validated_at=None,
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


def _normalize_date_str(raw_date: Optional[str]) -> str:
    """Padroniza datas para o formato DD/MM/AAAA."""
    if not raw_date:
        return date.today().strftime("%d/%m/%Y")
    s = str(raw_date).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        parts = s[:10].split("-")
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return s


async def create_patient(patient: Patient) -> Patient:
    await init_db()
    async with async_session_maker() as session:
        existing = await session.get(PatientRecord, patient.id)
        if existing:
            raise ValueError(f"Patient {patient.id} already exists")

        c_dict = _clinical_data_to_dict(patient.clinical_data) or {}
        # Se não houver histórico cognitivo explícito mas houver scores iniciais, cria o primeiro ponto
        if not c_dict.get("cognitive_history") and any(c_dict.get(k) is not None for k in ["mmse", "moca", "cdr", "cdrtot"]):
            c_dict["cognitive_history"] = [{
                "date": date.today().strftime("%d/%m/%Y"),
                "mmse": c_dict.get("mmse"),
                "moca": c_dict.get("moca"),
                "cdr": c_dict.get("cdr"),
                "cdrtot": c_dict.get("cdrtot"),
                "notes": "Avaliação inicial",
            }]

        record = PatientRecord(
            id=patient.id,
            name=patient.name,
            age=patient.age,
            sex=patient.sex,
            date_of_birth=patient.date_of_birth,
            clinical_data=c_dict,
            created_at=datetime.now(),
            last_prediction=patient.last_prediction,
            validated_diagnosis=patient.validated_diagnosis,
            validated_at=datetime.now() if patient.validated_diagnosis else None,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return _patient_to_schema(record)


async def update_patient_clinical_data(patient_id: str, update: ClinicalDataUpdate) -> Patient:
    """Atualiza dados clínicos e agrega a nova avaliação ao histórico cognitivo longitudinal."""
    await init_db()
    async with async_session_maker() as session:
        patient = await session.get(PatientRecord, patient_id)
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")

        clinical = dict(patient.clinical_data or {})

        # Atualiza campos clínicos diretos
        if update.mmse is not None:
            clinical["mmse"] = update.mmse
        if update.moca is not None:
            clinical["moca"] = update.moca
        if update.cdr is not None:
            clinical["cdr"] = update.cdr
        if update.cdrtot is not None:
            clinical["cdrtot"] = update.cdrtot
        if update.symptoms is not None:
            clinical["symptoms"] = update.symptoms
        if update.medications is not None:
            clinical["medications"] = update.medications
        if update.comorbidities is not None:
            clinical["comorbidities"] = update.comorbidities
        if update.biomarkers is not None:
            clinical["biomarkers"] = update.biomarkers
        if update.family_history is not None:
            clinical["family_history"] = update.family_history
        if update.education_years is not None:
            clinical["education_years"] = update.education_years

        # Recupera histórico cognitivo existente preservando todos os registros anteriores
        raw_history = list(clinical.get("cognitive_history") or [])
        cog_history = [dict(item) if isinstance(item, dict) else item.model_dump() for item in raw_history]

        assessment_date = _normalize_date_str(update.assessment_date)

        # Se houver scores cognitivos enviados, agrega ao histórico de evolução
        has_cog_update = any([
            update.mmse is not None,
            update.moca is not None,
            update.cdr is not None,
            update.cdrtot is not None,
            update.notes is not None,
            update.assessment_date is not None,
        ])

        if has_cog_update:
            new_entry = {
                "date": assessment_date,
                "mmse": float(update.mmse) if update.mmse is not None else clinical.get("mmse"),
                "moca": float(update.moca) if update.moca is not None else clinical.get("moca"),
                "cdr": float(update.cdr) if update.cdr is not None else clinical.get("cdr"),
                "cdrtot": float(update.cdrtot) if update.cdrtot is not None else clinical.get("cdrtot"),
                "notes": update.notes or "Consulta de acompanhamento",
            }

            # Procura se já existe avaliação para exatamente a mesma data para atualizar, senão adiciona
            found_idx = None
            for idx, item in enumerate(cog_history):
                if _normalize_date_str(item.get("date")) == assessment_date:
                    found_idx = idx
                    break

            if found_idx is not None:
                for k, v in new_entry.items():
                    if v is not None:
                        cog_history[found_idx][k] = v
            else:
                cog_history.append(new_entry)

            # Ordenação cronológica estável por data
            def _date_sort_key(entry):
                d_str = _normalize_date_str(entry.get("date", ""))
                try:
                    parts = d_str.split("/")
                    if len(parts) == 3:
                        return (int(parts[2]), int(parts[1]), int(parts[0]))
                except Exception:
                    pass
                return (9999, 12, 31)

            cog_history.sort(key=_date_sort_key)
            clinical["cognitive_history"] = cog_history

        patient.clinical_data = clinical
        await session.commit()
        await session.refresh(patient)
        return _patient_to_schema(patient)



async def validate_patient_diagnosis(patient_id: str, diagnosis: str, notes: Optional[str] = None) -> Patient:
    """
    Registra a validação do diagnóstico clínico pelo médico especialista.
    Se o paciente possuir arquivo MRI, integra a anotação validada como ground-truth
    no arquivo data/unified_metadata.csv para futuros treinamentos.
    """
    await init_db()
    async with async_session_maker() as session:
        patient = await session.get(PatientRecord, patient_id)
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")

        patient.validated_diagnosis = diagnosis
        patient.validated_at = datetime.now()
        await session.commit()
        await session.refresh(patient)

        # Integração com dataset de treinamento
        _integrate_feedback_into_training(patient_id, diagnosis, patient.clinical_data)

        return _patient_to_schema(patient)


def _integrate_feedback_into_training(patient_id: str, diagnosis: str, clinical_data: dict) -> None:
    """Adiciona o exame validado ao dataset unified_metadata.csv para treino contínuo."""
    csv_file = Path("data/unified_metadata.csv")
    if not csv_file.exists():
        return

    mri_files = _normalize_mri_files(clinical_data.get("mri_file") if clinical_data else None)
    if not mri_files:
        return

    filename = mri_files[0].get("filename")
    if not filename:
        return

    mri_path = str(_patient_mri_file_path(patient_id, filename))
    if not Path(mri_path).exists():
        logger.info(f"Exame MRI {mri_path} não encontrado fisicamente no disco. Ignorando inclusão no dataset de treino.")
        return

    label_map = {"CN": 0, "MCI": 1, "DEM": 2}
    label_id = label_map.get(diagnosis, 0)

    try:
        df = pd.read_csv(csv_file)
        c_dict = dict(clinical_data or {})
        
        age_val = float(c_dict.get("age", 70.0)) if c_dict.get("age") is not None else 70.0
        mmse_val = float(c_dict.get("mmse", 28.0)) if c_dict.get("mmse") is not None else (29.0 if label_id == 0 else (24.0 if label_id == 1 else 18.0))
        moca_val = float(c_dict.get("moca", round(min(30.0, max(0.0, 0.88 * mmse_val + 1.1)), 1))) if c_dict.get("moca") is not None else round(min(30.0, max(0.0, 0.88 * mmse_val + 1.1)), 1)
        cdr_val = float(c_dict.get("cdr", 0.0 if label_id == 0 else (0.5 if label_id == 1 else 1.0))) if c_dict.get("cdr") is not None else (0.0 if label_id == 0 else (0.5 if label_id == 1 else 1.0))
        cdrsb_val = float(c_dict.get("cdrtot", cdr_val * 2.5)) if c_dict.get("cdrtot") is not None else (cdr_val * 2.5)
        educ_val = float(c_dict.get("education_years", 16.0)) if c_dict.get("education_years") is not None else 16.0

        if mri_path in df["image"].values:
            df.loc[df["image"] == mri_path, "label_name"] = diagnosis
            df.loc[df["image"] == mri_path, "label_id"] = label_id
            df.loc[df["image"] == mri_path, "split"] = "train"
            df.loc[df["image"] == mri_path, "AGE"] = age_val
            df.loc[df["image"] == mri_path, "MMSE"] = mmse_val
            df.loc[df["image"] == mri_path, "MOCA"] = moca_val
            df.loc[df["image"] == mri_path, "CDR"] = cdr_val
            df.loc[df["image"] == mri_path, "CDRSB"] = cdrsb_val
            df.loc[df["image"] == mri_path, "EDUC"] = educ_val
            df.loc[df["image"] == mri_path, "CDRTOT"] = cdr_val
        else:
            new_row = {
                "image": mri_path,
                "label_id": label_id,
                "label_name": diagnosis,
                "PTID": patient_id,
                "split": "train",
                "AGE": age_val,
                "MMSE": mmse_val,
                "MOCA": moca_val,
                "CDR": cdr_val,
                "CDRSB": cdrsb_val,
                "EDUC": educ_val,
                "CDRTOT": cdr_val,
                "source": "CLINICAL_FEEDBACK",
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        logger.info(f"Feedback integrado ao dataset para o paciente {patient_id}: {diagnosis}")
    except Exception as e:
        logger.error(f"Erro ao integrar feedback no dataset: {e}")



async def add_patient_mri_file(patient_id: str, uploaded_file) -> MRIFile:
    await init_db()
    mri_meta = await store_patient_mri_file(patient_id, uploaded_file)

    async with async_session_maker() as session:
        patient = await session.get(PatientRecord, patient_id)
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")

        clinical_data = dict(patient.clinical_data or {})
        mri_files = _normalize_mri_files(clinical_data.get("mri_file"))
        mri_files.append(mri_meta.model_dump(mode="json"))
        clinical_data["mri_file"] = mri_files
        patient.clinical_data = clinical_data

        await session.commit()

    return mri_meta


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