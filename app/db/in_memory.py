from typing import Dict, List, Optional
from app.schemas.patient import Patient, PatientDetail, PatientResponse
from datetime import datetime, date

# Store em memória para MVP — substituir por PostgreSQL em produção
_patients: Dict[str, Patient] = {}
_predictions: Dict[str, List[dict]] = {}

# ---------------------------------------------------------------------------
# Predições de seed — cobrindo todos os campos de PredictionOutput
# ---------------------------------------------------------------------------
_seed_predictions: Dict[str, List[dict]] = {
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

# Seed com pacientes de exemplo
_seed = [
    Patient(id="pat-01", name="Eleanor Vance", age=74, sex="F",
            date_of_birth=date(1950, 3, 14),
            clinical_data={"mmse": 22, "moca": 19, "cdr": 0.5, "cdrtot": 2.5,
               "biomarkers": ["abeta42_low", "tau_high"],
               "symptoms": ["memory_loss", "disorientation"],
               "medications": ["donepezil", "losartan"],
               "mri_file": {
                   "filename": "pat-01-mri.nii.gz",
                   "content_type": "application/gzip",
                   "size": 15,
               },
                           "comorbidities": ["hypertension"], "family_history": True,
                           "education_years": 12}),
    Patient(id="pat-02", name="Robert Chen", age=68, sex="M",
            date_of_birth=date(1956, 7, 22),
            clinical_data={"mmse": 28, "moca": 27, "cdr": 0.0, "cdrtot": 0.0,
               "biomarkers": ["abeta42_normal", "ptau_normal"],
               "symptoms": ["mild_forgetfulness"],
               "medications": ["atorvastatin"],
               "mri_file": {
                   "filename": "pat-02-mri.nii.gz",
                   "content_type": "application/gzip",
                   "size": 15,
               },
                           "comorbidities": [], "family_history": False,
                           "education_years": 16}),
    Patient(id="pat-03", name="Maria Santos", age=81, sex="F",
            date_of_birth=date(1943, 11, 5),
            clinical_data={"mmse": 17, "moca": 13, "cdr": 1.0, "cdrtot": 5.0,
               "biomarkers": ["tau_very_high", "hippocampal_atrophy"],
               "symptoms": ["memory_loss", "language_difficulty", "apathy"],
               "medications": ["memantine", "sertraline", "metformin"],
               "mri_file": {
                   "filename": "pat-03-mri.nii.gz",
                   "content_type": "application/gzip",
                   "size": 15,
               },
                           "comorbidities": ["diabetes", "depression"],
                           "family_history": True, "education_years": 8}),
]
for p in _seed:
    p.created_at = datetime.now().isoformat()
    preds = _seed_predictions.get(p.id, [])
    _predictions[p.id] = preds
    if preds:
        p.last_prediction = preds[-1]
    _patients[p.id] = p


def list_patients() -> List[Patient]:
    return list(_patients.values())


def get_patient(patient_id: str) -> Optional[PatientDetail]:
    patient = _patients.get(patient_id)
    if not patient:
        return None
    return PatientDetail(
        patient=PatientResponse(**patient.model_dump()),
        predictions=_predictions.get(patient_id, []),
    )


def create_patient(patient: Patient) -> Patient:
    patient.created_at = datetime.now().isoformat()
    _patients[patient.id] = patient
    return patient


def save_prediction(patient_id: str, prediction: dict):
    if patient_id not in _predictions:
        _predictions[patient_id] = []
    _predictions[patient_id].append(prediction)
    if patient_id in _patients:
        _patients[patient_id].last_prediction = prediction
