from typing import Dict, List, Optional
from app.schemas.patient import Patient, PatientDetail, PatientResponse
from datetime import datetime, date

# Store em memória para MVP — substituir por PostgreSQL em produção
_patients: Dict[str, Patient] = {}
_predictions: Dict[str, List[dict]] = {}

# Seed com pacientes de exemplo
_seed = [
    Patient(id="pat-01", name="Eleanor Vance", age=74, sex="F",
            date_of_birth=date(1950, 3, 14),
            clinical_data={"mmse": 22, "moca": 19, "cdr": 0.5, "cdrtot": 2.5,
                           "comorbidities": ["hypertension"], "family_history": True,
                           "education_years": 12}),
    Patient(id="pat-02", name="Robert Chen", age=68, sex="M",
            date_of_birth=date(1956, 7, 22),
            clinical_data={"mmse": 28, "moca": 27, "cdr": 0.0, "cdrtot": 0.0,
                           "comorbidities": [], "family_history": False,
                           "education_years": 16}),
    Patient(id="pat-03", name="Maria Santos", age=81, sex="F",
            date_of_birth=date(1943, 11, 5),
            clinical_data={"mmse": 17, "moca": 13, "cdr": 1.0, "cdrtot": 5.0,
                           "comorbidities": ["diabetes", "depression"],
                           "family_history": True, "education_years": 8}),
]
for p in _seed:
    p.created_at = datetime.now().isoformat()
    _patients[p.id] = p


def list_patients() -> List[Patient]:
    return list(_patients.values())


def get_patient(patient_id: str) -> Optional[PatientDetail]:
    patient = _patients.get(patient_id)
    if not patient:
        return None
    # PatientDetail.patient exige PatientResponse; converte via model_validate
    patient_response = PatientResponse.model_validate(patient.model_dump())
    return PatientDetail(
        patient=patient_response,
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
