from fastapi import APIRouter, HTTPException
from typing import List

from app.schemas.patient import Patient, PatientCreate, PatientDetail
from app.db.in_memory import list_patients, get_patient, create_patient

router = APIRouter()


@router.get("", response_model=List[Patient])
async def get_patients():
    """Lista todos os pacientes registrados."""
    return list_patients()


@router.get("/{patient_id}", response_model=PatientDetail)
async def get_patient_detail(patient_id: str):
    """Retorna perfil completo + histórico de predições de um paciente."""
    detail = get_patient(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return detail


@router.post("", response_model=Patient, status_code=201)
async def create_new_patient(payload: PatientCreate):
    """Cria novo paciente no workspace clínico."""
    patient = Patient(**payload.model_dump())
    return create_patient(patient)
