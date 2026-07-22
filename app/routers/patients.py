from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.schemas.patient import ClinicalData, Patient, PatientDetail, PatientResponse
from app.db.in_memory import list_patients, get_patient, create_patient

router = APIRouter()


@router.get("", response_model=List[PatientResponse])
async def get_patients():
    """Lista todos os pacientes registrados com datas no formato dd/mm/aaaa."""
    return list_patients()


@router.get("/{patient_id}", response_model=PatientDetail)
async def get_patient_detail(patient_id: str):
    """Retorna perfil completo + histórico de predições de um paciente."""
    detail = get_patient(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return detail


@router.post("", response_model=PatientResponse, status_code=201)
async def create_new_patient(
    name: str = Form(...),
    age: int = Form(..., ge=18, le=120),
    sex: str = Form(..., pattern="^[MF]$"),
    date_of_birth: Optional[date] = Form(None),
    clinical_data: str = Form(...),
    mri_file: Optional[UploadFile] = File(None),
):
    """Cria novo paciente via multipart/form-data com MRI upload real."""

    clinical_data_payload = ClinicalData.model_validate_json(clinical_data)
    if mri_file is not None:
        file_bytes = await mri_file.read()
        clinical_data_payload.mri_file = {
            "filename": mri_file.filename,
            "content_type": mri_file.content_type,
            "size": len(file_bytes),
        }

    patient = Patient(
        name=name,
        age=age,
        sex=sex,
        date_of_birth=date_of_birth,
        clinical_data=clinical_data_payload,
    )
    return create_patient(patient)
