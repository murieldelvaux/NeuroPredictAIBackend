from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.schemas.patient import ClinicalData, Patient, PatientDetail, PatientResponse
from app.db.in_memory import list_patients, get_patient, create_patient, store_patient_mri_file, _patient_mri_file_path

router = APIRouter()


def _attach_absolute_mri_url(patient: Patient, request: Request) -> Patient:
    clinical_data = patient.clinical_data
    absolute_url = str(request.url_for("download_patient_mri_file", patient_id=patient.id))

    if isinstance(clinical_data, dict):
        mri_file = clinical_data.get("mri_file")
        if isinstance(mri_file, dict) and mri_file.get("filename"):
            mri_file["url"] = absolute_url
    elif hasattr(clinical_data, "mri_file"):
        mri_file = clinical_data.mri_file
        if mri_file is not None:
            if isinstance(mri_file, dict):
                if mri_file.get("filename"):
                    mri_file["url"] = absolute_url
            elif hasattr(mri_file, "url") and getattr(mri_file, "filename", None):
                mri_file.url = absolute_url
    return patient


@router.get("", response_model=List[PatientResponse])
async def get_patients(request: Request):
    """Lista todos os pacientes registrados com datas no formato dd/mm/aaaa."""
    patients = await list_patients()
    return [_attach_absolute_mri_url(patient, request) for patient in patients]


@router.get("/{patient_id}", response_model=PatientDetail)
async def get_patient_detail(patient_id: str, request: Request):
    """Retorna perfil completo + histórico de predições de um paciente."""
    detail = await get_patient(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    patient = detail.patient
    clinical_data = patient.clinical_data
    if isinstance(clinical_data, dict):
        mri_file = clinical_data.get("mri_file")
        if isinstance(mri_file, dict) and mri_file.get("filename"):
            mri_file["url"] = str(request.url_for("download_patient_mri_file", patient_id=patient_id))
    return detail


@router.post("", response_model=PatientResponse, status_code=201)
async def create_new_patient(
    request: Request,
    name: str = Form(...),
    age: int = Form(..., ge=18, le=120),
    sex: str = Form(..., pattern="^[MF]$"),
    date_of_birth: Optional[date] = Form(None),
    clinical_data: str = Form(...),
    mri_file: Optional[UploadFile] = File(None),
):
    """Cria novo paciente via multipart/form-data com MRI upload real."""

    clinical_data_payload = ClinicalData.model_validate_json(clinical_data)
    patient = Patient(
        name=name,
        age=age,
        sex=sex,
        date_of_birth=date_of_birth,
        clinical_data=clinical_data_payload,
    )

    if mri_file is not None:
        clinical_data_payload.mri_file = await store_patient_mri_file(patient.id, mri_file)
        patient.mri_file = clinical_data_payload.mri_file

    created_patient = await create_patient(patient)
    return _attach_absolute_mri_url(created_patient, request)


@router.get("/{patient_id}/mri-file")
async def download_patient_mri_file(patient_id: str):
    """Retorna o MRI salvo do paciente para consumo direto pelo frontend/NiiVue."""

    detail = await get_patient(patient_id)
    if not detail or not detail.patient.clinical_data.mri_file:
        raise HTTPException(status_code=404, detail=f"MRI file for patient {patient_id} not found")

    file_path = _patient_mri_file_path(patient_id, detail.patient.clinical_data.mri_file.filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"MRI file for patient {patient_id} not found")

    return FileResponse(
        path=str(file_path),
        media_type="application/octet-stream",
        filename=file_path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
