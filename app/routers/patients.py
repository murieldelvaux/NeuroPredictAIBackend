from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.schemas.patient import (
    ClinicalData,
    ClinicalDataUpdate,
    DiagnosisValidationRequest,
    MRIFile,
    Patient,
    PatientDetail,
    PatientResponse,
)
from app.db.in_memory import (
    list_patients,
    get_patient,
    create_patient,
    store_patient_mri_file,
    add_patient_mri_file,
    update_patient_clinical_data,
    validate_patient_diagnosis,
    _patient_mri_file_path,
)

router = APIRouter()


def _attach_absolute_mri_url(patient: Patient, request: Request) -> Patient:
    clinical_data = patient.clinical_data
    def _relative_file_url(file_name: str) -> str:
        return f"/patients/{patient.id}/{file_name}"

    if isinstance(clinical_data, dict):
        mri_files = clinical_data.get("mri_file")
        if isinstance(mri_files, list):
            for item in mri_files:
                if isinstance(item, dict) and item.get("filename"):
                    item["url"] = _relative_file_url(item["filename"])
    elif hasattr(clinical_data, "mri_file"):
        mri_files = clinical_data.mri_file
        if isinstance(mri_files, list):
            for item in mri_files:
                if isinstance(item, dict) and item.get("filename"):
                    item["url"] = _relative_file_url(item["filename"])
                elif hasattr(item, "filename") and item.filename:
                    item.url = _relative_file_url(item.filename)

    if isinstance(patient.mri_file, list):
        for item in patient.mri_file:
            if item.filename:
                item.url = _relative_file_url(item.filename)

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
    _attach_absolute_mri_url(detail.patient, request)
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
        created_mri = await store_patient_mri_file(patient.id, mri_file)
        clinical_data_payload.mri_file = [created_mri]
        patient.mri_file = [created_mri]

    created_patient = await create_patient(patient)
    return _attach_absolute_mri_url(created_patient, request)


@router.patch("/{patient_id}/clinical-data", response_model=PatientResponse)
async def update_clinical_data(
    patient_id: str,
    update_data: ClinicalDataUpdate,
    request: Request,
):
    """
    Atualiza dados clínicos do paciente (MMSE, MoCA, CDR, sintomas, comorbidades).
    Se assessment_date for informada, adiciona a avaliação ao histórico de evolução cognitiva.
    """
    try:
        updated = await update_patient_clinical_data(patient_id, update_data)
        return _attach_absolute_mri_url(updated, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{patient_id}/validate-diagnosis", response_model=PatientResponse)
async def validate_diagnosis(
    patient_id: str,
    payload: DiagnosisValidationRequest,
    request: Request,
):
    """
    Valida o diagnóstico clínico do paciente como CN, MCI ou DEM.
    Salva a confirmação médica e integra o exame validado no dataset de treino.
    """
    try:
        updated = await validate_patient_diagnosis(patient_id, payload.diagnosis, payload.notes)
        
        # Dispara re-treinamento contínuo em background de forma segura
        from app.services.training_worker import training_worker
        training_worker.trigger_training(epochs=2, lr=1e-5)
        
        return _attach_absolute_mri_url(updated, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc




@router.post("/{patient_id}/mri-file", response_model=MRIFile)
async def update_patient_mri_file(
    patient_id: str,
    mri_file: UploadFile = File(...),
):
    """Adiciona um novo MRI ao paciente e mantém histórico dos arquivos."""

    try:
        return await add_patient_mri_file(patient_id, mri_file)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{patient_id}/{filename}", name="download_patient_mri_file")
async def download_patient_mri_file(patient_id: str, filename: str):
    """Retorna o MRI salvo do paciente para consumo direto pelo frontend/NiiVue."""

    detail = await get_patient(patient_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"MRI file for patient {patient_id} not found")

    file_path = _patient_mri_file_path(patient_id, filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"MRI file for patient {patient_id} not found")

    return FileResponse(
        path=str(file_path),
        media_type="application/octet-stream",
        filename=file_path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
