from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import tempfile
import shutil
import os
from datetime import date

from app.schemas.prediction import PredictionOutput, ClinicalFeatures
from app.services.ai_service import model_service
from app.db.in_memory import save_prediction

router = APIRouter()


def _nii_suffix(filename: Optional[str]) -> str:
    """
    Returns the correct suffix for the temp file based on the original filename.
    Supports: .nii.gz  .nii  .gz  (nibabel detects format by extension)
    """
    if not filename:
        return ".nii.gz"
    name = filename.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    # fallback — keep whatever extension was sent
    _, ext = os.path.splitext(name)
    return ext or ".nii.gz"


@router.post("", response_model=PredictionOutput)
async def predict(
    patient_id: str = Form(...),
    prediction_date: date = Form(..., description="Data da predição no formato YYYY-MM-DD"),
    mri_file: Optional[UploadFile] = File(None),
    age: Optional[float] = Form(None),
    mmse: Optional[float] = Form(None),
    cdr: Optional[float] = Form(None),
    cdrtot: Optional[float] = Form(None),
):
    """
    Roda inferência 3D ResNet no MRI e retorna classificação CN/MCI/AD.
    O arquivo .nii ou .nii.gz é obrigatório para predição completa.
    Features clínicas são opcionais e enriquecem a explicação SHAP.
    `prediction_date` é obrigatório e deve ser enviado no formato YYYY-MM-DD.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Check checkpoint path.")

    if mri_file is None:
        raise HTTPException(
            status_code=400,
            detail="MRI file (.nii or .nii.gz) is required for prediction."
        )

    suffix = _nii_suffix(mri_file.filename)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(mri_file.file, tmp)
        tmp_path = tmp.name

    try:
        clinical = ClinicalFeatures(age=age, mmse=mmse, cdr=cdr, cdrtot=cdrtot)
        result = model_service.predict(tmp_path, clinical, prediction_date=prediction_date)
        result.patient_id = patient_id

        await save_prediction(patient_id, result.model_dump(mode="json"))
        return result
    finally:
        if tmp_path:
            os.unlink(tmp_path)
