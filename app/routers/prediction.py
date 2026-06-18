from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import tempfile
import shutil
import os

from app.schemas.prediction import PredictionOutput, ClinicalFeatures
from app.services.ai_service import model_service
from app.db.in_memory import save_prediction

router = APIRouter()


@router.post("", response_model=PredictionOutput)
async def predict(
    patient_id: str = Form(...),
    mri_file: Optional[UploadFile] = File(None),
    age: Optional[float] = Form(None),
    mmse: Optional[float] = Form(None),
    cdr: Optional[float] = Form(None),
    cdrtot: Optional[float] = Form(None),
):
    """
    Roda inferência 3D ResNet no MRI e retorna classificação CN/MCI/AD.
    O arquivo .nii.gz é obrigatório para predição completa.
    Features clínicas são opcionais e enriquecem a explicação SHAP.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Check checkpoint path.")

    if mri_file is None:
        raise HTTPException(
            status_code=400,
            detail="MRI file (.nii.gz) is required for prediction."
        )

    # Salva o upload em arquivo temporário para o MONAI processar
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        shutil.copyfileobj(mri_file.file, tmp)
        tmp_path = tmp.name

    try:
        clinical = ClinicalFeatures(age=age, mmse=mmse, cdr=cdr, cdrtot=cdrtot)
        result = model_service.predict(tmp_path, clinical)
        result.patient_id = patient_id

        # Persiste predição no histórico do paciente
        save_prediction(patient_id, result.model_dump())
        return result

    finally:
        os.unlink(tmp_path)
