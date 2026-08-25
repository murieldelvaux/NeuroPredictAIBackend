from pydantic import BaseModel, Field, model_serializer
from typing import Optional, List
from datetime import date, datetime
import uuid

_SEX_LABEL = {"M": "Masculino", "F": "Feminino"}


class CognitiveAssessment(BaseModel):
    date: str = Field(..., description="Data da avaliação (DD/MM/AAAA ou YYYY-MM-DD)")
    mmse: Optional[float] = Field(None, ge=0, le=30, description="Mini-Mental State Exam (0-30)")
    moca: Optional[float] = Field(None, ge=0, le=30, description="Montreal Cognitive Assessment (0-30)")
    cdr: Optional[float] = Field(None, description="Clinical Dementia Rating (0, 0.5, 1, 2, 3)")
    cdrtot: Optional[float] = Field(None, description="CDR Sum of Boxes")
    notes: Optional[str] = Field(None, description="Observações clínicas")


class MRIFile(BaseModel):
    filename: str
    content_type: Optional[str] = None
    size: int
    url: Optional[str] = None


class ClinicalData(BaseModel):
    mmse: Optional[float] = Field(None, ge=0, le=30, description="Mini-Mental State Exam (0-30)")
    moca: Optional[float] = Field(None, ge=0, le=30, description="Montreal Cognitive Assessment (0-30)")
    cdr: Optional[float] = Field(None, description="Clinical Dementia Rating (0, 0.5, 1, 2, 3)")
    cdrtot: Optional[float] = Field(None, description="CDR Sum of Boxes")
    biomarkers: List[str] = Field(default_factory=list, description="Lista de biomarcadores")
    symptoms: List[str] = Field(default_factory=list, description="Lista de sintomas atuais")
    medications: List[str] = Field(default_factory=list, description="Lista de medicamentos em uso")
    mri_file: List[MRIFile] = Field(default_factory=list, description="Lista de arquivos de MRI")
    comorbidities: List[str] = Field(default_factory=list)
    family_history: Optional[bool] = None
    education_years: Optional[int] = None
    cognitive_history: List[CognitiveAssessment] = Field(
        default_factory=list,
        description="Histórico temporal de avaliações cognitivas (MMSE, MoCA, CDR)"
    )


class ClinicalDataUpdate(BaseModel):
    mmse: Optional[float] = Field(None, ge=0, le=30)
    moca: Optional[float] = Field(None, ge=0, le=30)
    cdr: Optional[float] = None
    cdrtot: Optional[float] = None
    biomarkers: Optional[List[str]] = None
    symptoms: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    comorbidities: Optional[List[str]] = None
    family_history: Optional[bool] = None
    education_years: Optional[int] = None
    assessment_date: Optional[str] = Field(
        None,
        description="Data da nova avaliação para registrar no histórico de evolução (DD/MM/AAAA ou YYYY-MM-DD)"
    )
    notes: Optional[str] = None


class DiagnosisValidationRequest(BaseModel):
    diagnosis: str = Field(..., pattern="^(CN|MCI|DEM)$", description="Diagnóstico validado pelo médico (CN, MCI, DEM)")
    notes: Optional[str] = Field(None, description="Observações da validação clínica")


class PatientCreate(BaseModel):
    name: str
    age: int = Field(..., ge=18, le=120)
    sex: str = Field(..., pattern="^[MF]$")
    date_of_birth: Optional[date] = None
    clinical_data: ClinicalData
    mri_file: List[MRIFile] = Field(default_factory=list)


class Patient(PatientCreate):
    id: str = Field(default_factory=lambda: f"pat-{uuid.uuid4().hex[:8]}")
    created_at: str = ""
    last_prediction: Optional[dict] = None
    validated_diagnosis: Optional[str] = Field(None, description="Diagnóstico confirmado pelo especialista (CN/MCI/DEM)")
    validated_at: Optional[str] = Field(None, description="Data da validação do diagnóstico")
    validation_notes: Optional[str] = None


def _format_prediction_dict(pred: dict) -> dict:
    """Formata prediction_date dentro de um dict de predição para dd/mm/aaaa."""
    if not pred:
        return pred
    pred = dict(pred)
    raw = pred.get("prediction_date")
    if raw:
        try:
            if isinstance(raw, (date, datetime)):
                pred["prediction_date"] = raw.strftime("%d/%m/%Y")
            else:
                from datetime import date as date_type
                d = date_type.fromisoformat(str(raw)[:10])
                pred["prediction_date"] = d.strftime("%d/%m/%Y")
        except (ValueError, TypeError):
            pass
    return pred


class PatientResponse(Patient):
    """Schema de resposta com datas formatadas como dd/mm/aaaa."""

    @model_serializer(mode="wrap")
    def serialize_with_formatted_dates(self, handler) -> dict:
        data = handler(self)

        # Formata created_at de ISO para dd/mm/aaaa
        if data.get("created_at"):
            try:
                dt = datetime.fromisoformat(data["created_at"])
                data["created_at"] = dt.strftime("%d/%m/%Y")
            except (ValueError, TypeError):
                pass

        # Formata date_of_birth
        if data.get("date_of_birth"):
            try:
                dob = self.date_of_birth
                if isinstance(dob, (date, datetime)):
                    data["date_of_birth"] = dob.strftime("%d/%m/%Y")
            except Exception:
                pass

        # Formata validated_at
        if data.get("validated_at"):
            try:
                dt = datetime.fromisoformat(data["validated_at"])
                data["validated_at"] = dt.strftime("%d/%m/%Y %H:%M")
            except Exception:
                pass

        # Expande sexo: M -> Masculino, F -> Feminino
        if data.get("sex") in _SEX_LABEL:
            data["sex"] = _SEX_LABEL[data["sex"]]

        # Formata prediction_date dentro de last_prediction
        if data.get("last_prediction"):
            data["last_prediction"] = _format_prediction_dict(data["last_prediction"])

        # Mantém uma cópia top-level para compatibilidade com o frontend
        clinical_data = data.get("clinical_data")
        if isinstance(clinical_data, dict):
            data["mri_file"] = clinical_data.get("mri_file", [])

        return data


class PatientDetail(BaseModel):
    patient: PatientResponse
    predictions: List[dict] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_with_formatted_predictions(self, handler) -> dict:
        data = handler(self)
        if data.get("predictions"):
            data["predictions"] = [_format_prediction_dict(p) for p in data["predictions"]]
        return data
