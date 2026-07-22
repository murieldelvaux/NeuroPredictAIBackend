from pydantic import BaseModel, Field, model_serializer
from typing import Optional, List
from datetime import date
import uuid

_SEX_LABEL = {"M": "Masculino", "F": "Feminino"}


class ClinicalData(BaseModel):
    mmse: Optional[float] = Field(None, ge=0, le=30, description="Mini-Mental State Exam (0-30)")
    moca: Optional[float] = Field(None, ge=0, le=30, description="Montreal Cognitive Assessment (0-30)")
    cdr: Optional[float] = Field(None, description="Clinical Dementia Rating (0, 0.5, 1, 2, 3)")
    cdrtot: Optional[float] = Field(None, description="CDR Sum of Boxes")
    biomarkers: List[str] = Field(default_factory=list, description="Lista de biomarcadores")
    symptoms: List[str] = Field(default_factory=list, description="Lista de sintomas atuais")
    medications: List[str] = Field(default_factory=list, description="Lista de medicamentos em uso")
    mri_file: Optional[dict] = Field(None, description="Metadados e conteúdo do arquivo de MRI")
    comorbidities: List[str] = Field(default_factory=list)
    family_history: Optional[bool] = None
    education_years: Optional[int] = None


class PatientCreate(BaseModel):
    name: str
    age: int = Field(..., ge=18, le=120)
    sex: str = Field(..., pattern="^[MF]$")
    date_of_birth: Optional[date] = None
    clinical_data: ClinicalData


class Patient(PatientCreate):
    id: str = Field(default_factory=lambda: f"pat-{uuid.uuid4().hex[:8]}")
    created_at: str = ""
    last_prediction: Optional[dict] = None


def _format_prediction_dict(pred: dict) -> dict:
    """Formata prediction_date dentro de um dict de predição para dd/mm/aaaa."""
    if not pred:
        return pred
    pred = dict(pred)
    raw = pred.get("prediction_date")
    if raw:
        try:
            if isinstance(raw, date):
                pred["prediction_date"] = raw.strftime("%d/%m/%Y")
            else:
                # Pode vir como string ISO (YYYY-MM-DD) após model_dump()
                from datetime import date as date_type
                d = date_type.fromisoformat(str(raw))
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
                from datetime import datetime
                dt = datetime.fromisoformat(data["created_at"])
                data["created_at"] = dt.strftime("%d/%m/%Y")
            except ValueError:
                pass

        # Formata date_of_birth
        if data.get("date_of_birth"):
            try:
                dob = self.date_of_birth
                if isinstance(dob, date):
                    data["date_of_birth"] = dob.strftime("%d/%m/%Y")
            except Exception:
                pass

        # Expande sexo: M -> Masculino, F -> Feminino
        if data.get("sex") in _SEX_LABEL:
            data["sex"] = _SEX_LABEL[data["sex"]]

        # Formata prediction_date dentro de last_prediction
        if data.get("last_prediction"):
            data["last_prediction"] = _format_prediction_dict(data["last_prediction"])

        return data


class PatientDetail(BaseModel):
    patient: PatientResponse
    predictions: List[dict] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_with_formatted_predictions(self, handler) -> dict:
        data = handler(self)
        # Formata prediction_date em cada predição da lista
        if data.get("predictions"):
            data["predictions"] = [_format_prediction_dict(p) for p in data["predictions"]]
        return data
