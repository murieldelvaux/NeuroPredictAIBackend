from pydantic import BaseModel, Field, model_serializer
from typing import Optional, List
from datetime import date
import uuid

_SEX_LABEL = {"M": "Male", "F": "Female"}

class ClinicalData(BaseModel):
    mmse: Optional[float] = Field(None, ge=0, le=30, description="Mini-Mental State Exam (0-30)")
    moca: Optional[float] = Field(None, ge=0, le=30, description="Montreal Cognitive Assessment (0-30)")
    cdr: Optional[float] = Field(None, description="Clinical Dementia Rating (0, 0.5, 1, 2, 3)")
    cdrtot: Optional[float] = Field(None, description="CDR Sum of Boxes")
    comorbidities: List[str] = Field(default_factory=list)
    family_history: Optional[bool] = None
    education_years: Optional[int] = None


class PatientCreate(BaseModel):
    name: str
    age: int = Field(..., ge=18, le=120)
    sex: str = Field(..., pattern="^[MF]$")
    date_of_birth: Optional[date] = None
    clinical_data: Optional[ClinicalData] = None


class Patient(PatientCreate):
    id: str = Field(default_factory=lambda: f"pat-{uuid.uuid4().hex[:8]}")
    created_at: str = ""
    last_prediction: Optional[dict] = None


class PatientResponse(Patient):
    """Schema de resposta com datas formatadas como dd/mm/aaaa."""

    @model_serializer(mode="wrap")
    def serialize_with_formatted_dates(self, handler) -> dict:
        data = handler(self)
        # Formata created_at de ISO para dd/mm/aaaa (ignora a parte de hora se houver)
        if data.get("created_at"):
            try:
                from datetime import datetime
                raw = data["created_at"]
                # Suporta tanto date ISO puro quanto datetime ISO completo
                dt = datetime.fromisoformat(raw)
                data["created_at"] = dt.strftime("%d/%m/%Y")
            except ValueError:
                pass  # mantém o valor original se não conseguir parsear
        # Formata date_of_birth de date ISO para dd/mm/aaaa
        if data.get("date_of_birth"):
            try:
                from datetime import date as date_type
                dob = self.date_of_birth
                if isinstance(dob, date_type):
                    data["date_of_birth"] = dob.strftime("%d/%m/%Y")
            except Exception:
                pass
        # Expande sexo: M -> Male, F -> Female
        if data.get("sex") in _SEX_LABEL:
            data["sex"] = _SEX_LABEL[data["sex"]]

        return data


class PatientDetail(BaseModel):
    patient: PatientResponse
    predictions: List[dict] = Field(default_factory=list)
