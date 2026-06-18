from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
import uuid


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


class PatientDetail(BaseModel):
    patient: Patient
    predictions: List[dict] = Field(default_factory=list)
