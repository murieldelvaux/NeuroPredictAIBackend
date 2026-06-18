from pydantic import BaseModel, Field
from typing import Optional, List


class ClinicalFeatures(BaseModel):
    """Features clínicas opcionais para enriquecer a predição."""
    age: Optional[float] = None
    mmse: Optional[float] = None
    cdr: Optional[float] = None
    cdrtot: Optional[float] = None


class FeatureImportance(BaseModel):
    feature: str
    impact: float
    direction: str  # "risk" | "protective"


class PredictionOutput(BaseModel):
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    classification: str  # "CN" | "MCI" | "AD"
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict  # {"CN": 0.1, "MCI": 0.2, "AD": 0.7}
    explanation: Optional[List[FeatureImportance]] = None
    model_version: str = "resnet3d-oasis3-v1"
