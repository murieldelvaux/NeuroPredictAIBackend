from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class ClinicalFeatures(BaseModel):
    """Features clínicas opcionais para alimentar o MLP Tabular e enriquecer a predição multimodal."""
    age: Optional[float] = None
    mmse: Optional[float] = None
    moca: Optional[float] = None
    cdr: Optional[float] = None
    cdrtot: Optional[float] = None
    cdrsb: Optional[float] = None
    education_years: Optional[float] = None
    escolaridade: Optional[float] = None


class FeatureImportance(BaseModel):
    feature: str
    impact: float
    direction: str  # "risk" | "protective"


class ROIStructureVolume(BaseModel):
    name: str = Field(..., description="Nome da estrutura anatômica (Hipocampo, Ventrículo)")
    volume_mm3: float = Field(..., description="Volume exato em milímetros cúbicos (mm³)")
    reference_range_mm3: Optional[str] = Field(None, description="Faixa normativa de referência")
    status: str = Field("normal", description="Status volumétrico (normal, mild_atrophy, severe_atrophy, enlarged)")


class VolumetricReport(BaseModel):
    left_hippocampus_mm3: float = Field(..., description="Volume do Hipocampo Esquerdo em mm³")
    right_hippocampus_mm3: float = Field(..., description="Volume do Hipocampo Direito em mm³")
    total_hippocampus_mm3: float = Field(..., description="Volume Total do Hipocampo em mm³")
    left_lateral_ventricle_mm3: float = Field(..., description="Volume do Ventrículo Lateral Esquerdo em mm³")
    right_lateral_ventricle_mm3: float = Field(..., description="Volume do Ventrículo Lateral Direito em mm³")
    total_lateral_ventricles_mm3: float = Field(..., description="Volume Total dos Ventrículos Laterais em mm³")
    estimated_icv_mm3: float = Field(..., description="Volume Intracraniano Total Estimado (eTIV / ICV) em mm³")
    hippocampal_occupancy_ratio: float = Field(..., description="Taxa de ocupação hipocampal (Total Hipocampo / ICV * 1000)")
    ventricular_enlargement_ratio: float = Field(..., description="Índice de aumento ventricular (Total Ventrículos / ICV * 100)")
    hippocampal_atrophy_index: float = Field(..., description="Índice de atrofia hipocampal comparado a controles normativos (0.0 a 1.0)")
    hippocampal_asymmetry_index: float = Field(..., description="Índice percentual de assimetria inter-hemisférica (%)")
    atrophy_stage: str = Field(..., description="Estadiamento da atrofia (Preservado, Atrofia Leve, Atrofia Moderada, Atrofia Severa)")
    structures: List[ROIStructureVolume] = Field(default_factory=list, description="Detalhamento individual das estruturas anatômicas")


class PredictionOutput(BaseModel):
    patient_id: str
    prediction_date: date = Field(..., description="Data em que a predição foi realizada (dd/mm/aaaa)")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    classification: str  # "CN" | "MCI" | "DEM" | "AD"
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict  # {"CN": 0.1, "MCI": 0.2, "DEM": 0.7}
    explanation: Optional[List[FeatureImportance]] = None
    volumetry: Optional[VolumetricReport] = None
    model_version: str = "hybrid-cnn-resnet-vit-v1"

    model_config = {
        "protected_namespaces": (),
        "json_encoders": {date: lambda d: d.strftime("%d/%m/%Y")},
    }

