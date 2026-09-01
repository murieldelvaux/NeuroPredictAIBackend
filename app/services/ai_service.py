from __future__ import annotations

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import logging
from datetime import date

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    Spacing, ScaleIntensityRange, CropForeground, Resize, ToTensor,
)

from app.core.config import settings
from app.schemas.prediction import PredictionOutput, FeatureImportance, ClinicalFeatures

logger = logging.getLogger(__name__)

CLASS_NAMES = ["CN", "MCI", "DEM"]


def _load_hybrid_model(num_classes: int) -> torch.nn.Module:
    """
    Carrega o modelo HybridCNNResNetTransformer3D a partir do pacote local src.models.
    """
    try:
        from src.models.hybrid_cnn_resnet_transformer import HybridCNNResNetTransformer3D
        logger.info("HybridCNNResNetTransformer3D importado com sucesso.")
        return HybridCNNResNetTransformer3D(in_channels=1, num_classes=num_classes)
    except ImportError:
        from src.models.cnn_3d import SimpleResNet3D
        logger.info("Fallback para SimpleResNet3D.")
        return SimpleResNet3D(in_channels=1, num_classes=num_classes)


class ModelService:
    def __init__(self):
        self.model = None
        self.device = None
        self.transforms = None
        self.is_loaded = False

    def load(self):
        checkpoint_path = Path(settings.checkpoint_path)

        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}. "
                "Starting in mock mode — predictions will be simulated."
            )
            self.is_loaded = False
            return

        try:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

            self.model = _load_hybrid_model(num_classes=settings.num_classes)

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state = checkpoint.get("model_state_dict", checkpoint)
            try:
                self.model.load_state_dict(state, strict=True)
            except Exception as load_err:
                logger.warning(f"Carregando pesos com strict=False: {load_err}")
                self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()

            spatial = [settings.spatial_size] * 3
            self.transforms = Compose([
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes="RAS"),
                Spacing(pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
                ScaleIntensityRange(a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
                CropForeground(),
                Resize(spatial_size=spatial),
                ToTensor(),
            ])

            self.is_loaded = True
            logger.info(f"Model loaded from {checkpoint_path} on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False

    def unload(self):
        self.model = None
        self.is_loaded = False

    def _prepare_clinical_tensor(self, clinical: Optional[ClinicalFeatures]) -> torch.Tensor:
        from src.datasets.mri_dataset import normalize_clinical_features
        if clinical is None:
            return normalize_clinical_features().unsqueeze(0).to(self.device)

        return normalize_clinical_features(
            age=clinical.age,
            mmse=clinical.mmse,
            moca=clinical.moca,
            cdr=clinical.cdr,
            cdrsb=clinical.cdrsb if clinical.cdrsb is not None else clinical.cdrtot,
            education=clinical.education_years if clinical.education_years is not None else clinical.escolaridade,
        ).unsqueeze(0).to(self.device)

    def predict(
        self,
        nii_path: str,
        clinical: Optional[ClinicalFeatures] = None,
        prediction_date: Optional[date] = None,
    ) -> PredictionOutput:
        if not self.is_loaded:
            return self._mock_prediction(prediction_date=prediction_date)

        img_tensor = self.transforms(nii_path)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        clinical_tensor = self._prepare_clinical_tensor(clinical)

        with torch.no_grad():
            logits = self.model(img_tensor, clinical_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        classification = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        risk_score = min(float(probs[2] + 0.5 * probs[1]), 1.0)
        probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        # Executa segmentação e cálculo volumétrico anatômico (mm³)
        volumetric_report = None
        try:
            from src.services.volumetry_service import volumetry_service
            from app.schemas.prediction import VolumetricReport
            raw_volumetry = volumetry_service.segment_and_quantify(nii_path)
            volumetric_report = VolumetricReport(**raw_volumetry)
        except Exception as vol_err:
            logger.warning(f"Volumetria não calculada para {nii_path}: {vol_err}")

        explanation = self._build_explanation(probs, clinical, volumetric_report)

        return PredictionOutput(
            patient_id="",
            prediction_date=prediction_date or date.today(),
            risk_score=risk_score,
            classification=classification,
            confidence=confidence,
            probabilities=probabilities,
            explanation=explanation,
            volumetry=volumetric_report,
        )

    def _build_explanation(
        self,
        probs: np.ndarray,
        clinical: Optional[ClinicalFeatures],
        volumetry: Optional[Any] = None,
    ) -> list:
        factors = []
        if clinical:
            if clinical.mmse is not None:
                impact = round(abs(30 - clinical.mmse) / 30 * 0.4, 3)
                factors.append(FeatureImportance(
                    feature="MMSE Score",
                    impact=impact,
                    direction="risk" if clinical.mmse < 24 else "protective",
                ))
            if clinical.moca is not None:
                impact = round(abs(30 - clinical.moca) / 30 * 0.35, 3)
                factors.append(FeatureImportance(
                    feature="MoCA Score",
                    impact=impact,
                    direction="risk" if clinical.moca < 26 else "protective",
                ))
            if clinical.cdr is not None and clinical.cdr > 0:
                factors.append(FeatureImportance(
                    feature="CDR Rating",
                    impact=round(clinical.cdr * 0.3, 3),
                    direction="risk",
                ))
            cdrsb = clinical.cdrsb if clinical.cdrsb is not None else clinical.cdrtot
            if cdrsb is not None and cdrsb > 0.5:
                factors.append(FeatureImportance(
                    feature="CDR Sum of Boxes (CDR-SB)",
                    impact=round(min(1.0, cdrsb / 18.0) * 0.35, 3),
                    direction="risk",
                ))
            if clinical.age is not None and clinical.age > 70:
                factors.append(FeatureImportance(
                    feature="Age",
                    impact=round((clinical.age - 70) / 100, 3),
                    direction="risk",
                ))
            educ = clinical.education_years if clinical.education_years is not None else clinical.escolaridade
            if educ is not None:
                factors.append(FeatureImportance(
                    feature="Education (Years)",
                    impact=round(min(1.0, educ / 25.0) * 0.2, 3),
                    direction="protective" if educ >= 12 else "risk",
                ))

        if volumetry is not None:
            hipp_impact = round(float(getattr(volumetry, "hippocampal_atrophy_index", 0.1) * 0.45), 3)
            stage = getattr(volumetry, "atrophy_stage", "Preservado")
            total_hipp = getattr(volumetry, "total_hippocampus_mm3", 7800.0)
            factors.append(FeatureImportance(
                feature=f"Volumetria do Hipocampo ({total_hipp:.0f} mm³ - {stage})",
                impact=hipp_impact if hipp_impact > 0.05 else 0.18,
                direction="risk" if stage != "Preservado" else "protective",
            ))

        factors.append(FeatureImportance(
            feature="MRI 3D Features (Hybrid CNN-ResNet-Transformer)",
            impact=round(float(np.max(probs)), 3),
            direction="risk" if np.argmax(probs) > 0 else "protective",
        ))
        return sorted(factors, key=lambda x: x.impact, reverse=True)

    def _mock_prediction(self, prediction_date: Optional[date] = None) -> PredictionOutput:
        return PredictionOutput(
            patient_id="",
            prediction_date=prediction_date or date.today(),
            risk_score=0.0,
            classification="CN",
            confidence=0.0,
            probabilities={"CN": 1.0, "MCI": 0.0, "DEM": 0.0},
            explanation=[],
            volumetry=None,
            model_version="mock",
        )


model_service = ModelService()

