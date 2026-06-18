import torch
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    Spacing, ScaleIntensityRange, CropForeground, Resize, ToTensor
)
from monai.networks.nets import resnet10

from app.core.config import settings
from app.schemas.prediction import PredictionOutput, FeatureImportance, ClinicalFeatures

logger = logging.getLogger(__name__)

CLASS_NAMES = ["CN", "MCI", "AD"]


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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model = resnet10(
                pretrained=False,
                n_input_channels=1,
                num_classes=settings.num_classes,
                spatial_dims=3,
            )

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state)
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

    def predict(self, nii_path: str, clinical: Optional[ClinicalFeatures] = None) -> PredictionOutput:
        if not self.is_loaded:
            return self._mock_prediction()

        img_tensor = self.transforms(nii_path)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        classification = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        risk_score = min(float(probs[2] + 0.5 * probs[1]), 1.0)
        probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        explanation = self._build_explanation(probs, clinical)

        return PredictionOutput(
            patient_id="",
            risk_score=risk_score,
            classification=classification,
            confidence=confidence,
            probabilities=probabilities,
            explanation=explanation,
        )

    def _build_explanation(self, probs: np.ndarray, clinical: Optional[ClinicalFeatures]) -> list:
        factors = []
        if clinical:
            if clinical.mmse is not None:
                impact = round((30 - clinical.mmse) / 30 * 0.4, 3)
                factors.append(FeatureImportance(
                    feature="MMSE Score",
                    impact=impact,
                    direction="risk" if clinical.mmse < 24 else "protective"
                ))
            if clinical.cdr is not None and clinical.cdr > 0:
                factors.append(FeatureImportance(
                    feature="CDR Rating",
                    impact=round(clinical.cdr * 0.3, 3),
                    direction="risk"
                ))
            if clinical.age is not None and clinical.age > 70:
                factors.append(FeatureImportance(
                    feature="Age",
                    impact=round((clinical.age - 70) / 100, 3),
                    direction="risk"
                ))
        factors.append(FeatureImportance(
            feature="MRI 3D Features (ResNet)",
            impact=round(float(np.max(probs)), 3),
            direction="risk" if np.argmax(probs) > 0 else "protective"
        ))
        return sorted(factors, key=lambda x: x.impact, reverse=True)

    def _mock_prediction(self) -> PredictionOutput:
        return PredictionOutput(
            patient_id="",
            risk_score=0.0,
            classification="CN",
            confidence=0.0,
            probabilities={"CN": 1.0, "MCI": 0.0, "AD": 0.0},
            explanation=[],
            model_version="mock",
        )


model_service = ModelService()
