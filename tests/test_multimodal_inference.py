# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
import torch
from pathlib import Path
# pyrefly: ignore [missing-import]
from httpx import AsyncClient, ASGITransport

from src.models.hybrid_cnn_resnet_transformer import HybridCNNResNetTransformer3D
from src.datasets.mri_dataset import MRIDataset, normalize_clinical_features
from app.main import app
from app.services.ai_service import model_service


def test_hybrid_model_multimodal_dimensions():
    """Valida o fluxo arquitetural de Late Fusion (ViT 256-d + MLP Clínico 64-d = 320-d -> 3 classes)."""
    model = HybridCNNResNetTransformer3D(
        in_channels=1,
        num_classes=3,
        embed_dim=256,
        clinical_dim=6,
        clinical_hidden_dim=64,
        spatial_size=128,
    )
    model.eval()

    # Batch de 2 exames MRI 3D e 2 vetores de 6 variáveis clínicas
    batch_mri = torch.randn(2, 1, 128, 128, 128)
    batch_clinical = torch.randn(2, 6)

    # Teste de forward multimodal
    logits = model(batch_mri, batch_clinical)
    assert logits.shape == (2, 3), f"Esperado shape (2, 3), obtido {logits.shape}"

    # Teste de extração tabular direta
    clinical_rep = model.clinical_mlp(batch_clinical)
    assert clinical_rep.shape == (2, 64), f"Esperado MLP tabular (2, 64), obtido {clinical_rep.shape}"


def test_hybrid_model_fallback_without_clinical():
    """Valida que o modelo executa perfeitamente sem variáveis clínicas (fallback neutro)."""
    model = HybridCNNResNetTransformer3D(
        in_channels=1,
        num_classes=3,
        embed_dim=256,
        clinical_dim=6,
        clinical_hidden_dim=64,
        spatial_size=128,
    )
    model.eval()

    batch_mri = torch.randn(1, 1, 128, 128, 128)
    logits = model(batch_mri, clinical=None)
    assert logits.shape == (1, 3)


def test_clinical_normalizer_and_imputation():
    """Testa a normalização das 6 variáveis clínicas e imputação para dados faltantes."""
    # Caso 1: Dados clínicos completos com aliases
    full_data = {
        "AGE": 75.0,
        "mmse": 22.0,
        "moca": 19.0,
        "cdr": 0.5,
        "cdrsb": 2.5,
        "education_years": 16,
    }
    vec = normalize_clinical_features(full_data)
    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (6,)
    assert round(float(vec[0]), 4) == round((75.0 - 70.0) / 10.0, 4)  # Age: 0.5
    assert round(float(vec[1]), 4) == round(22.0 / 30.0, 4)           # MMSE: ~0.7333
    assert round(float(vec[2]), 4) == round(19.0 / 30.0, 4)           # MoCA: ~0.6333
    assert round(float(vec[3]), 4) == round(0.5 / 3.0, 4)             # CDR: ~0.1667
    assert round(float(vec[4]), 4) == round(2.5 / 18.0, 4)            # CDR-SB: ~0.1389
    assert round(float(vec[5]), 4) == round(16.0 / 25.0, 4)           # Education: 0.64

    # Caso 2: Dados vazios (deve usar referências clínicas padronizadas sem gerar NaN)
    empty_vec = normalize_clinical_features({})
    assert empty_vec.shape == (6,)
    assert not torch.isnan(empty_vec).any()
    assert float(empty_vec[0]) == 0.0  # (70.0 - 70.0)/10 = 0.0


def test_mri_dataset_multimodal_sample():
    """Valida que MRIDataset retorna o tensor de variáveis clínicas normalizadas."""
    csv_path = "data/unified_metadata.csv"
    assert Path(csv_path).exists()

    ds = MRIDataset(metadata_csv=csv_path, split="train")
    assert len(ds) > 0

    sample = ds[0]
    assert "image" in sample
    assert "label" in sample
    assert "clinical" in sample
    assert isinstance(sample["clinical"], torch.Tensor)
    assert sample["clinical"].shape == (6,)
    assert not torch.isnan(sample["clinical"]).any()


@pytest.mark.asyncio
async def test_e2e_predict_endpoint_full_multimodal():
    """Testa a rota FastAPI /predict com MRI 3D e todas as 6 variáveis clínicas."""
    model_service.load()
    assert model_service.is_loaded is True

    sample_mri = Path("data/sample_mri.nii.gz")
    assert sample_mri.exists()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with open(sample_mri, "rb") as f:
            files = {"mri_file": ("sample_mri.nii.gz", f, "application/gzip")}
            data = {
                "patient_id": "MULTIMODAL_PAT_01",
                "prediction_date": "2026-08-31",
                "age": "73.0",
                "mmse": "24.0",
                "moca": "21.0",
                "cdr": "0.5",
                "cdrsb": "2.0",
                "education_years": "14",
            }
            resp = await client.post("/predict", data=data, files=files)

        assert resp.status_code == 200
        result = resp.json()

        assert result["patient_id"] == "MULTIMODAL_PAT_01"
        assert result["classification"] in ["CN", "MCI", "DEM"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["risk_score"] <= 1.0
        assert "CN" in result["probabilities"]
        assert "MCI" in result["probabilities"]
        assert "DEM" in result["probabilities"]

        # Verifica explicações com fatores clínicos
        explanation_features = [item["feature"] for item in result["explanation"]]
        assert any("MMSE" in f for f in explanation_features)
        assert any("MoCA" in f for f in explanation_features)
        assert any("CDR" in f for f in explanation_features)
        assert any("MRI 3D" in f for f in explanation_features)


@pytest.mark.asyncio
async def test_e2e_predict_endpoint_fallback_without_clinical():
    """Testa a rota FastAPI /predict apenas com MRI 3D (fallback seguro)."""
    model_service.load()
    assert model_service.is_loaded is True

    sample_mri = Path("data/sample_mri.nii.gz")
    assert sample_mri.exists()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with open(sample_mri, "rb") as f:
            files = {"mri_file": ("sample_mri.nii.gz", f, "application/gzip")}
            data = {
                "patient_id": "FALLBACK_PAT_02",
                "prediction_date": "2026-08-31",
            }
            resp = await client.post("/predict", data=data, files=files)

        assert resp.status_code == 200
        result = resp.json()
        assert result["patient_id"] == "FALLBACK_PAT_02"
        assert result["classification"] in ["CN", "MCI", "DEM"]
        assert "probabilities" in result
