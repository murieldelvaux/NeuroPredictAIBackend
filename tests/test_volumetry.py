import pytest
import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from src.services.volumetry_service import volumetry_service, VolumetryService
from app.main import app
from app.services.ai_service import model_service


def test_volumetry_service_sample_mri():
    """Valida o cálculo quantitativo de volumetria em mm³ e índices de atrofia no exame real."""
    sample_mri = Path("data/sample_mri.nii.gz")
    assert sample_mri.exists()

    report = volumetry_service.segment_and_quantify(sample_mri)
    assert isinstance(report, dict)

    # Validação de Hipocampo
    assert report["left_hippocampus_mm3"] > 1000.0
    assert report["right_hippocampus_mm3"] > 1000.0
    assert abs(report["total_hippocampus_mm3"] - (report["left_hippocampus_mm3"] + report["right_hippocampus_mm3"])) < 0.2

    # Validação de Ventrículos Laterais
    assert report["left_lateral_ventricle_mm3"] > 1000.0
    assert report["right_lateral_ventricle_mm3"] > 1000.0
    assert abs(report["total_lateral_ventricles_mm3"] - (report["left_lateral_ventricle_mm3"] + report["right_lateral_ventricle_mm3"])) < 0.2

    # Validação de Volume Intracraniano Total (eTIV / ICV)
    assert report["estimated_icv_mm3"] >= 900000.0

    # Validação dos índices clínicos
    assert report["hippocampal_occupancy_ratio"] > 0.0
    assert report["ventricular_enlargement_ratio"] > 0.0
    assert 0.0 <= report["hippocampal_atrophy_index"] <= 1.0
    assert report["hippocampal_asymmetry_index"] >= 0.0
    assert report["atrophy_stage"] in ["Preservado", "Atrofia Leve", "Atrofia Moderada", "Atrofia Severa"]

    # Validação das 4 estruturas anatômicas detalhadas
    assert len(report["structures"]) == 4
    names = [s["name"] for s in report["structures"]]
    assert "Hipocampo Esquerdo" in names
    assert "Hipocampo Direito" in names
    assert "Ventrículo Lateral Esquerdo" in names
    assert "Ventrículo Lateral Direito" in names
    for struct in report["structures"]:
        assert struct["volume_mm3"] > 0.0
        assert struct["reference_range_mm3"]
        assert struct["status"] in ["normal", "mild_atrophy", "severe_atrophy", "enlarged_mild", "enlarged_severe"]


def test_volumetry_synthetic_nifti_custom_spacing():
    """Valida a precisão do cálculo numérico de volume em mm³ com espaçamento anisotrópico."""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Cria volume sintético 64x64x64 com zooms de (1.5, 1.5, 2.0) mm -> voxel = 4.5 mm³
        shape = (64, 64, 64)
        affine = np.diag([1.5, 1.5, 2.0, 1.0])
        synthetic_data = np.zeros(shape, dtype=np.float32)
        # Parênquima cerebral
        synthetic_data[16:48, 16:48, 16:48] = 500.0
        # Hipocampo sintético
        synthetic_data[18:24, 20:30, 20:26] = 300.0
        synthetic_data[40:46, 20:30, 20:26] = 300.0

        nii = nib.Nifti1Image(synthetic_data, affine)
        nib.save(nii, tmp_path)

        srv = VolumetryService()
        res = srv.segment_and_quantify(tmp_path)
        assert res["total_hippocampus_mm3"] > 0.0
        assert res["estimated_icv_mm3"] > 0.0
        assert "structures" in res
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_predict_endpoint_includes_volumetry_report():
    """Testa se a rota FastAPI /predict retorna o relatório volumétrico completo na resposta JSON."""
    model_service.load()
    assert model_service.is_loaded is True

    sample_mri = Path("data/sample_mri.nii.gz")
    assert sample_mri.exists()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with open(sample_mri, "rb") as f:
            files = {"mri_file": ("sample_mri.nii.gz", f, "application/gzip")}
            data = {
                "patient_id": "VOLUMETRY_PAT_01",
                "prediction_date": "2026-08-31",
                "age": "71.0",
                "mmse": "25.0",
                "cdr": "0.5",
            }
            resp = await client.post("/predict", data=data, files=files)

        assert resp.status_code == 200
        body = resp.json()

        assert "volumetry" in body
        vol = body["volumetry"]
        assert vol is not None
        assert "left_hippocampus_mm3" in vol
        assert "right_hippocampus_mm3" in vol
        assert "total_hippocampus_mm3" in vol
        assert "left_lateral_ventricle_mm3" in vol
        assert "right_lateral_ventricle_mm3" in vol
        assert "total_lateral_ventricles_mm3" in vol
        assert "hippocampal_occupancy_ratio" in vol
        assert "atrophy_stage" in vol
        assert len(vol["structures"]) == 4

        # Verifica se o relatório de volumetria também foi integrado à explicação XAI
        explanations = [item["feature"] for item in body["explanation"]]
        assert any("Volumetria" in exp for exp in explanations)
