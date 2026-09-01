from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Default clinical reference values for imputation when values are missing
CLINICAL_DEFAULTS = {
    "age": 70.0,
    "mmse": 28.0,
    "moca": 26.0,
    "cdr": 0.0,
    "cdrsb": 0.0,
    "education": 16.0,
}


def _get_val(data: Mapping[str, Any] | pd.Series, keys: list[str], default: float) -> float:
    """Extrai o primeiro valor existente e não-nulo para uma lista de chaves sinônimas."""
    for k in keys:
        if k in data:
            v = data[k]
            if v is not None and not pd.isna(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
    return default


def normalize_clinical_features(
    data: Mapping[str, Any] | pd.Series | None = None,
    age: Optional[float] = None,
    mmse: Optional[float] = None,
    moca: Optional[float] = None,
    cdr: Optional[float] = None,
    cdrsb: Optional[float] = None,
    education: Optional[float] = None,
) -> torch.Tensor:
    """
    Normaliza as 6 variáveis clínicas para o MLP Tabular (6-dimensões):
      [0] Idade (AGE): Centralizada em 70.0 anos, escala 10.0 -> ~[-2.5, +2.5]
      [1] MMSE: Escala normalizada [0, 30] -> [0.0, 1.0]
      [2] MoCA: Escala normalizada [0, 30] -> [0.0, 1.0]
      [3] CDR: Escala [0, 3] -> [0.0, 1.0]
      [4] CDR-SB (CDRSUM): Escala [0, 18] -> [0.0, 1.0]
      [5] Escolaridade (EDUC / anos): Escala [0, 25] -> [0.0, 1.0]
      
    Trata automaticamente NaNs e valores ausentes via imputação com medianas de referência.
    """
    d = data if data is not None else {}

    v_age = age if age is not None else _get_val(d, ["AGE", "age", "Idade", "idade"], CLINICAL_DEFAULTS["age"])
    v_mmse = mmse if mmse is not None else _get_val(d, ["MMSE", "mmse", "mmse_score"], CLINICAL_DEFAULTS["mmse"])
    
    # Se MoCA não estiver presente mas MMSE estiver, estima MoCA via regressão clínica
    moca_default = round(min(30.0, max(0.0, 0.88 * v_mmse + 1.1)), 1) if v_mmse is not None else CLINICAL_DEFAULTS["moca"]
    v_moca = moca if moca is not None else _get_val(d, ["MOCA", "moca", "mocatots", "MoCA"], moca_default)
    
    v_cdr = cdr if cdr is not None else _get_val(d, ["CDR", "cdr", "CDRTOT", "cdrtot"], CLINICAL_DEFAULTS["cdr"])
    
    # Se CDR-SB não estiver presente, estima a partir do CDR global (CDR * 2.5)
    cdrsb_default = v_cdr * 2.5 if v_cdr is not None else CLINICAL_DEFAULTS["cdrsb"]
    v_cdrsb = cdrsb if cdrsb is not None else _get_val(d, ["CDRSB", "cdrsb", "CDRSUM", "cdr_sb", "cdrtot"], cdrsb_default)
    
    v_educ = education if education is not None else _get_val(d, ["EDUC", "educ", "education_years", "escolaridade", "Escolaridade"], CLINICAL_DEFAULTS["education"])

    # Normalizações numéricas com bounds seguros
    norm_age = (float(v_age) - 70.0) / 10.0
    norm_mmse = np.clip(float(v_mmse) / 30.0, 0.0, 1.0)
    norm_moca = np.clip(float(v_moca) / 30.0, 0.0, 1.0)
    norm_cdr = np.clip(float(v_cdr) / 3.0, 0.0, 1.0)
    norm_cdrsb = np.clip(float(v_cdrsb) / 18.0, 0.0, 1.0)
    norm_educ = np.clip(float(v_educ) / 25.0, 0.0, 1.0)

    vec = np.array([norm_age, norm_mmse, norm_moca, norm_cdr, norm_cdrsb, norm_educ], dtype=np.float32)
    return torch.from_numpy(vec)


class MRIDataset(Dataset):
    """
    Dataset multimodal para carregar imagens volumétricas MRI T1 e variáveis clínicas tabulares.
    """
    def __init__(self, metadata_csv: str, split: str, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        # Filtra apenas imagens que existem em disco para prevenir erros de I/O em tempo de execução
        self.metadata = self.metadata[self.metadata["image"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        self.transform = transform


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = Path(row["image"])
        label = int(row["label_id"])
        clinical_tensor = normalize_clinical_features(row)
        
        sample = {
            "image": str(image_path),
            "label": label,
            "clinical": clinical_tensor,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # Garante que "clinical" e "label" estejam sempre presentes após as transformações MONAI
        if isinstance(sample, dict):
            if "clinical" not in sample:
                sample["clinical"] = clinical_tensor
            if "label" not in sample:
                sample["label"] = label

        return sample
