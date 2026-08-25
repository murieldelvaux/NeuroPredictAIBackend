                                                                                                                                                                                                                                                                                       #!/usr/bin/env python3
"""
prepare_dxsum.py

Limpa e padroniza a tabela DXSUM (ADNI Diagnostic Summary) para uso em pipeline longitudinal.

Entrada (raw):
  - CSV baixado na ADNI (ex.: data/adni/adni_metadata.csv)

Saída (interim):
  - data/adni/dxsum_clean.csv  (ou .parquet)
  - com colunas essenciais + derivadas (label_name, visit_month, is_baseline)

Uso:
  python -m src.data_prep.prepare_dxsum \
    --input data/adni/adni_metadata.csv \
    --output data/adni/dxsum_clean.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


# --- Config ---------------------------------------------------------------

DEFAULT_MISSING_SENTINELS = [-4, -1, 9, 99, 999, 9999]

DIAG_MAP = {
    1: "CN",   # cognitively unimpaired/normal
    2: "MCI",
    3: "DEM",  # dementia (ADNI usa "Dementia" como guarda-chuva)
}


@dataclass(frozen=True)
class Summary:
    n_rows_in: int
    n_rows_out: int
    n_ptid: int
    label_counts: dict
    n_missing_dates: int
    n_dropped_missing_label: int
    n_dropped_dupes: int


# --- Helpers --------------------------------------------------------------

def _safe_mkdirs(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _normalize_ptid(ptid: str) -> str:
    # PTID costuma ser algo como "002_S_0295"
    if ptid is None:
        return ptid
    ptid = str(ptid).strip()
    ptid = re.sub(r"\s+", "", ptid)
    return ptid.upper()


def _normalize_viscode2(vis: str) -> str:
    if vis is None:
        return vis
    vis = str(vis).strip()
    vis = re.sub(r"\s+", "", vis)
    return vis.lower()


def _parse_visit_month(viscode2: Optional[str]) -> Optional[int]:
    """
    Converte VISCODE2 em mês aproximado.
    - "bl" -> 0
    - "m06" -> 6
    - "m12" -> 12
    - "sc" (screening) -> -1 (opcionalmente)
    - outros -> None
    """
    if viscode2 is None or pd.isna(viscode2):
        return None
    v = str(viscode2).lower().strip()
    if v == "bl":
        return 0
    if v == "sc":
        return -1
    m = re.fullmatch(r"m(\d+)", v)
    if m:
        return int(m.group(1))
    return None


def _replace_missing_sentinels(df: pd.DataFrame, sentinels: list[int]) -> pd.DataFrame:
    # Substitui sentinelas numéricas por NA.
    # Atenção: só faz sentido para colunas numéricas/objetos com esses valores.
    return df.replace(sentinels, pd.NA)


def _dedupe_visits(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicatas prováveis por (PTID, VISCODE2, EXAMDATE).
    Mantém a última linha após ordenar por EXAMDATE e PHASE (quando existir).
    """
    before = len(df)

    sort_cols = [c for c in ["PTID", "EXAMDATE", "PHASE"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    subset = [c for c in ["PTID", "VISCODE2", "EXAMDATE"] if c in df.columns]
    if len(subset) >= 2:
        df = df.drop_duplicates(subset=subset, keep="last")

    dropped = before - len(df)
    return df, dropped


def _compute_summary(
    n_rows_in: int,
    df_out: pd.DataFrame,
    n_dropped_missing_label: int,
    n_dropped_dupes: int
) -> Summary:
    label_counts = df_out["label_name"].value_counts(dropna=False).to_dict() if "label_name" in df_out.columns else {}
    return Summary(
        n_rows_in=n_rows_in,
        n_rows_out=len(df_out),
        n_ptid=df_out["PTID"].nunique(dropna=True) if "PTID" in df_out.columns else 0,
        label_counts=label_counts,
        n_missing_dates=int(df_out["EXAMDATE"].isna().sum()) if "EXAMDATE" in df_out.columns else 0,
        n_dropped_missing_label=n_dropped_missing_label,
        n_dropped_dupes=n_dropped_dupes,
    )


# --- Core -----------------------------------------------------------------

def prepare_dxsum(
    input_path: str,
    output_path: str,
    output_format: str = "csv",
    drop_missing_label: bool = True,
    missing_sentinels: Optional[list[int]] = None,
) -> Summary:
    missing_sentinels = missing_sentinels or DEFAULT_MISSING_SENTINELS

    df = pd.read_csv(input_path, low_memory=False)
    n_rows_in = len(df)

    # 1) Normaliza sentinelas de missing da ADNI
    df = _replace_missing_sentinels(df, missing_sentinels)

    # 2) Normaliza identificadores
    if "PTID" in df.columns:
        df["PTID"] = df["PTID"].apply(_normalize_ptid)

    if "VISCODE2" in df.columns:
        df["VISCODE2"] = df["VISCODE2"].apply(_normalize_viscode2)
    elif "VISCODE" in df.columns:
        # Se não tiver VISCODE2, ainda vale normalizar VISCODE (mas VISCODE varia entre fases)
        df["VISCODE"] = df["VISCODE"].apply(_normalize_viscode2)

    # 3) Datas
    if "EXAMDATE" in df.columns:
        df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

    # 4) Diagnóstico (label)
    if "DIAGNOSIS" in df.columns:
        # garantir numérico onde possível
        df["DIAGNOSIS"] = pd.to_numeric(df["DIAGNOSIS"], errors="coerce")
        df["label_name"] = df["DIAGNOSIS"].map(DIAG_MAP)
    else:
        df["label_name"] = pd.NA  # evita quebra em downstream

    # 5) Derivadas úteis
    base_col = "VISCODE2" if "VISCODE2" in df.columns else ("VISCODE" if "VISCODE" in df.columns else None)
    if base_col:
        df["visit_month"] = df[base_col].apply(_parse_visit_month)
        df["is_baseline"] = df[base_col].astype("string").str.lower().eq("bl")
    else:
        df["visit_month"] = pd.NA
        df["is_baseline"] = pd.NA

    # 6) Remover linhas sem PTID
    if "PTID" in df.columns:
        df = df.dropna(subset=["PTID"])

    # 7) (Opcional) remover linhas sem label
    n_dropped_missing_label = 0
    if drop_missing_label:
        before = len(df)
        df = df.dropna(subset=["label_name"])
        n_dropped_missing_label = before - len(df)

    # 8) Deduplicar visitas prováveis
    df, n_dropped_dupes = _dedupe_visits(df)

    # 9) Selecionar colunas essenciais + manter o restante (opcional)
    essential = [c for c in ["PTID", "RID", "PHASE", "VISCODE", "VISCODE2", "EXAMDATE", "DIAGNOSIS", "label_name", "DXCONFID",
                             "visit_month", "is_baseline"] if c in df.columns]
    # mantém essenciais primeiro, depois o resto
    remaining = [c for c in df.columns if c not in essential]
    df_out = df[essential + remaining].copy()

    # 10) Salvar
    _safe_mkdirs(output_path)
    fmt = output_format.lower().strip()
    if fmt == "csv" or output_path.lower().endswith(".csv"):
        df_out.to_csv(output_path, index=False)
    elif fmt == "parquet" or output_path.lower().endswith(".parquet"):
        df_out.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"output_format inválido: {output_format}. Use csv ou parquet.")

    summary = _compute_summary(n_rows_in, df_out, n_dropped_missing_label, n_dropped_dupes)
    return summary


# --- CLI ------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Limpeza/padronização do DXSUM (ADNI).")
    p.add_argument("--input", required=True, help="Caminho do CSV bruto (data/adni/adni_metadata.csv)")
    p.add_argument("--output", required=True, help="Caminho do arquivo limpo (data/adni/dxsum_clean.csv|parquet)")
    p.add_argument("--format", default="csv", choices=["csv", "parquet"], help="Formato de saída")
    p.add_argument("--keep-missing-label", action="store_true",
                   help="Não remover linhas sem label_name (default remove).")
    p.add_argument("--missing-sentinels", default=None,
                   help="JSON list de sentinelas para missing (ex.: '[-4,-1,9]'). Default: -4,-1,9,99,999,9999")
    p.add_argument("--summary-json", default=None,
                   help="Opcional: salvar um resumo em JSON (ex.: data/adni/dxsum_summary.json)")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    missing_sentinels = None
    if args.missing_sentinels:
        missing_sentinels = json.loads(args.missing_sentinels)

    summary = prepare_dxsum(
        input_path=args.input,
        output_path=args.output,
        output_format=args.format,
        drop_missing_label=not args.keep_missing_label,
        missing_sentinels=missing_sentinels,
    )

    print("\n=== DXSUM CLEAN SUMMARY ===")
    print(f"Rows in:  {summary.n_rows_in:,}")
    print(f"Rows out: {summary.n_rows_out:,}")
    print(f"PTID uniq:{summary.n_ptid:,}")
    print(f"Missing EXAMDATE: {summary.n_missing_dates:,}")
    print(f"Dropped missing label: {summary.n_dropped_missing_label:,}")
    print(f"Dropped duplicates: {summary.n_dropped_dupes:,}")
    print("Label counts:", summary.label_counts)

    if args.summary_json:
        _safe_mkdirs(args.summary_json)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to: {args.summary_json}")


if __name__ == "__main__":
    main()