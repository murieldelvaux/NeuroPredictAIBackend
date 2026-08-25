#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Prioridade do "pior estágio" para estratificação por paciente
STAGE_PRIORITY = {"CN": 0, "MCI": 1, "DEM": 2}

def patient_stage_from_visits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna 1 linha por PTID com um stage para estratificar:
    - DEM se o paciente tiver qualquer visita DEM
    - senão MCI se tiver qualquer visita MCI
    - senão CN
    """
    tmp = df[["PTID", "label_name"]].dropna().copy()
    tmp["stage_rank"] = tmp["label_name"].map(STAGE_PRIORITY)
    # pega o pior (maior rank)
    pat = tmp.groupby("PTID", as_index=False)["stage_rank"].max()
    inv = {v: k for k, v in STAGE_PRIORITY.items()}
    pat["patient_stage"] = pat["stage_rank"].map(inv)
    return pat[["PTID", "patient_stage"]]

def summarize_split(df_visits: pd.DataFrame, ptids: list[str], split_name: str) -> dict:
    sub = df_visits[df_visits["PTID"].isin(ptids)]
    return {
        "split": split_name,
        "ptids": int(pd.Series(ptids).nunique()),
        "visits": int(len(sub)),
        "label_counts_visits": sub["label_name"].value_counts().to_dict(),
    }

def main():
    ap = argparse.ArgumentParser(description="Cria splits train/val/test por PTID (anti-leakage).")
    ap.add_argument("--input", required=True, help="CSV rotulado (ex.: data/adni/dxsum_labeled.csv)")
    ap.add_argument("--output", required=True, help="JSON de saída (ex.: data/adni/splits.json)")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df = df.dropna(subset=["PTID", "label_name"]).copy()

    pat = patient_stage_from_visits(df)

    # Primeiro separa teste
    train_val, test = train_test_split(
        pat,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=pat["patient_stage"],
    )

    # Agora separa validação do restante
    # val-size é fração do total; precisamos converter para fração do train_val
    val_frac_of_trainval = args.val_size / (1.0 - args.test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_frac_of_trainval,
        random_state=args.seed,
        stratify=train_val["patient_stage"],
    )

    splits = {
        "train_ptids": sorted(train["PTID"].tolist()),
        "val_ptids": sorted(val["PTID"].tolist()),
        "test_ptids": sorted(test["PTID"].tolist()),
        "meta": {
            "seed": args.seed,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "stratify_by": "patient_stage(worst_observed)",
            "stage_priority": STAGE_PRIORITY,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    # Relatório no terminal
    print("Saved:", args.output)
    print("Total PTIDs:", pat["PTID"].nunique())
    print("Patient-stage distribution:\n", pat["patient_stage"].value_counts())

    for name, ptid_list in [
        ("train", splits["train_ptids"]),
        ("val", splits["val_ptids"]),
        ("test", splits["test_ptids"]),
    ]:
        s = summarize_split(df, ptid_list, name)
        print("\n==", name.upper(), "==")
        print("PTIDs:", s["ptids"], "| Visits:", s["visits"])
        print("Visit label counts:", s["label_counts_visits"])

if __name__ == "__main__":
    main()
