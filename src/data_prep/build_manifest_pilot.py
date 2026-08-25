#!/usr/bin/env python3
import argparse, json, os
import pandas as pd

def load_splits(path: str):
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    split_map = {}
    for ptid in s["train_ptids"]:
        split_map[ptid] = "train"
    for ptid in s["val_ptids"]:
        split_map[ptid] = "val"
    for ptid in s["test_ptids"]:
        split_map[ptid] = "test"
    return split_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-index", required=True)
    ap.add_argument("--dxsum-labeled", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-days", type=int, default=60, help="tolerância (dias) para casar scan_date com EXAMDATE")
    args = ap.parse_args()

    img = pd.read_csv(args.images_index)
    img["scan_date"] = pd.to_datetime(img["scan_date"], errors="coerce")
    img["ptid"] = img["ptid"].astype(str).str.upper()

    dx = pd.read_csv(args.dxsum_labeled, low_memory=False)
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["PTID"] = dx["PTID"].astype(str).str.upper()

    # Vamos casar por PTID e data mais próxima usando merge_asof
    img = img.dropna(subset=["ptid", "scan_date"]).copy()
    dx  = dx.dropna(subset=["PTID", "EXAMDATE"]).copy()

    img = img.sort_values(["scan_date", "ptid"]).reset_index(drop=True)
    dx  = dx.sort_values(["EXAMDATE", "PTID"]).reset_index(drop=True)

    merged = pd.merge_asof(
        img,
        dx,
        left_on="scan_date",
        right_on="EXAMDATE",
        left_by="ptid",
        right_by="PTID",
        direction="nearest",
        tolerance=pd.Timedelta(days=args.max_days),
    )

    # Calcula delta em dias para auditoria
    merged["delta_days"] = (merged["scan_date"] - merged["EXAMDATE"]).abs().dt.days

    # Remove casos que não casaram (sem label)
    merged = merged.dropna(subset=["label_id", "label_name"]).copy()
    merged["label_id"] = merged["label_id"].astype(int)

    # Adiciona split por PTID (paciente)
    split_map = load_splits(args.splits)
    merged["split"] = merged["PTID"].map(split_map)

    # Colunas finais para MONAI
    out = merged.rename(columns={"file_path": "image"})[
        ["image", "label_id", "label_name", "PTID", "VISCODE2", "EXAMDATE", "scan_date", "delta_days", "split"]
    ].copy()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Rows (images with labels):", len(out))
    print("Split counts:\n", out["split"].value_counts(dropna=False))
    print("Label counts:\n", out["label_name"].value_counts())
    print("Max delta_days:", out["delta_days"].max())

if __name__ == "__main__":
    main()