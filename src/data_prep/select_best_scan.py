#!/usr/bin/env python3
import argparse, os
import pandas as pd

PREF = [
    "GradWarp__B1_Correction__N3__Scaled",
    "GradWarp__N3__Scaled",
    "N3__Scaled",
]

def score(series: str) -> int:
    if not isinstance(series, str):
        return 999
    for i, key in enumerate(PREF):
        if key in series:
            return i
    return 999

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
    df["pref_score"] = df["series"].apply(score)

    # 1 por PTID+scan_date: mantém o menor pref_score (melhor)
    df = df.sort_values(["ptid", "scan_date", "pref_score"])
    best = df.drop_duplicates(subset=["ptid", "scan_date"], keep="first").copy()
    best["selected"] = True

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Selected images:", len(best))
    print("Unique PTID:", best["ptid"].nunique())

if __name__ == "__main__":
    main()