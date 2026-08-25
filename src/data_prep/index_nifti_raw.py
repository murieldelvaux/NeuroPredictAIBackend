#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
import pandas as pd

PTID_RE = re.compile(r"(\d{3}_S_\d{4})", re.IGNORECASE)
IMGID_RE = re.compile(r"(I\d+)", re.IGNORECASE)
DATE_RE  = re.compile(r"(\d{4}-\d{2}-\d{2})")

def parse_from_path(p: Path):
    s = str(p)
    ptid = PTID_RE.search(s)
    imgid = IMGID_RE.search(s)
    date = DATE_RE.search(s)

    # "series" é a pasta logo após PTID (ex.: MPR__GradWarp__B1_Correction__N3__Scaled)
    parts = p.parts
    series = None
    if "ADNI" in parts:
        i = parts.index("ADNI")
        # padrão: .../ADNI/<PTID>/<SERIES>/<DATE_TIME>/<IMGID>/<file>
        if len(parts) > i + 2:
            series = parts[i + 2]

    return {
        "ptid": (ptid.group(1).upper() if ptid else None),
        "scan_date": (date.group(1) if date else None),
        "image_id": (imgid.group(1).upper() if imgid else None),
        "series": series,
        "file_path": str(p),
        "filename": p.name,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    rows = []
    for p in in_dir.rglob("*"):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            rows.append(parse_from_path(p))

    df = pd.DataFrame(rows)
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Images:", len(df))
    print("PTID coverage:", df["ptid"].notna().mean())
    print("Date coverage:", df["scan_date"].notna().mean())
    print("Unique PTID:", df["ptid"].nunique())

if __name__ == "__main__":
    main()