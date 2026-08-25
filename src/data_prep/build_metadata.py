from __future__ import annotations

import pandas as pd


def build_unified_metadata(adni_csv: str, oasis_csv: str, output_csv: str) -> None:
    adni = pd.read_csv(adni_csv)
    oasis = pd.read_csv(oasis_csv)

    adni = adni.assign(dataset="adni")
    oasis = oasis.assign(dataset="oasis")

    columns = sorted(set(adni.columns) | set(oasis.columns))
    adni = adni.reindex(columns=columns)
    oasis = oasis.reindex(columns=columns)

    unified = pd.concat([adni, oasis], axis=0, ignore_index=True)
    unified.to_csv(output_csv, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build unified metadata CSV from ADNI and OASIS metadata.")
    parser.add_argument("--adni", required=True)
    parser.add_argument("--oasis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_unified_metadata(args.adni, args.oasis, args.output)
