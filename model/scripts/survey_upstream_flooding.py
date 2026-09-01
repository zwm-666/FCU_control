"""Survey every upstream candidate for REAL flooding rows.

Read-only. Answers one question: which upstream file carries genuine flooding
samples (either an explicit label column, or enough real rows to select from),
so the 3-class dataset can be built from real measurements instead of
KNN-interpolated synthetic rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = [
    r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets.csv",
    r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv",
    r"D:/python/pythonProject3/classified_fuel_cell_data/full_test_data.csv",
    r"D:/python/pythonProject3/classified_fuel_cell_data/201703021126_RATSSingleCell.CSV",
]

LABEL_HINTS = ("state", "label", "class", "fault", "status", "condition",
               "标签", "类别", "状态", "故障", "工况")


def survey(path_str: str) -> None:
    path = Path(path_str)
    print("=" * 78)
    print(path)
    if not path.exists():
        print("  !! MISSING")
        return
    print(f"  size = {path.stat().st_size/1e6:.2f} MB")

    # sniff separator / encoding
    df = None
    for enc in ("utf-8", "gbk", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                probe = pd.read_csv(path, encoding=enc, sep=sep, nrows=5,
                                    engine="python")
                if probe.shape[1] > 1:
                    df = pd.read_csv(path, encoding=enc, sep=sep, low_memory=False)
                    print(f"  parsed: encoding={enc} sep={sep!r}")
                    break
            except Exception:
                continue
        if df is not None:
            break
    if df is None:
        print("  !! could not parse")
        return

    print(f"  shape = {df.shape}")
    print(f"  columns ({len(df.columns)}):")
    for k in range(0, len(df.columns), 6):
        print("    " + " | ".join(f"{c}" for c in df.columns[k:k + 6]))

    # any label-ish column?
    label_cols = [c for c in df.columns
                  if any(h in str(c).lower() for h in LABEL_HINTS)]
    print(f"  label-like columns: {label_cols if label_cols else 'NONE'}")
    for c in label_cols:
        vc = df[c].value_counts(dropna=False)
        print(f"    -- {c} ({df[c].nunique(dropna=False)} distinct) --")
        for val, cnt in vc.head(12).items():
            print(f"       {str(val)[:44]:44s} {cnt:7d}")

    # low-cardinality non-numeric or small-int columns could be hidden labels
    print("  low-cardinality candidates (<=8 distinct, not already listed):")
    found = False
    for c in df.columns:
        if c in label_cols:
            continue
        nu = df[c].nunique(dropna=False)
        if 1 < nu <= 8:
            found = True
            vals = df[c].value_counts(dropna=False).head(8).to_dict()
            print(f"    {str(c)[:30]:30s} n={nu}  {vals}")
    if not found:
        print("    none")

    print(f"  first 2 rows:")
    with pd.option_context("display.max_columns", 40, "display.width", 200):
        print(df.head(2).to_string()[:1200])


def main() -> None:
    for c in CANDIDATES:
        survey(c)
        print()

    # extra: does the 4-class public dataset drop cleanly to 3 classes?
    p = Path(CANDIDATES[0])
    if p.exists():
        print("=" * 78)
        print("KEY QUESTION: is 'Public datasets.csv' already labeled and balanced?")
        print("=" * 78)
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                df = pd.read_csv(p, encoding=enc, low_memory=False)
                break
            except Exception:
                continue
        obj = [c for c in df.columns if df[c].dtype == object or df[c].nunique() <= 10]
        print(f"candidate label columns: {obj}")
        for c in obj:
            print(f"\n{c}:")
            print(df[c].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
