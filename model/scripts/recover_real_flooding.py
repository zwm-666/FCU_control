"""Reverse-engineer the public dataset's Flooding criterion and find real rows.

Findings so far that this script builds on:
  * Public datasets.csv Flooding = 2500 rows but only 1346 UNIQUE -> upstream
    already duplicated rows to hit 2500. Only Flooding is duplicated; the other
    three classes are 2500/2500 unique.
  * All 1346 unique rows trace 100% back to processed_fuel_cell_data.csv, so
    they ARE real measurements.
  * The public Flooding profile is far more extreme than the two-axis rule's:
    RH_Air 99.64 vs 88.27, iA 26.58 vs 15.86, U 0.4511 vs 0.5392.
    -> upstream used a SATURATION + HIGH-CURRENT definition.

Goal: recover that criterion, then count how many real rows in the 37,899-row
processed source satisfy it, so 4,000 real Flooding rows can be selected with
no synthesis and no duplication.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PUBLIC = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets.csv")
PROCESSED = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv")

SHARED = [
    "U_totV", "iA", "PW", "m_Air", "m_H2", "RH_Air", "RH_H2",
    "P_Air_supply", "P_H2_supply", "P_Air_inlet", "P_H2_inlet",
    "T_1", "T_2", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet",
    "T_Stack_inlet", "T_Heater", "m_Air_write", "m_H2_write",
    "Heater_power", "i_write",
]


def desc(tag, df, cols=("U_totV", "iA", "PW", "RH_Air", "RH_H2", "T_Stack_inlet")):
    print(f"  {tag}  n={len(df)}")
    for c in cols:
        a = df[c].to_numpy(float)
        print(f"    {c:15s} min={a.min():8.3f} q05={np.percentile(a,5):8.3f} "
              f"q50={np.percentile(a,50):8.3f} q95={np.percentile(a,95):8.3f} "
              f"max={a.max():8.3f}")


def main() -> None:
    pub = pd.read_csv(PUBLIC)
    proc = pd.read_csv(PROCESSED)

    print("=" * 78)
    print("1. DUPLICATION AUDIT OF THE PUBLIC DATASET")
    print("=" * 78)
    for label in ("Normal", "Flooding", "Membrane_Drying", "Thermal_Management_Fault"):
        sub = pub.loc[pub["State_Label"] == label]
        uniq = len(sub.drop_duplicates(subset=SHARED))
        dup = len(sub) - uniq
        print(f"{label:26s} rows={len(sub):5d} unique={uniq:5d} duplicated={dup:5d} "
              f"({100*dup/len(sub):5.1f}%)")
        if dup:
            vc = sub.groupby(SHARED).size().sort_values(ascending=False)
            print(f"    max copies of one row = {vc.iloc[0]}, "
                  f"rows appearing >1x = {(vc > 1).sum()}")

    pubf = pub.loc[pub["State_Label"] == "Flooding"].drop_duplicates(subset=SHARED)
    print("\n" + "=" * 78)
    print("2. WHAT DEFINES THE PUBLIC FLOODING ROWS (1346 unique real rows)")
    print("=" * 78)
    desc("public Flooding", pubf)
    print()
    desc("whole processed source", proc)

    # saturation is the obvious signature
    print("\n-- saturation signature --")
    for thr in (100.0, 99.9, 99.0, 98.0, 95.0):
        pf = (pubf["RH_Air"] >= thr).mean()
        pr = (proc["RH_Air"] >= thr).mean()
        print(f"  RH_Air >= {thr:5.1f}: public Flooding {100*pf:5.1f}%   "
              f"source {100*pr:5.1f}%  ({int((proc['RH_Air']>=thr).sum())} rows)")
    print()
    for thr in (100.0, 99.9, 99.0, 98.0, 95.0):
        pf = (pubf["RH_H2"] >= thr).mean()
        pr = (proc["RH_H2"] >= thr).mean()
        print(f"  RH_H2  >= {thr:5.1f}: public Flooding {100*pf:5.1f}%   "
              f"source {100*pr:5.1f}%  ({int((proc['RH_H2']>=thr).sum())} rows)")
    print()
    for thr in (20, 22, 25, 26, 28):
        pf = (pubf["iA"] >= thr).mean()
        pr = (proc["iA"] >= thr).mean()
        print(f"  iA     >= {thr:5.1f}: public Flooding {100*pf:5.1f}%   "
              f"source {100*pr:5.1f}%  ({int((proc['iA']>=thr).sum())} rows)")

    print("\n" + "=" * 78)
    print("3. CANDIDATE SATURATION-ANCHORED FLOODING RULES ON THE SOURCE")
    print("=" * 78)
    v = proc["U_totV"].to_numpy(float)
    i = proc["iA"].to_numpy(float)
    ra = proc["RH_Air"].to_numpy(float)
    rh = proc["RH_H2"].to_numpy(float)
    t = proc["T_Stack_inlet"].to_numpy(float)

    # current-conditioned voltage residual
    edges = np.unique(np.quantile(i, np.linspace(0, 1, 11)))
    cb = np.clip(np.digitize(i, edges[1:-1]), 0, len(edges) - 2)
    du = np.empty_like(v)
    for b in range(len(edges) - 1):
        m = cb == b
        if m.any():
            du[m] = v[m] - np.median(v[m])
    du_f = np.percentile(du, 40)

    feat = proc[SHARED].round(9)
    cands = {
        "RH_Air>=99.9 & RH_H2>=99.9":            (ra >= 99.9) & (rh >= 99.9),
        "RH_Air>=99.9":                          (ra >= 99.9),
        "RH_Air>=99 & dU<q40":                   (ra >= 99.0) & (du < du_f),
        "RH_Air>=99.9 & dU<q40":                 (ra >= 99.9) & (du < du_f),
        "RH_Air>=99.9 & iA>=20":                 (ra >= 99.9) & (i >= 20),
        "RH_Air>=99.9 & iA>=20 & dU<q40":        (ra >= 99.9) & (i >= 20) & (du < du_f),
        "RH_Air>=98 & RH_H2>=98 & dU<q40":       (ra >= 98) & (rh >= 98) & (du < du_f),
        "RH_Air>=95 & RH_H2>=95 & dU<q40":       (ra >= 95) & (rh >= 95) & (du < du_f),
    }
    print(f"{'rule':42s} {'rows':>7} {'unique':>7} {'>=4000':>7}")
    print("-" * 68)
    for name, mask in cands.items():
        n = int(mask.sum())
        u = int((~feat.loc[mask].duplicated()).sum()) if n else 0
        print(f"{name:42s} {n:>7} {u:>7} {'YES' if u >= 4000 else 'no':>7}")

    print("\n" + "=" * 78)
    print("4. RECOVERY CHECK: does a saturation rule recover the public rows?")
    print("=" * 78)
    pk = set(map(tuple, pubf[SHARED].to_numpy(float).round(6)))
    for name, mask in cands.items():
        sel = set(map(tuple, proc.loc[mask, SHARED].to_numpy(float).round(6)))
        rec = len(pk & sel)
        print(f"{name:42s} recovers {rec:5d}/{len(pk)} "
              f"({100*rec/len(pk):5.1f}%) of public Flooding")

    print("\n" + "=" * 78)
    print("5. THE CHOSEN POOL: RH_Air>=99.9 & RH_H2>=99.9  (pure saturation)")
    print("=" * 78)
    mask = (ra >= 99.9) & (rh >= 99.9)
    pool = proc.loc[mask].drop_duplicates(subset=SHARED)
    print(f"unique real rows available = {len(pool)}")
    desc("saturated pool", pool)
    print(f"\n  dU on this pool: mean={du[mask].mean():+.4f}  "
          f"share with dU<0: {100*(du[mask]<0).mean():.1f}%")
    print(f"  overlap with public Flooding: "
          f"{len(pk & set(map(tuple, pool[SHARED].to_numpy(float).round(6))))}/{len(pk)}")

    # Where do saturated rows sit in time? contiguous flooding episodes?
    ts = np.sort(proc.loc[mask, "tsec"].to_numpy(float))
    gaps = np.diff(ts)
    breaks = int((gaps > 5.0).sum())
    print(f"\n  tsec span [{ts.min():.1f}, {ts.max():.1f}], "
          f"contiguous episodes (gap>5s) = {breaks + 1}")
    print(f"  -> real flooding is {'episodic (good, physical events)' if breaks > 0 else 'one block'}")


if __name__ == "__main__":
    main()
