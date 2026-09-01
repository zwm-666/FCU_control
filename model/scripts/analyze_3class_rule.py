"""Diagnose the label cascade so a genuine 3-class rule can be chosen.

Read-only. Prints the facts needed to justify a 3-class rule redesign.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE = r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv"


def q(a, p):
    return float(np.percentile(a, p))


def prof(name, arr, mask):
    if mask.sum() == 0:
        print(f"{name:15s} (empty)")
        return
    sub = arr[mask]
    print(f"{name:15s} min={sub.min():9.3f} q25={q(sub,25):9.3f} q50={q(sub,50):9.3f} "
          f"q75={q(sub,75):9.3f} max={sub.max():9.3f}")


def main() -> None:
    df = pd.read_csv(SOURCE)
    v = df["U_totV"].to_numpy(float)
    i = df["iA"].to_numpy(float)
    p = df["PW"].to_numpy(float)
    ra = df["RH_Air"].to_numpy(float)
    rh = df["RH_H2"].to_numpy(float)
    t = df["T_Stack_inlet"].to_numpy(float)

    vq25, vq50, vq75 = q(v, 25), q(v, 50), q(v, 75)
    aq25, aq50, aq75 = q(ra, 25), q(ra, 50), q(ra, 75)
    hq25, hq50, hq75 = q(rh, 25), q(rh, 50), q(rh, 75)
    tq25, tq50, tq75 = q(t, 25), q(t, 50), q(t, 75)
    iq50, pq50 = q(i, 50), q(p, 50)

    print("=" * 70)
    print("A. WHY the thermal primary branch is DEAD")
    print("=" * 70)
    hi = tq75 + (tq75 - tq50)
    lo = tq25 - (tq50 - tq25)
    print(f"thermal needs  T > {hi:.3f}  or  T < {lo:.3f}")
    print(f"actual T range         [{t.min():.3f}, {t.max():.3f}]")
    print(f"=> rows above hi: {(t > hi).sum()},  below lo: {(t < lo).sum()}  "
          f"-> primary thermal is UNSATISFIABLE")

    thermal = ((t > hi) | (t < lo)) & (v < vq50) & (p < pq50)
    flooding = (ra > aq75) & (rh > hq75) & (v < vq50) & (i > iq50) & (~thermal)
    drying = ((ra < aq50) & (rh < hq50) & (t >= tq50) & (t <= hi)
              & (v >= vq25) & (v <= vq75) & (~thermal) & (~flooding))
    normal = ((v > vq50) & (ra >= aq25) & (ra <= aq75) & (rh >= hq25) & (rh <= hq75)
              & (t >= tq25) & (t <= tq75) & (~thermal) & (~flooding) & (~drying))

    print("\n" + "=" * 70)
    print("B. WHERE the 10,984 'thermal' rows actually come from (the fallback)")
    print("=" * 70)
    states = np.zeros(len(df), dtype=np.int64)
    states[thermal] = 3
    states[flooding] = 1
    states[drying] = 2
    states[normal] = 0
    unclassified = (states == 0) & (~normal)
    high_voltage = unclassified & (v > vq75)
    temp_abnormal = unclassified & (~high_voltage) & ((t > tq75) | (t < tq25))
    low_voltage = unclassified & (~high_voltage) & (~temp_abnormal) & (v <= vq50)
    leftover = unclassified & (~high_voltage) & (~temp_abnormal) & (~low_voltage)
    print(f"primary  normal={normal.sum():6d} flooding={flooding.sum():6d} "
          f"drying={drying.sum():6d} thermal={thermal.sum():6d}")
    print(f"unclassified rows entering fallback: {unclassified.sum():6d}")
    print(f"  fallback -> Normal  (v>vq75)          {high_voltage.sum():6d}")
    print(f"  fallback -> THERMAL (T outside q25/75) {temp_abnormal.sum():6d}  <== the entire class 3")
    print(f"  fallback -> Drying  (v<=vq50)          {low_voltage.sum():6d}")
    print(f"  fallback -> Normal  (leftover)         {leftover.sum():6d}")

    print("\n-- profile of the 'thermal' bucket vs the whole dataset --")
    for name, arr in [("U_totV", v), ("iA", i), ("RH_Air", ra), ("RH_H2", rh),
                      ("T_Stack_inlet", t)]:
        prof(f"thermal {name}", arr, temp_abnormal)
    print()
    for name, arr in [("U_totV", v), ("iA", i), ("RH_Air", ra), ("RH_H2", rh),
                      ("T_Stack_inlet", t)]:
        prof(f"ALL     {name}", arr, np.ones(len(df), bool))

    print("\n" + "=" * 70)
    print("C. WHY flooding starves to 1,805 (conjunction collapse)")
    print("=" * 70)
    c1 = ra > aq75
    c2 = rh > hq75
    c3 = v < vq50
    c4 = i > iq50
    print(f"RH_Air>q75                {c1.sum():6d}")
    print(f"  & RH_H2>q75             {(c1&c2).sum():6d}")
    print(f"  & U<q50                 {(c1&c2&c3).sum():6d}")
    print(f"  & i>q50   (= flooding)  {(c1&c2&c3&c4).sum():6d}")
    print(f"corr(RH_Air, RH_H2) = {np.corrcoef(ra, rh)[0,1]:.4f}")
    print(f"i>iq50 share: {c4.mean():.4f}  (iq50={iq50:.4f} equals q75={q(i,75):.4f} "
          f"-> current is heavily tied at the top)")

    print("\n" + "=" * 70)
    print("D. CANDIDATE 3-class axes: voltage residual vs current + humidity direction")
    print("=" * 70)
    # Severity axis: voltage deficit relative to same-current median (polarization residual)
    bins = np.quantile(i, np.linspace(0, 1, 11))
    bins = np.unique(bins)
    idx = np.clip(np.digitize(i, bins[1:-1]), 0, len(bins) - 2)
    resid = np.zeros(len(df))
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() > 0:
            resid[m] = v[m] - np.median(v[m])
    print(f"voltage residual: mean={resid.mean():.5f} std={resid.std():.5f} "
          f"q10={q(resid,10):.5f} q25={q(resid,25):.5f} q50={q(resid,50):.5f} "
          f"q75={q(resid,75):.5f} q90={q(resid,90):.5f}")
    print(f"current bins used: {len(bins)-1}, edges={np.round(bins,3).tolist()}")

    # Direction axis: cathode humidity, and a temperature-corrected variant
    print(f"\nRH_Air deciles: {[round(q(ra,x),2) for x in range(0,101,10)]}")
    print(f"RH_H2  deciles: {[round(q(rh,x),2) for x in range(0,101,10)]}")
    print(f"T      deciles: {[round(q(t,x),2) for x in range(0,101,10)]}")

    # How balanced is a severity x direction partition?
    print("\n-- trial partition: |resid| small => Normal; else sign of RH_Air deviation --")
    for lo_p, hi_p in [(33, 67), (30, 70), (25, 75), (40, 60)]:
        r_lo, r_hi = q(resid, lo_p), q(resid, hi_p)
        faulty = (resid < r_lo)          # underperforming at its current
        wet = faulty & (ra > aq50)
        dry = faulty & (ra <= aq50)
        norm = ~faulty
        print(f"resid<q{lo_p:<3d}  Normal={norm.sum():6d} Flooding(wet)={wet.sum():6d} "
              f"Drying(dry)={dry.sum():6d}")

    print("\n-- trial partition: pure RH_Air tertiles (direction only) --")
    for lo_p, hi_p in [(33, 67), (30, 70), (25, 75)]:
        a_lo, a_hi = q(ra, lo_p), q(ra, hi_p)
        dry = ra <= a_lo
        wet = ra >= a_hi
        norm = (~dry) & (~wet)
        print(f"RH_Air tertile q{lo_p}/q{hi_p}: Normal={norm.sum():6d} "
              f"Flooding={wet.sum():6d} Drying={dry.sum():6d}")


if __name__ == "__main__":
    main()
