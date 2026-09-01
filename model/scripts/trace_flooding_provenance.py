"""Trace the provenance of every Flooding row available upstream.

Question: can we get 4,000 REAL flooding rows without synthesis, and where
from? Checks three sources and whether their rows trace back to actual
measurements in the raw table.

  S1  Public datasets.csv                      2500 labeled Flooding rows
  S2  processed_fuel_cell_data.csv (37899)     rule-labeled, no label column
  S3  201703021126_RATSSingleCell.CSV (79720)  the true raw bench log

Read-only. Writes nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PUBLIC = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets.csv")
PROCESSED = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv")
RAW = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/201703021126_RATSSingleCell.CSV")

SHARED = [
    "U_totV", "iA", "PW", "m_Air", "m_H2", "RH_Air", "RH_H2",
    "P_Air_supply", "P_H2_supply", "P_Air_inlet", "P_H2_inlet",
    "T_1", "T_2", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet",
    "T_Stack_inlet", "T_Heater", "m_Air_write", "m_H2_write",
    "Heater_power", "i_write",
]


def keys(df: pd.DataFrame, cols, nd=6) -> set:
    return set(map(tuple, df[cols].to_numpy(float).round(nd)))


def main() -> None:
    pub = pd.read_csv(PUBLIC)
    proc = pd.read_csv(PROCESSED)
    print(f"public    {pub.shape}")
    print(f"processed {proc.shape}")

    print("\n" + "=" * 78)
    print("1. ARE THE 2500 PUBLIC 'Flooding' ROWS REAL MEASUREMENTS?")
    print("=" * 78)
    proc_keys_full = keys(proc, SHARED)
    proc_keys_core = keys(proc, ["U_totV", "iA", "RH_Air", "RH_H2", "T_Stack_inlet"])

    for label in ("Normal", "Flooding", "Membrane_Drying", "Thermal_Management_Fault"):
        sub = pub.loc[pub["State_Label"] == label]
        k_full = keys(sub, SHARED)
        k_core = keys(sub, ["U_totV", "iA", "RH_Air", "RH_H2", "T_Stack_inlet"])
        hit_full = len(k_full & proc_keys_full)
        hit_core = len(k_core & proc_keys_core)
        print(f"{label:26s} n={len(sub):5d} uniq23={len(k_full):5d} "
              f"traced_23ch={hit_full:5d} ({100*hit_full/max(len(k_full),1):5.1f}%)  "
              f"traced_5ch={hit_core:5d} ({100*hit_core/max(len(k_core),1):5.1f}%)")

    # tsec traceability: an interpolated row has a tsec not present in the source
    print("\n-- tsec traceability (synthetic rows get interpolated timestamps) --")
    proc_tsec = set(np.round(proc["tsec"].to_numpy(float), 3))
    for label in ("Normal", "Flooding", "Membrane_Drying", "Thermal_Management_Fault"):
        sub = pub.loc[pub["State_Label"] == label]
        ts = np.round(sub["tsec"].to_numpy(float), 3)
        hit = sum(1 for x in ts if x in proc_tsec)
        print(f"{label:26s} tsec found in source: {hit:5d}/{len(ts):5d} "
              f"({100*hit/len(ts):5.1f}%)")

    print("\n" + "=" * 78)
    print("2. HOW MANY REAL FLOODING ROWS DOES EACH SOURCE OFFER?")
    print("=" * 78)

    # --- S1: public dataset, its own labels ---
    pub_flood = pub.loc[pub["State_Label"] == "Flooding"]
    print(f"S1 public labeled Flooding                : {len(pub_flood):6d} rows "
          f"({len(keys(pub_flood, SHARED))} unique)")

    # --- S2: processed source under the new two-axis rule ---
    v = proc["U_totV"].to_numpy(float)
    i = proc["iA"].to_numpy(float)
    ra = proc["RH_Air"].to_numpy(float)
    rh = proc["RH_H2"].to_numpy(float)
    t = proc["T_Stack_inlet"].to_numpy(float)

    edges = np.unique(np.quantile(i, np.linspace(0, 1, 11)))
    cb = np.clip(np.digitize(i, edges[1:-1]), 0, len(edges) - 2)
    du = np.empty_like(v)
    for b in range(len(edges) - 1):
        m = cb == b
        if m.any():
            du[m] = v[m] - np.median(v[m])

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-12)

    w = z(ra) + z(rh) - z(t)
    du_f, du_n = np.percentile(du, 40), np.percentile(du, 50)
    w_hi, w_lo = np.percentile(w, 55), np.percentile(w, 45)
    new_flood = (du < du_f) & (w > w_hi)
    print(f"S2 processed, two-axis rule Flooding      : {int(new_flood.sum()):6d} rows")

    # --- S2b: the old conjunction rule, for contrast ---
    vq50 = np.percentile(v, 50)
    aq75, hq75 = np.percentile(ra, 75), np.percentile(rh, 75)
    iq50 = np.percentile(i, 50)
    old_flood = (ra > aq75) & (rh > hq75) & (v < vq50) & (i > iq50)
    print(f"S2 processed, OLD conjunction Flooding    : {int(old_flood.sum()):6d} rows"
          f"   <- the 1805 that forced synthesis")

    # --- S3: raw bench log ---
    print()
    try:
        raw = pd.read_csv(RAW, low_memory=False)
        raw.columns = [str(c).strip() for c in raw.columns]
        ren = {"t (sec)": "tsec", "U_tot (V)": "U_totV", "i (A)": "iA", "P (W)": "PW",
               "T_Stack_outlet": "T_Stack_inlet"}
        raw = raw.rename(columns=ren)
        for c in ("U_totV", "iA", "RH_Air", "RH_H2", "T_Stack_inlet", "PW"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw = raw.dropna(subset=["U_totV", "iA", "RH_Air", "RH_H2", "T_Stack_inlet"])
        # drop the obvious pre-start garbage (negative RH, zero voltage)
        valid = raw.loc[(raw["RH_Air"] > 0) & (raw["U_totV"] > 0.1) & (raw["iA"] > -1)]
        print(f"S3 raw bench log rows                     : {len(raw):6d} "
              f"(valid after gating: {len(valid)})")
        rv = valid["U_totV"].to_numpy(float)
        ri = valid["iA"].to_numpy(float)
        rra = valid["RH_Air"].to_numpy(float)
        rrh = valid["RH_H2"].to_numpy(float)
        rt = valid["T_Stack_inlet"].to_numpy(float)
        re_ = np.unique(np.quantile(ri, np.linspace(0, 1, 11)))
        rcb = np.clip(np.digitize(ri, re_[1:-1]), 0, len(re_) - 2)
        rdu = np.empty_like(rv)
        for b in range(len(re_) - 1):
            m = rcb == b
            if m.any():
                rdu[m] = rv[m] - np.median(rv[m])
        rw = z(rra) + z(rrh) - z(rt)
        rflood = (rdu < np.percentile(rdu, 40)) & (rw > np.percentile(rw, 55))
        print(f"S3 raw, two-axis rule Flooding            : {int(rflood.sum()):6d} rows")
        print(f"   raw RH_Air range [{rra.min():.2f}, {rra.max():.2f}]  "
              f"T range [{rt.min():.2f}, {rt.max():.2f}]")
        # is the raw log a superset of processed?
        rk = keys(valid, ["U_totV", "iA", "RH_Air", "RH_H2"], nd=3)
        pk = keys(proc, ["U_totV", "iA", "RH_Air", "RH_H2"], nd=3)
        print(f"   processed rows found in raw: {len(pk & rk)}/{len(pk)} "
              f"({100*len(pk & rk)/len(pk):.1f}%)")
    except Exception as exc:
        print(f"S3 raw parse failed: {exc}")

    print("\n" + "=" * 78)
    print("3. UNION: real flooding rows from public + processed, deduplicated")
    print("=" * 78)
    pub_f_keys = keys(pub_flood, SHARED)
    proc_f = proc.loc[new_flood]
    proc_f_keys = keys(proc_f, SHARED)
    print(f"public Flooding unique      : {len(pub_f_keys):6d}")
    print(f"processed Flooding unique   : {len(proc_f_keys):6d}")
    print(f"overlap                     : {len(pub_f_keys & proc_f_keys):6d}")
    print(f"UNION                       : {len(pub_f_keys | proc_f_keys):6d}"
          f"   -> {'4000 reachable, NO synthesis' if len(pub_f_keys | proc_f_keys) >= 4000 else 'still short'}")

    print("\n-- do the public-labeled Flooding rows agree with the two-axis rule? --")
    pf = pub_flood
    pv, pi = pf["U_totV"].to_numpy(float), pf["iA"].to_numpy(float)
    pra, prh = pf["RH_Air"].to_numpy(float), pf["RH_H2"].to_numpy(float)
    pt = pf["T_Stack_inlet"].to_numpy(float)
    print(f"public Flooding profile: U={pv.mean():.4f} iA={pi.mean():.2f} "
          f"RH_Air={pra.mean():.2f} RH_H2={prh.mean():.2f} T={pt.mean():.2f}")
    nf = proc.loc[new_flood]
    print(f"rule   Flooding profile: U={nf['U_totV'].mean():.4f} iA={nf['iA'].mean():.2f} "
          f"RH_Air={nf['RH_Air'].mean():.2f} RH_H2={nf['RH_H2'].mean():.2f} "
          f"T={nf['T_Stack_inlet'].mean():.2f}")


if __name__ == "__main__":
    main()
