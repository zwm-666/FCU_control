"""Final rule design: all-real, no-synthesis, no-duplication 3-class dataset.

Two facts established by the provenance trace that drive this design:

 F1  Public datasets.csv Flooding = 2500 rows but only 1346 UNIQUE (46.2%
     duplicated, one row copied 7x). Normal/Drying/Thermal are 2500/2500
     unique. => the UPSTREAM public dataset could not find 2500 real flooding
     rows either, and padded by row duplication. The shortage is a property of
     the source, not of our rule.

 F2  The public Flooding rows sit at RH_Air~100, RH_H2~100, iA q50=28.5,
     U q50=0.444, T q50=45.1 (min 43.1). They are SATURATION + HIGH-CURRENT
     events at HIGH temperature.
     => the earlier two-axis rule's `w = z(RH_Air)+z(RH_H2)-z(T)` term is
     wrong for this data: RH is already normalized to saturation at the local
     temperature, so subtracting z(T) double-counts temperature and actively
     EXCLUDES the genuine hot flooding episodes (rule profile T=28.7 vs real
     T=45.1). The corrected direction axis must not subtract T.

This script sizes the largest balanced all-real dataset and reports the
per-class pools for a corrected, physics-anchored rule.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv")
PUBLIC = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets.csv")

SHARED = [
    "U_totV", "iA", "PW", "m_Air", "m_H2", "RH_Air", "RH_H2",
    "P_Air_supply", "P_H2_supply", "P_Air_inlet", "P_H2_inlet",
    "T_1", "T_2", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet",
    "T_Stack_inlet", "T_Heater", "m_Air_write", "m_H2_write",
    "Heater_power", "i_write",
]


def main() -> None:
    proc = pd.read_csv(PROCESSED)
    pub = pd.read_csv(PUBLIC)
    n = len(proc)

    v = proc["U_totV"].to_numpy(float)
    i = proc["iA"].to_numpy(float)
    ra = proc["RH_Air"].to_numpy(float)
    rh = proc["RH_H2"].to_numpy(float)
    t = proc["T_Stack_inlet"].to_numpy(float)

    # severity axis: current-conditioned voltage residual
    edges = np.unique(np.quantile(i, np.linspace(0, 1, 11)))
    cb = np.clip(np.digitize(i, edges[1:-1]), 0, len(edges) - 2)
    du = np.empty_like(v)
    for b in range(len(edges) - 1):
        m = cb == b
        if m.any():
            du[m] = v[m] - np.median(v[m])

    feat = proc[SHARED].round(9)
    dup_mask = feat.duplicated().to_numpy()
    print(f"source rows={n}, exact duplicate rows within source={int(dup_mask.sum())}")

    def uniq(mask):
        return int((mask & ~dup_mask).sum())

    print("\n" + "=" * 78)
    print("A. FLOODING POOL SIZE vs CRITERION STRICTNESS (unique real rows)")
    print("=" * 78)
    print(f"{'flooding criterion':52s} {'unique':>7} {'recall':>7}")
    print("-" * 70)
    pubf = pub.loc[pub["State_Label"] == "Flooding"].drop_duplicates(subset=SHARED)
    pk = set(map(tuple, pubf[SHARED].to_numpy(float).round(6)))

    def recall(mask):
        sel = set(map(tuple, proc.loc[mask, SHARED].to_numpy(float).round(6)))
        return 100 * len(pk & sel) / len(pk)

    flood_variants = {
        "RH_Air>=99.9 & RH_H2>=99.9 (双路饱和)":       (ra >= 99.9) & (rh >= 99.9),
        "RH_Air>=99.9":                                 (ra >= 99.9),
        "RH_Air>=99":                                   (ra >= 99.0),
        "RH_Air>=98":                                   (ra >= 98.0),
        "RH_Air>=95 & RH_H2>=95":                       (ra >= 95) & (rh >= 95),
        "RH_Air>=95 & RH_H2>=95 & dU<0":                (ra >= 95) & (rh >= 95) & (du < 0),
        "RH_Air>=97 & RH_H2>=95 & dU<0":                (ra >= 97) & (rh >= 95) & (du < 0),
        "RH_Air>=98 & iA>=15":                          (ra >= 98) & (i >= 15),
        "RH_Air>=95 & RH_H2>=95 & iA>=15":              (ra >= 95) & (rh >= 95) & (i >= 15),
        "RH_Air>=95 & RH_H2>=95 & iA>=15 & dU<0":       (ra >= 95) & (rh >= 95) & (i >= 15) & (du < 0),
    }
    for name, mask in flood_variants.items():
        print(f"{name:52s} {uniq(mask):>7} {recall(mask):>6.1f}%")

    print("\n" + "=" * 78)
    print("B. CORRECTED RULE (no -z(T) term); largest balanced ALL-REAL dataset")
    print("=" * 78)
    # direction axis without temperature: mean humidity z-score
    def z(a):
        return (a - a.mean()) / (a.std() + 1e-12)
    wet = 0.5 * (z(ra) + z(rh))

    print(f"{'sat_thr':>8} {'iA_min':>7} {'dU_q':>5} | {'Flood':>6} {'Dry':>6} {'Normal':>7} "
          f"| {'balanced_N':>10} {'total':>7}")
    print("-" * 78)
    best = None
    for sat in (95.0, 96.0, 97.0, 98.0, 99.0):
        for ia_min in (0.0, 10.0, 15.0):
            for dq in (40, 50):
                du_f = np.percentile(du, dq)
                du_n = np.percentile(du, min(dq + 10, 100))
                flood = (ra >= sat) & (rh >= 95.0) & (i >= ia_min) & (du < du_f)
                dry = (du < du_f) & (wet < np.percentile(wet, 35)) & (~flood)
                norm = (du >= du_n) & (ra < sat)
                fu, dy, no = uniq(flood), uniq(dry), uniq(norm)
                bal = min(fu, dy, no)
                print(f"{sat:>8.1f} {ia_min:>7.1f} {dq:>5} | {fu:>6} {dy:>6} {no:>7} "
                      f"| {bal:>10} {bal*3:>7}")
                if best is None or bal > best[0]:
                    best = (bal, sat, ia_min, dq)

    bal, SAT, IA, DQ = best
    print(f"\n=> best balanced all-real size = {bal}/class ({bal*3} rows total)")
    print(f"   at sat_thr={SAT}, iA_min={IA}, dU_q={DQ}")

    print("\n" + "=" * 78)
    print("C. RECOMMENDED CONFIG: verify class physics at a round target")
    print("=" * 78)
    du_f = np.percentile(du, DQ)
    du_n = np.percentile(du, min(DQ + 10, 100))
    w_lo = np.percentile(wet, 35)
    flood = (ra >= SAT) & (rh >= 95.0) & (i >= IA) & (du < du_f)
    dry = (du < du_f) & (wet < w_lo) & (~flood)
    norm = (du >= du_n) & (ra < SAT)

    label = np.full(n, "DISCARD", dtype=object)
    label[norm] = "Normal"
    label[dry] = "Membrane_Drying"
    label[flood] = "Flooding"
    print(f"overlaps: flood&dry={int((flood&dry).sum())} "
          f"flood&norm={int((flood&norm).sum())} dry&norm={int((dry&norm).sum())} (want 0)")

    hdr = (f"{'class':18s} {'uniq':>6} {'dU':>9} {'U':>8} {'iA':>7} "
           f"{'RH_Air':>8} {'RH_H2':>8} {'T':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in ("Normal", "Flooding", "Membrane_Drying", "DISCARD"):
        m = (label == name) & ~dup_mask
        if not m.any():
            continue
        print(f"{name:18s} {int(m.sum()):>6} {du[m].mean():>+9.4f} {v[m].mean():>8.4f} "
              f"{i[m].mean():>7.2f} {ra[m].mean():>8.2f} {rh[m].mean():>8.2f} {t[m].mean():>7.2f}")

    print("\n-- reference: the real labeled Flooding rows we are trying to match --")
    print(f"{'public Flooding':18s} {len(pubf):>6} {'':>9} "
          f"{pubf['U_totV'].mean():>8.4f} {pubf['iA'].mean():>7.2f} "
          f"{pubf['RH_Air'].mean():>8.2f} {pubf['RH_H2'].mean():>8.2f} "
          f"{pubf['T_Stack_inlet'].mean():>7.2f}")
    print(f"flooding recall vs public labels: {recall(flood):.1f}%")

    # episodic structure -> is a block split needed?
    print("\n" + "=" * 78)
    print("D. TEMPORAL STRUCTURE (does a random split leak?)")
    print("=" * 78)
    for name in ("Normal", "Flooding", "Membrane_Drying"):
        m = (label == name) & ~dup_mask
        ts = np.sort(proc.loc[m, "tsec"].to_numpy(float))
        if len(ts) < 2:
            continue
        gaps = np.diff(ts)
        eps = int((gaps > 5.0).sum()) + 1
        print(f"{name:18s} n={int(m.sum()):6d} tsec[{ts.min():8.1f},{ts.max():8.1f}] "
              f"episodes(gap>5s)={eps:4d}  median_gap={np.median(gaps):.3f}s")

    print("\n" + "=" * 78)
    print("E. SEPARABILITY at the balanced all-real size")
    print("=" * 78)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    target = min(bal, 2000)
    rng = np.random.default_rng(42)
    idxs = []
    for name in ("Normal", "Flooding", "Membrane_Drying"):
        pool = np.flatnonzero((label == name) & ~dup_mask)
        idxs.append(rng.permutation(pool)[:target])
    keep = np.concatenate(idxs)
    X = proc.loc[keep, SHARED].to_numpy(float)
    y = label[keep]
    print(f"balanced all-real subset: {X.shape} @ {target}/class")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    knn = KNeighborsClassifier(n_neighbors=1).fit(sc.transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    print(f"1-NN test acc = {knn.score(sc.transform(Xte), yte):.4f}")
    print(f"RF   test acc = {rf.score(Xte, yte):.4f}")
    tr_keys = set(map(tuple, np.round(Xtr, 9)))
    twins = sum(1 for r in np.round(Xte, 9) if tuple(r) in tr_keys)
    print(f"exact train/test twins = {twins} (want 0)")

    print("\n" + "=" * 78)
    print("FINAL CONSTANTS")
    print("=" * 78)
    print(f"current_bin_edges     = {np.round(edges,6).tolist()}")
    print(f"RH_Air_saturation_thr = {SAT}")
    print(f"RH_H2_min_thr         = 95.0")
    print(f"iA_min_thr            = {IA}")
    print(f"dU_fault_thr          = {du_f:.6f}  (q{DQ})")
    print(f"dU_normal_thr         = {du_n:.6f}  (q{min(DQ+10,100)}, guard band)")
    print(f"wet_drying_thr        = {w_lo:.6f}  (q35 of 0.5*(z(RH_Air)+z(RH_H2)))")
    print(f"balanced_all_real_N   = {bal}/class")


if __name__ == "__main__":
    main()
