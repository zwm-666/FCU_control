"""Validate a physics-anchored 3-class rule (Normal / Flooding / Membrane_Drying).

Read-only diagnosis. Tests the two-axis design:
  axis 1 (severity) : current-conditioned voltage residual dU
  axis 2 (direction): wet index  w = z(RH_Air) + z(RH_H2) - z(T_Stack_inlet)
with guard bands on both axes so ambiguous rows are DISCARDED, not forced.

Reports: class counts, unique-row counts after feature dedup, whether 4000
real rows/class is reachable without synthesis, class physical profiles, the
overlap with the old rule's dead 'thermal' bucket, and separability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE = r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv"

MODEL_FEATURES = [
    "U_totV", "iA", "PW", "m_Air", "m_H2", "RH_Air", "RH_H2",
    "P_Air_supply", "P_H2_supply", "P_Air_inlet", "P_H2_inlet",
    "T_1", "T_2", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet",
    "T_Stack_inlet", "T_Heater", "m_Air_write", "m_H2_write",
    "Heater_power", "i_write",
]


def q(a, p):
    return float(np.percentile(a, p))


def z(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def current_conditioned_residual(v, i, n_bins=10):
    """dU = U - median(U | same current bin). Removes the operating point."""
    edges = np.unique(np.quantile(i, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.digitize(i, edges[1:-1]), 0, len(edges) - 2)
    resid = np.empty_like(v)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            resid[m] = v[m] - np.median(v[m])
    return resid, edges, idx


def main() -> None:
    df = pd.read_csv(SOURCE)
    v = df["U_totV"].to_numpy(float)
    i = df["iA"].to_numpy(float)
    ra = df["RH_Air"].to_numpy(float)
    rh = df["RH_H2"].to_numpy(float)
    t = df["T_Stack_inlet"].to_numpy(float)
    n = len(df)

    du, edges, cbin = current_conditioned_residual(v, i)
    w = z(ra) + z(rh) - z(t)

    print("=" * 74)
    print("AXIS STATISTICS")
    print("=" * 74)
    print(f"rows = {n}   current bins = {len(edges)-1}  edges = {np.round(edges,3).tolist()}")
    print(f"dU  : q05={q(du,5):+.4f} q25={q(du,25):+.4f} q50={q(du,50):+.4f} "
          f"q75={q(du,75):+.4f} q95={q(du,95):+.4f}  std={du.std():.4f}")
    print(f"w   : q05={q(w,5):+.4f} q25={q(w,25):+.4f} q50={q(w,50):+.4f} "
          f"q75={q(w,75):+.4f} q95={q(w,95):+.4f}  std={w.std():.4f}")
    print(f"corr(dU, w) = {np.corrcoef(du, w)[0,1]:+.4f}   "
          f"corr(RH_Air,RH_H2)={np.corrcoef(ra,rh)[0,1]:+.4f}  "
          f"corr(RH_Air,T)={np.corrcoef(ra,t)[0,1]:+.4f}")

    print("\n" + "=" * 74)
    print("GRID SEARCH: severity threshold x direction threshold (with guard bands)")
    print("=" * 74)
    print(f"{'sev_p':>6} {'dir_p':>6} | {'Normal':>7} {'Flood':>7} {'Dry':>7} "
          f"{'kept':>7} {'drop':>7} | {'min_class':>9} {'>=4000?':>8}")
    print("-" * 74)
    best = None
    for sev_p in (20, 25, 30, 33, 35, 40):
        for dir_p in (55, 60, 65, 70):
            du_fault = q(du, sev_p)              # fault side threshold
            du_norm = q(du, sev_p + 10)          # normal side threshold -> guard band between
            w_hi, w_lo = q(w, dir_p), q(w, 100 - dir_p)
            fault = du < du_fault
            flood = fault & (w > w_hi)
            dry = fault & (w < w_lo)
            normal = du >= du_norm
            kept = int(flood.sum() + dry.sum() + normal.sum())
            counts = (int(normal.sum()), int(flood.sum()), int(dry.sum()))
            mn = min(counts)
            ok = "YES" if mn >= 4000 else "no"
            print(f"{sev_p:>6} {dir_p:>6} | {counts[0]:>7} {counts[1]:>7} {counts[2]:>7} "
                  f"{kept:>7} {n-kept:>7} | {mn:>9} {ok:>8}")
            if mn >= 4000 and (best is None or mn > best[0]):
                best = (mn, sev_p, dir_p)

    if best is None:
        print("\n!! no configuration reaches 4000 real rows per class")
        return
    _, SEV_P, DIR_P = best
    print(f"\nselected: severity=q{SEV_P} (guard to q{SEV_P+10}), direction=q{DIR_P}/q{100-DIR_P}")

    # ---- realize the selected rule ----
    du_fault, du_norm = q(du, SEV_P), q(du, SEV_P + 10)
    w_hi, w_lo = q(w, DIR_P), q(w, 100 - DIR_P)
    fault = du < du_fault
    flood = fault & (w > w_hi)
    dry = fault & (w < w_lo)
    normal = du >= du_norm

    label = np.full(n, "DISCARD", dtype=object)
    label[normal] = "Normal"
    label[dry] = "Membrane_Drying"
    label[flood] = "Flooding"          # flooding last: wettest wins on any tie
    overlap = int((flood & dry).sum())
    print(f"flood/dry mutual overlap = {overlap} (must be 0)")
    print(f"normal/fault overlap     = {int((normal & fault).sum())} (must be 0 via guard band)")

    print("\n" + "=" * 74)
    print("CLASS COUNTS  (real rows, before and after feature dedup)")
    print("=" * 74)
    feat = df[MODEL_FEATURES].round(12)
    for name in ("Normal", "Flooding", "Membrane_Drying", "DISCARD"):
        m = label == name
        uniq = int((~feat.loc[m].duplicated()).sum()) if m.any() else 0
        print(f"{name:18s} raw={int(m.sum()):6d}   unique={uniq:6d}   "
              f"{'OK for 4000' if uniq >= 4000 else ''}")

    print("\n" + "=" * 74)
    print("PHYSICAL PROFILE PER CLASS  (is each class the fault it claims to be?)")
    print("=" * 74)
    hdr = f"{'class':18s} {'n':>6} {'dU':>9} {'U':>8} {'iA':>7} {'RH_Air':>8} {'RH_H2':>8} {'T':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name in ("Normal", "Flooding", "Membrane_Drying", "DISCARD"):
        m = label == name
        if not m.any():
            continue
        print(f"{name:18s} {int(m.sum()):>6} {du[m].mean():>+9.4f} {v[m].mean():>8.4f} "
              f"{i[m].mean():>7.2f} {ra[m].mean():>8.2f} {rh[m].mean():>8.2f} {t[m].mean():>7.2f}")

    # ---- old rule, for the comparison the thesis needs ----
    vq25, vq50, vq75 = q(v, 25), q(v, 50), q(v, 75)
    aq25, aq50, aq75 = q(ra, 25), q(ra, 50), q(ra, 75)
    hq25, hq50, hq75 = q(rh, 25), q(rh, 50), q(rh, 75)
    tq25, tq50, tq75 = q(t, 25), q(t, 50), q(t, 75)
    iq50, pq50 = q(i, 50), q(df["PW"].to_numpy(float), 50)
    p = df["PW"].to_numpy(float)
    hi = tq75 + (tq75 - tq50)
    o_thermal = ((t > hi) | (t < tq25 - (tq50 - tq25))) & (v < vq50) & (p < pq50)
    o_flood = (ra > aq75) & (rh > hq75) & (v < vq50) & (i > iq50) & (~o_thermal)
    o_dry = ((ra < aq50) & (rh < hq50) & (t >= tq50) & (t <= hi) & (v >= vq25)
             & (v <= vq75) & (~o_thermal) & (~o_flood))
    o_norm = ((v > vq50) & (ra >= aq25) & (ra <= aq75) & (rh >= hq25) & (rh <= hq75)
              & (t >= tq25) & (t <= tq75) & (~o_thermal) & (~o_flood) & (~o_dry))
    o_state = np.zeros(n, np.int64)
    o_state[o_flood] = 1
    o_state[o_dry] = 2
    o_state[o_norm] = 0
    unc = (o_state == 0) & (~o_norm)
    o_hv = unc & (v > vq75)
    o_temp = unc & (~o_hv) & ((t > tq75) | (t < tq25))
    o_lv = unc & (~o_hv) & (~o_temp) & (v <= vq50)
    o_state[o_temp] = 3
    o_state[o_lv] = 2
    old_label = np.array(["Normal", "Flooding", "Membrane_Drying",
                          "Thermal_Management_Fault"], dtype=object)[o_state]

    print("\n" + "=" * 74)
    print("WHAT THE OLD 'THERMAL' BUCKET REALLY WAS (new-rule labels of those rows)")
    print("=" * 74)
    tb = old_label == "Thermal_Management_Fault"
    print(f"old thermal bucket n = {int(tb.sum())}")
    for name in ("Normal", "Flooding", "Membrane_Drying", "DISCARD"):
        c = int((tb & (label == name)).sum())
        print(f"  -> new {name:18s} {c:6d}  ({100*c/max(tb.sum(),1):5.1f}%)")
    print(f"\nold thermal bucket profile: RH_Air mean={ra[tb].mean():.2f} "
          f"T mean={t[tb].mean():.2f} U mean={v[tb].mean():.4f} dU mean={du[tb].mean():+.4f}")

    print("\ncross-tab old (rows) x new (cols):")
    print(pd.crosstab(pd.Series(old_label, name="old"), pd.Series(label, name="new")))

    # ---- separability sanity ----
    print("\n" + "=" * 74)
    print("SEPARABILITY (balanced 4000/class subsample, stratified 80/20, seed 42)")
    print("=" * 74)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    keep_idx = []
    for name in ("Normal", "Flooding", "Membrane_Drying"):
        pool = np.flatnonzero(label == name)
        pool = pool[~feat.iloc[pool].duplicated().to_numpy()]
        keep_idx.append(rng.permutation(pool)[:4000])
    keep = np.concatenate(keep_idx)
    X = df.loc[keep, MODEL_FEATURES].to_numpy(float)
    y = label[keep]
    print(f"balanced subset: {X.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    knn = KNeighborsClassifier(n_neighbors=1).fit(sc.transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    print(f"1-NN  test acc = {knn.score(sc.transform(Xte), yte):.4f}")
    print(f"RF    test acc = {rf.score(Xte, yte):.4f}")

    # exact-duplicate twins across the split (leakage probe)
    tr_keys = set(map(tuple, np.round(Xtr, 12)))
    twins = sum(1 for r in np.round(Xte, 12) if tuple(r) in tr_keys)
    print(f"exact train/test feature twins = {twins}  (leakage probe, want 0)")

    print("\nRF top-10 features (does the label just read back its own definition?):")
    order = np.argsort(rf.feature_importances_)[::-1][:10]
    for k in order:
        print(f"  {MODEL_FEATURES[k]:16s} {rf.feature_importances_[k]:.4f}")

    print("\n" + "=" * 74)
    print("FINAL RULE CONSTANTS (paste into the builder)")
    print("=" * 74)
    print(f"current_bin_edges      = {np.round(edges, 6).tolist()}")
    print(f"dU_fault_threshold     = {du_fault:.6f}   (= q{SEV_P} of dU)")
    print(f"dU_normal_threshold    = {du_norm:.6f}   (= q{SEV_P+10} of dU, guard band)")
    print(f"w_flooding_threshold   = {w_hi:.6f}   (= q{DIR_P} of w)")
    print(f"w_drying_threshold     = {w_lo:.6f}   (= q{100-DIR_P} of w)")
    print(f"discarded_rows         = {int((label=='DISCARD').sum())} "
          f"({100*(label=='DISCARD').mean():.1f}%)")


if __name__ == "__main__":
    main()
