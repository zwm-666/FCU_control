"""Settle the final design: load-controlled, physics-labeled 3-class dataset.

Established, non-negotiable physical facts:

  F1  lambda_air = K/i with K = 360.8 A. Bench max current 28.82 A, so
      lambda_air >= 12.31 ALWAYS. Textbook flooding needs lambda < 2, i.e.
      i > 180 A. THIS BENCH CANNOT FLOOD BY STOICHIOMETRY STARVATION.
      Cathode air flow is fixed (m_Air ~ 6.18 slpm, corr(m_Air,i) = +0.09).

  F2  Rigorous cathode water balance says the rows LABELED Flooding are
      subsaturated: RH_out_at_cell = 79.3%, a_w = 0.624, 100% of them below
      a_w = 0.70. They are physically DRY, not flooded.

  F3  The rows LABELED Membrane_Drying sit at lambda_air = 1419 and
      eta = +0.49 V, with iA q50 = 0.255 A. That is a near-zero-load / OCV
      condition, not a dehydrated membrane under load.

  F4  A depth-4 tree on (iA, U_totV) alone recovers 89.5% of the four public
      labels; adding RH_Air and T_Stack_inlet gives 96.7%. The labels are
      largely an OPERATING POINT, not an independent fault state.

  F5  Polarization baseline is physical and fits well:
        U(i) = 0.8565 - 0.0685*log10(i) - 0.011064*i
      E0 = 0.856 V, Tafel = 68.5 mV/dec, R = 11.1 mOhm, rms = 46.7 mV.

Consequence: a first-principles criterion CANNOT reproduce the published
labels. So the deliverable is a criterion that is physically correct on its own
terms. Because water management at fixed air flow is dominated by cell
temperature, the criterion is evaluated TWICE:

  Design A  free operating point   -- all 37,899 rows, current varies 0-28.8 A
  Design B  load-controlled        -- only the i in [12,16] A plateau, so
            current is held ~constant and the classes differ ONLY in their
            water/thermal state. This is the defensible fault-diagnosis task:
            same load, different cathode water state.

Read-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv")
PUB = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets.csv")

F = 96485.0
V_STD = 22.414
X_O2 = 0.20946
E0, TAFEL_B, R_OHM = 0.856459, 0.068525, 0.011064

A_FLOOD, A_DRY = 1.00, 0.70     # water activity: condensation / dehydration onset


def p_sat(T):
    T = np.asarray(T, float)
    return 0.61094 * np.exp(17.625 * T / (T + 243.04))


def dew_point(p):
    p = np.clip(np.asarray(p, float), 1e-9, None)
    ln = np.log(p / 0.61094)
    return 243.04 * ln / (17.625 - ln)


def physics(df):
    i = df["iA"].to_numpy(float)
    u = df["U_totV"].to_numpy(float)
    m_air = df["m_Air"].to_numpy(float)
    rh_in = df["RH_Air"].to_numpy(float)
    T_probe = df["T_Air_inlet"].to_numpy(float)
    T_cell = df["T_Stack_inlet"].to_numpy(float)
    P_kpa = df["P_Air_inlet"].to_numpy(float) * 100.0
    il = np.clip(i, 0.0, None)

    p_h2o_in = np.clip(rh_in / 100.0, 0.0, 1.0) * p_sat(T_probe)
    x_in = np.clip(p_h2o_in / P_kpa, 0.0, 0.99)
    n_wet = m_air / 60.0 / V_STD
    n_dry = n_wet * (1.0 - x_in)
    n_h2o_in = n_wet * x_in
    n_o2 = il / (4.0 * F)
    n_gen = il / (2.0 * F)

    lam = np.where(il > 0.05, (n_dry * X_O2) / np.where(il > 0.05, n_o2, np.nan), np.inf)
    n_h2o_out = n_h2o_in + n_gen
    n_dry_out = np.clip(n_dry - n_o2, 1e-12, None)
    p_h2o_out = n_h2o_out / (n_dry_out + n_h2o_out) * P_kpa
    rh_out = 100.0 * p_h2o_out / p_sat(T_cell)
    rh_in_cell = 100.0 * p_h2o_in / p_sat(T_cell)
    a_w = 0.5 * (rh_in_cell + rh_out) / 100.0

    # liquid water rate when the outlet is supersaturated [mol/s]
    x_sat = np.clip(p_sat(T_cell) / P_kpa, 0.0, 0.99)
    n_h2o_cap = x_sat / (1.0 - x_sat) * n_dry_out
    n_liq = np.clip(n_h2o_out - n_h2o_cap, 0.0, None)

    u_fit = E0 - TAFEL_B * np.log10(np.clip(il, 0.05, None)) - R_OHM * il
    return pd.DataFrame({
        "i": i, "U": u, "lambda_air": lam, "a_w": a_w, "RH_out": rh_out,
        "RH_in_cell": rh_in_cell, "eta": u_fit - u, "n_liq": n_liq,
        "T_cell": T_cell, "T_dew_out": dew_point(p_h2o_out),
        "dP": df["P_Air_inlet"].to_numpy(float) - df["P_Air_supply"].to_numpy(float),
    }, index=df.index)


def evaluate(name, df, live, tag_extra=""):
    """Label by water activity, then report capacity / physics / separability."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    ph = physics(df)
    a_w = ph["a_w"].to_numpy(float)
    eta = ph["eta"].to_numpy(float)

    flood = a_w >= A_FLOOD
    dry = a_w <= A_DRY
    norm = (~flood) & (~dry)

    label = np.full(len(df), "?", dtype=object)
    label[norm] = "Normal"
    label[dry] = "Membrane_Drying"
    label[flood] = "Flooding"

    feat = df[live].round(9)
    dup = feat.duplicated().to_numpy()
    sizes = {n: int(((label == n) & ~dup).sum())
             for n in ("Normal", "Flooding", "Membrane_Drying")}
    bal = min(sizes.values())

    print(f"\n{'='*92}")
    print(f"{name}  {tag_extra}")
    print("=" * 92)
    print(f"  rows = {len(df)}   unique = {int((~dup).sum())}")
    print(f"  class sizes (unique real) = {sizes}")
    print(f"  balanced all-real = {bal}/class = {3*bal} rows")

    hdr = (f"  {'class':18s} {'n':>7s} {'a_w':>7s} {'RH_out':>8s} {'RH_in@c':>8s} "
           f"{'lambda':>8s} {'eta':>8s} {'U':>7s} {'i':>6s} {'T_cell':>7s} {'n_liq':>10s}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for n in ("Normal", "Flooding", "Membrane_Drying"):
        m = (label == n) & ~dup
        if not m.any():
            continue
        print(f"  {n:18s} {int(m.sum()):>7d} {np.nanmedian(a_w[m]):>7.3f} "
              f"{np.nanmedian(ph['RH_out'].to_numpy()[m]):>8.2f} "
              f"{np.nanmedian(ph['RH_in_cell'].to_numpy()[m]):>8.2f} "
              f"{np.nanmedian(ph['lambda_air'].to_numpy()[m]):>8.2f} "
              f"{np.nanmedian(eta[m]):>+8.4f} "
              f"{np.nanmedian(ph['U'].to_numpy()[m]):>7.4f} "
              f"{np.nanmedian(ph['i'].to_numpy()[m]):>6.2f} "
              f"{np.nanmedian(ph['T_cell'].to_numpy()[m]):>7.2f} "
              f"{np.nanmedian(ph['n_liq'].to_numpy()[m]):>10.3e}")

    if bal < 150:
        print("  !! too few rows for a balanced set")
        return None

    rng = np.random.default_rng(42)
    idx = []
    for n in ("Normal", "Flooding", "Membrane_Drying"):
        pool = np.flatnonzero((label == n) & ~dup)
        idx.append(rng.permutation(pool)[:bal])
    keep = np.concatenate(idx)
    X = df.iloc[keep][live].to_numpy(float)
    y = label[keep]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    knn = KNeighborsClassifier(n_neighbors=1).fit(sc.transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    tk = set(map(tuple, np.round(Xtr, 9)))
    twins = sum(1 for r in np.round(Xte, 9) if tuple(r) in tk)
    print(f"\n  1-NN test acc = {knn.score(sc.transform(Xte), yte):.4f}   "
          f"RF test acc = {rf.score(Xte, yte):.4f}   twins = {twins}")

    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    print("  RF top-8: " + "  ".join(f"{live[k]}={imp[k]:.3f}" for k in order[:8]))
    temp_share = sum(imp[k] for k in range(len(live)) if live[k].startswith("T_"))
    print(f"  temperature-channel importance share = {100*temp_share:.1f}%")

    # current spread per class: is the label just the operating point?
    print("  current per class (want identical -> label is NOT the operating point):")
    for n in ("Normal", "Flooding", "Membrane_Drying"):
        m = (label == n) & ~dup
        a = ph["i"].to_numpy(float)[m]
        print(f"    {n:18s} i q05={np.percentile(a,5):6.2f} q50={np.percentile(a,50):6.2f} "
              f"q95={np.percentile(a,95):6.2f}")

    ts = df["tsec"].to_numpy(float)
    print("  temporal span per class (overlap -> no time/label collinearity):")
    for n in ("Normal", "Flooding", "Membrane_Drying"):
        m = (label == n) & ~dup
        a = np.sort(ts[m])
        print(f"    {n:18s} tsec[{a.min():8.1f},{a.max():8.1f}] "
              f"episodes={int((np.diff(a) > 5).sum())+1:4d}")
    return {"sizes": sizes, "bal": bal, "label": label, "dup": dup}


def main() -> None:
    proc = pd.read_csv(SRC)
    pub = pd.read_csv(PUB)
    SHARED = [c for c in proc.columns if c != "tsec"]
    live = [c for c in SHARED if proc[c].nunique() > 1]
    print(f"source {proc.shape}; live features {len(live)} "
          f"(dropped dead: {[c for c in SHARED if c not in live]})")

    print("\n" + "=" * 92)
    print("0. THE STOICHIOMETRY WALL (why textbook flooding is impossible here)")
    print("=" * 92)
    ph = physics(proc)
    lam = ph["lambda_air"].to_numpy(float)
    fin = np.isfinite(lam)
    print(f"  lambda_air over all load rows: min={np.nanmin(lam[fin]):.2f} "
          f"q50={np.nanmedian(lam[fin]):.2f} max={np.nanmax(lam[fin]):.1f}")
    print(f"  design flooding regime is lambda < 2.0 -> requires i > "
          f"{np.nanmedian(lam[fin]*proc['iA'].to_numpy(float)[fin])/2.0:.0f} A; "
          f"bench max = {proc['iA'].max():.2f} A")
    print("  => flooding in this dataset cannot be a stoichiometry-starvation fault.")
    print("     The only physical water-management lever left is CELL TEMPERATURE")
    print("     acting on vapour-carrying capacity, which is what the criterion uses.")

    print("\n" + "=" * 92)
    print("1. WHAT THE PUBLISHED LABELS ARE, PHYSICALLY")
    print("=" * 92)
    hdr = (f"  {'published label':26s} {'n':>6s} {'i q50':>7s} {'lambda':>9s} "
           f"{'a_w':>7s} {'RH_out':>8s} {'eta':>8s}  verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) + 20))
    verdicts = {
        "Flooding": "subsaturated (a_w<0.7) -> NOT flooded",
        "Normal": "driest class of all -> mislabelled",
        "Membrane_Drying": "i~0.26 A, eta +0.49 V -> no-load/OCV",
        "Thermal_Management_Fault": "a_w 1.38 -> the only truly CONDENSING class",
    }
    for n in ("Flooding", "Normal", "Membrane_Drying", "Thermal_Management_Fault"):
        sub = pub.loc[pub["State_Label"] == n].drop_duplicates(subset=SHARED)
        p = physics(sub)
        print(f"  {n:26s} {len(sub):>6d} {np.nanmedian(p['i']):>7.2f} "
              f"{np.nanmedian(p['lambda_air']):>9.1f} {np.nanmedian(p['a_w']):>7.3f} "
              f"{np.nanmedian(p['RH_out']):>8.1f} {np.nanmedian(p['eta']):>+8.4f}"
              f"  {verdicts[n]}")

    # ---- Design A: free operating point ----
    evaluate("DESIGN A - free operating point (all rows)", proc, live,
             "current varies 0-28.8 A")

    # ---- Design B: load-controlled plateau ----
    i = proc["iA"].to_numpy(float)
    plateau = (i >= 12.0) & (i <= 16.0)
    print(f"\n  load plateau i in [12,16] A -> {int(plateau.sum())} rows "
          f"({100*plateau.mean():.1f}% of source)")
    sub = proc.loc[plateau].reset_index(drop=True)
    evaluate("DESIGN B - load-controlled (i in [12,16] A)", sub, live,
             "current held ~constant; classes differ only in water/thermal state")

    print("\n" + "=" * 92)
    print("FINAL PHYSICAL CONSTANTS")
    print("=" * 92)
    print("  N_cell = 1                     (PW = U*I verified to 4e-2 W max err)")
    print("  F = 96485 C/mol, Vm = 22.414 L/mol (slpm), X_O2 = 0.20946")
    print("  p_sat(T) = 0.61094*exp(17.625*T/(T+243.04))  [kPa]  (Magnus)")
    print("  T_probe  = T_Air_inlet     (humidifier outlet, 29.5-30.9 C)")
    print("  T_cell   = T_Stack_inlet   (= raw T_Stack_OUTLET, 18.8-45.5 C)")
    print("  P_cath   = P_Air_inlet     (= raw P_Air_OUTLET) [bar abs]")
    print("  dP_cath  = P_Air_inlet - P_Air_supply  (>0 for 100% of rows)")
    print(f"  U_fit(i) = {E0:.4f} - {TAFEL_B:.4f}*log10(i) - {R_OHM:.6f}*i")
    print("             E0 0.856 V | Tafel 68.5 mV/dec | R 11.1 mOhm | rms 46.7 mV")
    print("  a_w = 0.5*(RH_in@cell + RH_out@cell)/100   [channel-mean water activity]")
    print(f"  FLOODING        a_w >= {A_FLOOD:.2f}   (vapour saturated -> liquid water)")
    print(f"  MEMBRANE_DRYING a_w <= {A_DRY:.2f}   (ionomer water content < ~9 H2O/SO3-)")
    print(f"  NORMAL          {A_DRY:.2f} < a_w < {A_FLOOD:.2f}")
    print("  dead channels dropped: m_H2 (const 1.298), T_2 (const 10.0)")


if __name__ == "__main__":
    main()
