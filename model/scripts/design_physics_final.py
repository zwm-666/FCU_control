"""Final physics-based 3-class criterion for the PEMFC public dataset.

RESOLVED PHYSICAL FACTS (from probe_physics_units.py + audit_label_semantics.py)

  P1  Single cell: PW = U*I exactly (implied N_cell = 1.0000 +/- 0.0006).
      U_totV is a per-cell voltage -> absolute electrochemical thresholds apply.

  P2  Column provenance. The raw bench file header is
        ... P_Air_inlet, P_H2_inlet, P_Air_outlet, P_H2_outlet, T_1..T_4,
            T_Air_inlet, T_H2_inlet, T_Stack_outlet, T_Heater
      The processed table renamed the SAME positions to
        ... P_Air_supply, P_H2_supply, P_Air_inlet, P_H2_inlet, T_1..T_4,
            T_Air_inlet, T_H2_inlet, T_Stack_inlet, T_Heater
      => processed "P_Air_inlet" is really the cathode OUTLET pressure and
         processed "T_Stack_inlet" is really the stack OUTLET temperature.
      This is why P_Air_supply - P_Air_inlet was negative for 100% of rows:
      it was (inlet - outlet) with the names swapped. Corrected cathode
      pressure drop = P_Air_inlet - P_Air_supply  (outlet - inlet, positive).

  P3  Humidity probe temperature. RH_Air is measured at the humidifier outlet,
      whose temperature is T_Air_inlet (29.5-30.9 C, sigma=0.38, corr(i)=+0.68
      -> a controlled setpoint). T_Stack_inlet (18.8-45.5 C, sigma=9.48) is the
      cell/stack temperature. Using T_Air_inlet as the probe temperature keeps
      p_H2O <= P_total for 100% of rows.

  P4  Polarization curve is real and physical:
        U(i) = 0.8565 - 0.0685*log10(i) - 0.01106*i        (n=30530, i>1A)
      E0=0.857 V, Tafel b=68.5 mV/dec, R=11.1 mOhm, rms=46.7 mV. All three
      inside textbook PEMFC ranges. This is the health baseline.

  P5  m_H2 (1.298) and T_2 (10.0) are CONSTANT -> dead channels, must be
      dropped from the feature set.

  P6  m_Air is fixed at ~6.18 slpm and does NOT follow load (corr = +0.09).
      Cathode stoichiometry is therefore uncontrolled: lambda_air ~ 1/I.
      This is the physical mechanism behind the labels -- at high current the
      fixed air flow can no longer purge product water (flooding); at near-zero
      current the same flow massively over-purges and dries the membrane.
      lambda_air is a genuine physical quantity, not a fitted statistic.

THE CRITERION  (no quantiles, no thresholds tuned on labels)

      lambda_air = (m_Air/(60*22.414))*(1-x_H2O_in)*0.20946 / (I/(4F))

      FLOODING         lambda_air <  LAM_FLOOD   (purge deficit)
      MEMBRANE_DRYING  lambda_air >  LAM_DRY     (purge excess)
      NORMAL           in between AND eta small  (on its polarization curve)

  LAM_FLOOD = 2.0 and LAM_DRY = 3.0 are the standard PEMFC cathode
  stoichiometry design envelope, not data-derived. Below 2.0 the cathode
  cannot remove product water; above 3.0 the excess dry gas strips membrane
  water. Reported below for the actual data, together with an alternative
  water-activity form.

Read-only: prints and validates. Writes nothing.
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
# polarization baseline fitted on all genuine load rows (P4)
E0, TAFEL_B, R_OHM = 0.856459, 0.068525, 0.011064


def p_sat(T):
    T = np.asarray(T, float)
    return 0.61094 * np.exp(17.625 * T / (T + 243.04))


def dew_point(p):
    p = np.clip(np.asarray(p, float), 1e-9, None)
    ln = np.log(p / 0.61094)
    return 243.04 * ln / (17.625 - ln)


def physics(df):
    """All physical quantities, corrected column semantics (P2, P3)."""
    i = df["iA"].to_numpy(float)
    u = df["U_totV"].to_numpy(float)
    m_air = df["m_Air"].to_numpy(float)
    rh_in = df["RH_Air"].to_numpy(float)
    T_probe = df["T_Air_inlet"].to_numpy(float)      # humidifier outlet (P3)
    T_cell = df["T_Stack_inlet"].to_numpy(float)     # really stack outlet (P2)
    P_out = df["P_Air_inlet"].to_numpy(float)        # really cathode outlet (P2)
    P_in = df["P_Air_supply"].to_numpy(float)        # really cathode inlet (P2)

    P_kpa = P_out * 100.0                            # bar(abs) -> kPa
    il = np.clip(i, 0.0, None)

    # inlet humidification
    p_h2o_in = np.clip(rh_in / 100.0, 0.0, 1.0) * p_sat(T_probe)
    x_h2o_in = np.clip(p_h2o_in / P_kpa, 0.0, 0.99)

    # molar flows, per cell
    n_wet_in = m_air / 60.0 / V_STD
    n_dry_in = n_wet_in * (1.0 - x_h2o_in)
    n_h2o_in = n_wet_in * x_h2o_in
    n_o2_req = il / (4.0 * F)
    n_o2_used = n_o2_req
    n_h2o_gen = il / (2.0 * F)

    lam = np.where(il > 0.05, (n_dry_in * X_O2) / np.where(il > 0.05, n_o2_req, np.nan), np.inf)

    # cathode outlet state
    n_h2o_out = n_h2o_in + n_h2o_gen
    n_dry_out = np.clip(n_dry_in - n_o2_used, 1e-12, None)
    x_h2o_out = n_h2o_out / (n_dry_out + n_h2o_out)
    p_h2o_out = x_h2o_out * P_kpa
    rh_out = 100.0 * p_h2o_out / p_sat(T_cell)

    # water activity averaged over the channel (membrane hydration driver)
    a_w = 0.5 * (100.0 * p_h2o_in / p_sat(T_cell) + rh_out) / 100.0

    # polarization overpotential vs the fitted health baseline (P4)
    u_fit = E0 - TAFEL_B * np.log10(np.clip(il, 0.05, None)) - R_OHM * il
    eta = u_fit - u

    return pd.DataFrame({
        "i": i, "U": u, "lambda_air": lam, "eta": eta, "u_fit": u_fit,
        "RH_out_at_cell": rh_out, "a_w": a_w,
        "T_dew_out": dew_point(p_h2o_out), "T_cell": T_cell,
        "dT_cond": dew_point(p_h2o_out) - T_cell,
        "dP_cathode": P_out - P_in,
        "n_h2o_gen": n_h2o_gen, "water_frac": np.where(
            n_h2o_out > 0, n_h2o_gen / np.clip(n_h2o_out, 1e-15, None), 0.0),
    }, index=df.index)


def main() -> None:
    proc = pd.read_csv(SRC)
    pub = pd.read_csv(PUB)
    SHARED = [c for c in proc.columns if c != "tsec"]
    ref = {n: pub.loc[pub["State_Label"] == n].drop_duplicates(subset=SHARED)
           for n in ("Flooding", "Normal", "Membrane_Drying", "Thermal_Management_Fault")}

    print("=" * 92)
    print("1. CORRECTED CATHODE PRESSURE DROP (validates the P2 column remap)")
    print("=" * 92)
    ph_all = physics(proc)
    dP = ph_all["dP_cathode"].to_numpy(float)
    i = proc["iA"].to_numpy(float)
    print(f"  dP = P_Air_inlet - P_Air_supply (outlet - inlet, corrected)")
    print(f"    q05={np.percentile(dP,5):.4f} q50={np.percentile(dP,50):.4f} "
          f"q95={np.percentile(dP,95):.4f} bar   frac>0 = {100*(dP>0).mean():.1f}%")
    print(f"    corr(dP, i)      = {np.corrcoef(dP, i)[0,1]:+.4f}")
    print(f"    corr(dP, m_Air)  = {np.corrcoef(dP, proc['m_Air'])[0,1]:+.4f}")
    print("  -> a single positive-definite pressure drop rising with flow is")
    print("     physically correct; the pre-remap version was negative for 100% of rows.")

    print("\n" + "=" * 92)
    print("2. PHYSICAL QUANTITIES ON THE LABELED REFERENCE ROWS")
    print("=" * 92)
    cols = ["lambda_air", "RH_out_at_cell", "a_w", "eta", "dT_cond", "water_frac"]
    print(f"  {'class':26s} {'n':>6s}", end="")
    for c in cols:
        print(f" {c:>15s}", end="")
    print()
    print("  " + "-" * 118)
    for name, sub in ref.items():
        p = physics(sub)
        print(f"  {name:26s} {len(sub):>6d}", end="")
        for c in cols:
            a = p[c].to_numpy(float)
            a = a[np.isfinite(a)]
            print(f" {np.median(a):>15.4f}", end="")
        print()

    print("\n  -- lambda_air separation (the physical mechanism) --")
    for name, sub in ref.items():
        a = physics(sub)["lambda_air"].to_numpy(float)
        a = a[np.isfinite(a)]
        print(f"    {name:26s} q01={np.percentile(a,1):9.2f} q25={np.percentile(a,25):9.2f} "
              f"q50={np.percentile(a,50):9.2f} q75={np.percentile(a,75):9.2f} "
              f"q99={np.percentile(a,99):9.2f}")

    print("\n  -- eta = U_fit(i) - U, deviation from the fitted polarization curve --")
    for name, sub in ref.items():
        a = physics(sub)["eta"].to_numpy(float)
        a = a[np.isfinite(a)]
        print(f"    {name:26s} q25={np.percentile(a,25):+8.4f} q50={np.percentile(a,50):+8.4f} "
              f"q75={np.percentile(a,75):+8.4f} V   frac>0={100*(a>0).mean():5.1f}%")

    print("\n" + "=" * 92)
    print("3. WHERE THE DATA ACTUALLY SITS IN lambda_air")
    print("=" * 92)
    lam = ph_all["lambda_air"].to_numpy(float)
    finite = np.isfinite(lam)
    print(f"  rows with a defined lambda (i > 0.05 A): {int(finite.sum())} / {len(proc)}")
    for lo, hi, tag in [(0, 1.5, "severe purge deficit"),
                        (1.5, 2.0, "deficit"),
                        (2.0, 2.5, "design low"),
                        (2.5, 3.0, "design nominal"),
                        (3.0, 5.0, "design high"),
                        (5.0, 20.0, "over-purge"),
                        (20.0, 100.0, "gross over-purge"),
                        (100.0, np.inf, "near-zero load")]:
        m = finite & (lam >= lo) & (lam < hi)
        print(f"    lambda in [{lo:6.1f},{hi:6.1f}) {tag:22s} {int(m.sum()):6d} rows")

    print("\n" + "=" * 92)
    print("4. THE PHYSICAL CRITERION")
    print("=" * 92)
    LAM_FLOOD, LAM_DRY = 2.0, 3.0
    ETA_TOL = 0.05
    print(f"  FLOODING        : lambda_air < {LAM_FLOOD}   (cathode cannot purge product water)")
    print(f"  MEMBRANE_DRYING : lambda_air > {LAM_DRY}   (excess dry gas strips membrane water)")
    print(f"  NORMAL          : {LAM_FLOOD} <= lambda_air <= {LAM_DRY} and |eta| <= {ETA_TOL} V")
    eta = ph_all["eta"].to_numpy(float)
    flood = finite & (lam < LAM_FLOOD)
    dry = finite & (lam > LAM_DRY)
    norm = finite & (lam >= LAM_FLOOD) & (lam <= LAM_DRY) & (np.abs(eta) <= ETA_TOL)
    print(f"\n  result on the 37,899-row source:")
    print(f"    Flooding {int(flood.sum()):6d}   Drying {int(dry.sum()):6d}   "
          f"Normal {int(norm.sum()):6d}")
    if flood.sum() == 0 or norm.sum() == 0:
        print("    !! the textbook envelope does not partition THIS bench:")
        print(f"       lambda_air min = {np.nanmin(lam[finite]):.2f} -> the bench never")
        print("       runs below the design envelope, so 'flooding' here cannot mean")
        print("       lambda<2. The mechanism must be expressed RELATIVE to this bench.")

    print("\n" + "=" * 92)
    print("5. BENCH-REFERENCED PHYSICAL CRITERION")
    print("=" * 92)
    print("  m_Air is CONSTANT by design, so lambda_air is a pure function of load:")
    print("      lambda_air(i) = K / i,  K = m_Air*(1-x_H2O)*X_O2*4F/(60*22.414)")
    K = lam[finite] * i[finite]
    print(f"    K = {np.median(K):.2f} A  (spread q01-q99: "
          f"{np.percentile(K,1):.2f} - {np.percentile(K,99):.2f})")
    print(f"    => lambda = 2.0 at i = {np.median(K)/2.0:.2f} A"
          f"  (bench max = {i.max():.2f} A)")
    print(f"    => lambda = 3.0 at i = {np.median(K)/3.0:.2f} A")
    print("  The bench simply never reaches lambda<2. What it DOES do is sweep")
    print("  water production against a fixed purge, so the physical state variable")
    print("  is the CATHODE OUTLET WATER ACTIVITY at the cell temperature.")

    rh_out = ph_all["RH_out_at_cell"].to_numpy(float)
    a_w = ph_all["a_w"].to_numpy(float)
    print(f"\n  RH_out_at_cell over the source: q05={np.percentile(rh_out,5):.1f} "
          f"q50={np.percentile(rh_out,50):.1f} q95={np.percentile(rh_out,95):.1f} %")
    print(f"  a_w (channel-mean water activity): q05={np.percentile(a_w,5):.3f} "
          f"q50={np.percentile(a_w,50):.3f} q95={np.percentile(a_w,95):.3f}")

    print("\n  Absolute physical thresholds on water activity:")
    print("    a_w >= 1.00  : vapour saturated -> liquid water forms       -> FLOODING")
    print("    a_w <= 0.70  : ionomer lambda_H2O < ~9, conductivity falls  -> DRYING")
    print("    0.70 < a_w < 1.00 : membrane hydrated, no liquid            -> NORMAL")
    for name, sub in ref.items():
        a = physics(sub)["a_w"].to_numpy(float)
        a = a[np.isfinite(a)]
        print(f"    labeled {name:26s} a_w q25={np.percentile(a,25):.3f} "
              f"q50={np.percentile(a,50):.3f} q75={np.percentile(a,75):.3f}  "
              f"frac>=1.0={100*np.mean(a>=1.0):5.1f}%  frac<=0.7={100*np.mean(a<=0.7):5.1f}%")

    A_FLOOD, A_DRY = 1.00, 0.70
    f2 = a_w >= A_FLOOD
    d2 = a_w <= A_DRY
    n2 = (a_w > A_DRY) & (a_w < A_FLOOD) & (np.abs(eta) <= ETA_TOL)
    print(f"\n  partition on the source: Flooding={int(f2.sum())} "
          f"Drying={int(d2.sum())} Normal={int(n2.sum())}")

    feat = proc[SHARED].round(9)
    dup = feat.duplicated().to_numpy()
    print(f"  unique real rows       : Flooding={int((f2&~dup).sum())} "
          f"Drying={int((d2&~dup).sum())} Normal={int((n2&~dup).sum())}")
    sizes = [int((m & ~dup).sum()) for m in (f2, d2, n2)]
    print(f"  balanced all-real size : {min(sizes)}/class = {3*min(sizes)} rows")

    def keyset(fr):
        return set(map(tuple, fr[SHARED].to_numpy(float).round(6)))
    print("\n  recall vs labeled reference rows:")
    for name, mask in (("Flooding", f2), ("Membrane_Drying", d2), ("Normal", n2)):
        tgt = keyset(ref[name])
        sel = keyset(proc.loc[mask])
        print(f"    {name:18s} {len(sel & tgt):5d}/{len(tgt):5d} "
              f"({100*len(sel & tgt)/len(tgt):5.1f}%)")

    print("\n" + "=" * 92)
    print("6. CLASS PROFILES UNDER THE WATER-ACTIVITY CRITERION")
    print("=" * 92)
    hdr = (f"  {'class':18s} {'n':>7s} {'a_w':>7s} {'RH_out':>8s} {'lambda':>9s} "
           f"{'eta':>8s} {'U':>7s} {'i':>7s} {'T_cell':>7s} {'dP':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, mask in (("Normal", n2), ("Flooding", f2), ("Membrane_Drying", d2)):
        if not mask.any():
            continue
        print(f"  {name:18s} {int(mask.sum()):>7d} {np.nanmedian(a_w[mask]):>7.3f} "
              f"{np.nanmedian(rh_out[mask]):>8.2f} {np.nanmedian(lam[mask]):>9.2f} "
              f"{np.nanmedian(eta[mask]):>+8.4f} {np.median(proc['U_totV'].to_numpy(float)[mask]):>7.4f} "
              f"{np.median(i[mask]):>7.2f} "
              f"{np.median(proc['T_Stack_inlet'].to_numpy(float)[mask]):>7.2f} "
              f"{np.nanmedian(dP[mask]):>8.4f}")

    print("\n" + "=" * 92)
    print("7. SEPARABILITY AND LEAKAGE")
    print("=" * 92)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    label = np.full(len(proc), "DISCARD", dtype=object)
    label[n2] = "Normal"
    label[d2] = "Membrane_Drying"
    label[f2] = "Flooding"
    live = [c for c in SHARED if proc[c].nunique() > 1]
    print(f"  live features (dead channels dropped): {len(live)} of {len(SHARED)}")
    print(f"  dropped: {[c for c in SHARED if c not in live]}")
    target = min(sizes)
    if target >= 200:
        rng = np.random.default_rng(42)
        idx = []
        for name in ("Normal", "Flooding", "Membrane_Drying"):
            pool = np.flatnonzero((label == name) & ~dup)
            idx.append(rng.permutation(pool)[:target])
        keep = np.concatenate(idx)
        X = proc.loc[keep, live].to_numpy(float)
        y = label[keep]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=42)
        sc = StandardScaler().fit(Xtr)
        knn = KNeighborsClassifier(n_neighbors=1).fit(sc.transform(Xtr), ytr)
        rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                    n_jobs=-1).fit(Xtr, ytr)
        print(f"  balanced subset {X.shape} @ {target}/class")
        print(f"  1-NN test acc = {knn.score(sc.transform(Xte), yte):.4f}")
        print(f"  RF   test acc = {rf.score(Xte, yte):.4f}")
        tk = set(map(tuple, np.round(Xtr, 9)))
        print(f"  exact train/test twins = "
              f"{sum(1 for r in np.round(Xte,9) if tuple(r) in tk)}")
        print("  RF top-8:")
        for k in np.argsort(rf.feature_importances_)[::-1][:8]:
            print(f"    {live[k]:16s} {rf.feature_importances_[k]:.4f}")

    print("\n  temporal structure (episode = tsec gap > 5 s):")
    for name in ("Normal", "Flooding", "Membrane_Drying"):
        m = (label == name) & ~dup
        if m.sum() < 2:
            continue
        ts = np.sort(proc.loc[m, "tsec"].to_numpy(float))
        print(f"    {name:18s} n={int(m.sum()):6d} tsec[{ts.min():8.1f},{ts.max():8.1f}] "
              f"episodes={int((np.diff(ts) > 5).sum()) + 1:4d}")

    print("\n" + "=" * 92)
    print("8. FINAL CONSTANTS (all physical, none tuned on labels)")
    print("=" * 92)
    print(f"  N_cell            = 1                      (PW = U*I verified)")
    print(f"  Faraday F         = 96485 C/mol")
    print(f"  molar volume      = 22.414 L/mol (slpm)")
    print(f"  X_O2              = {X_O2}")
    print(f"  Magnus p_sat      = 0.61094*exp(17.625*T/(T+243.04)) kPa")
    print(f"  T_probe (RH_Air)  = T_Air_inlet          (humidifier, 29.5-30.9 C)")
    print(f"  T_cell            = T_Stack_inlet        (= raw T_Stack_OUTLET)")
    print(f"  P_cathode         = P_Air_inlet          (= raw P_Air_OUTLET), bar abs")
    print(f"  polarization      = U(i) = {E0:.4f} - {TAFEL_B:.4f}*log10(i) - {R_OHM:.5f}*i")
    print(f"                      E0 {E0:.3f} V | Tafel {1000*TAFEL_B:.1f} mV/dec | "
          f"R {1000*R_OHM:.1f} mOhm")
    print(f"  a_w flooding thr  = {A_FLOOD:.2f}   (vapour saturation, liquid water)")
    print(f"  a_w drying thr    = {A_DRY:.2f}   (ionomer dehydration onset)")
    print(f"  eta tolerance     = {ETA_TOL:.2f} V  (on the polarization baseline)")
    print(f"  dead channels     = m_H2 (const 1.298), T_2 (const 10.0)")


if __name__ == "__main__":
    main()
