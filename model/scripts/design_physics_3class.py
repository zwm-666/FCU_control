"""Physics-based 3-class criterion: condensation thermodynamics + water balance.

Established by probe_physics_units.py:
  * SINGLE CELL: PW = U*I exactly (implied N_cell = 1.0000). U_totV is a
    per-cell voltage, so absolute electrochemical thresholds apply.
  * m_H2 is CONSTANT (1.298, 1 distinct value)  -> dead channel, and H2 is
    fixed-flow / dead-end. T_2 is CONSTANT (10.0) -> dead channel.
  * m_Air ~ 6.18 slpm with corr(m_Air, i) = +0.09 -> air flow is NOT
    load-following. Cathode stoichiometry is therefore uncontrolled and falls
    as ~1/I. lambda_Air(labeled Flooding) = 13.1 vs 24.9 for Normal/Drying.
  * Raw bench file names the channel T_Stack_OUTLET; the processed table
    renamed it T_Stack_inlet. It is the hottest stack-side probe (q50 36 C,
    max 45.5 C) -> used here as the cell temperature.

The criterion below uses no quantiles. It is built from:

  (1) DEW POINT  (inverse Magnus)
        p_H2O = RH/100 * p_sat(T_sensor);  T_dew = Magnus^-1(p_H2O)
      Condensation margin  dT_cond = T_dew - T_cell.
      dT_cond > 0  =>  vapour is supersaturated at the cell  =>  liquid water.
      This is an ABSOLUTE, dataset-independent threshold at 0 C.

  (2) CATHODE WATER BALANCE  (per-cell, Faraday)
        n_air_dry = m_Air/(60*22.414) * (1 - x_H2O_in)      mol/s
        n_H2O_gen = I/(2F)                                  mol/s
        n_O2_used = I/(4F)                                  mol/s
      outlet vapour fraction, then RH at the cell temperature:
        RH_ca_out = p_H2O_out / p_sat(T_cell) * 100
      RH_ca_out > 100 %  =>  condensation at the cathode outlet  =>  FLOODING.

  (3) CATHODE STOICHIOMETRY  lambda_Air = n_air_supplied / (I/(4F*0.21))
      The purge capacity for product water.

  (4) VOLTAGE LOSS vs a fitted polarization baseline
        eta = U_fit(i) - U   (fit on the healthiest rows only)
      Separates "a fault is present" from "which fault".

Validated against the 1,346 unique real labeled Flooding rows, 2,500 Normal
and 2,500 Membrane_Drying rows in Public datasets.csv.

Read-only: prints the criterion and its validation. Writes nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv")
PUB = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets.csv")

F = 96485.0
V_MOLAR_STD = 22.414      # L/mol, slpm convention
X_O2 = 0.20946

# Magnus coefficients over water (0-60 C), kPa
MAG_A, MAG_B, MAG_C = 0.61094, 17.625, 243.04


def p_sat(T_c):
    """Saturation vapour pressure [kPa] (Magnus/Alduchov-Eskridge)."""
    return MAG_A * np.exp(MAG_B * np.asarray(T_c, float) / (np.asarray(T_c, float) + MAG_C))


def dew_point(p_kpa):
    """Inverse Magnus: dew point [C] from vapour pressure [kPa]."""
    p = np.clip(np.asarray(p_kpa, float), 1e-9, None)
    ln = np.log(p / MAG_A)
    return MAG_C * ln / (MAG_B - ln)


def add_physics(df, p_total_bar_abs, t_cell_col, t_sensor_col):
    """Attach the physical quantities. p_total in bar absolute."""
    out = {}
    i = df["iA"].to_numpy(float)
    u = df["U_totV"].to_numpy(float)
    m_air = df["m_Air"].to_numpy(float)
    rh_in = df["RH_Air"].to_numpy(float)
    t_cell = df[t_cell_col].to_numpy(float)
    t_sens = df[t_sensor_col].to_numpy(float)

    P_kpa = np.asarray(p_total_bar_abs, float) * 100.0

    # ---- (1) dew point & condensation margin ----
    p_h2o_in = np.clip(rh_in / 100.0, 0, 1) * p_sat(t_sens)
    t_dew_in = dew_point(p_h2o_in)
    out["p_h2o_in_kpa"] = p_h2o_in
    out["T_dew_in"] = t_dew_in
    out["dT_cond_in"] = t_dew_in - t_cell          # >0 => condensing at cell
    out["RH_at_cell"] = 100.0 * p_h2o_in / p_sat(t_cell)

    # ---- (2) cathode water balance, per cell ----
    n_wet_in = m_air / 60.0 / V_MOLAR_STD                 # mol/s total wet gas
    x_h2o_in = np.clip(p_h2o_in / P_kpa, 0, 0.99)
    n_dry_in = n_wet_in * (1.0 - x_h2o_in)                # mol/s dry air
    n_h2o_in = n_wet_in * x_h2o_in
    i_load = np.clip(i, 0.0, None)
    n_h2o_gen = i_load / (2.0 * F)
    n_o2_used = i_load / (4.0 * F)

    n_h2o_out = n_h2o_in + n_h2o_gen
    n_dry_out = np.clip(n_dry_in - n_o2_used, 1e-12, None)
    n_tot_out = n_dry_out + n_h2o_out
    x_h2o_out = n_h2o_out / n_tot_out
    p_h2o_out = x_h2o_out * P_kpa

    out["p_h2o_out_kpa"] = p_h2o_out
    out["RH_ca_out"] = 100.0 * p_h2o_out / p_sat(t_cell)
    out["T_dew_out"] = dew_point(p_h2o_out)
    out["dT_cond_out"] = dew_point(p_h2o_out) - t_cell
    # liquid water condensed at the outlet, mol/s (0 when subsaturated)
    x_sat_cell = np.clip(p_sat(t_cell) / P_kpa, 0, 0.99)
    n_h2o_max = x_sat_cell / (1.0 - x_sat_cell) * n_dry_out
    out["n_liquid"] = np.clip(n_h2o_out - n_h2o_max, 0.0, None)

    # ---- (3) stoichiometry ----
    n_o2_req = np.where(i_load > 0.05, i_load / (4.0 * F), np.nan)
    out["lambda_air"] = (n_dry_in * X_O2) / n_o2_req
    out["lambda_h2"] = (df["m_H2"].to_numpy(float) / 60.0 / V_MOLAR_STD) / np.where(
        i_load > 0.05, i_load / (2.0 * F), np.nan)

    # ---- (4) current density (single cell, active area unknown -> use A) ----
    out["i_A"] = i
    out["U"] = u
    return pd.DataFrame(out, index=df.index)


def fit_polarization(u, i, healthy_mask):
    """U = E0 - b*log10(max(i,i0)) - R*i  fitted on healthy rows only."""
    from scipy.optimize import curve_fit

    def model(i_, E0, b, R):
        return E0 - b * np.log10(np.clip(i_, 0.05, None)) - R * i_

    m = healthy_mask & np.isfinite(u) & np.isfinite(i) & (i > 0.05)
    popt, _ = curve_fit(model, i[m], u[m], p0=[0.95, 0.06, 0.005], maxfev=20000)
    return popt, model


def report(tag, phys, mask, cols):
    sub = phys.loc[mask]
    print(f"  {tag:26s} n={int(mask.sum()):6d}", end="")
    for c in cols:
        a = sub[c].to_numpy(float)
        a = a[np.isfinite(a)]
        med = np.median(a) if a.size else np.nan
        print(f"  {c}={med:9.3f}", end="")
    print()


def main() -> None:
    proc = pd.read_csv(SRC)
    pub = pd.read_csv(PUB)
    SHARED = [c for c in proc.columns if c != "tsec"]

    print("=" * 88)
    print("0. DEAD CHANNELS (constant -> must be dropped from the feature set)")
    print("=" * 88)
    for c in proc.columns:
        if proc[c].nunique() == 1:
            print(f"  {c:16s} constant = {proc[c].iloc[0]}")

    # ---- resolve pressure convention + sensor temperature by physical sanity ----
    print("\n" + "=" * 88)
    print("1. RESOLVE CONVENTIONS: which (P_total, T_sensor, T_cell) is physical?")
    print("=" * 88)
    pubf = pub.loc[pub["State_Label"] == "Flooding"].drop_duplicates(subset=SHARED)
    pubn = pub.loc[pub["State_Label"] == "Normal"].drop_duplicates(subset=SHARED)
    pubd = pub.loc[pub["State_Label"] == "Membrane_Drying"].drop_duplicates(subset=SHARED)
    print(f"  reference rows: Flooding={len(pubf)} Normal={len(pubn)} Drying={len(pubd)}")

    combos = []
    for pconv, pname in ((1.0, "P_Air_inlet as bar(abs)"), (2.0, "P_Air_inlet as bar(gauge)+1")):
        for tsens in ("T_Air_inlet", "T_Stack_inlet"):
            for tcell in ("T_Stack_inlet", "T_3"):
                if tsens == tcell:
                    continue
                combos.append((pconv, pname, tsens, tcell))

    print(f"\n  {'P conv':28s} {'T_sensor':15s} {'T_cell':15s} "
          f"{'flood dTc':>10s} {'norm dTc':>10s} {'dry dTc':>10s} {'sep':>7s}")
    print("  " + "-" * 100)
    best = None
    for pconv, pname, tsens, tcell in combos:
        rows = []
        for sub in (pubf, pubn, pubd):
            P = sub["P_Air_inlet"].to_numpy(float) + (1.0 if pconv == 2.0 else 0.0)
            ph = add_physics(sub, P, tcell, tsens)
            rows.append(np.nanmedian(ph["dT_cond_out"].to_numpy(float)))
        fl, no, dr = rows
        # physical requirement: flooding condenses (>0), drying does not (<0)
        sep = fl - max(no, dr)
        flag = "OK" if (fl > 0 and dr < 0) else ""
        print(f"  {pname:28s} {tsens:15s} {tcell:15s} "
              f"{fl:>10.3f} {no:>10.3f} {dr:>10.3f} {sep:>7.3f} {flag}")
        if best is None or sep > best[0]:
            best = (sep, pconv, pname, tsens, tcell)

    _, PCONV, PNAME, TSENS, TCELL = best
    print(f"\n  => selected: {PNAME}, T_sensor={TSENS}, T_cell={TCELL}")

    def phys_of(sub):
        P = sub["P_Air_inlet"].to_numpy(float) + (1.0 if PCONV == 2.0 else 0.0)
        return add_physics(sub, P, TCELL, TSENS)

    print("\n" + "=" * 88)
    print("2. PHYSICAL QUANTITIES ON THE LABELED REFERENCE ROWS")
    print("=" * 88)
    cols = ["dT_cond_out", "RH_ca_out", "lambda_air", "n_liquid", "T_dew_out"]
    print(f"  {'class':26s} {'n':>8s}", end="")
    for c in cols:
        print(f"  {c:>14s}", end="")
    print()
    print("  " + "-" * 110)
    for tag, sub in (("Flooding (real)", pubf), ("Normal (real)", pubn),
                     ("Membrane_Drying (real)", pubd)):
        ph = phys_of(sub)
        print(f"  {tag:26s} {len(sub):>8d}", end="")
        for c in cols:
            a = ph[c].to_numpy(float)
            a = a[np.isfinite(a)]
            print(f"  {np.median(a):>14.4f}", end="")
        print()

    print("\n  -- distribution overlap on the key quantity dT_cond_out [C] --")
    for tag, sub in (("Flooding", pubf), ("Normal", pubn), ("Drying", pubd)):
        a = phys_of(sub)["dT_cond_out"].to_numpy(float)
        a = a[np.isfinite(a)]
        print(f"    {tag:10s} q01={np.percentile(a,1):8.3f} q25={np.percentile(a,25):8.3f} "
              f"q50={np.percentile(a,50):8.3f} q75={np.percentile(a,75):8.3f} "
              f"q99={np.percentile(a,99):8.3f}   frac>0={100*(a>0).mean():5.1f}%")

    print("\n  -- RH_ca_out [%], the water-balance saturation level --")
    for tag, sub in (("Flooding", pubf), ("Normal", pubn), ("Drying", pubd)):
        a = phys_of(sub)["RH_ca_out"].to_numpy(float)
        a = a[np.isfinite(a)]
        print(f"    {tag:10s} q01={np.percentile(a,1):8.2f} q25={np.percentile(a,25):8.2f} "
              f"q50={np.percentile(a,50):8.2f} q75={np.percentile(a,75):8.2f} "
              f"q99={np.percentile(a,99):8.2f}   frac>100={100*(a>100).mean():5.1f}%")

    print("\n  -- lambda_air, cathode purge capacity --")
    for tag, sub in (("Flooding", pubf), ("Normal", pubn), ("Drying", pubd)):
        a = phys_of(sub)["lambda_air"].to_numpy(float)
        a = a[np.isfinite(a)]
        print(f"    {tag:10s} q01={np.percentile(a,1):8.2f} q25={np.percentile(a,25):8.2f} "
              f"q50={np.percentile(a,50):8.2f} q75={np.percentile(a,75):8.2f} "
              f"q99={np.percentile(a,99):8.2f}")

    # ---- polarization baseline on the source ----
    print("\n" + "=" * 88)
    print("3. POLARIZATION BASELINE (fitted on thermodynamically healthy rows)")
    print("=" * 88)
    ph = phys_of(proc)
    u = proc["U_totV"].to_numpy(float)
    i = proc["iA"].to_numpy(float)
    rh_out = ph["RH_ca_out"].to_numpy(float)
    dtc = ph["dT_cond_out"].to_numpy(float)
    lam = ph["lambda_air"].to_numpy(float)

    healthy = (rh_out > 40) & (rh_out < 95) & (i > 0.05)
    print(f"  healthy calibration rows (40% < RH_ca_out < 95%): {int(healthy.sum())}")
    try:
        (E0, b, Rint), model = fit_polarization(u, i, healthy)
        print(f"  U(i) = {E0:.4f} - {b:.4f}*log10(i) - {Rint:.6f}*i")
        print(f"    E0   = {E0:.4f} V      (OCV-ish intercept, physical range 0.9-1.05)")
        print(f"    b    = {b:.4f} V/dec  (Tafel slope, physical range 0.04-0.12)")
        print(f"    R    = {Rint*1000:.3f} mOhm    (area-specific ohmic term)")
        eta = model(i, E0, b, Rint) - u
        ph["eta"] = eta
        print(f"  eta = U_fit - U : q25={np.percentile(eta[i>0.05],25):+.4f} "
              f"q50={np.percentile(eta[i>0.05],50):+.4f} "
              f"q75={np.percentile(eta[i>0.05],75):+.4f} V")
    except Exception as exc:
        print(f"  fit failed: {exc}")
        ph["eta"] = np.nan

    print("\n" + "=" * 88)
    print("4. THE PHYSICAL CRITERION APPLIED TO ALL 37,899 SOURCE ROWS")
    print("=" * 88)
    n = len(proc)
    eta = ph["eta"].to_numpy(float)

    # Absolute, physics-anchored thresholds:
    #   condensation  : dT_cond_out > 0   (vapour supersaturated at cell)
    #   subsaturation : RH_ca_out < 100 - margin
    #   fault present : eta > 0 (underperforming its own polarization curve)
    COND_MARGIN = 0.0     # C, thermodynamic equality point
    DRY_RH = 80.0         # %, below this the ionomer starts losing conductivity
    ETA_MIN = 0.0         # V

    flooding = (dtc > COND_MARGIN) & (i > 0.05)
    drying = (rh_out < DRY_RH) & (i > 0.05) & (~flooding)
    normal = (~flooding) & (~drying) & (i > 0.05)

    print(f"  raw thermodynamic partition (no eta gate, no quantiles):")
    print(f"    Flooding  (dT_cond_out > 0 C)      {int(flooding.sum()):6d}")
    print(f"    Drying    (RH_ca_out < {DRY_RH:.0f} %)      {int(drying.sum()):6d}")
    print(f"    Normal    (remainder, under load)  {int(normal.sum()):6d}")
    print(f"    excluded  (i <= 0.05 A, no load)   {int((i <= 0.05).sum()):6d}")

    print(f"\n  recall against the labeled reference rows:")
    def keyset(frame):
        return set(map(tuple, frame[SHARED].to_numpy(float).round(6)))
    ks = {name: keyset(s) for name, s in
          (("Flooding", pubf), ("Normal", pubn), ("Drying", pubd))}
    for name, mask in (("Flooding", flooding), ("Normal", normal), ("Drying", drying)):
        sel = keyset(proc.loc[mask])
        tgt = ks[name]
        print(f"    {name:10s} recovers {len(sel & tgt):5d}/{len(tgt):5d} "
              f"({100*len(sel & tgt)/len(tgt):5.1f}%)")

    print(f"\n  cross-check: do labeled Flooding rows satisfy dT_cond_out>0 ?")
    for name, sub in (("Flooding", pubf), ("Normal", pubn), ("Drying", pubd)):
        a = phys_of(sub)["dT_cond_out"].to_numpy(float)
        print(f"    labeled {name:10s} frac(dT_cond_out>0) = {100*np.nanmean(a>0):5.1f}%")

    print("\n" + "=" * 88)
    print("5. CLASS PROFILES UNDER THE PHYSICAL CRITERION")
    print("=" * 88)
    hdr = (f"  {'class':18s} {'n':>7s} {'dT_cond':>9s} {'RH_ca':>8s} {'lambda':>8s} "
           f"{'eta':>8s} {'U':>7s} {'iA':>7s} {'T_cell':>7s} {'RH_Air':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    tcell_a = proc[TCELL].to_numpy(float)
    for name, mask in (("Normal", normal), ("Flooding", flooding), ("Drying", drying)):
        if not mask.any():
            continue
        print(f"  {name:18s} {int(mask.sum()):>7d} "
              f"{np.nanmedian(dtc[mask]):>9.3f} {np.nanmedian(rh_out[mask]):>8.2f} "
              f"{np.nanmedian(lam[mask]):>8.2f} {np.nanmedian(eta[mask]):>8.4f} "
              f"{np.median(u[mask]):>7.4f} {np.median(i[mask]):>7.2f} "
              f"{np.median(tcell_a[mask]):>7.2f} "
              f"{np.median(proc['RH_Air'].to_numpy(float)[mask]):>7.2f}")

    print("\n" + "=" * 88)
    print("6. BALANCED ALL-REAL CAPACITY")
    print("=" * 88)
    feat = proc[SHARED].round(9)
    dup = feat.duplicated().to_numpy()
    sizes = {}
    for name, mask in (("Normal", normal), ("Flooding", flooding), ("Drying", drying)):
        sizes[name] = int((mask & ~dup).sum())
    print(f"  unique real rows: {sizes}")
    print(f"  balanced all-real size = {min(sizes.values())}/class "
          f"= {3*min(sizes.values())} rows total")

    print("\n" + "=" * 88)
    print("7. FINAL PHYSICAL CONSTANTS")
    print("=" * 88)
    print(f"  cell count                = 1 (PW = U*I verified)")
    print(f"  pressure convention       = {PNAME}")
    print(f"  T_sensor (humidity probe) = {TSENS}")
    print(f"  T_cell                    = {TCELL}")
    print(f"  Magnus                    = 0.61094*exp(17.625*T/(T+243.04)) kPa")
    print(f"  Faraday                   = 96485 C/mol, single cell")
    print(f"  molar volume              = 22.414 L/mol (slpm)")
    print(f"  X_O2                      = {X_O2}")
    print(f"  FLOODING  threshold       = dT_cond_out > {COND_MARGIN} C  (condensation)")
    print(f"  DRYING    threshold       = RH_ca_out  < {DRY_RH} %  (ionomer dehydration)")
    print(f"  dead channels to drop     = m_H2, T_2")


if __name__ == "__main__":
    main()
