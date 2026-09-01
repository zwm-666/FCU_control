"""Probe channel units so real physics (not quantiles) can be computed.

Everything downstream depends on knowing what m_Air / m_H2 / pressures
actually are. This resolves them by Faraday consistency: for a single cell,
    H2 consumption  = I/(2F)  mol/s
    O2 consumption  = I/(4F)  mol/s
    air requirement = I/(4F*0.21) mol/s
A physically sane bench runs stoichiometry lambda in ~1.2-3 (sometimes higher
on small single cells used for thermal stability). Whichever unit assumption
puts lambda in a sane band is the right one.

Read-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv")
PUB = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets.csv")

F = 96485.0          # C/mol
R = 8.314            # J/(mol K)
V_MOLAR_STD = 22.414 # L/mol at 0 C, 1 atm  (slpm convention)
V_MOLAR_NORM = 24.055  # L/mol at 25 C, 1 atm (nlpm convention)


def stats(tag, a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"  {tag:34s} (empty)")
        return
    print(f"  {tag:34s} min={a.min():10.4f} q25={np.percentile(a,25):10.4f} "
          f"q50={np.percentile(a,50):10.4f} q75={np.percentile(a,75):10.4f} "
          f"max={a.max():10.4f}")


def main() -> None:
    df = pd.read_csv(SRC)
    pub = pd.read_csv(PUB)
    print(f"processed {df.shape}   public {pub.shape}")

    print("\n" + "=" * 78)
    print("1. RAW CHANNEL RANGES (what are we actually holding?)")
    print("=" * 78)
    for c in ["U_totV", "iA", "PW", "m_Air", "m_H2", "RH_Air", "RH_H2",
              "P_Air_supply", "P_H2_supply", "P_Air_inlet", "P_H2_inlet",
              "T_1", "T_2", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet",
              "T_Stack_inlet", "T_Heater", "m_Air_write", "m_H2_write",
              "Heater_power", "i_write"]:
        stats(c, df[c])

    print("\n-- distinct-value counts (which channels are setpoints vs measured?) --")
    for c in df.columns:
        nu = df[c].nunique()
        kind = "CONSTANT" if nu == 1 else ("setpoint-like" if nu <= 12 else "measured")
        print(f"  {c:16s} distinct={nu:6d}  {kind}")

    print("\n" + "=" * 78)
    print("2. IS PW = U*I ? (confirms U is per-cell or stack, and cell count)")
    print("=" * 78)
    u, i, p = (df[c].to_numpy(float) for c in ("U_totV", "iA", "PW"))
    for ncell in (1, 2, 3, 5, 10):
        pred = u * i * ncell
        err = np.abs(pred - p)
        print(f"  N_cell={ncell:3d}: max|U*I*N - PW| = {err.max():.6e}  "
              f"mean={err.mean():.6e}")
    # implied cell count
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = p / (u * i)
    implied = implied[np.isfinite(implied)]
    print(f"  implied N_cell = PW/(U*I): q05={np.percentile(implied,5):.4f} "
          f"q50={np.percentile(implied,50):.4f} q95={np.percentile(implied,95):.4f}")

    print("\n" + "=" * 78)
    print("3. FARADAY CONSISTENCY -> resolve m_Air / m_H2 units")
    print("=" * 78)
    i_pos = np.where(i > 0.5, i, np.nan)          # only meaningful under load
    mAir, mH2 = df["m_Air"].to_numpy(float), df["m_H2"].to_numpy(float)

    # required molar flows for a single cell
    n_h2_req = i_pos / (2 * F)                     # mol/s
    n_air_req = i_pos / (4 * F * 0.21)             # mol/s

    print(f"  under load (i>0.5A): n={np.isfinite(i_pos).sum()}, "
          f"i q50={np.nanpercentile(i_pos,50):.3f} A, max={np.nanmax(i_pos):.3f} A")
    print(f"  required H2  @ i_max: {np.nanmax(n_h2_req)*60*V_MOLAR_STD*1000:.2f} sccm "
          f"= {np.nanmax(n_h2_req)*60*V_MOLAR_STD:.4f} slpm")
    print(f"  required Air @ i_max: {np.nanmax(n_air_req)*60*V_MOLAR_STD:.4f} slpm")

    print("\n  -- lambda under each unit hypothesis --")
    hyps = {
        "slpm (22.414 L/mol)":   lambda m: m / 60.0 / V_MOLAR_STD,
        "nlpm (24.055 L/mol)":   lambda m: m / 60.0 / V_MOLAR_NORM,
        "sccm (=1e-3 slpm)":     lambda m: m / 1000.0 / 60.0 / V_MOLAR_STD,
        "g/min H2 (2.016 g/mol)": None,   # handled separately
    }
    for name, conv in hyps.items():
        if conv is None:
            continue
        lam_air = conv(mAir) / n_air_req
        lam_h2 = conv(mH2) / n_h2_req
        print(f"  {name:26s} lam_Air q50={np.nanpercentile(lam_air,50):9.3f} "
              f"lam_H2 q50={np.nanpercentile(lam_h2,50):9.3f}")
    # mass-flow hypotheses
    lam_h2_mass = (mH2 / 2.016 / 60.0) / n_h2_req
    lam_air_mass = (mAir / 28.96 / 60.0) / n_air_req
    print(f"  {'g/min (mass flow)':26s} lam_Air q50={np.nanpercentile(lam_air_mass,50):9.3f} "
          f"lam_H2 q50={np.nanpercentile(lam_h2_mass,50):9.3f}")

    print("\n  -- m_Air vs current: is air flow actively controlled with load? --")
    ok = np.isfinite(i_pos)
    print(f"  corr(m_Air, iA) under load = {np.corrcoef(mAir[ok], i_pos[ok])[0,1]:+.4f}")
    print(f"  corr(m_H2 , iA) under load = {np.corrcoef(mH2[ok], i_pos[ok])[0,1]:+.4f}")
    for lo, hi in [(0.5, 5), (5, 12), (12, 16), (16, 22), (22, 29)]:
        m = (i > lo) & (i <= hi)
        if m.sum() > 10:
            print(f"    i in ({lo:4.1f},{hi:4.1f}] n={int(m.sum()):6d}  "
                  f"m_Air q50={np.percentile(mAir[m],50):8.4f}  "
                  f"m_H2 q50={np.percentile(mH2[m],50):8.4f}")

    print("\n" + "=" * 78)
    print("4. PRESSURES: which pair gives a usable pressure DROP?")
    print("=" * 78)
    for a, b in [("P_Air_supply", "P_Air_inlet"), ("P_H2_supply", "P_H2_inlet")]:
        d = df[a].to_numpy(float) - df[b].to_numpy(float)
        stats(f"{a} - {b}", d)
        print(f"    corr(drop, iA)={np.corrcoef(d, i)[0,1]:+.4f}  "
              f"corr(drop, m_Air)={np.corrcoef(d, mAir)[0,1]:+.4f}  "
              f"frac<0={100*(d<0).mean():.1f}%")
    print("  (a flooding channel should show pressure drop RISING as liquid blocks it)")

    print("\n" + "=" * 78)
    print("5. TEMPERATURES: which channel is the electrochemically relevant one?")
    print("=" * 78)
    for c in ["T_1", "T_2", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet",
              "T_Stack_inlet", "T_Heater"]:
        a = df[c].to_numpy(float)
        print(f"  {c:16s} q50={np.percentile(a,50):8.3f} "
              f"corr(U)={np.corrcoef(a,u)[0,1]:+.4f} corr(i)={np.corrcoef(a,i)[0,1]:+.4f} "
              f"distinct={df[c].nunique()}")

    print("\n" + "=" * 78)
    print("6. SATURATION / DEW POINT FEASIBILITY (Magnus)")
    print("=" * 78)
    def p_sat_kpa(T):
        return 0.61094 * np.exp(17.625 * T / (T + 243.04))  # kPa

    for tc in ["T_Air_inlet", "T_Stack_inlet", "T_3"]:
        T = df[tc].to_numpy(float)
        ps = p_sat_kpa(T)
        stats(f"p_sat({tc}) [kPa]", ps)
    print()
    stats("P_Air_inlet raw", df["P_Air_inlet"])
    print("  -> if P_Air_inlet ~1.0 it is likely bar(g) or bar(a); p_sat at 45C = "
          f"{p_sat_kpa(45.0):.3f} kPa = {p_sat_kpa(45.0)/100:.5f} bar")

    print("\n" + "=" * 78)
    print("7. LABELED FLOODING vs the physics candidates (public 1346 unique rows)")
    print("=" * 78)
    SHARED = [c for c in df.columns if c != "tsec"]
    pf = pub.loc[pub["State_Label"] == "Flooding"].drop_duplicates(subset=SHARED)
    pn = pub.loc[pub["State_Label"] == "Normal"].drop_duplicates(subset=SHARED)
    pd_ = pub.loc[pub["State_Label"] == "Membrane_Drying"].drop_duplicates(subset=SHARED)
    for tag, sub in (("Flooding", pf), ("Normal", pn), ("Drying", pd_)):
        ii = sub["iA"].to_numpy(float)
        ma = sub["m_Air"].to_numpy(float)
        n_air_r = np.where(ii > 0.5, ii, np.nan) / (4 * F * 0.21)
        lam = (ma / 60.0 / V_MOLAR_STD) / n_air_r
        print(f"  {tag:10s} n={len(sub):5d}  lam_Air q25={np.nanpercentile(lam,25):8.3f} "
              f"q50={np.nanpercentile(lam,50):8.3f} q75={np.nanpercentile(lam,75):8.3f}")


if __name__ == "__main__":
    main()
