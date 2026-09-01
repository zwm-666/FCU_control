"""Audit what the public labels actually encode, before designing any rule.

The previous attempt picked (P, T_sensor, T_cell) by maximising class
separation -- that is fitting the convention to the labels, the same statistical
crutch quantiles were. And it produced impossible values (RH_ca_out = 271%,
Tafel slope -0.68 V/dec).

Step back. Two questions decide whether a physical rule is even possible:

  Q1  Are the labels an OPERATING POINT (load setpoint) rather than a fault?
      m_Air is fixed at ~6.18 slpm and does NOT follow load (corr=+0.09), so
      lambda_air is forced to ~1/I. If each class sits at its own current, then
      "class" == "current level" and any rule, physical or not, just reads i.

  Q2  Which temperature is the cell? Resolve it by physics, not by separation:
      a humidity sensor reads RH at its own probe temperature, and the raw
      bench file names the channel T_Stack_OUTLET (renamed T_Stack_inlet in
      the processed table). Column ORDER in the raw file also suggests
      processed P_Air_inlet is really the raw P_Air_OUTLET.

Read-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv")
PUB = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets.csv")
RAW = Path(r"D:/python/pythonProject3/classified_fuel_cell_data/201703021126_RATSSingleCell.CSV")

F = 96485.0
V_STD = 22.414
X_O2 = 0.20946


def p_sat(T):
    T = np.asarray(T, float)
    return 0.61094 * np.exp(17.625 * T / (T + 243.04))


def main() -> None:
    proc = pd.read_csv(SRC)
    pub = pd.read_csv(PUB)
    SHARED = [c for c in proc.columns if c != "tsec"]
    cls = {
        "Flooding": pub.loc[pub["State_Label"] == "Flooding"].drop_duplicates(subset=SHARED),
        "Normal": pub.loc[pub["State_Label"] == "Normal"].drop_duplicates(subset=SHARED),
        "Membrane_Drying": pub.loc[pub["State_Label"] == "Membrane_Drying"].drop_duplicates(subset=SHARED),
        "Thermal_Management_Fault": pub.loc[pub["State_Label"] == "Thermal_Management_Fault"].drop_duplicates(subset=SHARED),
    }

    print("=" * 90)
    print("Q1. IS THE LABEL JUST THE OPERATING POINT?")
    print("=" * 90)
    print(f"  {'class':26s} {'n':>6s} {'i_write values (share)':<44s}")
    print("  " + "-" * 80)
    for name, sub in cls.items():
        vc = sub["i_write"].value_counts(normalize=True).head(4)
        s = "  ".join(f"{v:g}A:{100*p:.0f}%" for v, p in vc.items())
        print(f"  {name:26s} {len(sub):>6d} {s:<44s}")

    print(f"\n  {'class':26s} {'iA q25':>9s} {'iA q50':>9s} {'iA q75':>9s} "
          f"{'U q50':>8s} {'lambda q50':>11s}")
    print("  " + "-" * 80)
    for name, sub in cls.items():
        i = sub["iA"].to_numpy(float)
        ma = sub["m_Air"].to_numpy(float)
        lam = (ma / 60.0 / V_STD * X_O2) / np.where(i > 0.05, i / (4 * F), np.nan)
        print(f"  {name:26s} {np.percentile(i,25):>9.3f} {np.percentile(i,50):>9.3f} "
              f"{np.percentile(i,75):>9.3f} {np.median(sub['U_totV']):>8.4f} "
              f"{np.nanmedian(lam):>11.2f}")

    print("\n  -- can current ALONE reproduce the labels? --")
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score
    lab_all = pub["State_Label"].to_numpy()
    for feats in (["iA"], ["i_write"], ["iA", "U_totV"],
                  ["iA", "U_totV", "RH_Air", "T_Stack_inlet"]):
        X = pub[feats].to_numpy(float)
        sc = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0),
                             X, lab_all, cv=5)
        print(f"    depth-4 tree on {str(feats):48s} acc = {sc.mean():.4f}")
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        pub[["iA"]].to_numpy(float), lab_all)
    print("\n    tree on iA alone:")
    for line in export_text(tree, feature_names=["iA"]).splitlines():
        print("      " + line)

    print("\n" + "=" * 90)
    print("Q2. WHICH TEMPERATURE IS THE CELL? (resolve by physics, not separation)")
    print("=" * 90)
    print("  raw bench file column order vs processed column order:")
    try:
        rawhdr = pd.read_csv(RAW, nrows=0).columns.tolist()
        print(f"    raw      : {rawhdr[8:20]}")
        print(f"    processed: {proc.columns.tolist()[8:20]}")
        print("    -> positional map: processed P_Air_supply <- raw P_Air_inlet,")
        print("                       processed P_Air_inlet  <- raw P_Air_OUTLET,")
        print("                       processed T_Stack_inlet <- raw T_Stack_OUTLET")
    except Exception as exc:
        print(f"    raw header read failed: {exc}")

    print("\n  temperature channel behaviour (a cell temp must respond to load):")
    i = proc["iA"].to_numpy(float)
    u = proc["U_totV"].to_numpy(float)
    hdr = f"  {'channel':16s} {'q50':>8s} {'range':>16s} {'corr(i)':>9s} {'corr(U)':>9s} {'std':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in ["T_1", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet", "T_Stack_inlet", "T_Heater"]:
        a = proc[c].to_numpy(float)
        print(f"  {c:16s} {np.median(a):>8.2f} {f'[{a.min():.1f},{a.max():.1f}]':>16s} "
              f"{np.corrcoef(a,i)[0,1]:>+9.3f} {np.corrcoef(a,u)[0,1]:>+9.3f} {a.std():>8.2f}")

    print("\n  ordering check: for the humidity sensor to be physical we need")
    print("  p_H2O = RH/100 * p_sat(T_probe) <= P_total, i.e. RH<=100 at the probe.")
    for tprobe in ("T_Air_inlet", "T_Stack_inlet", "T_3"):
        T = proc[tprobe].to_numpy(float)
        ph = proc["RH_Air"].to_numpy(float) / 100.0 * p_sat(T)
        for pname, P in (("P_Air_inlet(abs)", proc["P_Air_inlet"].to_numpy(float) * 100),
                         ("P_Air_inlet+1bar", (proc["P_Air_inlet"].to_numpy(float) + 1) * 100)):
            frac = 100 * np.mean(ph < P)
            print(f"    probe={tprobe:15s} P={pname:18s} "
                  f"p_H2O<P in {frac:5.1f}% of rows   p_H2O q50={np.median(ph):6.2f} kPa")

    print("\n" + "=" * 90)
    print("Q3. WHAT DOES THE WATER BALANCE SAY WITH THE PHYSICAL CHOICE?")
    print("=" * 90)
    print("  physical choice: RH_Air probe at T_Air_inlet (humidifier, tightly")
    print("  controlled 29.5-30.9 C); cell at T_Stack_inlet (renamed stack OUTLET,")
    print("  the hottest stack-side probe, 18.8-45.5 C, responds to load).")

    def water_balance(sub, P_bar_abs):
        i_ = sub["iA"].to_numpy(float)
        ma = sub["m_Air"].to_numpy(float)
        rh = sub["RH_Air"].to_numpy(float)
        t_probe = sub["T_Air_inlet"].to_numpy(float)
        t_cell = sub["T_Stack_inlet"].to_numpy(float)
        P = np.asarray(P_bar_abs, float) * 100.0
        p_in = np.clip(rh / 100, 0, 1) * p_sat(t_probe)
        n_wet = ma / 60.0 / V_STD
        x_in = np.clip(p_in / P, 0, 0.99)
        n_dry = n_wet * (1 - x_in)
        n_h2o_in = n_wet * x_in
        il = np.clip(i_, 0, None)
        n_gen = il / (2 * F)
        n_o2 = il / (4 * F)
        n_h2o_out = n_h2o_in + n_gen
        n_dry_out = np.clip(n_dry - n_o2, 1e-12, None)
        x_out = n_h2o_out / (n_dry_out + n_h2o_out)
        p_out = x_out * P
        return {
            "lambda": (n_dry * X_O2) / np.where(il > 0.05, il / (4 * F), np.nan),
            "RH_out_at_cell": 100 * p_out / p_sat(t_cell),
            "RH_in_at_cell": 100 * p_in / p_sat(t_cell),
            "water_gen_frac": n_gen / np.clip(n_h2o_out, 1e-15, None),
            "p_out": p_out,
        }

    for pconv, pname in ((1.0, "P_Air_inlet as bar(abs) ~1.08 bar"),
                         (2.0, "P_Air_inlet+1 as bar(abs) ~2.08 bar")):
        print(f"\n  --- {pname} ---")
        print(f"  {'class':26s} {'RH_in@cell':>11s} {'RH_out@cell':>12s} "
              f"{'lambda':>9s} {'gen frac':>9s}")
        print("  " + "-" * 74)
        for name, sub in cls.items():
            P = sub["P_Air_inlet"].to_numpy(float) + (1.0 if pconv == 2.0 else 0.0)
            wb = water_balance(sub, P)
            print(f"  {name:26s} {np.nanmedian(wb['RH_in_at_cell']):>11.2f} "
                  f"{np.nanmedian(wb['RH_out_at_cell']):>12.2f} "
                  f"{np.nanmedian(wb['lambda']):>9.2f} "
                  f"{np.nanmedian(wb['water_gen_frac']):>9.3f}")

    print("\n" + "=" * 90)
    print("Q4. POLARIZATION: fit only on genuine load rows, check Tafel sanity")
    print("=" * 90)
    from scipy.optimize import curve_fit

    def model(i_, E0, b, R):
        return E0 - b * np.log10(np.clip(i_, 0.05, None)) - R * i_

    for tag, mask in (("all i>1A", i > 1.0),
                      ("i>1A & RH_Air in [60,90]",
                       (i > 1.0) & (proc["RH_Air"].to_numpy(float) > 60)
                       & (proc["RH_Air"].to_numpy(float) < 90))):
        m = mask & np.isfinite(u)
        if m.sum() < 50:
            print(f"  {tag}: too few rows")
            continue
        try:
            popt, _ = curve_fit(model, i[m], u[m], p0=[0.95, 0.06, 0.005], maxfev=40000)
            E0, b, R = popt
            resid = u[m] - model(i[m], *popt)
            ok = (0.85 < E0 < 1.10) and (0.03 < b < 0.15) and (R > 0)
            print(f"  {tag:28s} n={int(m.sum()):6d}  E0={E0:7.4f}  b={b:+7.4f}  "
                  f"R={R*1000:8.2f} mOhm  rms={resid.std():.4f}  "
                  f"{'PHYSICAL' if ok else 'NON-PHYSICAL'}")
        except Exception as exc:
            print(f"  {tag}: fit failed {exc}")

    print("\n  U vs i scatter by current level (is there a polarization curve at all?):")
    for lo, hi in [(0.05, 1), (1, 6), (6, 12), (12, 16), (16, 22), (22, 29)]:
        m = (i > lo) & (i <= hi)
        if m.sum() > 5:
            print(f"    i in ({lo:5.2f},{hi:5.2f}] n={int(m.sum()):6d}  "
                  f"U q25={np.percentile(u[m],25):.4f} q50={np.percentile(u[m],50):.4f} "
                  f"q75={np.percentile(u[m],75):.4f}  spread={u[m].max()-u[m].min():.4f}")


if __name__ == "__main__":
    main()
