"""Build a 3-class PEMFC dataset labeled by first-principles thermodynamics.

ROUTE 1: the published State_Label column is DISCARDED. Labels are recomputed
from a closed cathode water balance. Every row is real; nothing is synthesised
and nothing is duplicated.

WHY THE PUBLISHED LABELS ARE DISCARDED
  Rigorous per-row water balance on the 1,346 unique rows labeled "Flooding"
  gives water activity a_w = 0.624 (RH at cell outlet 79.3%), i.e. 100% of them
  are SUBSATURATED -- physically dry, not flooded. Rows labeled "Normal" are the
  driest of all (a_w = 0.508). Rows labeled "Membrane_Drying" sit at iA = 0.25 A
  and eta = +0.49 V, an open-circuit condition. The only genuinely condensing
  class (a_w = 1.381, RH_out = 172%) is the one labeled
  "Thermal_Management_Fault". A depth-4 tree on (iA, U_totV) alone recovers
  89.5% of the four published labels, so they largely encode the OPERATING
  POINT rather than an independent fault state.

RESOLVED COLUMN SEMANTICS (positional map against the raw bench header)
  raw       : P_Air_inlet  P_H2_inlet  P_Air_outlet  P_H2_outlet ... T_Stack_outlet
  processed : P_Air_supply P_H2_supply P_Air_inlet   P_H2_inlet  ... T_Stack_inlet
  => processed "P_Air_inlet"   is the cathode OUTLET pressure
     processed "T_Stack_inlet" is the stack  OUTLET temperature
  Consequence: cathode pressure drop is P_Air_inlet - P_Air_supply, positive on
  100% of rows. The un-remapped difference was negative on 100% of rows.

THE CRITERION -- purely row-wise, no global statistic, no quantile
  Single cell (PW = U*I verified, implied N_cell = 1.0000).
      p_sat(T)   = 0.61094*exp(17.625*T/(T+243.04))            [kPa, Magnus]
      p_H2O,in   = RH_Air/100 * p_sat(T_Air_inlet)
      n_wet,in   = m_Air/(60*22.414)                           [mol/s, slpm]
      n_dry,in   = n_wet,in * (1 - p_H2O,in/P_cathode)
      n_H2O,gen  = I/(2F)      n_O2,used = I/(4F)              [Faraday]
      n_H2O,out  = n_H2O,in + n_H2O,gen
      n_dry,out  = n_dry,in - n_O2,used
      p_H2O,out  = n_H2O,out/(n_dry,out + n_H2O,out) * P_cathode
      a_w        = 0.5*(p_H2O,in + p_H2O,out)/p_sat(T_cell)

      FLOODING         a_w >= 1.00   vapour saturated -> liquid water forms
      MEMBRANE_DRYING  a_w <= 0.70   ionomer water content below ~9 H2O/SO3-
      NORMAL           0.70 < a_w < 1.00

  Both thresholds are absolute physical constants (thermodynamic saturation;
  Nafion sorption-isotherm conductivity knee). Neither is tuned on labels or
  derived from this dataset's distribution, so the criterion transfers to
  another bench and can be evaluated online row by row.

LOAD CONTROL (design B, the default)
  m_Air is fixed at ~6.18 slpm and does not follow load (corr(m_Air,i)=+0.09),
  so lambda_air = 360.8/I >= 12.31 always: this bench CANNOT flood by
  stoichiometric starvation (that needs I > 180 A vs a 28.8 A maximum).
  Restricting to the i in [12,16] A plateau holds current constant, so the
  three classes differ only in water/thermal state and the label cannot be a
  proxy for the operating point. Design A keeps every row for comparison.

Diagnostics behind these facts: scripts/probe_physics_units.py,
audit_label_semantics.py, design_physics_final.py, design_physics_settle.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

SOURCE_PATH = Path(
    r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv"
)
OUT_DIR = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件")

# ---- physical constants (none fitted) ----
FARADAY = 96485.0            # C/mol
V_MOLAR_SLPM = 22.414        # L/mol at 0 C 1 atm, slpm convention
X_O2 = 0.20946               # dry-air oxygen mole fraction
MAGNUS_A, MAGNUS_B, MAGNUS_C = 0.61094, 17.625, 243.04   # kPa, Alduchov-Eskridge

A_W_FLOOD = 1.00             # vapour saturation -> liquid water
A_W_DRY = 0.70               # ionomer dehydration onset

# ---- resolved channel semantics ----
COL_RH_IN = "RH_Air"
COL_T_PROBE = "T_Air_inlet"      # humidifier outlet, where RH_Air is measured
COL_T_CELL = "T_Stack_inlet"     # = raw T_Stack_outlet
COL_P_CATH = "P_Air_inlet"       # = raw P_Air_outlet, bar absolute
COL_P_CATH_IN = "P_Air_supply"   # = raw P_Air_inlet
COL_M_AIR = "m_Air"

# ---- polarization baseline, fitted on all genuine load rows (i > 1 A, n=30530)
# Reported only; NOT used to assign labels, so labels stay row-wise pure.
POLARIZATION = {"E0": 0.856459, "tafel_b": 0.068525, "r_ohm": 0.011064,
                "rms_V": 0.0467, "n_fit_rows": 30530}

FEATURE_COLUMNS = [
    "tsec", "U_totV", "iA", "PW", "m_Air", "m_H2", "RH_Air", "RH_H2",
    "P_Air_supply", "P_H2_supply", "P_Air_inlet", "P_H2_inlet",
    "T_1", "T_2", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet",
    "T_Stack_inlet", "T_Heater", "m_Air_write", "m_H2_write",
    "Heater_power", "i_write",
]
MODEL_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c != "tsec"]
OUTPUT_COLUMNS = FEATURE_COLUMNS + ["State", "State_Label"]

CLASS_ORDER = ["Normal", "Flooding", "Membrane_Drying"]
STATE_MAPPING = {"Normal": 0, "Flooding": 1, "Membrane_Drying": 2}

PLATEAU_LO, PLATEAU_HI = 12.0, 16.0     # A, the load-controlled window


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p_sat_kpa(t_celsius: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water [kPa]."""
    t = np.asarray(t_celsius, dtype=float)
    return MAGNUS_A * np.exp(MAGNUS_B * t / (t + MAGNUS_C))


def dew_point_c(p_kpa: np.ndarray) -> np.ndarray:
    """Inverse Magnus: dew point [C] from vapour pressure [kPa]."""
    p = np.clip(np.asarray(p_kpa, dtype=float), 1e-9, None)
    ln = np.log(p / MAGNUS_A)
    return MAGNUS_C * ln / (MAGNUS_B - ln)


def cathode_water_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Closed per-row cathode water balance. No global statistics are used."""
    i = df["iA"].to_numpy(float)
    m_air = df[COL_M_AIR].to_numpy(float)
    rh_in = df[COL_RH_IN].to_numpy(float)
    t_probe = df[COL_T_PROBE].to_numpy(float)
    t_cell = df[COL_T_CELL].to_numpy(float)
    p_cath_kpa = df[COL_P_CATH].to_numpy(float) * 100.0

    i_load = np.clip(i, 0.0, None)

    # inlet humidification, evaluated at the probe temperature
    p_h2o_in = np.clip(rh_in / 100.0, 0.0, 1.0) * p_sat_kpa(t_probe)
    x_h2o_in = np.clip(p_h2o_in / p_cath_kpa, 0.0, 0.99)

    # molar flows, per cell
    n_wet_in = m_air / 60.0 / V_MOLAR_SLPM
    n_dry_in = n_wet_in * (1.0 - x_h2o_in)
    n_h2o_in = n_wet_in * x_h2o_in
    n_o2_used = i_load / (4.0 * FARADAY)
    n_h2o_gen = i_load / (2.0 * FARADAY)

    # cathode outlet state
    n_h2o_out = n_h2o_in + n_h2o_gen
    n_dry_out = np.clip(n_dry_in - n_o2_used, 1e-12, None)
    p_h2o_out = n_h2o_out / (n_dry_out + n_h2o_out) * p_cath_kpa

    p_sat_cell = p_sat_kpa(t_cell)
    rh_in_at_cell = 100.0 * p_h2o_in / p_sat_cell
    rh_out_at_cell = 100.0 * p_h2o_out / p_sat_cell
    a_w = 0.5 * (rh_in_at_cell + rh_out_at_cell) / 100.0

    # condensed liquid water rate when the outlet is supersaturated [mol/s]
    x_sat_cell = np.clip(p_sat_cell / p_cath_kpa, 0.0, 0.99)
    n_h2o_capacity = x_sat_cell / (1.0 - x_sat_cell) * n_dry_out
    n_liquid = np.clip(n_h2o_out - n_h2o_capacity, 0.0, None)

    # cathode stoichiometry and the polarization residual (diagnostics only)
    lam = np.where(i_load > 0.05,
                   (n_dry_in * X_O2) / np.where(i_load > 0.05, n_o2_used, np.nan),
                   np.inf)
    u_fit = (POLARIZATION["E0"]
             - POLARIZATION["tafel_b"] * np.log10(np.clip(i_load, 0.05, None))
             - POLARIZATION["r_ohm"] * i_load)

    return pd.DataFrame({
        "a_w": a_w,
        "RH_in_at_cell": rh_in_at_cell,
        "RH_out_at_cell": rh_out_at_cell,
        "T_dew_out": dew_point_c(p_h2o_out),
        "dT_cond": dew_point_c(p_h2o_out) - t_cell,
        "n_liquid_mol_s": n_liquid,
        "lambda_air": lam,
        "eta_V": u_fit - df["U_totV"].to_numpy(float),
        "dP_cathode_bar": (df[COL_P_CATH].to_numpy(float)
                           - df[COL_P_CATH_IN].to_numpy(float)),
    }, index=df.index)


def classify_by_physics(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Assign Normal/Flooding/Membrane_Drying from water activity alone."""
    phys = cathode_water_balance(df)
    a_w = phys["a_w"].to_numpy(float)

    flooding = a_w >= A_W_FLOOD
    drying = a_w <= A_W_DRY
    normal = (~flooding) & (~drying)

    if int((flooding & drying).sum()):
        raise AssertionError("Flooding and Membrane_Drying overlap")
    if int((flooding & normal).sum()) or int((drying & normal).sum()):
        raise AssertionError("Normal overlaps a fault class")
    if not (flooding | drying | normal).all():
        raise AssertionError("Some rows received no label")

    labels = np.empty(len(df), dtype=object)
    labels[normal] = "Normal"
    labels[flooding] = "Flooding"
    labels[drying] = "Membrane_Drying"
    return labels, phys


def _validate_source(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for c in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"Source column {c!r} is not numeric")


def build_dataset(
    design: str = "B",
    output_path: str | Path | None = None,
    seed: int = 42,
    source_path: str | Path = SOURCE_PATH,
    per_class: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the physics-labeled balanced dataset. design 'A' or 'B'."""
    design = design.upper()
    if design not in {"A", "B"}:
        raise ValueError("design must be 'A' or 'B'")
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if output_path is None:
        output_path = OUT_DIR / f"Public datasets_physics3_design{design}.csv"
    output_path = Path(output_path)
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Refusing to overwrite the source file")

    source = pd.read_csv(source_path)
    _validate_source(source, source_path)

    labels_all, phys_all = classify_by_physics(source)
    raw_counts_all = {c: int((labels_all == c).sum()) for c in CLASS_ORDER}

    # ---- load control ----
    i = source["iA"].to_numpy(float)
    if design == "B":
        window = (i >= PLATEAU_LO) & (i <= PLATEAU_HI)
    else:
        window = np.ones(len(source), dtype=bool)
    pool = source.loc[window].reset_index(drop=True)
    labels = labels_all[window]
    phys = phys_all.loc[window].reset_index(drop=True)

    # ---- deduplicate on model features, then balance ----
    dup = pool[MODEL_FEATURE_COLUMNS].round(9).duplicated().to_numpy()
    unique_counts = {c: int(((labels == c) & ~dup).sum()) for c in CLASS_ORDER}
    capacity = min(unique_counts.values())
    n_per_class = capacity if per_class is None else int(per_class)
    if n_per_class > capacity:
        raise ValueError(
            f"per_class={n_per_class} exceeds all-real capacity {capacity}; "
            "this builder never synthesises or duplicates rows"
        )

    rng = np.random.default_rng(int(seed))
    pieces = []
    for name in CLASS_ORDER:
        idx = np.flatnonzero((labels == name) & ~dup)
        chosen = rng.permutation(idx)[:n_per_class]
        part = pool.loc[chosen, FEATURE_COLUMNS].copy()
        part["State"] = STATE_MAPPING[name]
        part["State_Label"] = name
        pieces.append(part[OUTPUT_COLUMNS])

    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8", float_format="%.17g")
    reloaded = pd.read_csv(output_path)

    # ---- verification on the serialized file ----
    class_counts = {c: int((reloaded["State_Label"] == c).sum()) for c in CLASS_ORDER}
    if len(reloaded) != 3 * n_per_class:
        raise AssertionError(f"row count {len(reloaded)} != {3 * n_per_class}")
    if class_counts != {c: n_per_class for c in CLASS_ORDER}:
        raise AssertionError(f"unbalanced classes: {class_counts}")
    dup_out = int(reloaded[MODEL_FEATURE_COLUMNS].round(9).duplicated().sum())
    if dup_out:
        raise AssertionError(f"serialized output has {dup_out} duplicate feature rows")

    # PW carries the bench's own measurement/rounding error against U*I. Do not
    # "repair" it -- that would fabricate values. Require only that the export
    # is no worse than the source it was drawn from.
    power_err = float(np.abs(reloaded["PW"]
                             - reloaded["U_totV"] * reloaded["iA"]).max())
    source_power_err = float(np.abs(source["PW"]
                                   - source["U_totV"] * source["iA"]).max())
    if power_err > source_power_err + 1e-9:
        raise AssertionError(
            f"power identity error {power_err} exceeds the source's own "
            f"{source_power_err}; the export altered measured values"
        )

    # relabel the serialized rows from scratch and require an exact match
    recheck, phys_out = classify_by_physics(reloaded)
    if not (recheck == reloaded["State_Label"].to_numpy()).all():
        raise AssertionError("physics relabel of the serialized file disagrees")

    dead = [c for c in MODEL_FEATURE_COLUMNS if reloaded[c].nunique() == 1]

    def class_stat(col: str) -> Dict[str, float]:
        return {c: round(float(np.nanmedian(
            phys_out.loc[reloaded["State_Label"] == c, col].to_numpy(float))), 6)
            for c in CLASS_ORDER}

    episodes = {}
    for c in CLASS_ORDER:
        ts = np.sort(reloaded.loc[reloaded["State_Label"] == c, "tsec"].to_numpy(float))
        episodes[c] = {
            "tsec_min": round(float(ts.min()), 3),
            "tsec_max": round(float(ts.max()), 3),
            "episodes_gap_gt_5s": int((np.diff(ts) > 5.0).sum()) + 1,
        }

    i_out = reloaded["iA"].to_numpy(float)
    manifest: Dict[str, Any] = {
        "dataset_name": f"PEMFC 3-class, physics-labeled, design {design}",
        "route": "published State_Label discarded; labels recomputed from a "
                 "closed cathode water balance",
        "design": design,
        "design_meaning": ("B: load-controlled, only the i in "
                           f"[{PLATEAU_LO},{PLATEAU_HI}] A plateau, so classes "
                           "cannot proxy the operating point"
                           if design == "B" else
                           "A: free operating point, all source rows"),
        "source_path": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "source_rows": int(len(source)),
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "seed": int(seed),
        "total_rows": int(len(reloaded)),
        "class_counts": class_counts,
        "class_order": CLASS_ORDER,
        "state_mapping": STATE_MAPPING,

        "synthetic_counts": {c: 0 for c in CLASS_ORDER},
        "duplicated_row_counts": {c: 0 for c in CLASS_ORDER},
        "all_rows_real": True,
        "duplicate_feature_rows": dup_out,
        "power_identity_max_abs_error": power_err,
        "source_power_identity_max_abs_error": source_power_err,
        "power_identity_note": "PW is a measured bench channel; its residual "
                              "against U*I is inherited from the source and is "
                              "not corrected, so no value is fabricated",

        "criterion": {
            "kind": "first-principles cathode water balance; row-wise; "
                    "no global statistic and no quantile",
            "state_variable": "a_w = 0.5*(RH_in@cell + RH_out@cell)/100",
            "flooding_threshold": f"a_w >= {A_W_FLOOD}",
            "drying_threshold": f"a_w <= {A_W_DRY}",
            "normal_band": f"{A_W_DRY} < a_w < {A_W_FLOOD}",
            "threshold_provenance": "thermodynamic saturation (1.00) and the "
                                    "Nafion sorption-isotherm conductivity "
                                    "knee (0.70); neither tuned on this data",
            "constants": {
                "faraday_C_per_mol": FARADAY,
                "molar_volume_L_per_mol_slpm": V_MOLAR_SLPM,
                "x_O2_dry_air": X_O2,
                "magnus_kPa": "0.61094*exp(17.625*T/(T+243.04))",
                "n_cell": 1,
            },
            "channel_semantics": {
                "RH_probe_temperature": COL_T_PROBE,
                "cell_temperature": COL_T_CELL,
                "cathode_pressure": f"{COL_P_CATH} (= raw P_Air_outlet), bar abs",
                "cathode_inlet_pressure": f"{COL_P_CATH_IN} (= raw P_Air_inlet)",
                "pressure_drop": f"{COL_P_CATH} - {COL_P_CATH_IN}, positive",
                "note": "the processed table renamed the raw columns by one "
                        "position; corrected here",
            },
            "polarization_baseline_reported_only": POLARIZATION,
        },

        "discarded_published_labels": {
            "reason": "water balance contradicts them",
            "labeled_Flooding_a_w_median": 0.624,
            "labeled_Flooding_subsaturated_share": 1.0,
            "labeled_Normal_a_w_median": 0.508,
            "labeled_Membrane_Drying_median_current_A": 0.255,
            "labeled_Thermal_Management_Fault_a_w_median": 1.381,
            "tree_depth4_on_current_and_voltage_recovers": 0.8953,
        },

        "stoichiometry_wall": {
            "m_Air_slpm_median": round(float(np.median(source[COL_M_AIR])), 4),
            "corr_m_Air_current": 0.0914,
            "lambda_air_min_over_source": 12.31,
            "current_needed_for_lambda_2_A": 180.4,
            "bench_max_current_A": round(float(source["iA"].max()), 3),
            "conclusion": "cathode air flow is fixed, so lambda_air = 360.8/I "
                          ">= 12.31; stoichiometric-starvation flooding is "
                          "impossible on this bench",
        },

        "load_control": {
            "window_A": [PLATEAU_LO, PLATEAU_HI] if design == "B" else None,
            "current_q05": round(float(np.percentile(i_out, 5)), 4),
            "current_q50": round(float(np.percentile(i_out, 50)), 4),
            "current_q95": round(float(np.percentile(i_out, 95)), 4),
            "current_per_class_q50": {
                c: round(float(np.percentile(
                    i_out[reloaded["State_Label"] == c], 50)), 4)
                for c in CLASS_ORDER},
        },

        "raw_physics_label_counts_full_source": raw_counts_all,
        "unique_real_pool_after_load_control": unique_counts,
        "all_real_balanced_capacity_per_class": int(capacity),
        "selected_per_class": int(n_per_class),

        "class_physics_medians": {
            "a_w": class_stat("a_w"),
            "RH_out_at_cell_pct": class_stat("RH_out_at_cell"),
            "RH_in_at_cell_pct": class_stat("RH_in_at_cell"),
            "lambda_air": class_stat("lambda_air"),
            "eta_V": class_stat("eta_V"),
            "n_liquid_mol_s": class_stat("n_liquid_mol_s"),
            "dP_cathode_bar": class_stat("dP_cathode_bar"),
            "T_cell_C": {c: round(float(np.median(
                reloaded.loc[reloaded["State_Label"] == c, COL_T_CELL])), 4)
                for c in CLASS_ORDER},
        },

        "temporal_structure": episodes,
        "feature_columns": FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "dead_channels_constant": dead,
        "metadata_columns": ["tsec"],
        "preprocessing_note": "the training pipeline drops State and tsec and "
                              "uses State_Label (last column) as the target; "
                              "power columns are dropped by policy",
        "verification": {
            "relabel_of_serialized_file_matches": True,
            "duplicate_feature_rows": dup_out,
            "power_identity_max_abs_error": power_err,
            "class_overlap": 0,
        },
    }

    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return reloaded, manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default="B", choices=("A", "B", "both"))
    parser.add_argument("--source", default=str(SOURCE_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=None)
    args = parser.parse_args()

    designs = ["A", "B"] if args.design == "both" else [args.design]
    for design in designs:
        out = args.output if (args.output and args.design != "both") else None
        frame, manifest = build_dataset(
            design=design, output_path=out, seed=args.seed,
            source_path=args.source, per_class=args.per_class,
        )
        print(json.dumps({
            "design": design,
            "output": manifest["output_path"],
            "rows": int(len(frame)),
            "class_counts": manifest["class_counts"],
            "synthetic_counts": manifest["synthetic_counts"],
            "duplicated_row_counts": manifest["duplicated_row_counts"],
            "all_real_capacity_per_class": manifest["all_real_balanced_capacity_per_class"],
            "current_per_class_q50": manifest["load_control"]["current_per_class_q50"],
            "a_w_medians": manifest["class_physics_medians"]["a_w"],
            "n_liquid_mol_s": manifest["class_physics_medians"]["n_liquid_mol_s"],
            "dead_channels": manifest["dead_channels_constant"],
            "duplicate_feature_rows": manifest["duplicate_feature_rows"],
            "power_identity_max_abs_error": manifest["power_identity_max_abs_error"],
            "output_sha256": manifest["output_sha256"],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
