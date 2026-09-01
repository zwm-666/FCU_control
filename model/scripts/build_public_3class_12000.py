"""Build a three-class, 12,000-row public PEMFC dataset from processed data.

The source is the processed measurement table itself::

    D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv

Labels are generated with the same global-quartile cascade used by the
``数据集2`` scripts (Normal/Flooding/Membrane_Drying/Thermal management).
Thermal-management rows are excluded. The source contains 1,805 real Flooding
rows, so a balanced 4,000 rows per requested class requires an explicit
Flooding-only supplement of 2,195 synthetic rows. Normal and Membrane_Drying
are selected as real rows without replacement.

The output is a derived/augmented dataset, not an untouched benchmark. Source
and output hashes, the exact thresholds and class counts are written to the
sidecar manifest. Existing source files are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SOURCE_PATH = Path(
    r"D:/python/pythonProject3/classified_fuel_cell_data/processed_fuel_cell_data.csv"
)
DEFAULT_OUTPUT = Path(
    r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets_3class_12000.csv"
)

# The processed source has no label columns; these are the 23 physical channels
# plus tsec. tsec is retained in the exported file as metadata and excluded
# from the feature-space neighbour search used for synthetic rows.
FEATURE_COLUMNS = [
    "tsec", "U_totV", "iA", "PW", "m_Air", "m_H2", "RH_Air", "RH_H2",
    "P_Air_supply", "P_H2_supply", "P_Air_inlet", "P_H2_inlet",
    "T_1", "T_2", "T_3", "T_4", "T_Air_inlet", "T_H2_inlet",
    "T_Stack_inlet", "T_Heater", "m_Air_write", "m_H2_write",
    "Heater_power", "i_write",
]
MODEL_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column != "tsec"]
OUTPUT_COLUMNS = FEATURE_COLUMNS + ["State", "State_Label"]

STATE_MAPPING = {"Normal": 0, "Flooding": 1, "Membrane_Drying": 2}
CLASS_ORDER = ["Normal", "Flooding", "Membrane_Drying"]
TARGET_PER_CLASS = 4000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_key(values: Iterable[float]) -> Tuple[float, ...]:
    return tuple(np.asarray(values, dtype=np.float64).round(12).tolist())


def _validate_source(df: pd.DataFrame, path: Path) -> None:
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required source columns: {missing}")
    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Source column {column!r} is not numeric in {path}")


def classify_states(df: pd.DataFrame) -> tuple[np.ndarray, Dict[str, float]]:
    """Apply the dataset2 global-quartile cascade to processed measurements.

    The precedence and fallback are intentionally copied from
    ``build_balanced_dataset_smote.py``: thermal first, then flooding, drying,
    normal, and finally voltage/temperature fallback for unclassified rows.
    """
    v = df["U_totV"].to_numpy(dtype=float)
    i = df["iA"].to_numpy(dtype=float)
    p = df["PW"].to_numpy(dtype=float)
    ra = df["RH_Air"].to_numpy(dtype=float)
    rh = df["RH_H2"].to_numpy(dtype=float)
    t = df["T_Stack_inlet"].to_numpy(dtype=float)

    vq25, vq50, vq75 = np.percentile(v, [25, 50, 75])
    aq25, aq50, aq75 = np.percentile(ra, [25, 50, 75])
    hq25, hq50, hq75 = np.percentile(rh, [25, 50, 75])
    tq25, tq50, tq75 = np.percentile(t, [25, 50, 75])
    iq50, pq50 = np.percentile(i, 50), np.percentile(p, 50)

    states = np.zeros(len(df), dtype=np.int64)
    thermal = (
        ((t > tq75 + (tq75 - tq50)) | (t < tq25 - (tq50 - tq25)))
        & (v < vq50)
        & (p < pq50)
    )
    flooding = (
        (ra > aq75)
        & (rh > hq75)
        & (v < vq50)
        & (i > iq50)
        & (~thermal)
    )
    drying = (
        (ra < aq50)
        & (rh < hq50)
        & (t >= tq50)
        & (t <= tq75 + (tq75 - tq50))
        & (v >= vq25)
        & (v <= vq75)
        & (~thermal)
        & (~flooding)
    )
    normal = (
        (v > vq50)
        & (ra >= aq25)
        & (ra <= aq75)
        & (rh >= hq25)
        & (rh <= hq75)
        & (t >= tq25)
        & (t <= tq75)
        & (~thermal)
        & (~flooding)
        & (~drying)
    )

    states[thermal] = 3
    states[flooding] = 1
    states[drying] = 2
    states[normal] = 0

    unclassified = (states == 0) & (~normal)
    if unclassified.any():
        high_voltage = unclassified & (v > vq75)
        states[high_voltage] = 0
        temp_abnormal = unclassified & (~high_voltage) & (
            (t > tq75) | (t < tq25)
        )
        states[temp_abnormal] = 3
        low_voltage = (
            unclassified
            & (v <= vq50)
            & (~high_voltage)
            & (~temp_abnormal)
        )
        states[low_voltage] = 2
        # Remaining rows intentionally retain the Normal state 0, matching the
        # source dataset2 fallback.

    thresholds = {
        "voltage_q25": float(vq25),
        "voltage_q50": float(vq50),
        "voltage_q75": float(vq75),
        "air_rh_q25": float(aq25),
        "air_rh_q50": float(aq50),
        "air_rh_q75": float(aq75),
        "h2_rh_q25": float(hq25),
        "h2_rh_q50": float(hq50),
        "h2_rh_q75": float(hq75),
        "temperature_q25": float(tq25),
        "temperature_q50": float(tq50),
        "temperature_q75": float(tq75),
        "current_q50": float(iq50),
        "power_q50": float(pq50),
    }
    return states, thresholds


def _select_real_rows(
    labeled: pd.DataFrame,
    label: str,
    n_rows: int,
    used_keys: set[Tuple[float, ...]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = labeled.loc[labeled["State_Label"].eq(label), FEATURE_COLUMNS].copy()
    candidates = candidates.drop_duplicates(subset=MODEL_FEATURE_COLUMNS, keep="first")
    candidates = candidates.loc[
        [
            _feature_key(row)
            not in used_keys
            for row in candidates[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
        ]
    ]
    if len(candidates) < n_rows:
        raise ValueError(f"Need {n_rows} real {label} rows, only {len(candidates)} available")
    selected = candidates.iloc[rng.permutation(len(candidates))[:n_rows]].copy()
    selected["State"] = STATE_MAPPING[label]
    selected["State_Label"] = label
    return selected[OUTPUT_COLUMNS]


def _synthetic_flooding_rows(
    real_flooding: pd.DataFrame,
    n_rows: int,
    used_keys: set[Tuple[float, ...]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate unique Flooding rows from same-class processed measurements."""
    model_values = real_flooding[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    all_values = real_flooding[FEATURE_COLUMNS].to_numpy(dtype=float)
    center = np.nanmedian(model_values, axis=0)
    scale = np.nanstd(model_values, axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (model_values - center) / scale

    neighbour_count = min(8, len(model_values))
    neighbours = NearestNeighbors(n_neighbors=neighbour_count, metric="euclidean")
    neighbours.fit(normalized)
    neighbour_indices = neighbours.kneighbors(return_distance=False)

    lower = np.nanmin(model_values, axis=0)
    upper = np.nanmax(model_values, axis=0)
    voltage_idx = MODEL_FEATURE_COLUMNS.index("U_totV")
    current_idx = MODEL_FEATURE_COLUMNS.index("iA")
    power_idx = MODEL_FEATURE_COLUMNS.index("PW")
    tsec_idx = FEATURE_COLUMNS.index("tsec")
    model_positions = [FEATURE_COLUMNS.index(column) for column in MODEL_FEATURE_COLUMNS]

    generated: list[np.ndarray] = []
    generated_keys: set[Tuple[float, ...]] = set()
    for _ in range(n_rows):
        for _attempt in range(200):
            left = int(rng.integers(0, len(model_values)))
            choices = neighbour_indices[left, 1:]
            right = int(choices[int(rng.integers(0, len(choices)))])
            alpha = float(rng.uniform(0.2, 0.8))
            model_row = (1.0 - alpha) * model_values[left] + alpha * model_values[right]
            model_row += rng.normal(0.0, 0.002 * scale, size=model_row.shape)
            model_row = np.clip(model_row, lower, upper)
            model_row[power_idx] = model_row[voltage_idx] * model_row[current_idx]

            output_row = np.zeros(len(FEATURE_COLUMNS), dtype=float)
            # tsec is metadata: interpolate without adding physical noise.
            output_row[tsec_idx] = (
                (1.0 - alpha) * all_values[left, tsec_idx]
                + alpha * all_values[right, tsec_idx]
            )
            output_row[model_positions] = model_row
            key = _feature_key(model_row)
            if key not in used_keys and key not in generated_keys:
                generated_keys.add(key)
                generated.append(output_row)
                break
        else:
            raise RuntimeError("Unable to generate a unique synthetic Flooding row")

    frame = pd.DataFrame(np.asarray(generated), columns=FEATURE_COLUMNS)
    frame["State"] = STATE_MAPPING["Flooding"]
    frame["State_Label"] = "Flooding"
    return frame[OUTPUT_COLUMNS]


def build_dataset(
    output_path: str | Path = DEFAULT_OUTPUT,
    seed: int = 42,
    source_path: str | Path = SOURCE_PATH,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Classify processed data, select 12,000 rows, write, reload, and verify."""
    output_path = Path(output_path)
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Refusing to overwrite the processed source file")

    source = pd.read_csv(source_path)
    _validate_source(source, source_path)
    states, thresholds = classify_states(source)
    labels = np.array(
        [
            next((name for name, value in {**STATE_MAPPING, "Thermal_Management_Fault": 3}.items() if value == state), "Normal")
            for state in states
        ],
        dtype=object,
    )
    labeled = source.copy()
    labeled["State"] = states
    labeled["State_Label"] = labels

    raw_counts = {
        label: int((labels == label).sum())
        for label in ("Normal", "Flooding", "Membrane_Drying", "Thermal_Management_Fault")
    }
    if raw_counts != {
        "Normal": 16981,
        "Flooding": 1805,
        "Membrane_Drying": 8129,
        "Thermal_Management_Fault": 10984,
    }:
        raise AssertionError(f"Unexpected rule-label counts: {raw_counts}")

    rng = np.random.default_rng(int(seed))
    pieces: list[pd.DataFrame] = []
    used_keys: set[Tuple[float, ...]] = set()
    selected_real_counts: Dict[str, int] = {}

    for label, n_rows in (("Normal", 4000), ("Membrane_Drying", 4000)):
        part = _select_real_rows(labeled, label, n_rows, used_keys, rng)
        for row in part[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float):
            key = _feature_key(row)
            if key in used_keys:
                raise AssertionError(f"Real feature duplicate encountered in {label}")
            used_keys.add(key)
        pieces.append(part)
        selected_real_counts[label] = n_rows

    real_flooding = labeled.loc[
        labeled["State_Label"].eq("Flooding"), OUTPUT_COLUMNS
    ].copy()
    real_flooding["State"] = STATE_MAPPING["Flooding"]
    real_flooding["State_Label"] = "Flooding"
    real_flooding = real_flooding.drop_duplicates(subset=MODEL_FEATURE_COLUMNS, keep="first")
    if len(real_flooding) != raw_counts["Flooding"]:
        raise AssertionError("Flooding real-row count changed before supplementation")
    for row in real_flooding[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float):
        used_keys.add(_feature_key(row))
    pieces.append(real_flooding)
    selected_real_counts["Flooding"] = len(real_flooding)

    synthetic_count = TARGET_PER_CLASS - len(real_flooding)
    synthetic = _synthetic_flooding_rows(real_flooding, synthetic_count, used_keys, rng)
    pieces.append(synthetic)

    frame = pd.concat(pieces, ignore_index=True)
    frame["State"] = frame["State_Label"].map(STATE_MAPPING).astype(np.int64)
    frame["PW"] = frame["U_totV"].to_numpy(dtype=float) * frame["iA"].to_numpy(dtype=float)
    frame = frame.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    frame = frame[OUTPUT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8", float_format="%.17g")
    reloaded = pd.read_csv(output_path)

    class_counts = {
        label: int((reloaded["State_Label"] == label).sum()) for label in CLASS_ORDER
    }
    state_counts = {
        str(state): int((reloaded["State"] == state).sum())
        for state in range(3)
    }
    identity_error = float(
        np.abs(reloaded["PW"] - reloaded["U_totV"] * reloaded["iA"]).max()
    )
    duplicate_feature_rows = int(
        reloaded.duplicated(subset=MODEL_FEATURE_COLUMNS).sum()
    )
    if len(reloaded) != 12000 or class_counts != {label: 4000 for label in CLASS_ORDER}:
        raise AssertionError(f"Wrong serialized shape/class counts: {len(reloaded)}, {class_counts}")
    if set(reloaded["State_Label"]) != set(CLASS_ORDER):
        raise AssertionError("Unexpected class in serialized output")
    if duplicate_feature_rows != 0:
        raise AssertionError(f"Serialized output has {duplicate_feature_rows} feature duplicates")
    if identity_error > 1e-10:
        raise AssertionError(f"Power identity error too large: {identity_error}")

    manifest: Dict[str, Any] = {
        "dataset_name": "Public datasets 3-class 12000 from processed measurements",
        "source_kind": "processed_raw_measurements",
        "source_path": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "source_rows": int(len(source)),
        "seed": int(seed),
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "total_rows": int(len(reloaded)),
        "class_counts": class_counts,
        "state_counts": state_counts,
        "raw_rule_label_counts": raw_counts,
        "excluded_classes": ["Thermal_Management_Fault"],
        "excluded_class_counts": {"Thermal_Management_Fault": raw_counts["Thermal_Management_Fault"]},
        "class_order": CLASS_ORDER,
        "state_mapping": STATE_MAPPING,
        "feature_columns": FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "metadata_columns": ["tsec"],
        "selected_real_counts": {
            "Normal": selected_real_counts["Normal"],
            "Flooding": selected_real_counts["Flooding"],
            "Membrane_Drying": selected_real_counts["Membrane_Drying"],
        },
        "supplemental_real_counts": {"Normal": 0, "Flooding": 0, "Membrane_Drying": 0},
        "synthetic_counts": {"Normal": 0, "Flooding": synthetic_count, "Membrane_Drying": 0},
        "synthetic_method_class": "Flooding",
        "synthetic_method": "same-class standardized 8-nearest-neighbour interpolation; alpha in [0.2,0.8]; Gaussian noise 0.002*class_std on model features; empirical-range clipping; PW recomputed as U_totV*iA; tsec interpolated as metadata",
        "classification_rule": "global quartile cascade copied from 数据集2/build_balanced_dataset_smote.py; thermal precedence, then flooding, drying, normal, then voltage/temperature fallback",
        "classification_thresholds": thresholds,
        "power_identity_max_abs_error": identity_error,
        "duplicate_feature_rows": duplicate_feature_rows,
        "selection_policy": "4000 Normal real rows + 4000 Membrane_Drying real rows + all 1805 real Flooding rows + 2195 synthetic Flooding rows; Thermal_Management_Fault excluded",
        "preprocessing_note": "tsec is retained as metadata but current model preprocessing drops it; State is provenance encoding and State_Label is the supervised target",
    }
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return reloaded, manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame, manifest = build_dataset(
        output_path=args.output,
        seed=args.seed,
        source_path=args.source,
    )
    print(json.dumps({
        "source": manifest["source_path"],
        "output": manifest["output_path"],
        "manifest": str(Path(args.output).with_name(Path(args.output).stem + "_manifest.json").resolve()),
        "rows": len(frame),
        "raw_rule_label_counts": manifest["raw_rule_label_counts"],
        "class_counts": manifest["class_counts"],
        "synthetic_counts": manifest["synthetic_counts"],
        "excluded_classes": manifest["excluded_classes"],
        "duplicate_feature_rows": manifest["duplicate_feature_rows"],
        "power_identity_max_abs_error": manifest["power_identity_max_abs_error"],
        "output_sha256": manifest["output_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
