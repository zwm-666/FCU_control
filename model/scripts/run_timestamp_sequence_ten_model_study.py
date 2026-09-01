"""Frozen-checkpoint 10-model study on observed timestamp sequences.

Every sample is one full timestamp block (up to 60 raw high-frequency rows).
Keras models receive the observed [time-within-block, feature] tensor; classic
models receive summary statistics made from that exact same block. Development
uses only the validation timestamp blocks; the explicit test phase reloads
frozen checkpoints and never refits them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from scripts.comparison_models import KerasSequenceClassifier, TEN_MODEL_COMPARISON_MODELS, build_default_model_specs
from scripts.run_temporal_ten_model_study import (
    _fit_estimator,
    _metric_result,
    _predict_estimator,
    _write_json,
    aggregate_records,
    chronological_block_split,
    load_frozen_estimator,
    manifest_matches_requested_contract,
    save_frozen_estimator,
    sha256_file,
)


DROP_FEATURES = ("电堆功率", "power", "Power")


def _numeric_feature_columns(df: pd.DataFrame, label_col: str, time_col: str) -> list[str]:
    excluded = {label_col, time_col, "State", "state", "tsec", *DROP_FEATURES}
    columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        if pd.to_numeric(df[column], errors="coerce").notna().any():
            columns.append(column)
    if not columns:
        raise ValueError("No numeric sequence features remain after metadata/power removal")
    return columns


def _fixed_length_sequence(values: np.ndarray, sequence_length: int) -> np.ndarray:
    if values.shape[0] == sequence_length:
        return values
    if values.shape[0] > sequence_length:
        indices = np.linspace(0, values.shape[0] - 1, sequence_length).round().astype(int)
        return values[indices]
    return np.concatenate(
        [values, np.repeat(values[-1:, :], sequence_length - values.shape[0], axis=0)], axis=0
    )


def build_timestamp_sequence_dataset(
    df: pd.DataFrame,
    label_col: str = "类型",
    time_col: str = "测试时间",
    sequence_length: int = 60,
) -> Dict[str, Any]:
    """Aggregate raw high-frequency rows into label-pure timestamp sequences."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if label_col not in df or time_col not in df:
        raise ValueError("label_col and time_col are required")

    feature_names = _numeric_feature_columns(df, label_col, time_col)
    frame = df[[time_col, label_col, *feature_names]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors="raise")
    for feature in feature_names:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    sequences, summaries, labels, rows = [], [], [], []
    for timestamp, group in frame.groupby(time_col, sort=False):
        unique_labels = group[label_col].dropna().unique()
        if len(unique_labels) != 1:
            raise ValueError(f"Timestamp {timestamp} contains mixed labels: {unique_labels.tolist()}")
        numeric = group[feature_names].copy()
        numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
        sequence = _fixed_length_sequence(numeric.to_numpy(dtype=np.float32), sequence_length)
        diff = np.diff(sequence, axis=0)
        summaries.append(np.concatenate([
            sequence.mean(axis=0), sequence.std(axis=0), sequence.min(axis=0),
            sequence.max(axis=0), diff.mean(axis=0), diff.std(axis=0),
        ]).astype(np.float32))
        sequences.append(sequence)
        labels.append(unique_labels[0])
        rows.append({
            time_col: timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            label_col: unique_labels[0],
            "raw_rows": int(len(group)),
        })

    return {
        "X_sequence": np.stack(sequences).astype(np.float32),
        "X_summary": np.stack(summaries).astype(np.float32),
        "labels": np.asarray(labels),
        "feature_names": feature_names,
        "block_frame": pd.DataFrame(rows),
        "sequence_length": int(sequence_length),
    }


def _read_raw(data_path: str, label_col: str, time_col: str, sequence_length: int) -> Dict[str, Any]:
    path = Path(data_path)
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    dataset = build_timestamp_sequence_dataset(raw, label_col, time_col, sequence_length)
    encoder = LabelEncoder()
    dataset["y"] = encoder.fit_transform(dataset["labels"]).astype(np.int32)
    dataset["class_names"] = [str(item) for item in encoder.classes_]
    return dataset


def _prepare_arrays(dataset: Mapping[str, Any], split: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    X_seq = np.asarray(dataset["X_sequence"], dtype=np.float32)
    X_summary = np.asarray(dataset["X_summary"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.int32)
    seq_scaler = StandardScaler().fit(X_seq[split["train"]].reshape(-1, X_seq.shape[-1]))
    summary_scaler = StandardScaler().fit(X_summary[split["train"]])

    def seq_transform(indices: np.ndarray) -> np.ndarray:
        values = X_seq[indices]
        return seq_scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape).astype(np.float32)

    return {
        "X_sequence_train": seq_transform(split["train"]),
        "X_sequence_validation": seq_transform(split["validation"]),
        "X_sequence_test": seq_transform(split["test"]),
        "X_summary_train": summary_scaler.transform(X_summary[split["train"]]).astype(np.float32),
        "X_summary_validation": summary_scaler.transform(X_summary[split["validation"]]).astype(np.float32),
        "X_summary_test": summary_scaler.transform(X_summary[split["test"]]).astype(np.float32),
        "y_train": y[split["train"]],
        "y_validation": y[split["validation"]],
        "y_test": y[split["test"]],
    }


def _uses_sequence(estimator: Any) -> bool:
    return isinstance(estimator, KerasSequenceClassifier)


def _arrays_for_estimator(arrays: Mapping[str, np.ndarray], estimator: Any, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = "X_sequence" if _uses_sequence(estimator) else "X_summary"
    return arrays[f"{prefix}_{split_name}"], arrays[f"y_{split_name}"]


def _manifest(
    data_path: str,
    dataset: Mapping[str, Any],
    split: Mapping[str, np.ndarray],
    label_col: str,
    time_col: str,
    seeds: Sequence[int],
    models: Sequence[str],
    deep_epochs: int,
    comparison_strength: str,
    test_fraction: float,
    val_fraction: float,
    proposed_validation_split: float = 0.15,
    proposed_clipnorm: float = 1.0,
) -> Dict[str, Any]:
    block = dataset["block_frame"]
    return {
        "study_name": "observed_timestamp_sequence_10_model",
        "data_path": data_path,
        "data_sha256": sha256_file(data_path),
        "label_col": label_col,
        "time_col": time_col,
        "models": list(models),
        "seeds": [int(seed) for seed in seeds],
        "deep_epochs": int(deep_epochs),
        "comparison_strength": comparison_strength,
        "baseline_capacity": "reduced" if comparison_strength == "reduced_baselines" else "standard",
        "proposed_capacity": "standard",
        "proposed_validation_split": float(proposed_validation_split),
        "proposed_clipnorm": float(proposed_clipnorm),
        "test_fraction": float(test_fraction),
        "val_fraction": float(val_fraction),
        "sequence_length": int(dataset["sequence_length"]),
        "feature_count": len(dataset["feature_names"]),
        "feature_names": list(dataset["feature_names"]),
        "class_names": list(dataset["class_names"]),
        "split_protocol": "within-class chronological timestamp-block holdout",
        "split_sizes": {name: int(len(indices)) for name, indices in split.items()},
        "class_counts": {
            name: {str(k): int(v) for k, v in block.iloc[indices][label_col].value_counts().sort_index().items()}
            for name, indices in split.items()
        },
        "outer_test_evaluated": False,
        "artifact_semantics": {
            "development_evaluated_split": "validation",
            "outer_evaluated_split": "test",
            "selection_rule": "mean validation Macro-F1 across matched seeds",
            "model_input": "Keras models: observed 60-row timestamp sequences; classic models: summaries of same sequences",
        },
        "caveat": (
            "Timestamp blocks prevent row leakage and use real within-block dynamics, but labels still occupy "
            "distinct recording periods. This is chronological within-dataset evidence, not independent fault-session validation."
        ),
    }


def _development(
    specs_by_seed: Mapping[int, Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    models: Sequence[str],
    output_dir: Path,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    records, frozen = [], []
    for seed, specs in specs_by_seed.items():
        for name in models:
            spec = specs[name]
            estimator, fit_time = _fit_estimator(spec, *_arrays_for_estimator(arrays, spec.estimator_factory(), "train"), int(seed))
            X_val, y_val = _arrays_for_estimator(arrays, estimator, "validation")
            prediction, prediction_time = _predict_estimator(estimator, X_val)
            record = _metric_result(spec.name, spec.level, y_val, prediction, class_names, fit_time, prediction_time)
            record.update({"seed": int(seed), "evaluated_split": "validation"})
            records.append(record)
            frozen.append(save_frozen_estimator(estimator, output_dir, spec.name, int(seed)))
            print(f"[validation] seed={seed} {name}: macro_f1={record['f1_macro']:.5f}", flush=True)
    return records, frozen


def _test(
    frozen: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    specs: Mapping[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
) -> list[Dict[str, Any]]:
    records = []
    for checkpoint in frozen:
        estimator = load_frozen_estimator(checkpoint, output_dir)
        X_test, y_test = _arrays_for_estimator(arrays, estimator, "test")
        prediction, prediction_time = _predict_estimator(estimator, X_test)
        spec = specs[checkpoint["model"]]
        record = _metric_result(spec.name, spec.level, y_test, prediction, class_names, 0.0, prediction_time)
        record.update({
            "seed": int(checkpoint["seed"]),
            "evaluated_split": "test",
            "checkpoint": checkpoint["checkpoint"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        })
        records.append(record)
        print(f"[test frozen] seed={record['seed']} {spec.name}: macro_f1={record['f1_macro']:.5f}", flush=True)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", required=True, choices=("development", "test"))
    parser.add_argument("--label-col", default="类型")
    parser.add_argument("--time-col", default="测试时间")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--models", default="default")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--deep-epochs", type=int, default=60)
    parser.add_argument(
        "--comparison-strength",
        choices=("standard", "light", "reduced_baselines"),
        default="standard",
        help="reduced_baselines lowers only non-proposed baselines; DI/AB stay standard.",
    )
    parser.add_argument(
        "--proposed-validation-split",
        type=float,
        default=0.15,
        help="Train-internal validation fraction used only by DI/AB-EMSTGAT early stopping.",
    )
    parser.add_argument(
        "--proposed-clipnorm",
        type=float,
        default=1.0,
        help="Gradient clipping norm applied only to DI/AB-EMSTGAT.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "split_manifest.json"
    seeds = tuple(int(v.strip()) for v in args.seeds.split(",") if v.strip())
    models = TEN_MODEL_COMPARISON_MODELS if args.models == "default" else tuple(v.strip() for v in args.models.split(",") if v.strip())
    dataset = _read_raw(args.data, args.label_col, args.time_col, args.sequence_length)
    split = chronological_block_split(dataset["block_frame"], args.label_col, args.time_col, args.test_fraction, args.val_fraction)
    arrays = _prepare_arrays(dataset, split)
    manifest = _manifest(
        args.data, dataset, split, args.label_col, args.time_col, seeds, models,
        args.deep_epochs, args.comparison_strength, args.test_fraction, args.val_fraction,
        proposed_validation_split=args.proposed_validation_split,
        proposed_clipnorm=args.proposed_clipnorm,
    )

    all_specs = build_default_model_specs(
        seed=seeds[0], n_jobs=args.n_jobs, deep_epochs=args.deep_epochs,
        deep_validation_split=0.0, comparison_strength=args.comparison_strength,
        proposed_validation_split=args.proposed_validation_split,
        proposed_clipnorm=args.proposed_clipnorm,
    )
    missing = [name for name in models if name not in all_specs]
    if missing:
        raise ValueError(f"Unknown models: {missing}")

    if args.phase == "development":
        specs_by_seed = {
            seed: build_default_model_specs(
                seed=seed, n_jobs=args.n_jobs, deep_epochs=args.deep_epochs,
                deep_validation_split=0.0, comparison_strength=args.comparison_strength,
                proposed_validation_split=args.proposed_validation_split,
                proposed_clipnorm=args.proposed_clipnorm,
            )
            for seed in seeds
        }
        records, frozen = _development(specs_by_seed, arrays, dataset["class_names"], models, output_dir)
        manifest["frozen_checkpoints"] = frozen
        _write_json(manifest_path, manifest)
        _write_json(output_dir / "development_records.json", records)
        aggregate_records(records).to_csv(output_dir / "development_summary.csv", index=False, encoding="utf-8-sig")
        return 0

    if not manifest_path.exists():
        raise FileNotFoundError("Test phase requires a completed development manifest")
    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest_matches_requested_contract(
        frozen_manifest, manifest["data_sha256"], models, seeds, args.deep_epochs,
        args.comparison_strength, args.test_fraction, args.val_fraction,
    ):
        raise RuntimeError("Requested test phase differs from frozen development study contract")
    if frozen_manifest.get("outer_test_evaluated"):
        raise RuntimeError("This manifest already performed test evaluation")
    frozen = frozen_manifest.get("frozen_checkpoints") or []
    if {(r["model"], int(r["seed"])) for r in frozen} != {(name, seed) for name in models for seed in seeds}:
        raise RuntimeError("Frozen checkpoint set is incomplete")
    records = _test(frozen, arrays, all_specs, dataset["class_names"], output_dir)
    _write_json(output_dir / "outer_test_records.json", records)
    aggregate_records(records).to_csv(output_dir / "outer_test_summary.csv", index=False, encoding="utf-8-sig")
    frozen_manifest["outer_test_evaluated"] = True
    _write_json(manifest_path, frozen_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
