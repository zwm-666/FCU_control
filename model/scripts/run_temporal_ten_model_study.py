"""Reproducible 10-model PEMFC study with chronological timestamp blocks.

The self-test CSV is composed of many high-frequency samples per timestamp.
Random row splits leak near-duplicate operating points across partitions.  This
runner keeps all rows from one timestamp together and makes a per-class
chronological train/validation/test allocation.  Development selection never
reads the outer test block; test evaluation is an explicit second phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.comparison_models import (  # noqa: E402
    TEN_MODEL_COMPARISON_MODELS,
    KerasSequenceClassifier,
    _clean_feature_frame,
    build_default_model_specs,
)


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
ARTIFACT_SEMANTICS = {
    "development_evaluated_split": "validation",
    "outer_evaluated_split": "test",
    "selection_rule": "mean validation Macro-F1 across matched seeds",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamps_per_split(n_blocks: int, test_fraction: float, val_fraction: float) -> tuple[int, int, int]:
    if n_blocks < 3:
        raise ValueError("Each class requires at least three distinct timestamp blocks.")
    n_test = max(1, int(round(n_blocks * test_fraction)))
    n_val = max(1, int(round(n_blocks * val_fraction)))
    while n_blocks - n_test - n_val < 1:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("Cannot allocate non-empty train/validation/test blocks.")
    return n_blocks - n_val - n_test, n_val, n_test


def chronological_block_split(
    df: pd.DataFrame,
    label_col: str,
    time_col: str,
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
) -> Dict[str, np.ndarray]:
    """Split each class by whole chronological timestamp blocks.

    This prevents rows recorded in the same minute from straddling partitions.
    The split is class-stratified by construction, but it is still a
    within-condition temporal-block protocol—not a fully independent
    fault-session protocol when a class has only one recording session.
    """
    if not 0.0 < test_fraction < 0.5 or not 0.0 < val_fraction < 0.5:
        raise ValueError("test_fraction and val_fraction must be in (0, 0.5)")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction + val_fraction must be less than 1")
    if label_col not in df or time_col not in df:
        raise ValueError("label_col and time_col must exist in dataframe")

    parsed_time = pd.to_datetime(df[time_col], errors="coerce")
    if parsed_time.isna().any():
        raise ValueError(f"{time_col} contains unparseable timestamps")

    splits: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
    for label in sorted(df[label_col].unique().tolist()):
        class_indices = df.index[df[label_col] == label]
        blocks = (
            pd.DataFrame({"idx": class_indices, "time": parsed_time.loc[class_indices].values})
            .groupby("time", sort=True)["idx"]
            .apply(list)
            .tolist()
        )
        n_train, n_val, n_test = _timestamps_per_split(len(blocks), test_fraction, val_fraction)
        for block in blocks[:n_train]:
            splits["train"].extend(int(i) for i in block)
        for block in blocks[n_train : n_train + n_val]:
            splits["validation"].extend(int(i) for i in block)
        for block in blocks[n_train + n_val : n_train + n_val + n_test]:
            splits["test"].extend(int(i) for i in block)

    result = {name: np.asarray(sorted(indices), dtype=np.int64) for name, indices in splits.items()}
    all_indices = np.concatenate(list(result.values()))
    if len(all_indices) != len(df) or len(np.unique(all_indices)) != len(df):
        raise RuntimeError("Timestamp split did not assign each input row exactly once")
    for name, indices in result.items():
        labels = set(df.iloc[indices][label_col].tolist())
        if labels != set(df[label_col].unique().tolist()):
            raise RuntimeError(f"{name} split does not contain all target classes")
    return result


def _split_counts(df: pd.DataFrame, indices: np.ndarray, label_col: str) -> Dict[str, int]:
    return {str(key): int(value) for key, value in df.iloc[indices][label_col].value_counts().sort_index().items()}


def build_manifest(
    df: pd.DataFrame,
    split: Mapping[str, np.ndarray],
    data_path: str,
    label_col: str,
    time_col: str,
    seeds: Sequence[int],
    models: Sequence[str],
) -> Dict[str, Any]:
    return {
        "study_name": "temporal_timestamp_block_10_model",
        "data_path": data_path,
        "data_sha256": sha256_file(data_path) if Path(data_path).exists() else None,
        "label_col": label_col,
        "time_col": time_col,
        "models": list(models),
        "seeds": [int(seed) for seed in seeds],
        "split_protocol": "within-class chronological timestamp-block holdout",
        "split_sizes": {name: int(len(indices)) for name, indices in split.items()},
        "class_counts": {name: _split_counts(df, indices, label_col) for name, indices in split.items()},
        "timestamp_counts": {
            name: int(pd.to_datetime(df.iloc[indices][time_col]).nunique())
            for name, indices in split.items()
        },
        "outer_test_evaluated": False,
        "artifact_semantics": dict(ARTIFACT_SEMANTICS),
        "caveat": (
            "The source labels occupy distinct recording periods. Timestamp blocks stop same-minute leakage, "
            "but do not create independent cross-condition fault sessions. Results must be reported as "
            "within-dataset chronological-block evidence."
        ),
    }


def manifest_matches_requested_contract(
    manifest: Mapping[str, Any],
    data_sha256: str,
    models: Sequence[str],
    seeds: Sequence[int],
    deep_epochs: int,
    comparison_strength: str,
    test_fraction: float,
    val_fraction: float,
) -> bool:
    """Ensure test evaluation cannot silently drift from frozen development settings."""
    return (
        manifest.get("data_sha256") == data_sha256
        and manifest.get("models") == list(models)
        and manifest.get("seeds") == [int(seed) for seed in seeds]
        and int(manifest.get("deep_epochs", -1)) == int(deep_epochs)
        and manifest.get("comparison_strength") == comparison_strength
        and float(manifest.get("test_fraction", -1.0)) == float(test_fraction)
        and float(manifest.get("val_fraction", -1.0)) == float(val_fraction)
    )


def checkpoint_path(model_name: str, seed: int, checkpoint_kind: str) -> Path:
    suffix = ".weights.h5" if checkpoint_kind == "keras" else ".pkl"
    return Path("checkpoints") / f"{model_name}_seed{int(seed)}{suffix}"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save_frozen_estimator(estimator: Any, output_dir: Path, model_name: str, seed: int) -> Dict[str, Any]:
    """Freeze a fitted model and emit enough metadata for test-only restoration."""
    if isinstance(estimator, KerasSequenceClassifier) and estimator.backend_ == "keras":
        relative_path = checkpoint_path(model_name, seed, "keras")
        metadata = estimator.save_frozen(output_dir / relative_path)
        checkpoint_kind = "keras"
        metadata_path = relative_path.with_suffix(".json")
        _write_json(output_dir / metadata_path, metadata)
    else:
        relative_path = checkpoint_path(model_name, seed, "pickle")
        checkpoint_path_abs = output_dir / relative_path
        checkpoint_path_abs.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_path_abs.write_bytes(payload)
        checkpoint_kind = "pickle"
        metadata_path = None
    absolute_path = output_dir / relative_path
    return {
        "model": model_name,
        "seed": int(seed),
        "checkpoint": str(relative_path),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": sha256_file(absolute_path),
        "metadata": str(metadata_path) if metadata_path else None,
    }


def load_frozen_estimator(record: Mapping[str, Any], output_dir: Path) -> Any:
    checkpoint = output_dir / str(record["checkpoint"])
    if sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    if record["checkpoint_kind"] == "pickle":
        return pickle.loads(checkpoint.read_bytes())
    if record["checkpoint_kind"] == "keras":
        metadata_path = output_dir / str(record["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return KerasSequenceClassifier.load_frozen(metadata, checkpoint)
    raise ValueError(f"Unsupported checkpoint kind: {record['checkpoint_kind']}")


def _read_data(data_path: str, label_col: str, time_col: str) -> tuple[pd.DataFrame, List[str], np.ndarray, List[str]]:
    path = Path(data_path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    if label_col not in df or time_col not in df:
        raise ValueError(f"Dataset must contain {label_col!r} and {time_col!r}")
    df = df[df[label_col].notna()].copy().reset_index(drop=True)
    raw_features = df.drop(columns=[label_col, time_col, "State", "state", "tsec"], errors="ignore")
    feature_frame = _clean_feature_frame(raw_features)
    if feature_frame.empty:
        raise ValueError("No usable numeric features after removing metadata columns")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[label_col]).astype(np.int32)
    return df, list(feature_frame.columns), feature_frame.to_numpy(dtype=np.float32), [str(v) for v in encoder.classes_]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def _metric_result(
    name: str,
    level: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    fit_time: float,
    prediction_time: float,
) -> Dict[str, Any]:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return _json_safe(
        {
            "model": name,
            "level": level,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "per_class_recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "classification_report": report,
            "fit_time": fit_time,
            "prediction_time": prediction_time,
        }
    )


def _fit_estimator(
    spec: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[Any, float]:
    _seed_everything(seed)
    estimator = spec.estimator_factory()
    started = time.time()
    estimator.fit(X_train, y_train)
    return estimator, time.time() - started


def _predict_estimator(estimator: Any, X_eval: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.time()
    prediction = estimator.predict(X_eval)
    return np.asarray(prediction), time.time() - started


def _fit_predict(
    spec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    _seed_everything(seed)
    estimator = spec.estimator_factory()
    started = time.time()
    estimator.fit(X_train, y_train)
    fit_time = time.time() - started
    started = time.time()
    prediction = estimator.predict(X_eval)
    prediction_time = time.time() - started
    return np.asarray(prediction), fit_time, prediction_time


def prepare_arrays(
    X: np.ndarray,
    y: np.ndarray,
    split: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Fit scaler on train only and materialize all declared partitions."""
    scaler = StandardScaler().fit(X[split["train"]])
    return {
        "X_train": scaler.transform(X[split["train"]]),
        "y_train": y[split["train"]],
        "X_validation": scaler.transform(X[split["validation"]]),
        "y_validation": y[split["validation"]],
        "X_test": scaler.transform(X[split["test"]]),
        "y_test": y[split["test"]],
    }


def run_phase(
    specs: Iterable[Any],
    arrays: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    seeds: Sequence[int],
    evaluated_split: str,
) -> List[Dict[str, Any]]:
    if evaluated_split not in {"validation", "test"}:
        raise ValueError("evaluated_split must be validation or test")
    X_eval = arrays[f"X_{evaluated_split}"]
    y_eval = arrays[f"y_{evaluated_split}"]
    records: List[Dict[str, Any]] = []
    for seed in seeds:
        for spec in specs:
            prediction, fit_time, prediction_time = _fit_predict(
                spec, arrays["X_train"], arrays["y_train"], X_eval, int(seed)
            )
            record = _metric_result(
                spec.name, spec.level, y_eval, prediction, class_names, fit_time, prediction_time
            )
            record.update({"seed": int(seed), "evaluated_split": evaluated_split})
            records.append(record)
            print(
                f"[{evaluated_split}] seed={seed} {spec.name}: "
                f"macro_f1={record['f1_macro']:.5f} acc={record['accuracy']:.5f}",
                flush=True,
            )
    return records


def run_and_freeze_development(
    specs: Iterable[Any],
    arrays: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    seeds: Sequence[int],
    output_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fit on train, evaluate only validation, then persist frozen checkpoints."""
    records: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    for seed in seeds:
        for spec in specs:
            estimator, fit_time = _fit_estimator(spec, arrays["X_train"], arrays["y_train"], int(seed))
            prediction, prediction_time = _predict_estimator(estimator, arrays["X_validation"])
            record = _metric_result(
                spec.name, spec.level, arrays["y_validation"], prediction, class_names, fit_time, prediction_time
            )
            record.update({"seed": int(seed), "evaluated_split": "validation"})
            records.append(record)
            checkpoints.append(save_frozen_estimator(estimator, output_dir, spec.name, int(seed)))
            print(
                f"[validation] seed={seed} {spec.name}: "
                f"macro_f1={record['f1_macro']:.5f} acc={record['accuracy']:.5f}",
                flush=True,
            )
    return records, checkpoints


def evaluate_frozen_checkpoints(
    checkpoint_records: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    specs_by_name: Mapping[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Run the declared test split without refitting any checkpoint."""
    records: List[Dict[str, Any]] = []
    for checkpoint_record in checkpoint_records:
        estimator = load_frozen_estimator(checkpoint_record, output_dir)
        prediction, prediction_time = _predict_estimator(estimator, arrays["X_test"])
        spec = specs_by_name[checkpoint_record["model"]]
        record = _metric_result(
            spec.name,
            spec.level,
            arrays["y_test"],
            prediction,
            class_names,
            fit_time=0.0,
            prediction_time=prediction_time,
        )
        record.update({
            "seed": int(checkpoint_record["seed"]),
            "evaluated_split": "test",
            "checkpoint": checkpoint_record["checkpoint"],
            "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        })
        records.append(record)
        print(
            f"[test frozen] seed={record['seed']} {spec.name}: "
            f"macro_f1={record['f1_macro']:.5f} acc={record['accuracy']:.5f}",
            flush=True,
        )
    return records


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted")
    df = pd.DataFrame(records)
    for model, group in df.groupby("model", sort=False):
        row = {"model": model, "level": group["level"].iloc[0], "n_seeds": int(len(group))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("development", "test"), required=True)
    parser.add_argument("--label-col", default="类型")
    parser.add_argument("--time-col", default="测试时间")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--models", default="default")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--deep-epochs", type=int, default=60)
    parser.add_argument("--comparison-strength", choices=("standard", "light"), default="standard")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "split_manifest.json"
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("At least one seed is required")

    df, feature_names, X, class_names = _read_data(args.data, args.label_col, args.time_col)
    y = LabelEncoder().fit_transform(df[args.label_col]).astype(np.int32)
    split = chronological_block_split(
        df, args.label_col, args.time_col, args.test_fraction, args.val_fraction
    )
    requested = TEN_MODEL_COMPARISON_MODELS if args.models == "default" else tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    specs_by_name = build_default_model_specs(
        seed=seeds[0], n_jobs=args.n_jobs, deep_epochs=args.deep_epochs,
        deep_validation_split=0.0, comparison_strength=args.comparison_strength,
    )
    missing = [name for name in requested if name not in specs_by_name]
    if missing:
        raise ValueError(f"Unknown models: {missing}")

    manifest = build_manifest(
        df, split, args.data, args.label_col, args.time_col, seeds, requested
    )
    manifest["feature_count"] = len(feature_names)
    manifest["feature_names"] = feature_names
    manifest["class_names"] = class_names
    manifest["deep_epochs"] = args.deep_epochs
    manifest["comparison_strength"] = args.comparison_strength
    manifest["test_fraction"] = args.test_fraction
    manifest["val_fraction"] = args.val_fraction
    arrays = prepare_arrays(X, y, split)

    if args.phase == "development":
        records: List[Dict[str, Any]] = []
        frozen_checkpoints: List[Dict[str, Any]] = []
        for seed in seeds:
            seeded_specs = build_default_model_specs(
                seed=seed, n_jobs=args.n_jobs, deep_epochs=args.deep_epochs,
                deep_validation_split=0.0, comparison_strength=args.comparison_strength,
            )
            seed_records, seed_checkpoints = run_and_freeze_development(
                [seeded_specs[name] for name in requested],
                arrays,
                class_names,
                [seed],
                output_dir,
            )
            records.extend(seed_records)
            frozen_checkpoints.extend(seed_checkpoints)
        manifest["frozen_checkpoints"] = frozen_checkpoints
        _write_json(manifest_path, manifest)
        _write_json(output_dir / "development_records.json", records)
        aggregate_records(records).to_csv(output_dir / "development_summary.csv", index=False, encoding="utf-8-sig")
        print(f"Development artifacts and frozen checkpoints written to {output_dir}")
        return 0

    if not manifest_path.exists():
        raise FileNotFoundError("Test phase requires an existing development split_manifest.json")
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest_matches_requested_contract(
        stored_manifest,
        manifest["data_sha256"],
        requested,
        seeds,
        args.deep_epochs,
        args.comparison_strength,
        args.test_fraction,
        args.val_fraction,
    ):
        raise RuntimeError("Requested test phase drifts from frozen development manifest")
    if stored_manifest.get("outer_test_evaluated"):
        raise RuntimeError("Frozen manifest already records an outer-test evaluation")

    frozen_checkpoints = stored_manifest.get("frozen_checkpoints") or []
    expected_pairs = {(name, int(seed)) for name in requested for seed in seeds}
    actual_pairs = {(item.get("model"), int(item.get("seed"))) for item in frozen_checkpoints}
    if actual_pairs != expected_pairs:
        raise RuntimeError("Frozen development checkpoints are incomplete or do not match the study contract")

    test_specs = build_default_model_specs(
        seed=seeds[0],
        n_jobs=args.n_jobs,
        deep_epochs=args.deep_epochs,
        deep_validation_split=0.0,
        comparison_strength=args.comparison_strength,
    )
    records = evaluate_frozen_checkpoints(
        frozen_checkpoints,
        arrays,
        test_specs,
        class_names,
        output_dir,
    )
    _write_json(output_dir / "outer_test_records.json", records)
    aggregate_records(records).to_csv(output_dir / "outer_test_summary.csv", index=False, encoding="utf-8-sig")
    stored_manifest["outer_test_evaluated"] = True
    stored_manifest["outer_test_evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(manifest_path, stored_manifest)
    print(f"Outer-test artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
