"""Reproducible multi-seed RF / DI-EMSTGAT / AB-EMSTGAT evaluation.

The runner uses one fixed stratified outer test partition per seed.  All
preprocessing and early-stopping selection are fit on the corresponding
training partition only.  It is intentionally limited to the candidate and
control models needed to diagnose whether direct input or automatic branching
adds value; it does not alter the fixed 10-model thesis table.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "minority_recall")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def load_numeric_table(data_path: str, label_col: str, drop_columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(data_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")
    features = df.drop(columns=[label_col, *[name for name in drop_columns if name in df.columns]])
    features = features.apply(pd.to_numeric, errors="coerce").fillna(features.median(numeric_only=True)).fillna(0.0)
    encoder = LabelEncoder()
    return features.to_numpy(dtype=np.float32), encoder.fit_transform(df[label_col]).astype(np.int32), list(features.columns)


def calculate_metrics(model: str, seed: int, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "model": model,
        "seed": int(seed),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "minority_recall": float(recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]),
    }


def aggregate_seed_results(rows: Iterable[Dict[str, Any]], control_model: str = "rf_control") -> List[Dict[str, Any]]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    control = frame[frame["model"] == control_model].set_index("seed")
    metric_names = [metric for metric in METRIC_NAMES if metric in frame.columns]
    summaries: List[Dict[str, Any]] = []
    for model, group in frame.groupby("model", sort=False):
        summary: Dict[str, Any] = {"model": model, "n_seeds": int(group["seed"].nunique())}
        for metric in metric_names:
            summary[f"{metric}_mean"] = float(group[metric].mean())
            summary[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            paired = group.set_index("seed")[metric].sort_index()
            common = paired.index.intersection(control.index)
            if len(common):
                delta = paired.loc[common] - control.loc[common, metric]
                summary[f"{metric}_delta_vs_control_mean"] = float(delta.mean())
                summary[f"{metric}_delta_vs_control_std"] = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
            else:
                summary[f"{metric}_delta_vs_control_mean"] = float("nan")
                summary[f"{metric}_delta_vs_control_std"] = float("nan")
        summaries.append(summary)
    return summaries


def fit_predict_deep(
    branch_mode: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    epochs: int,
) -> np.ndarray:
    import tensorflow as tf
    from physics_decoupled_emstgat.adaptive_model import AdaptiveEMSTGAT, AdaptiveModelConfig

    set_seed(seed)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    lookup = dict(zip(np.unique(y_train).tolist(), class_weights.tolist()))
    weights = np.asarray([lookup[int(label)] for label in y_train], dtype=np.float32)
    weights /= weights.mean()
    model = AdaptiveEMSTGAT(
        AdaptiveModelConfig(
            input_dim=X_train.shape[1],
            num_classes=len(np.unique(y_train)),
            branch_mode=branch_mode,
            hidden_units=64,
            attention_heads=4,
            max_sequence_length=15,
            knn_top_k=6,
            dropout_rate=0.15,
            num_auto_branches=4,
            router_balance_weight=0.01,
            router_entropy_weight=0.001,
        )
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    model.fit(X_train, y_train, sample_weight=weights, validation_data=(X_val, y_val), epochs=epochs, batch_size=64, verbose=0, callbacks=callbacks)
    return np.argmax(model.predict(X_test, verbose=0), axis=1)


def run_one_seed(X: np.ndarray, y: np.ndarray, seed: int, test_size: float, val_size: float, epochs: int) -> List[Dict[str, Any]]:
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=val_size, stratify=y_train_all, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=1).fit(X_train, y_train)
    rows = [calculate_metrics("rf_control", seed, y_test, rf.predict(X_test))]
    for branch_mode, model_name in (("raw", "di_emstgat"), ("auto", "ab_emstgat")):
        prediction = fit_predict_deep(branch_mode, seed, X_train, y_train, X_val, y_val, X_test, epochs)
        rows.append(calculate_metrics(model_name, seed, y_test, prediction))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-seed RF vs DI/AB-EMSTGAT evaluation.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--drop-columns", default="测试时间")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    X, y, features = load_numeric_table(args.data, args.label_col, [name.strip() for name in args.drop_columns.split(",") if name.strip()])
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"[multiseed] seed={seed}", flush=True)
        rows.extend(run_one_seed(X, y, seed, args.test_size, args.val_size, args.epochs))
    summary = aggregate_seed_results(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(output / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    with (output / "experiment.json").open("w", encoding="utf-8") as handle:
        json.dump({"data": args.data, "label_col": args.label_col, "seeds": seeds, "test_size": args.test_size, "val_size": args.val_size, "epochs": args.epochs, "feature_count": len(features), "rows": rows, "summary": summary}, handle, ensure_ascii=False, indent=2)
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
