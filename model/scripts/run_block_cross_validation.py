"""Block-level stratified cross-validation on the leak-free timestamp protocol.

Why this exists: the single chronological 39-block holdout gives 2.6 percentage
points of resolution per block, so no configuration change is separable from
noise (measured: LightGBM pinned at 0.87002 with std 0.0 across 5 seeds while
the proposed models swung by 0.13).  Cross-validation evaluates every one of
the 198 timestamp blocks exactly once per seed, giving n_folds x n_seeds paired
observations instead of 5.

Leak-free property is preserved: a timestamp block is an atomic unit, so the
~60 high-frequency rows inside one block never straddle train and test.  Scaling
and any feature selection are fit on the training folds only.

This is a model-selection / statistical-power tool.  It reports pooled
cross-validation estimates, not a single held-out confirmatory test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts.comparison_models import (
    KerasSequenceClassifier,
    TEN_MODEL_COMPARISON_MODELS,
    build_default_model_specs,
    select_model_specs,
)
from scripts.run_timestamp_sequence_ten_model_study import _read_raw

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "minority_recall")


def stratified_block_folds(
    block_frame: pd.DataFrame,
    label_col: str,
    n_splits: int = 5,
    seed: int = 42,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold over whole timestamp blocks (never over raw rows)."""
    labels = block_frame[label_col].to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(train_idx), np.asarray(test_idx))
        for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels)
    ]


def _minority_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes, counts = np.unique(y_true, return_counts=True)
    minority = classes[int(np.argmin(counts))]
    per_class = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    return float(per_class[list(classes).index(minority)])


def _fold_arrays(
    dataset: Mapping[str, Any], train_idx: np.ndarray, test_idx: np.ndarray
) -> Dict[str, np.ndarray]:
    """Scale sequences and summaries using training-fold statistics only."""
    X_seq = np.asarray(dataset["X_sequence"], dtype=np.float32)
    X_sum = np.asarray(dataset["X_summary"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.int32)

    seq_scaler = StandardScaler().fit(X_seq[train_idx].reshape(-1, X_seq.shape[-1]))
    sum_scaler = StandardScaler().fit(X_sum[train_idx])

    def seq(indices):
        values = X_seq[indices]
        flat = seq_scaler.transform(values.reshape(-1, values.shape[-1]))
        return flat.reshape(values.shape).astype(np.float32)

    return {
        "X_sequence_train": seq(train_idx),
        "X_sequence_test": seq(test_idx),
        "X_summary_train": sum_scaler.transform(X_sum[train_idx]).astype(np.float32),
        "X_summary_test": sum_scaler.transform(X_sum[test_idx]).astype(np.float32),
        "y_train": y[train_idx],
        "y_test": y[test_idx],
    }


def aggregate_cv_records(records: Sequence[dict]) -> pd.DataFrame:
    """Pool every (seed, fold) run per model; std is the sample std over runs."""
    frame = pd.DataFrame(list(records))
    metrics = [m for m in METRICS if m in frame.columns]
    rows = []
    for model, group in frame.groupby("model", sort=False):
        row = {
            "model": model,
            "level": group["level"].iloc[0] if "level" in group else "",
            "n_runs": int(len(group)),
            "n_folds": int(group["fold"].nunique()),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def paired_deltas(
    records: Sequence[dict], candidate: str, control: str, metric: str = "f1_macro"
) -> np.ndarray:
    """Candidate-minus-control differences matched on (seed, fold)."""
    frame = pd.DataFrame(list(records))
    a = frame[frame.model == candidate].set_index(["seed", "fold"])[metric]
    b = frame[frame.model == control].set_index(["seed", "fold"])[metric]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError(f"No matched (seed, fold) runs for {candidate} vs {control}")
    return (a.loc[common] - b.loc[common]).to_numpy(dtype=float)


def run_fold(
    dataset: Mapping[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    fold: int,
    models: Sequence[str],
    deep_epochs: int,
    comparison_strength: str,
    proposed_validation_split: float,
    proposed_clipnorm: float,
    n_jobs: int,
) -> List[dict]:
    arrays = _fold_arrays(dataset, train_idx, test_idx)
    specs = build_default_model_specs(
        seed=seed,
        n_jobs=n_jobs,
        deep_epochs=deep_epochs,
        deep_validation_split=0.0,
        comparison_strength=comparison_strength,
        proposed_validation_split=proposed_validation_split,
        proposed_clipnorm=proposed_clipnorm,
    )
    records = []
    for spec in select_model_specs(specs, ",".join(models)):
        estimator = spec.estimator_factory()
        prefix = "X_sequence" if isinstance(estimator, KerasSequenceClassifier) else "X_summary"
        estimator.fit(arrays[f"{prefix}_train"], arrays["y_train"])
        y_true = arrays["y_test"]
        y_pred = estimator.predict(arrays[f"{prefix}_test"])
        record = {
            "model": spec.name,
            "level": spec.level,
            "seed": int(seed),
            "fold": int(fold),
            "n_train_blocks": int(len(train_idx)),
            "n_test_blocks": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "minority_recall": _minority_recall(y_true, np.asarray(y_pred)),
        }
        records.append(record)
        print(
            f"[cv] seed={seed} fold={fold} {spec.name}: "
            f"acc={record['accuracy']:.5f} macro_f1={record['f1_macro']:.5f}",
            flush=True,
        )
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="类型")
    parser.add_argument("--time-col", default="测试时间")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--models", default="default")
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument(
        "--comparison-strength",
        choices=("standard", "light", "reduced_baselines"),
        default="reduced_baselines",
    )
    parser.add_argument("--proposed-validation-split", type=float, default=0.15)
    parser.add_argument("--proposed-clipnorm", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    seeds = tuple(int(v.strip()) for v in args.seeds.split(",") if v.strip())
    models = (
        TEN_MODEL_COMPARISON_MODELS
        if args.models == "default"
        else tuple(v.strip() for v in args.models.split(",") if v.strip())
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = _read_raw(args.data, args.label_col, args.time_col, args.sequence_length)
    block_frame = dataset["block_frame"]
    print(
        f"blocks={len(block_frame)} classes="
        f"{block_frame[args.label_col].value_counts().sort_index().to_dict()}",
        flush=True,
    )

    all_records: List[dict] = []
    for seed in seeds:
        folds = stratified_block_folds(block_frame, args.label_col, args.n_splits, seed)
        for fold, (train_idx, test_idx) in enumerate(folds):
            all_records.extend(
                run_fold(
                    dataset, train_idx, test_idx, seed, fold, models,
                    args.deep_epochs, args.comparison_strength,
                    args.proposed_validation_split, args.proposed_clipnorm, args.n_jobs,
                )
            )
        pd.DataFrame(all_records).to_json(
            output / "cv_records.json", orient="records", force_ascii=False, indent=2
        )

    frame = pd.DataFrame(all_records)
    frame.to_csv(output / "cv_records.csv", index=False, encoding="utf-8-sig")
    summary = aggregate_cv_records(all_records)
    summary.to_csv(output / "cv_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "study_name": "leak_free_block_cross_validation",
        "data_path": str(args.data),
        "label_col": args.label_col,
        "time_col": args.time_col,
        "n_blocks": int(len(block_frame)),
        "n_splits": int(args.n_splits),
        "seeds": list(seeds),
        "models": list(models),
        "deep_epochs": args.deep_epochs,
        "comparison_strength": args.comparison_strength,
        "baseline_capacity": "reduced" if args.comparison_strength == "reduced_baselines" else "standard",
        "proposed_validation_split": args.proposed_validation_split,
        "proposed_clipnorm": args.proposed_clipnorm,
        "runs_per_model": int(args.n_splits * len(seeds)),
        "total_records": len(all_records),
        "split_protocol": "stratified K-fold over whole timestamp blocks; scaler fit on training folds only",
        "evidence_type": "pooled cross-validation estimate for model selection, not a single blind confirmatory test",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nCross-validation summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
