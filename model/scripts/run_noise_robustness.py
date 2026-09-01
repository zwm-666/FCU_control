"""Noise robustness study for the proposed DI/AB-EMSTGAT models.

Protocol (per the thesis work plan):
  * train once per (model, seed) on clean training data;
  * evaluate the same fitted model on the clean test set and on the full
    40 -> 5 dB SNR grid;
  * report multi-seed mean +/- sample standard deviation, never a single run.

Noise is added only to the evaluation inputs; training data is never noised,
so this measures deployment-time sensor degradation rather than augmentation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

from scripts.comparison_models import build_default_model_specs, resolve_data_path, select_model_specs
from scripts.run_main_comparison import _load_main_protocol_data

DEFAULT_SNR_GRID_DB = (40, 35, 30, 25, 20, 15, 10, 5)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_MODELS = ("di_emstgat", "ab_emstgat")


def add_gaussian_noise(X: np.ndarray, snr_db: Optional[float], seed: int) -> np.ndarray:
    """Add zero-mean Gaussian noise at the requested global SNR (dB).

    ``snr_db=None`` returns the clean array unchanged.  The noise standard
    deviation is derived from the mean signal power of ``X`` so the realised
    SNR matches the request, and the draw is deterministic for a given seed.
    """
    X = np.asarray(X, dtype=np.float32)
    if snr_db is None:
        return X
    signal_power = float(np.mean(np.square(X)))
    if signal_power <= 0.0:
        return X
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=X.shape)
    return (X + noise).astype(np.float32)


def _minority_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes, counts = np.unique(y_true, return_counts=True)
    minority = int(classes[int(np.argmin(counts))])
    per_class = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    return float(per_class[list(classes).index(minority)])


def evaluate_under_noise(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    snr_db: Optional[float],
    noise_seed: int,
) -> Dict[str, float]:
    X_noisy = add_gaussian_noise(X_test, snr_db, noise_seed)
    y_pred = estimator.predict(X_noisy)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "minority_recall": _minority_recall(np.asarray(y_test), np.asarray(y_pred)),
    }


def aggregate_noise_records(records: Sequence[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    metrics = [
        column
        for column in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "minority_recall")
        if column in frame.columns
    ]
    rows: List[dict] = []
    for (model, snr_db), group in frame.groupby(["model", "snr_db"], dropna=False, sort=False):
        row = {"model": model, "snr_db": snr_db, "n_seeds": int(len(group))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["model", "snr_db"], ascending=[True, False]).reset_index(drop=True)


def run_seed(
    data_path: str,
    test_size: float,
    seed: int,
    deep_epochs: int,
    comparison_strength: str,
    models: Sequence[str],
    snr_grid: Sequence[float],
    n_jobs: int,
    proposed_validation_split: float,
    proposed_clipnorm: float,
    proposed_capacity: Optional[str] = None,
    ab_capacity: Optional[str] = None,
    ab_epochs: Optional[int] = None,
    di_epochs: Optional[int] = None,
    proposed_boundary_weight: float = 0.0,
    proposed_boundary_margin: float = 0.05,
) -> List[dict]:
    data = _load_main_protocol_data(data_path, test_size=test_size, seed=seed)
    specs = build_default_model_specs(
        seed=seed,
        n_jobs=n_jobs,
        deep_epochs=deep_epochs,
        deep_validation_split=0.0,
        comparison_strength=comparison_strength,
        proposed_validation_split=proposed_validation_split,
        proposed_clipnorm=proposed_clipnorm,
        proposed_capacity=proposed_capacity,
        ab_capacity=ab_capacity,
        ab_epochs=ab_epochs,
        di_epochs=di_epochs,
        proposed_boundary_weight=proposed_boundary_weight,
        proposed_boundary_margin=proposed_boundary_margin,
    )
    selected = select_model_specs(specs, ",".join(models))
    records: List[dict] = []
    for spec in selected:
        estimator = spec.estimator_factory()
        estimator.fit(data["X_train"], data["y_train"])
        for snr_db in [None, *snr_grid]:
            metrics = evaluate_under_noise(
                estimator,
                data["X_test"],
                data["y_test"],
                snr_db,
                noise_seed=seed * 1000 + int(snr_db if snr_db is not None else 999),
            )
            record = {
                "model": spec.name,
                "level": spec.level,
                "seed": int(seed),
                "snr_db": "clean" if snr_db is None else float(snr_db),
                "evaluated_split": "random_stratified_test",
                **metrics,
            }
            records.append(record)
            print(
                f"[noise] seed={seed} {spec.name} snr={record['snr_db']}: "
                f"acc={metrics['accuracy']:.6f} macro_f1={metrics['f1_macro']:.6f} "
                f"minority_recall={metrics['minority_recall']:.6f}",
                flush=True,
            )
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="数据文件/自测原数据/测试数据.xlsx")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--snr-grid", default=",".join(str(v) for v in DEFAULT_SNR_GRID_DB))
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument(
        "--comparison-strength",
        choices=("standard", "light", "reduced_baselines", "designB_reduced",
                 "designB_reduced_plus"),
        default="reduced_baselines",
        help="Use 'designB_reduced_plus' to match the locked DesignB_plus_v1 "
             "baseline budget from 实验配置与方法说明.md section 3.1.")
    parser.add_argument(
        "--proposed-capacity", choices=("standard", "compact", "light", "micro"),
        default=None, help="DI/AB capacity; locked config uses 'standard'.")
    parser.add_argument(
        "--ab-capacity", choices=("standard", "compact", "light", "micro"),
        default=None, help="AB capacity only; DesignB_plus_v1 used 'micro', "
                           "DesignB_branch_v1 uses 'compact'.")
    parser.add_argument(
        "--ab-epochs", type=int, default=None,
        help="AB epoch budget. Omit to keep the historical derivation "
             "(designB_reduced/_plus force 1). Branch variant uses 100.")
    parser.add_argument(
        "--di-epochs", type=int, default=None,
        help="DI epoch budget only. Omit to use --deep-epochs. Needed because "
             "lowering --deep-epochs would also re-cap the baselines.")
    parser.add_argument(
        "--proposed-boundary-weight", type=float, default=0.0,
        help="Locked config keeps this at 0.0 (spec 3.2: boundary loss off).")
    parser.add_argument(
        "--proposed-boundary-margin", type=float, default=0.05)
    parser.add_argument("--proposed-validation-split", type=float, default=0.15)
    parser.add_argument("--proposed-clipnorm", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    seeds = tuple(int(v.strip()) for v in args.seeds.split(",") if v.strip())
    models = tuple(v.strip() for v in args.models.split(",") if v.strip())
    snr_grid = tuple(float(v.strip()) for v in args.snr_grid.split(",") if v.strip())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    for seed in seeds:
        all_records.extend(
            run_seed(
                args.data, args.test_size, seed, args.deep_epochs,
                args.comparison_strength, models, snr_grid, args.n_jobs,
                args.proposed_validation_split, args.proposed_clipnorm,
                proposed_capacity=args.proposed_capacity,
                ab_capacity=args.ab_capacity,
                ab_epochs=args.ab_epochs,
                di_epochs=args.di_epochs,
                proposed_boundary_weight=args.proposed_boundary_weight,
                proposed_boundary_margin=args.proposed_boundary_margin,
            )
        )
        pd.DataFrame(all_records).to_json(
            output / "per_seed_records.json", orient="records", force_ascii=False, indent=2
        )

    frame = pd.DataFrame(all_records)
    frame.to_csv(output / "per_seed_records.csv", index=False, encoding="utf-8-sig")
    summary = aggregate_noise_records(all_records)
    summary.to_csv(output / "aggregate_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "study_name": "noise_robustness_full_snr_grid",
        "data_path": resolve_data_path(args.data),
        "test_size": args.test_size,
        "seeds": list(seeds),
        "models": list(models),
        "snr_grid_db": list(snr_grid),
        "clean_evaluation_included": True,
        "deep_epochs": args.deep_epochs,
        "comparison_strength": args.comparison_strength,
        "proposed_capacity": args.proposed_capacity,
        "ab_capacity": args.ab_capacity,
        "ab_epochs": (int(args.ab_epochs) if args.ab_epochs is not None
                      else (1 if args.comparison_strength in {"designB_reduced", "designB_reduced_plus"}
                            else int(args.deep_epochs))),
        "ab_epochs_explicit": args.ab_epochs is not None,
        "di_epochs": int(args.di_epochs) if args.di_epochs is not None else int(args.deep_epochs),
        "di_epochs_explicit": args.di_epochs is not None,
        "proposed_boundary_weight": args.proposed_boundary_weight,
        "proposed_boundary_margin": args.proposed_boundary_margin,
        "proposed_validation_split": args.proposed_validation_split,
        "proposed_clipnorm": args.proposed_clipnorm,
        "noise_applied_to": "evaluation inputs only (training data stays clean)",
        "records": len(all_records),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nNoise robustness aggregate:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
