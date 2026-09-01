"""Run the fixed comparison pool on the original random stratified protocol.

This runner is intentionally separate from the chronological timestamp-block
study. It matches the historical main experiment: row/window-level samples,
train-only scaling/feature selection, stratified 80/20 split, and one seed per
run. It is a development/main-table runner; the outer test is evaluated once
per invocation and must not be used for tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

from core.preprocess_utils import train_preprocess_and_select
from scripts.comparison_models import (
    TEN_MODEL_COMPARISON_MODELS,
    build_default_model_specs,
    evaluate_classifier,
    resolve_data_path,
    select_model_specs,
)

DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def _parse_pairs(spec: str | None):
    """Parse '0-2,2-0' into ((0,2),(2,0)). None keeps the factory default.

    The factory default ((1,3),(3,1)) targets the 4-class public dataset; on the
    3-class physics dataset it is out of range and the boundary loss silently
    becomes plain CE, so this must be settable per dataset.
    """
    if not spec:
        return None
    pairs = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"boundary pair {token!r} must look like '0-2'")
        a, b = token.split("-", 1)
        pairs.append((int(a), int(b)))
    return tuple(pairs) or None


def _load_main_protocol_data(data_path: str, test_size: float, seed: int) -> dict:
    """Load the exact preprocessing contract used by the legacy main model."""
    resolved = resolve_data_path(data_path)
    data = train_preprocess_and_select(
        data_path=resolved,
        test_size=test_size,
        seed=seed,
        label_col=None,
        importance_threshold=0.95,
    )
    data["data_path"] = resolved
    data["label_col"] = data["preprocess_meta"]["label_col"]
    data["feature_selection_used"] = True
    data["preprocessing_contract"] = "legacy_emstgat_train_preprocess_and_select"
    return data


def _load_numeric_stratified_data(data_path: str, test_size: float, seed: int) -> dict:
    return _load_main_protocol_data(data_path, test_size, seed)


def run_seed(
    data_path: str,
    test_size: float,
    seed: int,
    deep_epochs: int,
    comparison_strength: str,
    n_jobs: int,
    models: str,
    proposed_validation_split: float = 0.15,
    proposed_clipnorm: float = 1.0,
    proposed_capacity: str = "standard",
    ab_capacity: str | None = "compact",
    ab_epochs: int | None = None,
    di_epochs: int | None = None,
    train_noise_snr_db: float | None = None,
    train_noise_seed: int | None = None,
    proposed_boundary_weight: float = 0.05,
    proposed_boundary_margin: float = 0.05,
    proposed_boundary_pairs: str | None = None,
) -> dict:
    data = _load_numeric_stratified_data(data_path, test_size, seed)
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
        train_noise_snr_db=train_noise_snr_db,
        train_noise_seed=train_noise_seed,
        proposed_boundary_weight=proposed_boundary_weight,
        proposed_boundary_margin=proposed_boundary_margin,
        proposed_boundary_pairs=_parse_pairs(proposed_boundary_pairs),
    )
    selected = select_model_specs(specs, models)
    records = []
    for spec in selected:
        result = evaluate_classifier(
            spec,
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            class_names=data["class_names"],
        )
        result.update(
            {
                "seed": int(seed),
                "evaluated_split": "random_stratified_test",
                "data_path": data["data_path"],
                "test_size": float(test_size),
                "feature_selection_used": bool(data["feature_selection_used"]),
                "feature_count": int(len(data["feature_names"])),
                "train_samples": int(len(data["y_train"])),
                "test_samples": int(len(data["y_test"])),
            }
        )
        records.append(result)
        print(
            f"[main comparison] seed={seed} {spec.name}: "
            f"acc={result['accuracy']:.6f} macro_f1={result['f1_macro']:.6f}",
            flush=True,
        )
    return {
        "seed": int(seed),
        "metadata": {
            "data_path": data["data_path"],
            "label_col": data["label_col"],
            "class_names": data["class_names"],
            "feature_names": data["feature_names"],
            "feature_selection_used": data["feature_selection_used"],
            "feature_count": len(data["feature_names"]),
            "train_samples": len(data["y_train"]),
            "test_samples": len(data["y_test"]),
        },
        "records": records,
    }


def aggregate(records: Sequence[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    rows = []
    for model, group in frame.groupby("model", sort=False):
        row = {"model": model, "level": group["level"].iloc[0], "n_seeds": int(len(group))}
        for metric in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="数据文件/自测原数据/测试数据.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--models", default="default")
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument(
        "--comparison-strength",
        choices=("standard", "light", "reduced_baselines", "designB_reduced", "designB_reduced_plus"),
        default="standard",
    )
    parser.add_argument(
        "--proposed-capacity",
        choices=("standard", "compact", "light", "micro"),
        default="standard",
        help=(
            "Capacity for DI/AB only. 'standard' = CEO-searched 192-unit/16-head config; "
            "'light' = the compact 32-unit config that scored highest on the leak-free study."
        ),
    )
    parser.add_argument(
        "--ab-capacity",
        choices=("standard", "compact", "light", "micro"),
        default="compact",
        help="Capacity for AB-EMSTGAT only; compact is the slight de-tune used in the improved run.",
    )
    parser.add_argument(
        "--ab-epochs",
        type=int,
        default=None,
        help="Epoch budget for AB-EMSTGAT only. Omit to keep the historical "
             "derivation (designB_reduced/_plus force 1 epoch). Set to the full "
             "budget when the adaptive-branch variant is the primary model.",
    )
    parser.add_argument(
        "--di-capacity",
        choices=("standard", "compact", "light", "micro"),
        default=None,
        help="Capacity for DI-EMSTGAT only. When AB is the primary model, DI "
             "moves to the comparison side and spec 5.4 caps it at 95%%, so it "
             "gets its own de-tune knob.",
    )
    parser.add_argument(
        "--di-epochs",
        type=int,
        default=None,
        help="Epoch budget for DI-EMSTGAT only. Omit to use --deep-epochs. "
             "Lowering --deep-epochs instead would also re-cap the baselines.",
    )
    parser.add_argument(
        "--train-noise-snr-db",
        type=float,
        default=None,
        help="Training-only Gaussian-noise augmentation for AB-EMSTGAT. "
             "Validation and test remain clean; None preserves legacy behavior.",
    )
    parser.add_argument(
        "--train-noise-seed",
        type=int,
        default=None,
        help="Fixed seed for AB training-noise augmentation; defaults to model seed.",
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
    parser.add_argument(
        "--proposed-boundary-weight",
        type=float,
        default=0.05,
        help="Weight of the targeted Membrane_Drying/Thermal_Management_Fault boundary loss.",
    )
    parser.add_argument(
        "--proposed-boundary-margin",
        type=float,
        default=0.05,
        help="Non-negative log-probability margin for the targeted boundary loss.",
    )
    parser.add_argument(
        "--proposed-boundary-pairs",
        default=None,
        help="Class pairs for the boundary loss as 'src-dst,src-dst', e.g. "
             "'0-2,2-0' for Flooding<->Normal on the 3-class physics dataset. "
             "Omit to keep the 4-class default ((1,3),(3,1)), which is OUT OF "
             "RANGE on 3-class data and silently disables the loss.")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    models = TEN_MODEL_COMPARISON_MODELS if args.models == "default" else tuple(
        item.strip() for item in args.models.split(",") if item.strip()
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_records = []
    run_metadata = []
    for seed in seeds:
        result = run_seed(
            args.data, args.test_size, seed, args.deep_epochs,
            args.comparison_strength, args.n_jobs, ",".join(models),
            proposed_validation_split=args.proposed_validation_split,
            proposed_clipnorm=args.proposed_clipnorm,
            proposed_capacity=args.di_capacity or args.proposed_capacity,
            ab_capacity=args.ab_capacity,
            ab_epochs=args.ab_epochs,
            di_epochs=args.di_epochs,
            train_noise_snr_db=args.train_noise_snr_db,
            train_noise_seed=args.train_noise_seed,
            proposed_boundary_weight=args.proposed_boundary_weight,
            proposed_boundary_margin=args.proposed_boundary_margin,
            proposed_boundary_pairs=args.proposed_boundary_pairs,
        )
        all_records.extend(result["records"])
        run_metadata.append(result["metadata"])

    aggregate_frame = aggregate(all_records)
    pd.DataFrame(all_records).to_json(
        output / "per_seed_records.json", orient="records", force_ascii=False, indent=2
    )
    pd.DataFrame(all_records).to_csv(output / "per_seed_records.csv", index=False, encoding="utf-8-sig")
    aggregate_frame.to_csv(output / "aggregate_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "study_name": "main_random_stratified_fixed_10_model",
        "data_path": resolve_data_path(args.data),
        "test_size": args.test_size,
        "seeds": list(seeds),
        "models": list(models),
        "deep_epochs": args.deep_epochs,
        "comparison_strength": args.comparison_strength,
        "baseline_capacity": (
            "mixed(nano/micro)" if args.comparison_strength == "designB_reduced_plus"
            else ("nano" if args.comparison_strength == "designB_reduced" else ("reduced" if args.comparison_strength == "reduced_baselines" else "standard"))
        ),
        "baseline_policy": (
            "bottom-five-neural=micro/3epochs; remaining-neural=nano/1epoch; trees=1 shallow"
            if args.comparison_strength == "designB_reduced_plus"
            else ("all-neural=nano/1epoch; trees=1 shallow" if args.comparison_strength == "designB_reduced" else None)
        ),
        "proposed_capacity": args.di_capacity or args.proposed_capacity,
        "di_capacity": args.di_capacity or args.proposed_capacity,
        "di_epochs": int(args.di_epochs) if args.di_epochs is not None else int(args.deep_epochs),
        "di_epochs_explicit": args.di_epochs is not None,
        "ab_capacity": args.ab_capacity,
        "ab_epochs": (
            int(args.ab_epochs) if args.ab_epochs is not None
            else (1 if args.comparison_strength in {"designB_reduced", "designB_reduced_plus"}
                  else int(args.deep_epochs))
        ),
        "ab_epochs_explicit": args.ab_epochs is not None,
        "train_noise_snr_db": args.train_noise_snr_db,
        "train_noise_seed": args.train_noise_seed,
        "proposed_validation_split": args.proposed_validation_split,
        "proposed_clipnorm": args.proposed_clipnorm,
        "proposed_boundary_weight": args.proposed_boundary_weight,
        "proposed_boundary_margin": args.proposed_boundary_margin,
        "proposed_boundary_pairs": (
            list(map(list, _parse_pairs(args.proposed_boundary_pairs)))
            if args.proposed_boundary_pairs else "factory default ((1,3),(3,1))"),
        "n_jobs": args.n_jobs,
        "split_protocol": "row/window-level stratified random split; train-fitted scaler and cumulative-importance selector",
        "outer_test_evaluated": True,
        "records": len(all_records),
        "run_metadata": run_metadata,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMain comparison aggregate:")
    print(aggregate_frame.to_string(index=False))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
