"""Does the robustness advantage survive EQUAL compute?

WHY THIS IS THE DECISIVE EXPERIMENT
  Clean accuracy under equal compute goes to the tree models (XGBoost/LightGBM
  ~99.6% vs AB 98.9%), so the thesis cannot claim first place there. The
  fallback claim is robustness: DI beat PatchTST by 2.07-2.44 pp at 15/10/5 dB
  with p<0.05, 5/5 seeds.

  But that noise study also used reduced baselines. If the robustness advantage
  disappears once baselines get the full 100-epoch budget, then the fallback
  claim collapses too and the thesis needs a different framing. This script
  settles it.

PROTOCOL
  Identical to run_noise_robustness: train once on clean data, evaluate the same
  fitted model on clean input and across the SNR grid. The ONLY change is
  comparison_strength="standard", i.e. every baseline gets the same 100 epochs
  as the proposed models. Noise draws are keyed per (seed, snr) exactly as in
  the original study so the two are directly comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.comparison_models import (
    build_default_model_specs, resolve_data_path, select_model_specs,
)
from scripts.run_main_comparison import _load_main_protocol_data
from scripts.run_noise_robustness import (
    add_gaussian_noise, aggregate_noise_records, evaluate_under_noise,
)

DEFAULT_GRID = (40, 35, 30, 25, 20, 15, 10, 5)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_MODELS = ("di_emstgat", "ab_emstgat", "patchtst", "mtgnn", "tcn",
                  "xgboost", "lightgbm", "itransformer", "transformer", "gatv2")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",
                        default="数据文件/Public datasets_physics3_designB.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--snr-grid", default=",".join(map(str, DEFAULT_GRID)))
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--proposed-validation-split", type=float, default=0.0,
        help="Train-internal validation fraction for DI/AB early stopping. "
             "Defaults to 0.0 so the proposed models train on ALL 6799 rows, "
             "the same as the tree baselines. The historical 0.15 removed 15%% "
             "of the training data from the proposed models only, costing a "
             "measured 0.4314 pp -- a protocol handicap, not a design choice.")
    args = parser.parse_args(argv)

    seeds = tuple(int(v) for v in args.seeds.split(",") if v.strip())
    models = tuple(v.strip() for v in args.models.split(",") if v.strip())
    grid = tuple(float(v) for v in args.snr_grid.split(",") if v.strip())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    for seed in seeds:
        data = _load_main_protocol_data(args.data, test_size=args.test_size,
                                        seed=seed)
        specs = build_default_model_specs(
            seed=seed, n_jobs=1, deep_epochs=args.deep_epochs,
            deep_validation_split=0.0,
            # THE key difference from the original study: no baseline de-tuning.
            comparison_strength="standard",
            proposed_validation_split=args.proposed_validation_split,
            proposed_clipnorm=1.0,
            proposed_capacity="standard", ab_capacity="compact",
            proposed_boundary_weight=0.05, proposed_boundary_margin=0.05,
        )
        for spec in select_model_specs(specs, ",".join(models)):
            est = spec.estimator_factory()
            est.fit(data["X_train"], data["y_train"])
            for snr in [None, *grid]:
                metrics = evaluate_under_noise(
                    est, data["X_test"], data["y_test"], snr,
                    noise_seed=seed * 1000 + int(snr if snr is not None else 999))
                records.append({
                    "model": spec.name, "level": spec.level, "seed": int(seed),
                    "snr_db": "clean" if snr is None else float(snr),
                    "comparison_strength": "standard_equal_compute",
                    **metrics,
                })
                print(f"[equal-noise] seed={seed} {spec.name} "
                      f"snr={records[-1]['snr_db']}: "
                      f"acc={metrics['accuracy']:.6f}", flush=True)
            pd.DataFrame(records).to_json(
                output / "per_seed_records.json", orient="records",
                force_ascii=False, indent=2)

    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_seed_records.csv", index=False,
                 encoding="utf-8-sig")
    aggregate_noise_records(records).to_csv(
        output / "aggregate_summary.csv", index=False, encoding="utf-8-sig")
    (output / "manifest.json").write_text(json.dumps({
        "study_name": "noise_robustness_equal_compute",
        "data_path": resolve_data_path(args.data),
        "seeds": list(seeds), "models": list(models),
        "snr_grid_db": list(grid), "deep_epochs": args.deep_epochs,
        "comparison_strength": "standard",
        "proposed_validation_split": args.proposed_validation_split,
        "validation_split_note": (
            "0.0 means the proposed models train on all 6799 rows, matching the "
            "tree baselines. The historical 0.15 handicapped only the proposed "
            "models by 15% of the training data, measured cost 0.4314 pp."),
        "difference_from_reduced_study": (
            "baselines receive the SAME epoch budget and capacity as the "
            "proposed models; everything else is identical, so this tests "
            "whether the robustness advantage depends on baseline de-tuning"),
        "noise_applied_to": "evaluation inputs only",
        "records": len(records),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
