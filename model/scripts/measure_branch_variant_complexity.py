"""Measure DI/AB trainable parameters under the branch-variant budgets.

Read-only diagnostic: fits each proposed variant for a single epoch on the real
preprocessed Design B features purely to build the graph, then reports the
trainable parameter count. Accuracy from this script is meaningless and is not
printed; only complexity is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocess_utils import train_preprocess_and_select  # noqa: E402
from scripts.comparison_models import (  # noqa: E402
    build_default_model_specs,
    model_complexity,
    resolve_data_path,
)

DATA = "数据文件/Public datasets_physics3_designB.csv"


def measure(ab_capacity: str, ab_epochs: int | None, X, y) -> dict:
    specs = build_default_model_specs(
        seed=42, n_jobs=1, deep_epochs=100, deep_validation_split=0.0,
        comparison_strength="designB_reduced_plus",
        proposed_validation_split=0.15, proposed_clipnorm=1.0,
        proposed_capacity="standard", ab_capacity=ab_capacity, ab_epochs=ab_epochs,
        proposed_boundary_weight=0.0, proposed_boundary_margin=0.05,
    )
    out = {}
    for name in ("di_emstgat", "ab_emstgat"):
        est = specs[name].estimator_factory()
        resolved = {"capacity": est.capacity, "epochs": est.epochs}
        est.epochs, est.validation_split, est.verbose = 1, 0.0, 0
        info = model_complexity(est.fit(X, y))
        out[name] = {**resolved, "param_count": int(info["param_count"]),
                     "config_note": info["config_note"]}
    return out


def main() -> int:
    data = train_preprocess_and_select(
        data_path=resolve_data_path(DATA), test_size=0.2, seed=42,
        label_col=None, importance_threshold=0.95,
    )
    X, y = data["X_train"], data["y_train"]

    branch = measure("compact", 100, X, y)
    locked = measure("micro", None, X, y)

    print("DesignB_branch_v1 (AB=compact/100):")
    for k, v in branch.items():
        print(f"  {k:12s} {v['capacity']:9s} {v['epochs']:>4d}ep  {v['param_count']:>9,d} params")
    print("DesignB_plus_v1 (AB=micro/1, 旧锁定):")
    for k, v in locked.items():
        print(f"  {k:12s} {v['capacity']:9s} {v['epochs']:>4d}ep  {v['param_count']:>9,d} params")

    di = branch["di_emstgat"]["param_count"]
    ab = branch["ab_emstgat"]["param_count"]
    print(f"\nAB(compact) / DI(standard) = {ab / di:.4f}  "
          f"-> AB uses {ab / di * 100:.1f}% of DI's parameters "
          f"({(1 - ab / di) * 100:.1f}% fewer)")

    dest = PROJECT_ROOT / "results" / "studies" / "physics3_designB_branch_v1" / "branch_variant_complexity.json"
    dest.write_text(json.dumps({
        "feature_count": int(len(data["feature_names"])),
        "train_samples": int(len(y)),
        "DesignB_branch_v1": branch,
        "DesignB_plus_v1_locked": locked,
        "ab_over_di_param_ratio": round(ab / di, 6),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
