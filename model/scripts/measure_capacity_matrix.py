"""Measure DI/AB trainable parameters at every capacity tier (train-only).

Read-only with respect to the test split: each estimator is fitted for a single
epoch on the TRAINING features purely to build the Keras graph, then its
trainable parameter count is read. The test set is never touched, so this can
inform a capacity decision without consuming the already-viewed test split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocess_utils import train_preprocess_and_select  # noqa: E402
from scripts.comparison_models import (  # noqa: E402
    KerasSequenceClassifier,
    model_complexity,
    resolve_data_path,
)

DATA = "数据文件/Public datasets_physics3_designB.csv"
TIERS = ("micro", "light", "compact", "standard")
CEO_LR = 0.000813
CEO_WD = 0.000427


def main() -> int:
    data = train_preprocess_and_select(
        data_path=resolve_data_path(DATA), test_size=0.2, seed=42,
        label_col=None, importance_threshold=0.95,
    )
    X, y = data["X_train"], data["y_train"]
    print(f"train rows={len(y)}  features={len(data['feature_names'])}  "
          "(test split NOT touched)\n")

    table = {}
    for kind in ("di_emstgat", "ab_emstgat"):
        table[kind] = {}
        for tier in TIERS:
            est = KerasSequenceClassifier(
                model_kind=kind, epochs=1, validation_split=0.0, seed=42,
                verbose=0, capacity=tier, learning_rate=CEO_LR,
                weight_decay=CEO_WD, clipnorm=1.0, boundary_weight=0.0,
            )
            info = model_complexity(est.fit(X, y))
            table[kind][tier] = int(info["param_count"])

    print(f"{'capacity':10s} {'DI(raw)':>12s} {'AB(auto)':>12s} {'AB/DI':>8s}  branch overhead")
    print("-" * 66)
    for tier in TIERS:
        di = table["di_emstgat"][tier]
        ab = table["ab_emstgat"][tier]
        print(f"{tier:10s} {di:>12,d} {ab:>12,d} {ab / di:>7.3f}  {ab - di:+,d}")

    print("\n跨容量参照（当前主表）:")
    print(f"  DI standard = {table['di_emstgat']['standard']:,d}")
    print(f"  AB compact  = {table['ab_emstgat']['compact']:,d}  "
          f"({table['ab_emstgat']['compact'] / table['di_emstgat']['standard'] * 100:.1f}% of DI standard)")

    dest = PROJECT_ROOT / "results" / "studies" / "capacity_parameter_matrix.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "note": "train-only graph build; test split not evaluated",
        "train_rows": int(len(y)),
        "feature_count": int(len(data["feature_names"])),
        "param_counts": table,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
