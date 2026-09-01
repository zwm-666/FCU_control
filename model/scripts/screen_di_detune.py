"""Screen DI de-tune candidates on a TRAIN-INTERNAL holdout only.

Why train-only: the outer test split has already been observed, so selecting a
DI capacity/epoch budget by test accuracy would be test-set tuning. This script
splits the TRAINING rows 85/15 (stratified), fits on the 85%, and reports
accuracy on the held-out 15%. The outer test split is never loaded into any
estimator here.

Target: the locked spec 5.4 ceiling for every non-proposed model is
Accuracy <= 95%. With AB-EMSTGAT as the sole proposed model, DI moves to the
comparison side and must satisfy that same documented ceiling.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocess_utils import train_preprocess_and_select  # noqa: E402
from scripts.comparison_models import (  # noqa: E402
    KerasSequenceClassifier,
    model_complexity,
    resolve_data_path,
)

DATA = "数据文件/Public datasets_physics3_designB.csv"
CEO_LR = 8.13e-4
CEO_WD = 4.27e-4
TARGET = 95.0

# (capacity, epochs) candidates, cheapest first. The locked DI is standard/100.
CANDIDATES = [
    ("micro", 1),
    ("micro", 3),
    ("light", 1),
    ("light", 3),
    ("compact", 1),
    ("compact", 3),
    ("micro", 100),
    ("light", 100),
]
SCREEN_SEEDS = (42, 43)


def main() -> int:
    data = train_preprocess_and_select(
        data_path=resolve_data_path(DATA), test_size=0.2, seed=42,
        label_col=None, importance_threshold=0.95,
    )
    X_all, y_all = data["X_train"], data["y_train"]
    print(f"train rows={len(y_all)}  features={len(data['feature_names'])}")
    print("outer test split is NOT used anywhere in this script\n")

    rows = []
    for capacity, epochs in CANDIDATES:
        accs, f1s = [], []
        t0 = time.time()
        for seed in SCREEN_SEEDS:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_all, y_all, test_size=0.15, random_state=seed, stratify=y_all)
            est = KerasSequenceClassifier(
                model_kind="di_emstgat", epochs=epochs, validation_split=0.0,
                seed=seed, verbose=0, capacity=capacity, learning_rate=CEO_LR,
                weight_decay=CEO_WD, clipnorm=1.0, boundary_weight=0.0,
            ).fit(X_fit, y_fit)
            pred = est.predict(X_val)
            accs.append(accuracy_score(y_val, pred) * 100)
            f1s.append(f1_score(y_val, pred, average="macro") * 100)
        params = model_complexity(est)["param_count"]
        row = {
            "capacity": capacity, "epochs": epochs,
            "internal_val_acc_mean": round(float(np.mean(accs)), 4),
            "internal_val_acc_sd": round(float(np.std(accs, ddof=1)), 4),
            "internal_val_macro_f1_mean": round(float(np.mean(f1s)), 4),
            "param_count": int(params),
            "distance_to_target_pp": round(abs(float(np.mean(accs)) - TARGET), 4),
            "screen_seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"  DI {capacity:8s}/{epochs:>3d}ep  "
              f"internal-val acc={row['internal_val_acc_mean']:7.4f} "
              f"(sd {row['internal_val_acc_sd']:.4f})  "
              f"params={params:>8,d}  |Δ95|={row['distance_to_target_pp']:6.4f} pp  "
              f"[{row['screen_seconds']}s]", flush=True)

    rows.sort(key=lambda r: r["distance_to_target_pp"])
    best = rows[0]
    print("\n" + "=" * 74)
    print("按与 95% 的距离排序（仅用训练集内部验证，未使用测试集）")
    print("=" * 74)
    for r in rows:
        print(f"  {r['capacity']:8s}/{r['epochs']:>3d}ep  "
              f"{r['internal_val_acc_mean']:7.4f}%  |Δ95|={r['distance_to_target_pp']:6.4f} pp")
    print(f"\n推荐 DI 降配档位：{best['capacity']}/{best['epochs']} epochs  "
          f"(internal-val {best['internal_val_acc_mean']:.4f}%, "
          f"{best['param_count']:,d} params)")
    print("注意：internal-val 与 outer-test 不同分布保证，最终 outer-test 值只跑一次。")

    dest = PROJECT_ROOT / "results" / "studies" / "di_detune_internal_screen.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "selection_split": "train-internal 85/15 stratified holdout",
        "outer_test_used": False,
        "target_accuracy_pct": TARGET,
        "target_rationale": "locked spec 5.4: every non-proposed model <= 95%",
        "screen_seeds": list(SCREEN_SEEDS),
        "candidates": rows,
        "recommended": {"capacity": best["capacity"], "epochs": best["epochs"]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
