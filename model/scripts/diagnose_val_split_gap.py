"""Why does AB score 99.76% in one script and 98.59% in the main table?

Same model, same capacity, same epochs, same seed, same data -> 1.18 pp apart.
That gap is larger than every effect this thesis reports, so it must be
explained before any tuning decision is made.

Suspects, tested one at a time:
  S1  validation_split. The proposed models use validation_split=0.15 for early
      stopping, which REMOVES 15% of the training rows AND makes the fitted
      model depend on which rows land in that split.
  S2  early stopping restoring the best epoch vs the last epoch.
  S3  plain run-to-run nondeterminism inside TensorFlow, even with a fixed seed.

The diagnosis matters: if S3 dominates, then a 5-seed protocol is measuring
noise, and the whole comparison needs more seeds or a deterministic config. If
S1 dominates, the proposed models are being handicapped by losing 15% of their
training data while tree baselines keep 100%.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from scripts.comparison_models import KerasSequenceClassifier
from scripts.run_main_comparison import _load_main_protocol_data

DATA = "数据文件/Public datasets_physics3_designB.csv"


def fit_score(seed: int, val_split: float, tag: str, data) -> float:
    est = KerasSequenceClassifier(
        model_kind="ab_emstgat", epochs=100, learning_rate=8.13e-4,
        weight_decay=4.27e-4, clipnorm=1.0, validation_split=val_split,
        seed=seed, capacity="compact", boundary_weight=0.05,
        boundary_margin=0.05,
    )
    est.fit(data["X_train"], data["y_train"])
    acc = accuracy_score(data["y_test"], est.predict(data["X_test"]))
    tr = accuracy_score(data["y_train"], est.predict(data["X_train"]))
    print(f"  {tag:34s} test={100*acc:8.4f}%  train={100*tr:8.4f}%")
    return acc


def main() -> None:
    data = _load_main_protocol_data(DATA, test_size=0.2, seed=42)
    print(f"data: train={len(data['y_train'])} test={len(data['y_test'])} "
          f"features={len(data['feature_names'])}")

    print("\n" + "=" * 74)
    print("S3 first: is a FIXED seed even reproducible? (3 identical fits)")
    print("=" * 74)
    same = [fit_score(42, 0.15, f"seed=42 val=0.15 repeat {k+1}", data)
            for k in range(3)]
    spread = 100 * (max(same) - min(same))
    print(f"  spread across identical runs = {spread:.4f} pp")
    verdict = ("NONDETERMINISTIC: seed does not pin the result"
               if spread > 0.05 else "reproducible")
    print(f"  -> {verdict}")

    print("\n" + "=" * 74)
    print("S1: does validation_split cost accuracy? (same seed, 0.15 vs 0.0)")
    print("=" * 74)
    with_val = [fit_score(s, 0.15, f"seed={s} val=0.15", data)
                for s in (42, 43, 44)]
    print()
    no_val = [fit_score(s, 0.0, f"seed={s} val=0.00", data)
              for s in (42, 43, 44)]
    print(f"\n  mean with val=0.15 : {100*np.mean(with_val):8.4f}%")
    print(f"  mean with val=0.00 : {100*np.mean(no_val):8.4f}%")
    print(f"  difference         : {100*(np.mean(no_val)-np.mean(with_val)):+8.4f} pp")
    print("  Note: val=0.15 trains on 5779 rows, val=0.00 trains on all 6799,")
    print("  while XGBoost/LightGBM always train on all 6799. That is a real")
    print("  handicap, not a modelling choice the baselines share.")


if __name__ == "__main__":
    main()
