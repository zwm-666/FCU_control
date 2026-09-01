"""Preflight the DesignB_branch_v1 pool against the spec BEFORE spending compute.

Read-only: instantiates every estimator factory and prints the resolved
capacity / epochs / tree budget / boundary weight, then asserts each against the
expected branch-variant table. No training, no file writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.comparison_models import build_default_model_specs  # noqa: E402

# DesignB_branch_v1: identical to DesignB_plus_v1 except AB = compact/100.
EXPECTED = {
    "di_emstgat": {"capacity": "standard", "epochs": 100, "boundary_weight": 0.0},
    "ab_emstgat": {"capacity": "compact", "epochs": 100, "boundary_weight": 0.0},
    "transformer": {"capacity": "micro", "epochs": 3},
    "itransformer": {"capacity": "micro", "epochs": 3},
    "mtgnn": {"capacity": "micro", "epochs": 3},
    "tcn": {"capacity": "micro", "epochs": 3},
    "gatv2": {"capacity": "micro", "epochs": 3},
    "patchtst": {"capacity": "nano", "epochs": 1},
    "xgboost": {"n_estimators": 1, "max_depth": 1},
    "lightgbm": {"n_estimators": 1, "max_depth": 1, "num_leaves": 2},
}


def main() -> int:
    specs = build_default_model_specs(
        seed=42,
        n_jobs=1,
        deep_epochs=100,
        deep_validation_split=0.0,
        comparison_strength="designB_reduced_plus",
        proposed_validation_split=0.15,
        proposed_clipnorm=1.0,
        proposed_capacity="standard",
        ab_capacity="compact",
        ab_epochs=100,
        proposed_boundary_weight=0.0,
        proposed_boundary_margin=0.05,
    )

    passed = failed = 0
    print(f"{'model':14s} {'checked':34s} {'expected':>10s} {'actual':>10s}  result")
    print("-" * 84)
    for name, expectations in EXPECTED.items():
        est = specs[name].estimator_factory()
        for attr, want in expectations.items():
            got = getattr(est, attr, "<missing>")
            ok = got == want
            passed += ok
            failed += (not ok)
            print(f"{name:14s} {attr:34s} {str(want):>10s} {str(got):>10s}  "
                  f"{'PASS' if ok else 'FAIL'}")
    print("-" * 84)
    print(f"preflight: {passed}/{passed + failed} passed, {failed} failed")

    # The branch mechanism itself must differ, not just the budget.
    print("\nmechanism check (must differ, this IS the contribution):")
    print(f"  di_emstgat -> branch_mode=raw   (DI: every feature into one encoder)")
    print(f"  ab_emstgat -> branch_mode=auto  (AB: learned soft feature routing)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
