"""Controlled ablation study for DI/AB-EMSTGAT.

PROTOCOL
  * One component removed per variant -- never two -- so each accuracy delta is
    attributable to exactly that component.
  * Everything else is held fixed: same dataset, same stratified 80/20 split,
    same seeds, same capacity, same epoch budget, same optimizer, same boundary
    loss. The only difference between the full model and a variant is one flag.
  * 5 seeds per variant; results reported as mean +/- sample std and as a
    paired per-seed comparison against the full model (same seeds), which is
    the correct test because the split is seed-determined.
  * The full model is re-run inside this study rather than copied from the main
    table, so the baseline shares the exact code path of the variants.

COMPONENTS
  DCC           dilated causal convolution temporal front end
  BiGRU         bidirectional GRU
  KNN-Graph     temporal KNN adjacency context injection
  SelfAttn      multi-head temporal self-attention
  AttnPool      attention pooling (replaced by mean pooling)
  SkipCls       tabular skip-logit fusion
  PosEmb        learned positional embedding
  Router-Gate   AB only: per-feature relevance gate
  Boundary      the targeted pairwise boundary loss (weight -> 0)

Ablating a component that is genuinely load-bearing should HURT. A variant that
matches or beats the full model is reported as such -- that is evidence the
component is not earning its parameters, and the thesis should say so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

from scripts.comparison_models import KerasSequenceClassifier, resolve_data_path
from scripts.run_main_comparison import _load_main_protocol_data

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_DATA = "数据文件/Public datasets_physics3_designB.csv"

# (variant key, chinese label, ablation flags, boundary weight override)
ABLATIONS: list[tuple[str, str, Dict[str, bool], Optional[float]]] = [
    ("full", "完整模型", {}, None),
    ("no_dcc", "去除膨胀因果卷积", {"use_dilated_causal_conv": False}, None),
    ("no_bigru", "去除双向GRU", {"use_bigru": False}, None),
    ("no_knn_graph", "去除时序KNN图", {"use_knn_graph": False}, None),
    ("no_self_attn", "去除多头自注意力", {"use_self_attention": False}, None),
    ("no_attn_pool", "注意力池化替为均值池化", {"use_attention_pooling": False}, None),
    ("no_skip_cls", "去除表格跳连分类器", {"use_skip_classifier": False}, None),
    ("no_pos_emb", "去除位置编码", {"use_positional_embedding": False}, None),
    ("no_boundary", "去除边界损失", {}, 0.0),
]
AB_ONLY = [("no_router_gate", "去除特征相关性门控", {"use_router_gate": False}, None)]

# Defaults kept for backwards compatibility with the earlier exploratory runs.
# The locked config (实验配置与方法说明.md 3.1-3.3) is AB=micro / 1 epoch /
# boundary_weight=0.0, selectable via --ab-capacity micro --ab-epochs 1
# --boundary-weight 0.0 so this study can match the main table's budget.
CAPACITY = {"di_emstgat": "standard", "ab_emstgat": "compact"}
LEARNING_RATE = 8.13e-4
WEIGHT_DECAY = 4.27e-4
BOUNDARY_WEIGHT = 0.05
BOUNDARY_MARGIN = 0.05


def variants_for(model: str):
    items = list(ABLATIONS)
    if model == "ab_emstgat":
        items = items + AB_ONLY
    return items


def add_gaussian_noise(X: np.ndarray, snr_db: Optional[float], seed: int) -> np.ndarray:
    """Add zero-mean Gaussian noise at a global SNR. None returns X unchanged.

    Identical formulation to run_noise_robustness.add_gaussian_noise so the two
    studies remain directly comparable.
    """
    X = np.asarray(X, dtype=np.float32)
    if snr_db is None:
        return X
    signal_power = float(np.mean(np.square(X)))
    if signal_power <= 0.0:
        return X
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0.0, np.sqrt(noise_power), size=X.shape)).astype(np.float32)


def subsample_train(X: np.ndarray, y: np.ndarray, fraction: float, seed: int):
    """Stratified subsample of the training set, for small-sample ablation."""
    if fraction >= 1.0:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        n = max(2, int(round(len(idx) * float(fraction))))
        keep.append(rng.permutation(idx)[:n])
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def build_estimator(model: str, seed: int, epochs: int,
                    ablation: Dict[str, bool], boundary_weight: Optional[float],
                    capacity: Optional[str] = None,
                    base_boundary_weight: float = BOUNDARY_WEIGHT):
    return KerasSequenceClassifier(
        model_kind=model,
        epochs=epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,
        validation_split=0.15,
        seed=seed,
        capacity=capacity or CAPACITY[model],
        boundary_weight=(float(base_boundary_weight) if boundary_weight is None
                         else float(boundary_weight)),
        boundary_margin=BOUNDARY_MARGIN,
        ablation=ablation,
    )


def run(data_path: str, output_dir: Path, seeds: Sequence[int], epochs: int,
        models: Sequence[str], test_size: float, eval_snr_db: Optional[float] = None,
        train_fraction: float = 1.0, noise_seed: int = 20260828,
        di_capacity: Optional[str] = None, ab_capacity: Optional[str] = None,
        ab_epochs: Optional[int] = None, di_epochs: Optional[int] = None,
        boundary_weight: float = BOUNDARY_WEIGHT) -> None:
    capacity_of = {"di_emstgat": di_capacity or CAPACITY["di_emstgat"],
                   "ab_emstgat": ab_capacity or CAPACITY["ab_emstgat"]}
    # DI and AB need independent epoch budgets: with AB as the primary model DI
    # is de-tuned to the comparison side (spec 5.4), so a single --epochs value
    # can no longer describe both.
    epochs_of = {"di_emstgat": epochs if di_epochs is None else int(di_epochs),
                 "ab_emstgat": epochs if ab_epochs is None else int(ab_epochs)}
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_data_path(data_path)

    # Resume safely from the per-fit checkpoint.  A previous interruption left
    # 50/95 train20pct records; re-running must skip only exact existing
    # (seed, model, variant) keys, never duplicate or overwrite them.
    checkpoint = output_dir / "per_seed_records.json"
    records: List[dict] = []
    seen_keys: set[tuple[int, str, str]] = set()
    if checkpoint.exists():
        loaded = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"checkpoint is not a JSON list: {checkpoint}")
        for old in loaded:
            key = (int(old["seed"]), str(old["model"]), str(old["variant"]))
            if key in seen_keys:
                raise ValueError(f"duplicate checkpoint key: {key}")
            expected_condition = ("clean" if eval_snr_db is None
                                  else f"snr{int(eval_snr_db)}dB")
            expected_condition += ("" if train_fraction >= 1.0
                                   else f"_train{int(round(100 * train_fraction))}pct")
            if old.get("condition") != expected_condition:
                raise ValueError(
                    f"checkpoint condition {old.get('condition')!r} does not "
                    f"match requested {expected_condition!r}: {checkpoint}")
            if abs(float(old.get("train_fraction", -1.0)) - float(train_fraction)) > 1e-12:
                raise ValueError(f"checkpoint train_fraction mismatch: {checkpoint}")
            records.append(old)
            seen_keys.add(key)
        print(f"[resume] loaded {len(records)} existing records from {checkpoint}",
              flush=True)

    for seed in seeds:
        data = _load_main_protocol_data(resolved, test_size=test_size, seed=seed)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        class_names = data["class_names"]
        # Degradation knobs exist to break the accuracy ceiling: at ~98.9% the
        # model misses only ~19 of 1700 rows while seed noise alone is +-6.6
        # rows, so no single component can reach significance. Training data is
        # never noised; only the evaluation input is, matching the noise study.
        if train_fraction < 1.0:
            X_train, y_train = subsample_train(X_train, y_train,
                                               train_fraction, seed)
        # The noise draw is deliberately NOT keyed on `seed`. Measured: one
        # fitted model evaluated under 15 different SNR-10 draws varies by
        # 0.51 pp, which is 32% of the observed across-seed spread. Keying the
        # draw on the training seed would fold that pure evaluation noise into
        # the paired comparison and mask the very component effects this study
        # is meant to resolve. A fixed draw makes every variant and every seed
        # face the IDENTICAL noised test set, so the paired test isolates the
        # component instead of the noise realisation.
        X_eval = add_gaussian_noise(X_test, eval_snr_db, seed=noise_seed)
        for model in models:
            for key, label, flags, bw in variants_for(model):
                checkpoint_key = (int(seed), str(model), str(key))
                if checkpoint_key in seen_keys:
                    print(f"[resume] skip seed={seed} {model} {key}", flush=True)
                    continue
                est = build_estimator(
                    model, seed, epochs_of[model], flags, bw,
                    capacity=capacity_of[model],
                    base_boundary_weight=boundary_weight)
                import time
                t0 = time.perf_counter()
                est.fit(X_train, y_train)
                fit_time = time.perf_counter() - t0
                t1 = time.perf_counter()
                pred = est.predict(X_eval)
                pred_time = time.perf_counter() - t1
                params = 0
                try:
                    params = int(est.model_.count_params())
                except Exception:
                    params = 0
                rec = {
                    "condition": ("clean" if eval_snr_db is None
                                  else f"snr{int(eval_snr_db)}dB")
                                 + ("" if train_fraction >= 1.0
                                    else f"_train{int(round(100*train_fraction))}pct"),
                    "eval_snr_db": ("clean" if eval_snr_db is None
                                    else float(eval_snr_db)),
                    "train_fraction": float(train_fraction),
                    "noise_seed": int(noise_seed),
                    "model": model,
                    "variant": key,
                    "variant_cn": label,
                    "seed": int(seed),
                    "removed_component": key if key != "full" else "none",
                    "ablation_flags": json.dumps(flags, ensure_ascii=False),
                    "boundary_weight": (float(boundary_weight) if bw is None
                                        else float(bw)),
                    "capacity": capacity_of[model],
                    "epochs": epochs_of[model],
                    "param_count": params,
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "f1_macro": float(f1_score(y_test, pred, average="macro",
                                               zero_division=0)),
                    "f1_weighted": float(f1_score(y_test, pred, average="weighted",
                                                  zero_division=0)),
                    "cohen_kappa": float(cohen_kappa_score(y_test, pred)),
                    "fit_time": float(fit_time),
                    "prediction_time": float(pred_time),
                    "train_samples": int(len(y_train)),
                    "test_samples": int(len(y_test)),
                    "feature_count": int(len(data["feature_names"])),
                }
                per_class = f1_score(y_test, pred, average=None, zero_division=0)
                for name, value in zip(class_names, per_class):
                    rec[f"f1_{name}"] = float(value)
                records.append(rec)
                seen_keys.add(checkpoint_key)
                print(f"[ablation] seed={seed} {model} {key:16s} "
                      f"acc={rec['accuracy']:.6f} macroF1={rec['f1_macro']:.6f} "
                      f"params={params}", flush=True)
                # checkpoint after every fit so a crash never loses progress
                pd.DataFrame(records).to_json(
                    output_dir / "per_seed_records.json", orient="records",
                    force_ascii=False, indent=2)

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "per_seed_records.csv", index=False,
                 encoding="utf-8-sig")

    # ---- aggregate + paired comparison against the full model ----
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
               "cohen_kappa"]
    rows = []
    for model in models:
        sub = frame[frame.model == model]
        full = sub[sub.variant == "full"].set_index("seed").sort_index()
        for key, label, _, _ in variants_for(model):
            g = sub[sub.variant == key].set_index("seed").sort_index()
            if g.empty:
                continue
            row = {"model": model, "variant": key, "variant_cn": label,
                   "n_seeds": int(len(g)),
                   "param_count": int(g["param_count"].iloc[0])}
            for m in metrics:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
            if key != "full":
                common = g.index.intersection(full.index)
                d = (g.loc[common, "accuracy"] - full.loc[common, "accuracy"]).to_numpy(float)
                row["delta_accuracy_points"] = float(d.mean()) * 100.0
                dm = (g.loc[common, "f1_macro"] - full.loc[common, "f1_macro"]).to_numpy(float)
                row["delta_f1_macro_points"] = float(dm.mean()) * 100.0
                if len(common) >= 2 and not np.allclose(d, 0):
                    row["paired_t_p"] = float(ttest_rel(g.loc[common, "accuracy"],
                                                        full.loc[common, "accuracy"]).pvalue)
                    try:
                        row["wilcoxon_p"] = float(wilcoxon(d).pvalue)
                    except ValueError:
                        row["wilcoxon_p"] = float("nan")
                else:
                    row["paired_t_p"] = 1.0
                    row["wilcoxon_p"] = 1.0
                row["variant_worse_seeds"] = f"{int((d < 0).sum())}/{len(d)}"
                row["param_delta"] = int(g["param_count"].iloc[0]
                                         - full["param_count"].iloc[0])
            else:
                row["delta_accuracy_points"] = 0.0
                row["delta_f1_macro_points"] = 0.0
                row["paired_t_p"] = float("nan")
                row["wilcoxon_p"] = float("nan")
                row["variant_worse_seeds"] = "-"
                row["param_delta"] = 0
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "aggregate_summary.csv", index=False,
                   encoding="utf-8-sig")

    manifest = {
        "study_name": "controlled_component_ablation",
        "condition": ("clean" if eval_snr_db is None
                      else f"snr{int(eval_snr_db)}dB")
                     + ("" if train_fraction >= 1.0
                        else f"_train{int(round(100*train_fraction))}pct"),
        "eval_snr_db": "clean" if eval_snr_db is None else float(eval_snr_db),
        "train_fraction": float(train_fraction),
        "noise_seed": int(noise_seed),
        "noise_seed_policy": (
            "FIXED across seeds and variants, on purpose. One fitted model "
            "evaluated under 15 different SNR-10 draws varied by 0.51 pp = 32% "
            "of the across-seed spread; keying the draw on the training seed "
            "injected that evaluation noise into the paired test and masked "
            "component effects. Every variant now faces the identical noised "
            "test set."),
        "degradation_rationale": (
            "at ~98.9% accuracy the model misses only ~19 of 1700 test rows "
            "while seed-to-seed noise is +-6.6 rows, so a paired test over 5 "
            "seeds cannot resolve any single component; evaluating under "
            "degradation restores discriminative power. Training data is never "
            "noised."),
        "data_path": resolved,
        "test_size": test_size,
        "seeds": list(map(int, seeds)),
        "models": list(models),
        "epochs": epochs,
        "capacity": capacity_of,
        "epochs_per_model": epochs_of,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "boundary_weight": float(boundary_weight),
        "boundary_margin": BOUNDARY_MARGIN,
        "variants": {k: {"cn": cn, "flags": f, "boundary_weight_override": bw}
                     for k, cn, f, bw in ABLATIONS + AB_ONLY},
        "protocol": ("one component removed per variant; all other settings "
                     "held fixed; the full model is re-run in this study so it "
                     "shares the variants' code path"),
        "records": len(records),
        "split_protocol": "row-level stratified random 80/20, seed-determined",
        "outer_test_evaluated": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ablation aggregate ===")
    show = summary[["model", "variant_cn", "param_count", "accuracy_mean",
                    "accuracy_std", "delta_accuracy_points", "paired_t_p",
                    "variant_worse_seeds"]].copy()
    for c in ("accuracy_mean", "accuracy_std"):
        show[c] = (100 * show[c]).round(4)
    show["delta_accuracy_points"] = show["delta_accuracy_points"].round(4)
    show["paired_t_p"] = show["paired_t_p"].round(6)
    print(show.to_string(index=False))
    print(f"\nSaved: {output_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--models", default="di_emstgat,ab_emstgat")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--eval-snr", type=float, default=None,
        help="Evaluate under Gaussian noise at this SNR in dB (training data "
             "stays clean). Omit for the clean condition.")
    parser.add_argument(
        "--train-fraction", type=float, default=1.0,
        help="Stratified fraction of the training set to keep, e.g. 0.2 for "
             "the small-sample ablation condition.")
    parser.add_argument(
        "--di-capacity", choices=("standard", "compact", "light", "micro"),
        default=None, help="DI capacity; locked config uses 'standard'.")
    parser.add_argument(
        "--ab-capacity", choices=("standard", "compact", "light", "micro"),
        default=None, help="AB capacity; locked config uses 'micro'.")
    parser.add_argument(
        "--ab-epochs", type=int, default=None,
        help="AB epoch budget; DesignB_plus_v1 used 1, DesignB_branch_v1 uses "
             "100. Defaults to --epochs.")
    parser.add_argument(
        "--di-epochs", type=int, default=None,
        help="DI epoch budget, independent of AB. Defaults to --epochs.")
    parser.add_argument(
        "--boundary-weight", type=float, default=BOUNDARY_WEIGHT,
        help="Boundary loss weight for the 'full' model and all variants except "
             "no_boundary. Locked config uses 0.0 (spec 3.2).")
    parser.add_argument(
        "--noise-seed", type=int, default=20260828,
        help="Seed for the evaluation noise draw. Held FIXED across training "
             "seeds and variants so the paired test isolates the component "
             "rather than the noise realisation.")
    args = parser.parse_args(argv)
    if not (0.0 < args.train_fraction <= 1.0):
        parser.error("--train-fraction must be in (0, 1]")

    run(
        data_path=args.data,
        output_dir=Path(args.output_dir),
        seeds=tuple(int(v) for v in args.seeds.split(",") if v.strip()),
        epochs=args.epochs,
        models=tuple(v.strip() for v in args.models.split(",") if v.strip()),
        test_size=args.test_size,
        eval_snr_db=args.eval_snr,
        train_fraction=args.train_fraction,
        noise_seed=args.noise_seed,
        di_capacity=args.di_capacity,
        ab_capacity=args.ab_capacity,
        ab_epochs=args.ab_epochs,
        di_epochs=args.di_epochs,
        boundary_weight=args.boundary_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
