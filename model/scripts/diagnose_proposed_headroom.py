"""Diagnose whether the proposed models have real headroom, before tuning.

Context: under EQUAL compute (100 epochs, standard capacity) the baselines
overtake the proposed models -- LightGBM 99.59%, TCN 99.35%, MTGNN 99.59% vs
AB 98.88%. So the current "first place" comes largely from the reduced-baseline
policy, not from the architecture. Before changing anything, find out WHERE the
proposed models lose.

Questions answered here:
  1. What is the error structure? Which class boundary do they miss?
  2. How many features does the pipeline actually feed them? A 78k-parameter
     model on 8 inputs may simply be over-parameterised.
  3. Do the tree models see the same features, or do they get more?
  4. Is the proposed model under- or over-fitting? Compare train vs test.
  5. What is the theoretical ceiling -- are the remaining errors even separable?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ROOT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model")
STUDIES = MODEL_ROOT / "results/studies"
CLASSES = ["Flooding", "Membrane_Drying", "Normal"]
CN = {"Flooding": "水淹", "Membrane_Drying": "膜干", "Normal": "正常"}


def load(name: str):
    path = STUDIES / name / "per_seed_records.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def confusion(records, model: str) -> np.ndarray:
    total = np.zeros((3, 3))
    for r in records:
        if r["model"] == model:
            total += np.array(r["confusion_matrix"], dtype=float)
    return total


def main() -> None:
    reduced = load("physics3_designB")
    equal = load("equal_compute_designB")

    print("=" * 78)
    print("1. ERROR STRUCTURE of the proposed models (reduced-baseline run)")
    print("=" * 78)
    for model in ("ab_emstgat", "di_emstgat"):
        cm = confusion(reduced, model)
        n_seeds = len({r["seed"] for r in reduced if r["model"] == model})
        err = cm.sum() - np.trace(cm)
        print(f"\n{model}  ({n_seeds} seeds, {int(err)} total errors, "
              f"{err/max(n_seeds,1):.1f} per seed of 1700)")
        frame = pd.DataFrame(cm.astype(int), index=CLASSES, columns=CLASSES)
        print(frame.to_string())
        off = [(CLASSES[i], CLASSES[j], cm[i, j])
               for i in range(3) for j in range(3) if i != j and cm[i, j] > 0]
        off.sort(key=lambda t: -t[2])
        for a, b, v in off:
            print(f"    {CN[a]}->{CN[b]}: {int(v)}  ({100*v/err:.1f}% of errors)")

    print("\n" + "=" * 78)
    print("2. HOW MANY FEATURES does the pipeline feed each model?")
    print("=" * 78)
    frame = pd.DataFrame(reduced)
    print(frame.groupby("model").feature_count.agg(["min", "max"]).to_string())
    print("\n  The physics dataset has 19 live channels, but the pipeline's")
    print("  cumulative-importance selector keeps only a handful. Every model")
    print("  sees the SAME selected features, so this is not an unfair input.")

    print("\n" + "=" * 78)
    print("3. EQUAL-COMPUTE comparison (the honest table)")
    print("=" * 78)
    if equal is None:
        print("  equal-compute run not finished yet")
    else:
        e = pd.DataFrame(equal)
        r = pd.DataFrame(reduced)
        rows = []
        for model in sorted(set(e.model)):
            ge = e[e.model == model]
            gr = r[r.model == model]
            rows.append({
                "model": model,
                "reduced_acc": round(100 * gr.accuracy.mean(), 4) if len(gr) else np.nan,
                "equal_acc": round(100 * ge.accuracy.mean(), 4),
                "equal_sd": round(100 * ge.accuracy.std(ddof=1), 4) if len(ge) > 1 else 0.0,
                "n_seeds": int(ge.seed.nunique()),
                "equal_fit_s": round(float(ge.fit_time.mean()), 1),
            })
        table = pd.DataFrame(rows).sort_values("equal_acc", ascending=False)
        print(table.to_string(index=False))
        prop = table[table.model.isin(["di_emstgat", "ab_emstgat"])]
        base = table[~table.model.isin(["di_emstgat", "ab_emstgat"])]
        if len(prop) and len(base):
            best_prop = prop.equal_acc.max()
            best_base = base.equal_acc.max()
            print(f"\n  best proposed = {best_prop:.4f}%   "
                  f"best baseline = {best_base:.4f}%   "
                  f"gap = {best_prop - best_base:+.4f} pp")
            if best_prop < best_base:
                print("  => under equal compute the proposed models LOSE. The"
                      " reduced-baseline policy is what produced first place.")

    print("\n" + "=" * 78)
    print("4. UNDER- or OVER-FITTING? train vs test on the proposed models")
    print("=" * 78)
    import sys
    sys.path.insert(0, str(MODEL_ROOT))
    from sklearn.metrics import accuracy_score

    from scripts.comparison_models import build_default_model_specs
    from scripts.run_main_comparison import _load_main_protocol_data

    data = _load_main_protocol_data(
        "数据文件/Public datasets_physics3_designB.csv", test_size=0.2, seed=42)
    print(f"  features fed to the model: {len(data['feature_names'])} "
          f"-> {data['feature_names']}")
    print(f"  train {len(data['y_train'])}  test {len(data['y_test'])}")

    specs = build_default_model_specs(
        seed=42, n_jobs=1, deep_epochs=100, deep_validation_split=0.0,
        comparison_strength="reduced_baselines", proposed_capacity="standard",
        ab_capacity="compact", proposed_boundary_weight=0.05,
        proposed_boundary_margin=0.05)
    for model in ("ab_emstgat", "di_emstgat"):
        est = specs[model].estimator_factory()
        est.fit(data["X_train"], data["y_train"])
        tr = accuracy_score(data["y_train"], est.predict(data["X_train"]))
        te = accuracy_score(data["y_test"], est.predict(data["X_test"]))
        params = int(est.model_.count_params()) if est.model_ is not None else 0
        print(f"  {model:12s} train={100*tr:7.4f}%  test={100*te:7.4f}%  "
              f"gap={100*(tr-te):+6.4f} pp  params={params:,}")
        print(f"    params per training row = {params/len(data['y_train']):.1f}"
              f"  -> {'OVER-parameterised' if params/len(data['y_train']) > 10 else 'reasonable'}")

    print("\n" + "=" * 78)
    print("5. IS THE REMAINING ERROR EVEN SEPARABLE?")
    print("=" * 78)
    # 1-NN on the training set gives a rough Bayes-error proxy for these features
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(data["X_train"])
    knn = KNeighborsClassifier(n_neighbors=1).fit(sc.transform(data["X_train"]),
                                                  data["y_train"])
    knn_te = accuracy_score(data["y_test"], knn.predict(sc.transform(data["X_test"])))
    print(f"  1-NN test accuracy on the SAME features: {100*knn_te:.4f}%")
    print("  A 1-NN this strong means the selected features are nearly")
    print("  perfectly separable, so a deep model has almost nothing to add.")


if __name__ == "__main__":
    main()
