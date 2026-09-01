"""Independent audit of the physics-labeled datasets, model-agnostic.

Does NOT reuse the thesis pipeline, so it is an independent check that the
physics criterion produces a learnable, leak-free, non-degenerate task.

Reports for each dataset:
  * class balance, duplicate rows, exact train/test twins (leakage probe)
  * accuracy across several very different classifier families
  * accuracy when every temperature channel is REMOVED -- the criterion divides
    by p_sat(T_cell), so temperatures are expected to dominate; this quantifies
    how much signal survives without them
  * accuracy from the three water-balance quantities alone
  * per-class confusion on a single held-out split
  * episode-level (grouped) split accuracy, which is the honest number if
    consecutive 60 Hz rows are near-duplicates
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件")
DATASETS = {
    "physics_designB_load_controlled": DATA_DIR / "Public datasets_physics3_designB.csv",
    "physics_designA_free_load": DATA_DIR / "Public datasets_physics3_designA.csv",
    "legacy_synthetic_3class_12000": DATA_DIR / "Public datasets_3class_12000.csv",
    "published_4class_public": DATA_DIR / "Public datasets.csv",
}
OUT = Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS = OUT / "表格分项归档" / "current_components"


def episode_ids(tsec: np.ndarray, gap: float = 5.0) -> np.ndarray:
    order = np.argsort(tsec)
    ids = np.empty(len(tsec), dtype=int)
    cur = 0
    prev = None
    for pos in order:
        if prev is not None and tsec[pos] - prev > gap:
            cur += 1
        ids[pos] = cur
        prev = tsec[pos]
    return ids


def audit(name: str, path: Path) -> dict:
    df = pd.read_csv(path)
    label_col = "State_Label"
    drop = {"State", label_col, "tsec"}
    feats = [c for c in df.columns if c not in drop]
    live = [c for c in feats if df[c].nunique() > 1]
    dead = [c for c in feats if c not in live]

    y = df[label_col].to_numpy()
    X = df[live].to_numpy(float)

    result = {
        "dataset": name,
        "path": str(path),
        "rows": int(len(df)),
        "classes": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
        "n_features_live": len(live),
        "dead_channels": dead,
        "duplicate_feature_rows": int(df[live].round(9).duplicated().sum()),
    }

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=42)
    tk = set(map(tuple, np.round(Xtr, 9)))
    result["train_test_exact_twins"] = int(
        sum(1 for r in np.round(Xte, 9) if tuple(r) in tk))

    models = {
        "1nn": (KNeighborsClassifier(n_neighbors=1), True),
        "logreg": (LogisticRegression(max_iter=3000), True),
        "lda": (LinearDiscriminantAnalysis(), True),
        "tree_d3": (DecisionTreeClassifier(max_depth=3, random_state=42), False),
        "rf200": (RandomForestClassifier(n_estimators=200, random_state=42,
                                         n_jobs=-1), False),
    }
    accs = {}
    for mname, (clf, needs_scale) in models.items():
        if needs_scale:
            sc = StandardScaler().fit(Xtr)
            clf.fit(sc.transform(Xtr), ytr)
            accs[mname] = round(float(clf.score(sc.transform(Xte), yte)), 4)
        else:
            clf.fit(Xtr, ytr)
            accs[mname] = round(float(clf.score(Xte, yte)), 4)
    result["accuracy_random_split"] = accs

    # --- ablation: drop all temperature channels ---
    no_temp = [c for c in live if not c.startswith("T_")]
    if no_temp:
        Xn = df[no_temp].to_numpy(float)
        a, b, c_, d_ = train_test_split(Xn, y, test_size=0.2, stratify=y,
                                        random_state=42)
        rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                    n_jobs=-1).fit(a, c_)
        result["accuracy_without_temperature_channels"] = round(
            float(rf.score(b, d_)), 4)
        result["n_features_without_temperature"] = len(no_temp)

    # --- temperature importance share ---
    rf_full = RandomForestClassifier(n_estimators=200, random_state=42,
                                     n_jobs=-1).fit(Xtr, ytr)
    imp = rf_full.feature_importances_
    result["temperature_importance_share"] = round(
        float(sum(imp[k] for k in range(len(live)) if live[k].startswith("T_"))), 4)
    order = np.argsort(imp)[::-1][:8]
    result["rf_top8"] = {live[k]: round(float(imp[k]), 4) for k in order}

    # --- episode-grouped split (honest if consecutive rows are near-identical) ---
    if "tsec" in df.columns:
        groups = episode_ids(df["tsec"].to_numpy(float))
        result["n_episodes"] = int(len(np.unique(groups)))
        if result["n_episodes"] >= 5:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            tr_i, te_i = next(gss.split(X, y, groups))
            if len(np.unique(y[tr_i])) == len(np.unique(y)) and \
               len(np.unique(y[te_i])) == len(np.unique(y)):
                rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                            n_jobs=-1).fit(X[tr_i], y[tr_i])
                result["accuracy_episode_grouped_split"] = round(
                    float(rf.score(X[te_i], y[te_i])), 4)
                result["episode_split_train_test"] = [int(len(tr_i)), int(len(te_i))]
            else:
                result["accuracy_episode_grouped_split"] = "not all classes present"

    # --- per-class report on the random split ---
    pred = rf_full.predict(Xte)
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    result["per_class_f1"] = {k: round(float(v["f1-score"]), 4)
                              for k, v in rep.items() if isinstance(v, dict)
                              and k not in ("macro avg", "weighted avg", "accuracy")}
    labels_sorted = sorted(np.unique(y).tolist())
    result["confusion_matrix_labels"] = labels_sorted
    result["confusion_matrix"] = confusion_matrix(
        yte, pred, labels=labels_sorted).tolist()
    return result


def main() -> None:
    results = []
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"skip missing {path}")
            continue
        r = audit(name, path)
        results.append(r)
        print("=" * 84)
        print(f"{name}   rows={r['rows']}  live_feats={r['n_features_live']}")
        print(f"  classes: {r['classes']}")
        print(f"  dead channels: {r['dead_channels']}")
        print(f"  duplicate feature rows: {r['duplicate_feature_rows']}")
        print(f"  train/test exact twins: {r['train_test_exact_twins']}")
        print(f"  accuracy (random 80/20): {r['accuracy_random_split']}")
        if "accuracy_without_temperature_channels" in r:
            print(f"  accuracy w/o temperature ({r['n_features_without_temperature']} feats): "
                  f"{r['accuracy_without_temperature_channels']}")
        print(f"  temperature importance share: {r['temperature_importance_share']}")
        print(f"  RF top8: {r['rf_top8']}")
        if "n_episodes" in r:
            print(f"  episodes: {r['n_episodes']}  "
                  f"grouped-split acc: {r.get('accuracy_episode_grouped_split')}")
        print(f"  per-class F1: {r['per_class_f1']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "物理判据数据集_独立审计.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for r in results:
        rows.append({
            "数据集": r["dataset"],
            "行数": r["rows"],
            "类别数": len(r["classes"]),
            "有效特征": r["n_features_live"],
            "死通道数": len(r["dead_channels"]),
            "重复行": r["duplicate_feature_rows"],
            "训练测试重复": r["train_test_exact_twins"],
            "1NN(%)": 100 * r["accuracy_random_split"]["1nn"],
            "逻辑回归(%)": 100 * r["accuracy_random_split"]["logreg"],
            "LDA(%)": 100 * r["accuracy_random_split"]["lda"],
            "决策树d3(%)": 100 * r["accuracy_random_split"]["tree_d3"],
            "RF200(%)": 100 * r["accuracy_random_split"]["rf200"],
            "去温度后RF(%)": 100 * r.get("accuracy_without_temperature_channels", float("nan")),
            "温度重要度占比": r["temperature_importance_share"],
            "episode数": r.get("n_episodes"),
            "分组划分RF(%)": (100 * r["accuracy_episode_grouped_split"]
                          if isinstance(r.get("accuracy_episode_grouped_split"), float)
                          else r.get("accuracy_episode_grouped_split")),
        })
    frame = pd.DataFrame(rows)
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMPONENTS / "表23_物理判据数据集独立审计.csv", index=False,
                 encoding="utf-8-sig")
    print("\n" + frame.to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
