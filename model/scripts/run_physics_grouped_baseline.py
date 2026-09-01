"""Episode-grouped (leak-free) evaluation of the physics-labeled dataset.

WHY THIS EXISTS
  The bench logs at ~3.5 Hz (median tsec gap 0.286 s). Consecutive rows inside
  one recording episode are near-identical, so a row-level random split puts
  near-twins on both sides and inflates every model. The independent audit
  measured the gap directly on design B:
      random 80/20 split        RF = 99.82 %
      episode-grouped split     RF = 88.05 %
  A 11.8-point drop. The grouped number is the honest one.

WHAT THIS DOES
  Splits by EPISODE (tsec gap > 5 s starts a new episode), never by row, so no
  episode contributes rows to both train and test. Repeats over several random
  group partitions and reports mean +/- std, plus a stratified-by-class variant
  that guarantees every class appears on both sides.

  Also reports the row-level random split on the same data and the same models,
  so the inflation is quantified rather than assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件")
OUT = Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS = OUT / "表格分项归档" / "current_components"
DATASETS = {
    "physics_designB": DATA_DIR / "Public datasets_physics3_designB.csv",
    "physics_designA": DATA_DIR / "Public datasets_physics3_designA.csv",
    "legacy_synthetic": DATA_DIR / "Public datasets_3class_12000.csv",
    "published_4class": DATA_DIR / "Public datasets.csv",
}
SEEDS = (42, 43, 44, 45, 46)
GAP_S = 5.0


def episode_ids(tsec: np.ndarray, gap: float = GAP_S) -> np.ndarray:
    order = np.argsort(tsec)
    ids = np.empty(len(tsec), dtype=int)
    cur, prev = 0, None
    for pos in order:
        if prev is not None and tsec[pos] - prev > gap:
            cur += 1
        ids[pos] = cur
        prev = tsec[pos]
    return ids


def class_stratified_group_split(groups, y, test_frac, rng):
    """Hold out whole episodes, sampling episodes within each class.

    Each episode is assigned the class of its majority label, then episodes are
    drawn per class so both sides always carry every class.
    """
    uniq = np.unique(groups)
    ep_class = {}
    for g in uniq:
        vals, cnts = np.unique(y[groups == g], return_counts=True)
        ep_class[g] = vals[np.argmax(cnts)]
    test_eps = []
    for cls in np.unique(y):
        eps = np.array([g for g in uniq if ep_class[g] == cls])
        if len(eps) == 0:
            continue
        n_test = max(1, int(round(len(eps) * test_frac)))
        n_test = min(n_test, len(eps) - 1) if len(eps) > 1 else 0
        if n_test > 0:
            test_eps.extend(rng.permutation(eps)[:n_test].tolist())
    test_mask = np.isin(groups, test_eps)
    return ~test_mask, test_mask


def eval_models(Xtr, ytr, Xte, yte):
    out = {}
    sc = StandardScaler().fit(Xtr)
    for name, clf, scale in (
        ("1nn", KNeighborsClassifier(n_neighbors=1), True),
        ("logreg", LogisticRegression(max_iter=3000), True),
        ("rf200", RandomForestClassifier(n_estimators=200, random_state=42,
                                         n_jobs=-1), False),
    ):
        a, b = (sc.transform(Xtr), sc.transform(Xte)) if scale else (Xtr, Xte)
        clf.fit(a, ytr)
        pred = clf.predict(b)
        out[name] = {
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "f1_macro": round(float(f1_score(yte, pred, average="macro",
                                             zero_division=0)), 4),
        }
    return out


def run(name: str, path: Path) -> dict:
    df = pd.read_csv(path)
    label_col = "State_Label"
    feats = [c for c in df.columns if c not in {"State", label_col, "tsec"}]
    live = [c for c in feats if df[c].nunique() > 1]
    X = df[live].to_numpy(float)
    y = df[label_col].to_numpy()
    groups = episode_ids(df["tsec"].to_numpy(float))
    n_ep = int(len(np.unique(groups)))

    res = {"dataset": name, "rows": int(len(df)), "features": len(live),
           "episodes": n_ep, "classes": sorted(np.unique(y).tolist())}

    # ---- row-level random split, for the inflation reference ----
    row_runs = []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=s)
        row_runs.append(eval_models(Xtr, ytr, Xte, yte))
    res["row_random_split"] = {
        m: {
            "accuracy_mean": round(float(np.mean([r[m]["accuracy"] for r in row_runs])), 4),
            "accuracy_std": round(float(np.std([r[m]["accuracy"] for r in row_runs], ddof=1)), 4),
            "f1_macro_mean": round(float(np.mean([r[m]["f1_macro"] for r in row_runs])), 4),
        } for m in row_runs[0]
    }

    # ---- episode-grouped split ----
    grp_runs, skipped = [], 0
    sizes = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        tr_m, te_m = class_stratified_group_split(groups, y, 0.3, rng)
        if (tr_m.sum() == 0 or te_m.sum() == 0
                or len(np.unique(y[tr_m])) < len(res["classes"])
                or len(np.unique(y[te_m])) < len(res["classes"])):
            skipped += 1
            continue
        sizes.append([int(tr_m.sum()), int(te_m.sum())])
        grp_runs.append(eval_models(X[tr_m], y[tr_m], X[te_m], y[te_m]))
    res["episode_grouped_runs"] = len(grp_runs)
    res["episode_grouped_skipped"] = skipped
    res["episode_split_sizes"] = sizes
    if grp_runs:
        res["episode_grouped_split"] = {
            m: {
                "accuracy_mean": round(float(np.mean([r[m]["accuracy"] for r in grp_runs])), 4),
                "accuracy_std": round(float(np.std([r[m]["accuracy"] for r in grp_runs], ddof=1))
                                      if len(grp_runs) > 1 else 0.0, 4),
                "f1_macro_mean": round(float(np.mean([r[m]["f1_macro"] for r in grp_runs])), 4),
            } for m in grp_runs[0]
        }
        rf_row = res["row_random_split"]["rf200"]["accuracy_mean"]
        rf_grp = res["episode_grouped_split"]["rf200"]["accuracy_mean"]
        res["rf_inflation_points"] = round(100 * (rf_row - rf_grp), 2)
    else:
        res["episode_grouped_split"] = None
        res["rf_inflation_points"] = None
        res["note"] = ("episode-grouped split impossible: a class occupies too "
                       "few episodes, i.e. label and recording episode are "
                       "collinear in this dataset")
    return res


def main() -> None:
    results = []
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"skip {path}")
            continue
        r = run(name, path)
        results.append(r)
        print("=" * 86)
        print(f"{r['dataset']}  rows={r['rows']} feats={r['features']} "
              f"episodes={r['episodes']} classes={len(r['classes'])}")
        for m in ("1nn", "logreg", "rf200"):
            row = r["row_random_split"][m]
            line = (f"  {m:8s} row-random acc={100*row['accuracy_mean']:6.2f}"
                    f"+-{100*row['accuracy_std']:.2f}")
            if r["episode_grouped_split"]:
                g = r["episode_grouped_split"][m]
                line += (f"   episode-grouped acc={100*g['accuracy_mean']:6.2f}"
                         f"+-{100*g['accuracy_std']:.2f}"
                         f"   macroF1={100*g['f1_macro_mean']:6.2f}")
            print(line)
        if r["episode_grouped_split"]:
            print(f"  RF inflation from row-level splitting: "
                  f"{r['rf_inflation_points']} points "
                  f"({r['episode_grouped_runs']}/{len(SEEDS)} usable partitions)")
        else:
            print(f"  {r['note']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "物理判据_分组划分诚实基线.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for r in results:
        g = r["episode_grouped_split"]
        rows.append({
            "数据集": r["dataset"],
            "行数": r["rows"],
            "类别数": len(r["classes"]),
            "episode数": r["episodes"],
            "按行随机_RF(%)": round(100 * r["row_random_split"]["rf200"]["accuracy_mean"], 2),
            "按行随机_RF标准差": round(100 * r["row_random_split"]["rf200"]["accuracy_std"], 2),
            "按块分组_RF(%)": round(100 * g["rf200"]["accuracy_mean"], 2) if g else "不可划分",
            "按块分组_RF标准差": round(100 * g["rf200"]["accuracy_std"], 2) if g else "-",
            "按块分组_MacroF1(%)": round(100 * g["rf200"]["f1_macro_mean"], 2) if g else "-",
            "按行随机_1NN(%)": round(100 * r["row_random_split"]["1nn"]["accuracy_mean"], 2),
            "按块分组_1NN(%)": round(100 * g["1nn"]["accuracy_mean"], 2) if g else "-",
            "虚高幅度(个百分点)": r["rf_inflation_points"] if g else "-",
            "可用分区数": f"{r['episode_grouped_runs']}/{len(SEEDS)}",
        })
    frame = pd.DataFrame(rows)
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMPONENTS / "表24_按块分组诚实基线对照.csv", index=False,
                 encoding="utf-8-sig")
    print("\n" + frame.to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
