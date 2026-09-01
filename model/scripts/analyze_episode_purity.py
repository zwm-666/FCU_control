"""Explain the episode-grouped accuracy gap: episode/label purity.

The grouped-split audit produced a counter-intuitive ranking:

  dataset            row-random RF   episode-grouped RF   inflation
  physics_designB        99.79              68.00           31.79
  physics_designA        99.75              63.45           36.30
  legacy_synthetic       99.94              96.56            3.38
  published_4class       99.82              90.01            9.81

A naive reading says the physics labels are "worse". That would be wrong. The
grouped split measures how much a label varies WITHIN a recording episode:

  * If episodes are PURE (one class per episode), holding out whole episodes
    forces extrapolation to operating regimes never seen in training, and the
    score collapses. Label and episode are collinear.
  * If episodes are MIXED (several classes inside one episode), every held-out
    episode still contains classes the model saw during training, so the
    grouped split is easy and the score stays high.

So a HIGH grouped score can mean the labels flip rapidly inside an episode,
which for a slowly-drifting physical state is a sign the label is tracking
sensor noise rather than a physical condition. This script measures episode
purity, class-per-episode counts, and within-episode label switching so the
thesis can state which situation each dataset is in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def analyse(name: str, path: Path) -> dict:
    df = pd.read_csv(path).sort_values("tsec").reset_index(drop=True)
    y = df["State_Label"].to_numpy()
    groups = episode_ids(df["tsec"].to_numpy(float))
    uniq = np.unique(groups)

    purities, n_classes_per_ep, ep_sizes = [], [], []
    for g in uniq:
        sub = y[groups == g]
        vals, cnts = np.unique(sub, return_counts=True)
        purities.append(cnts.max() / cnts.sum())
        n_classes_per_ep.append(len(vals))
        ep_sizes.append(len(sub))
    purities = np.array(purities)
    n_classes_per_ep = np.array(n_classes_per_ep)

    # within-episode label switches per 100 rows
    switches = []
    for g in uniq:
        sub = y[groups == g]
        if len(sub) > 1:
            switches.append(100.0 * np.mean(sub[1:] != sub[:-1]))
    switches = np.array(switches) if switches else np.array([0.0])

    # how many episodes does each class occupy?
    class_eps = {}
    for cls in np.unique(y):
        class_eps[str(cls)] = int(len(np.unique(groups[y == cls])))

    # consecutive-row similarity: how near-duplicate are neighbours?
    feats = [c for c in df.columns
             if c not in {"State", "State_Label", "tsec"} and df[c].nunique() > 1]
    Xs = (df[feats].to_numpy(float) - df[feats].to_numpy(float).mean(0)) / (
        df[feats].to_numpy(float).std(0) + 1e-12)
    same_ep = groups[1:] == groups[:-1]
    step = np.linalg.norm(Xs[1:] - Xs[:-1], axis=1)[same_ep]

    return {
        "dataset": name,
        "rows": int(len(df)),
        "episodes": int(len(uniq)),
        "median_episode_rows": int(np.median(ep_sizes)),
        "episode_purity_mean": round(float(purities.mean()), 4),
        "episode_purity_median": round(float(np.median(purities)), 4),
        "pure_episode_share": round(float(np.mean(purities > 0.99)), 4),
        "mean_classes_per_episode": round(float(n_classes_per_ep.mean()), 3),
        "single_class_episodes": int(np.sum(n_classes_per_ep == 1)),
        "multi_class_episodes": int(np.sum(n_classes_per_ep > 1)),
        "label_switches_per_100_rows_mean": round(float(switches.mean()), 3),
        "episodes_per_class": class_eps,
        "min_episodes_for_any_class": int(min(class_eps.values())),
        "consecutive_row_distance_median": round(float(np.median(step)), 4),
        "consecutive_row_distance_q05": round(float(np.percentile(step, 5)), 4),
    }


def main() -> None:
    results = [analyse(n, p) for n, p in DATASETS.items() if p.exists()]
    for r in results:
        print("=" * 88)
        print(f"{r['dataset']}  rows={r['rows']} episodes={r['episodes']} "
              f"median_episode_rows={r['median_episode_rows']}")
        print(f"  episode purity      mean={r['episode_purity_mean']:.4f} "
              f"median={r['episode_purity_median']:.4f} "
              f"pure(>0.99) share={r['pure_episode_share']:.4f}")
        print(f"  classes per episode mean={r['mean_classes_per_episode']:.3f}  "
              f"single-class eps={r['single_class_episodes']} "
              f"multi-class eps={r['multi_class_episodes']}")
        print(f"  label switches / 100 rows = {r['label_switches_per_100_rows_mean']:.3f}")
        print(f"  episodes per class = {r['episodes_per_class']} "
              f"(min {r['min_episodes_for_any_class']})")
        print(f"  consecutive-row standardized distance median="
              f"{r['consecutive_row_distance_median']:.4f} "
              f"q05={r['consecutive_row_distance_q05']:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "物理判据_episode纯度分析.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame([{
        "数据集": r["dataset"],
        "episode数": r["episodes"],
        "episode中位行数": r["median_episode_rows"],
        "episode纯度均值": r["episode_purity_mean"],
        "纯单类episode占比": r["pure_episode_share"],
        "单类episode数": r["single_class_episodes"],
        "多类episode数": r["multi_class_episodes"],
        "每百行标签跳变次数": r["label_switches_per_100_rows_mean"],
        "最少类占用episode数": r["min_episodes_for_any_class"],
        "相邻行标准化距离中位数": r["consecutive_row_distance_median"],
    } for r in results])
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMPONENTS / "表25_episode纯度与标签时间结构.csv", index=False,
                 encoding="utf-8-sig")
    print("\n" + frame.to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
