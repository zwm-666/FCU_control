"""Export the equal-compute comparison tables -- the honest headline table.

WHY THIS EXISTS
  The thesis' main table gives the proposed models first place, but the
  baselines there are deliberately de-tuned (TCN/MTGNN/PatchTST 3 epochs,
  Transformer/GATv2/iTransformer 5, LightGBM 6 trees, XGBoost 15) while DI/AB
  get standard capacity and the full 100 epochs. Any reviewer can ask what
  happens at equal compute, so it must be measured and reported, not avoided.

TWO EQUAL-COMPUTE VARIANTS ARE EXPORTED
  equal_compute_designB
      every baseline gets 100 epochs / standard capacity.
      DI/AB still carry proposed_validation_split=0.15, i.e. they fit only
      5,779 of 6,799 training rows while the tree models fit all 6,799.
  equal_compute_noval_designB
      identical, except proposed_validation_split=0.0 so DI/AB also fit all
      6,799 rows. Measured cost of the 0.15 split: 0.4314 pp (99.1176% ->
      99.5490% on seeds 42-44), so this is a fairness correction, not tuning.

Both are reported side by side. The reduced-baseline table is kept as the
"engineering-deployable configuration" reading, clearly labelled as such.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

STUDIES = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/studies")
OUT = Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS = OUT / "表格分项归档" / "current_components"

RUNS = [
    ("reduced", "降配基线（原主表）", "physics3_designB"),
    ("equal", "等算力（本文仍切15%验证）", "equal_compute_designB"),
    ("equal_fair", "等算力+公平数据量", "equal_compute_noval_designB"),
]
CN = {
    "di_emstgat": "DI-EMSTGAT（本文）", "ab_emstgat": "AB-EMSTGAT（本文）",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "patchtst": "PatchTST",
    "mtgnn": "MTGNN", "tcn": "TCN", "transformer": "Transformer",
    "itransformer": "iTransformer", "gatv2": "GATv2",
}
PROPOSED = {"di_emstgat", "ab_emstgat"}
METRICS = [("accuracy", "Accuracy"), ("f1_macro", "MacroF1"),
           ("cohen_kappa", "CohenKappa")]


def load(dirname: str) -> Optional[pd.DataFrame]:
    path = STUDIES / dirname / "per_seed_records.json"
    if not path.exists():
        return None
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return frame if len(frame) else None


def save(df: pd.DataFrame, name: str) -> None:
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPONENTS / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{name}.csv  rows={len(df)}")


def main() -> None:
    loaded = []
    for key, label, dirname in RUNS:
        frame = load(dirname)
        if frame is None:
            print(f".. {key} not available ({dirname})")
            continue
        n = frame.seed.nunique()
        print(f"loaded {key:11s} {len(frame):3d} records, {n} seeds"
              f"{'' if n >= 5 else '  (INCOMPLETE)'}")
        loaded.append((key, label, frame, n))
    if not loaded:
        return

    # ---------- per-condition ranking ----------
    rows = []
    for key, label, frame, n in loaded:
        for model in frame.model.unique():
            g = frame[frame.model == model]
            row = {"配置": label, "condition": key, "模型": CN.get(model, model),
                   "model_key": model, "种子数": int(g.seed.nunique()),
                   "是否本文": "本文" if model in PROPOSED else "基线"}
            for metric, name in METRICS:
                row[f"{name}(%)"] = round(100 * g[metric].mean(), 4)
                row[f"{name}_std"] = round(
                    100 * g[metric].std(ddof=1) if len(g) > 1 else 0.0, 4)
            row["训练耗时(s)"] = round(float(g.fit_time.mean()), 1)
            rows.append(row)
    table = pd.DataFrame(rows)
    ranked = []
    for key, label, _, _ in loaded:
        sub = table[table.condition == key].sort_values("Accuracy(%)",
                                                        ascending=False).copy()
        sub.insert(0, "排名", np.arange(1, len(sub) + 1))
        ranked.append(sub)
    ranked = pd.concat(ranked, ignore_index=True)
    save(ranked, "表44_等算力对比_三配置排名")

    for key, label, _, _ in loaded:
        sub = ranked[ranked.condition == key]
        print(f"\n=== {label} ===")
        print(sub[["排名", "模型", "Accuracy(%)", "Accuracy_std",
                   "MacroF1(%)", "训练耗时(s)", "是否本文"]].to_string(index=False))
        prop = sub[sub.model_key.isin(PROPOSED)]
        base = sub[~sub.model_key.isin(PROPOSED)]
        if len(prop) and len(base):
            bp, bb = prop["Accuracy(%)"].max(), base["Accuracy(%)"].max()
            best_base = base.loc[base["Accuracy(%)"].idxmax(), "模型"]
            best_prop = prop.loc[prop["Accuracy(%)"].idxmax(), "模型"]
            gap = bp - bb
            print(f"  最优本文 {best_prop} {bp:.4f}%  vs  最强基线 {best_base} "
                  f"{bb:.4f}%   差距 {gap:+.4f} pp"
                  f"  -> {'本文领先' if gap > 0 else '本文落后'}")

    # ---------- paired: proposed vs strongest baseline, per condition ----------
    rows = []
    for key, label, frame, n in loaded:
        base = frame[~frame.model.isin(PROPOSED)]
        if base.empty:
            continue
        best = base.groupby("model").accuracy.mean().idxmax()
        b = base[base.model == best].set_index("seed").sort_index()
        for model in ("di_emstgat", "ab_emstgat"):
            a = frame[frame.model == model].set_index("seed").sort_index()
            if a.empty:
                continue
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            for metric, name in METRICS:
                d = (a.loc[common, metric] - b.loc[common, metric]).to_numpy(float)
                if np.allclose(d, 0):
                    pt = pw = 1.0
                else:
                    pt = float(ttest_rel(a.loc[common, metric],
                                         b.loc[common, metric]).pvalue)
                    try:
                        pw = float(wilcoxon(d).pvalue)
                    except ValueError:
                        pw = float("nan")
                rows.append({
                    "配置": label, "condition": key, "本文模型": CN[model],
                    "指标": name, "最强基线": CN.get(best, best),
                    "本文(%)": round(100 * a.loc[common, metric].mean(), 4),
                    "基线(%)": round(100 * b.loc[common, metric].mean(), 4),
                    "差值(pp)": round(100 * float(d.mean()), 4),
                    "配对t检验p": round(pt, 6),
                    "Wilcoxon p": round(pw, 6) if np.isfinite(pw) else np.nan,
                    "本文胜出种子": f"{int((d > 0).sum())}/{len(d)}",
                    "结论": ("本文显著领先" if d.mean() > 0 and pt < 0.05
                           else "本文领先但不显著" if d.mean() > 0
                           else "本文显著落后" if pt < 0.05
                           else "本文落后但不显著"),
                })
    if rows:
        paired = pd.DataFrame(rows)
        save(paired, "表45_各配置下本文与最强基线的配对检验")
        print("\n=== 本文 vs 最强基线（各配置，Accuracy）===")
        print(paired[paired.指标 == "Accuracy"][
            ["配置", "本文模型", "最强基线", "本文(%)", "基线(%)",
             "差值(pp)", "配对t检验p", "本文胜出种子", "结论"]].to_string(index=False))

    # ---------- the validation-split fairness effect ----------
    eq = next((f for k, _, f, _ in loaded if k == "equal"), None)
    eqf = next((f for k, _, f, _ in loaded if k == "equal_fair"), None)
    if eq is not None and eqf is not None:
        rows = []
        for model in ("di_emstgat", "ab_emstgat"):
            a = eq[eq.model == model].set_index("seed").sort_index()
            b = eqf[eqf.model == model].set_index("seed").sort_index()
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            d = (b.loc[common, "accuracy"] - a.loc[common, "accuracy"]).to_numpy(float)
            pt = 1.0 if np.allclose(d, 0) else float(
                ttest_rel(b.loc[common, "accuracy"], a.loc[common, "accuracy"]).pvalue)
            rows.append({
                "模型": CN[model],
                "切15%验证集(%)": round(100 * a.loc[common, "accuracy"].mean(), 4),
                "用全部训练数据(%)": round(100 * b.loc[common, "accuracy"].mean(), 4),
                "提升(pp)": round(100 * float(d.mean()), 4),
                "配对t检验p": round(pt, 6),
                "改善种子": f"{int((d > 0).sum())}/{len(d)}",
                "说明": "仅改 proposed_validation_split，其余完全一致；"
                       "树模型始终用全部6799行，故这是公平性修正",
            })
        if rows:
            save(pd.DataFrame(rows), "表46_验证集切分的公平性影响")
            print("\n=== 验证集切分的影响（公平性修正）===")
            print(pd.DataFrame(rows).to_string(index=False))

    contract = {
        "conditions": [{"key": k, "label": l, "dir": d, "seeds": int(f.seed.nunique())}
                       for (k, l, f, _), (_, _, d) in
                       zip(loaded, [(a, b, c) for a, b, c in RUNS
                                    if any(a == k for k, _, _, _ in loaded)])],
        "baseline_policy_reduced": {
            "tcn_mtgnn_patchtst_epochs": 3,
            "transformer_gatv2_itransformer_epochs": 5,
            "lightgbm_trees": 6, "xgboost_trees": 15,
            "proposed": "standard capacity, 100 epochs",
        },
        "baseline_policy_equal": "all models 100 epochs / standard capacity",
        "validation_split_issue": (
            "proposed_validation_split=0.15 removed 15% of the training rows "
            "from DI/AB only (5,779 of 6,799) while tree baselines fit all "
            "6,799. Measured cost 0.4314 pp on seeds 42-44. Setting it to 0.0 "
            "is a fairness correction, not tuning."),
        "reporting_rule": (
            "the reduced-baseline table may be presented as the "
            "engineering-deployable configuration, but the equal-compute table "
            "must be reported alongside it; claiming first place from the "
            "reduced table alone is misleading"),
        "outer_test_evaluated": True,
    }
    (OUT / "等算力对比契约.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
