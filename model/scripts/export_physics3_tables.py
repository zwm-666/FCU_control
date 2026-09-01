"""Export the physics-labeled 3-class study tables for the thesis.

Reads stored per-seed JSON records only; runs no models. Produces the
candidate-vs-control comparison the thesis needs:

  candidate : 物理判据 design B (load-controlled, all-real)
  control   : 旧合成版 3class_12000 (Flooding 55% KNN-synthesised)

Both were run through the SAME pipeline (run_main_comparison.py), the same 10
models, the same 5 seeds, and the same reduced-baseline policy, so the model
configurations are matched and only the dataset differs.

IMPORTANT HONESTY NOTE carried into every output: the two datasets use
DIFFERENT label definitions, so a raw accuracy difference is NOT evidence that
one method is better. What is comparable, and what the thesis should claim, is
(a) synthetic-row share, (b) duplicate-row share, (c) criterion defensibility,
(d) episode-grouped honest accuracy, and (e) whether the label can proxy the
operating point.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/studies")
CAND = ROOT / "physics3_designB"
CTRL = ROOT / "legacy_synthetic_3class_control"
DATA_DIR = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件")
OUT = Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS = OUT / "表格分项归档" / "current_components"

MODEL_ORDER = ("di_emstgat", "ab_emstgat", "xgboost", "lightgbm", "patchtst",
               "mtgnn", "tcn", "transformer", "itransformer", "gatv2")
CN = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "tcn": "TCN",
    "transformer": "Transformer", "mtgnn": "MTGNN", "gatv2": "GATv2",
    "itransformer": "iTransformer", "patchtst": "PatchTST",
    "di_emstgat": "DI-EMSTGAT（本文）", "ab_emstgat": "AB-EMSTGAT（本文）",
}
LEVEL_CN = {
    "traditional_baseline": "传统机器学习",
    "deep_baseline": "深度序列基线",
    "graph_spatiotemporal_baseline": "图/时空基线",
    "modern_transformer_baseline": "现代Transformer基线",
    "proposed_direct_input": "本文-直接输入",
    "proposed_adaptive_branch": "本文-自动分支",
}
CLASSES = ["Flooding", "Membrane_Drying", "Normal"]
PROPOSED = {"di_emstgat", "ab_emstgat"}
METRICS = [
    ("accuracy", "Accuracy"), ("balanced_accuracy", "BalancedAcc"),
    ("precision_weighted", "Precision_weighted"), ("recall_weighted", "Recall_weighted"),
    ("f1_macro", "MacroF1"), ("f1_weighted", "F1_weighted"),
    ("cohen_kappa", "CohenKappa"),
]


def pct(v) -> float:
    return round(float(v) * 100.0, 4)


def save(df: pd.DataFrame, name: str) -> None:
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPONENTS / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{name}.csv  rows={len(df)}")


def load(root: Path) -> tuple[list[dict], dict]:
    recs = json.loads((root / "per_seed_records.json").read_text(encoding="utf-8"))
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return recs, man


def summary_table(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        g = frame[frame.model == model]
        if g.empty:
            continue
        row = {"模型": CN[model], "model_key": model,
               "层级": LEVEL_CN.get(g.level.iloc[0], g.level.iloc[0]),
               "数据集": tag, "运行次数": int(len(g))}
        for metric, label in METRICS:
            row[f"{label}_均值(%)"] = pct(g[metric].mean())
            row[f"{label}_标准差(%)"] = pct(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
        row["训练耗时均值(s)"] = round(float(g.fit_time.mean()), 4)
        row["推理耗时均值(s)"] = round(float(g.prediction_time.mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cand_recs, cand_man = load(CAND)
    cand = pd.DataFrame(cand_recs)
    have_ctrl = (CTRL / "per_seed_records.json").exists()
    if have_ctrl:
        ctrl_recs, ctrl_man = load(CTRL)
        ctrl = pd.DataFrame(ctrl_recs)
    else:
        ctrl_recs, ctrl_man, ctrl = [], {}, pd.DataFrame()
        print("!! control study not found; candidate-only tables will be written")

    # ---------- main aggregate ----------
    main_tbl = summary_table(cand, "物理判据designB")
    if have_ctrl:
        main_tbl = pd.concat([main_tbl, summary_table(ctrl, "旧合成版3class")],
                             ignore_index=True)
    save(main_tbl, "表26_物理判据数据集主表_10模型5种子")

    # ---------- per-seed detail ----------
    scalar = ["model", "level", "seed", "accuracy", "balanced_accuracy",
              "precision_weighted", "recall_weighted", "f1_macro", "f1_weighted",
              "cohen_kappa", "fit_time", "prediction_time", "feature_count",
              "train_samples", "test_samples"]
    detail = cand[scalar].copy()
    detail.insert(0, "模型", detail["model"].map(CN))
    detail.insert(1, "层级中文", detail["level"].map(lambda x: LEVEL_CN.get(x, x)))
    for metric, _ in METRICS:
        detail[metric] = detail[metric].map(pct)
    save(detail, "表27_物理判据逐种子全部指标")

    # ---------- class-level ----------
    class_rows, cm_rows = [], []
    for rec in cand_recs:
        rep, cm = rec["classification_report"], rec["confusion_matrix"]
        for cls in CLASSES:
            if cls not in rep:
                continue
            v = rep[cls]
            class_rows.append({
                "模型": CN[rec["model"]], "model_key": rec["model"],
                "seed": int(rec["seed"]), "类别": cls,
                "Precision(%)": pct(v["precision"]), "Recall(%)": pct(v["recall"]),
                "F1(%)": pct(v["f1-score"]), "支持数": int(v["support"]),
            })
        cm_rows.append({"模型": CN[rec["model"]], "model_key": rec["model"],
                        "seed": int(rec["seed"]),
                        "混淆矩阵": json.dumps(cm, ensure_ascii=False)})
    class_detail = pd.DataFrame(class_rows)
    save(class_detail, "表28_物理判据逐种子类别指标")
    save(pd.DataFrame(cm_rows), "表29_物理判据混淆矩阵")

    agg = []
    for (model, cls), g in class_detail.groupby(["model_key", "类别"], sort=False):
        agg.append({
            "模型": CN[model], "model_key": model, "类别": cls,
            "Precision均值(%)": round(float(g["Precision(%)"].mean()), 4),
            "Recall均值(%)": round(float(g["Recall(%)"].mean()), 4),
            "F1均值(%)": round(float(g["F1(%)"].mean()), 4),
            "F1标准差(%)": round(float(g["F1(%)"].std(ddof=1)), 4),
            "支持数(每种子)": int(g["支持数"].iloc[0]),
        })
    save(pd.DataFrame(agg), "表30_物理判据类别级指标")

    # ---------- paired candidate vs control, per model ----------
    if have_ctrl:
        paired = []
        for model in MODEL_ORDER:
            a = cand[cand.model == model].set_index("seed").sort_index()
            b = ctrl[ctrl.model == model].set_index("seed").sort_index()
            if a.empty or b.empty:
                continue
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            for metric, label in METRICS:
                x, yv = a.loc[common, metric], b.loc[common, metric]
                d = (x - yv).to_numpy(float)
                pt = 1.0 if np.allclose(d, 0) else float(ttest_rel(x, yv).pvalue)
                try:
                    pw = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
                except ValueError:
                    pw = float("nan")
                paired.append({
                    "模型": CN[model], "model_key": model, "指标": label,
                    "匹配种子数": int(len(common)),
                    "物理判据均值(%)": pct(x.mean()),
                    "旧合成版均值(%)": pct(yv.mean()),
                    "差值(个百分点)": round(float(d.mean()) * 100, 4),
                    "配对t检验p": round(pt, 6),
                    "Wilcoxon p": round(pw, 6),
                    "物理判据胜出种子": f"{int((d > 0).sum())}/{len(d)}",
                    "解读警告": "两数据集标签定义不同，差值不构成方法优劣证据",
                })
        save(pd.DataFrame(paired), "表31_物理判据_vs_旧合成版_配对对照")

    # ---------- proposed vs best baseline, within the physics dataset ----------
    win_rows = []
    for metric, label in METRICS:
        base = cand[~cand.model.isin(PROPOSED)]
        best_base_model = (base.groupby("model")[metric].mean().idxmax())
        bb = base[base.model == best_base_model].set_index("seed").sort_index()
        for model in ("di_emstgat", "ab_emstgat"):
            a = cand[cand.model == model].set_index("seed").sort_index()
            common = a.index.intersection(bb.index)
            x, yv = a.loc[common, metric], bb.loc[common, metric]
            d = (x - yv).to_numpy(float)
            pt = 1.0 if np.allclose(d, 0) else float(ttest_rel(x, yv).pvalue)
            win_rows.append({
                "本文模型": CN[model], "指标": label,
                "本文均值(%)": pct(x.mean()),
                "最强基线": CN[best_base_model],
                "最强基线均值(%)": pct(yv.mean()),
                "领先(个百分点)": round(float(d.mean()) * 100, 4),
                "配对t检验p": round(pt, 6),
                "胜出种子": f"{int((d > 0).sum())}/{len(d)}",
                "说明": "同一数据集、同一划分、同一种子；基线按论文要求降配",
            })
    save(pd.DataFrame(win_rows), "表32_物理判据_本文模型_vs_最强基线")

    # ---------- dataset provenance comparison ----------
    def dsinfo(path: Path, mpath: Path | None) -> dict:
        df = pd.read_csv(path)
        feats = [c for c in df.columns if c not in {"State", "State_Label", "tsec"}]
        live = [c for c in feats if df[c].nunique() > 1]
        info = {
            "行数": int(len(df)),
            "类别数": int(df["State_Label"].nunique()),
            "每类行数": json.dumps(
                {k: int(v) for k, v in df["State_Label"].value_counts().items()},
                ensure_ascii=False),
            "有效特征": len(live),
            "恒定死通道": json.dumps([c for c in feats if c not in live],
                                 ensure_ascii=False),
            "重复特征行": int(df[live].round(9).duplicated().sum()),
        }
        if mpath and mpath.exists():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            syn = m.get("synthetic_counts", {})
            info["合成行总数"] = int(sum(syn.values())) if syn else 0
            info["判据类型"] = (m.get("criterion", {}) or {}).get(
                "kind", m.get("classification_rule", "-"))
        return info

    rows = []
    for tag, csv, man in (
        ("物理判据 designB（本文）", DATA_DIR / "Public datasets_physics3_designB.csv",
         DATA_DIR / "Public datasets_physics3_designB_manifest.json"),
        ("物理判据 designA", DATA_DIR / "Public datasets_physics3_designA.csv",
         DATA_DIR / "Public datasets_physics3_designA_manifest.json"),
        ("旧合成版 3class_12000", DATA_DIR / "Public datasets_3class_12000.csv",
         DATA_DIR / "Public datasets_3class_12000_manifest.json"),
        ("原始公开 4 类", DATA_DIR / "Public datasets.csv", None),
    ):
        if not csv.exists():
            continue
        r = {"数据集": tag}
        r.update(dsinfo(csv, man))
        rows.append(r)
    save(pd.DataFrame(rows), "表33_数据集来源与构成对照")

    # ---------- contract ----------
    contract: dict[str, Any] = {
        "route": "route 1 -- published State_Label discarded; labels recomputed "
                 "from a closed cathode water balance",
        "candidate_root": str(CAND),
        "control_root": str(CTRL) if have_ctrl else None,
        "candidate_records": len(cand_recs),
        "control_records": len(ctrl_recs),
        "candidate_manifest": cand_man,
        "control_manifest": ctrl_man,
        "model_order": list(MODEL_ORDER),
        "classes": CLASSES,
        "matched": "same pipeline, same 10 models, same 5 seeds, same "
                   "reduced-baseline policy; only the dataset differs",
        "comparability_warning": (
            "The candidate and control use DIFFERENT label definitions. A raw "
            "accuracy difference between them is NOT evidence of method "
            "superiority. Comparable claims are limited to: synthetic-row share, "
            "duplicate-row share, criterion defensibility, episode-grouped "
            "honest accuracy, and operating-point confounding."),
        "physics_criterion": (cand_man.get("run_metadata") or [{}])[0],
        "outer_test_evaluated": True,
        "notes": [
            "Test metrics were observed; do not reuse them to tune a further candidate.",
            "Baselines are intentionally reduced per thesis policy; DI/AB keep "
            "standard capacity and the full epoch budget.",
        ],
    }
    (OUT / "物理判据实验契约与审计.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== candidate aggregate (physics design B) ===")
    show = summary_table(cand, "物理判据designB")[
        ["模型", "Accuracy_均值(%)", "Accuracy_标准差(%)", "MacroF1_均值(%)",
         "MacroF1_标准差(%)", "CohenKappa_均值(%)"]]
    print(show.to_string(index=False))
    if have_ctrl:
        print("\n=== control aggregate (legacy synthetic) ===")
        show2 = summary_table(ctrl, "旧合成版")[
            ["模型", "Accuracy_均值(%)", "Accuracy_标准差(%)", "MacroF1_均值(%)"]]
        print(show2.to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
