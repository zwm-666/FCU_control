"""Export noise-robustness and ablation tables for the thesis.

Reads stored per-seed records only; trains nothing.

Noise tables
  表34  accuracy vs SNR, all models, mean +/- std
  表35  macro-F1 vs SNR
  表36  degradation relative to clean, plus an area-under-curve robustness score
  表37  minority-class recall vs SNR (the class that fails first)
  表38  per-seed raw noise records

Ablation tables
  表39  component contribution: delta accuracy/macro-F1 vs the full model,
        paired t and Wilcoxon over the shared seeds, parameter cost
  表40  per-seed ablation records
  表41  per-class F1 by variant
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

STUDIES = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/studies")
# Spec 实验配置与方法说明.md section 13: the aligned runs use the locked
# DesignB_plus_v1 budget (designB_reduced_plus). Table numbers are shifted to
# 48+ so the earlier reduced_baselines tables 34-43 stay intact (spec 7.3.7).
NOISE = STUDIES / "noise_designB_plus_v1"
ABL = STUDIES / "ablation_designB_plus_v1_clean"
ABL_SNR10 = STUDIES / "ablation_designB_plus_v1_snr10"
ABL_TRAIN20 = STUDIES / "ablation_designB_plus_v1_train20pct"
OUT = Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS = OUT / "表格分项归档" / "current_components"

CN = {
    "di_emstgat": "DI-EMSTGAT（本文）", "ab_emstgat": "AB-EMSTGAT（本文）",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "patchtst": "PatchTST",
    "mtgnn": "MTGNN", "tcn": "TCN", "transformer": "Transformer",
    "itransformer": "iTransformer", "gatv2": "GATv2",
}
ORDER = ["di_emstgat", "ab_emstgat", "patchtst", "mtgnn", "tcn", "xgboost",
         "lightgbm", "itransformer", "transformer", "gatv2"]
CLASS_CN = {"Flooding": "水淹", "Membrane_Drying": "膜干", "Normal": "正常"}


def save(df: pd.DataFrame, name: str) -> None:
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPONENTS / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{name}.csv  rows={len(df)}")


def load(root: Path) -> Optional[pd.DataFrame]:
    path = root / "per_seed_records.json"
    if not path.exists():
        return None
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return frame if len(frame) else None


def snr_columns(frame: pd.DataFrame) -> List:
    grid = sorted({float(v) for v in frame.snr_db if v != "clean"}, reverse=True)
    return ["clean"] + grid


def noise_metric_table(frame: pd.DataFrame, metric: str, label: str) -> pd.DataFrame:
    cols = snr_columns(frame)
    rows = []
    for key in ORDER:
        g = frame[frame.model == key]
        if g.empty:
            continue
        row = {"模型": CN[key], "model_key": key,
               "种子数": int(g[g.snr_db == "clean"].shape[0])}
        for c in cols:
            sel = g[g.snr_db == c]
            name = "无噪" if c == "clean" else f"{int(c)}dB"
            row[f"{name}_均值(%)"] = round(100 * sel[metric].mean(), 4)
            row[f"{name}_标准差(%)"] = round(
                100 * sel[metric].std(ddof=1) if len(sel) > 1 else 0.0, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    noise = load(NOISE)
    if noise is not None:
        cols = snr_columns(noise)
        save(noise_metric_table(noise, "accuracy", "Accuracy"),
             "表56_噪声鲁棒性_准确率随SNR_锁定配置")
        save(noise_metric_table(noise, "f1_macro", "MacroF1"),
             "表57_噪声鲁棒性_宏F1随SNR_锁定配置")
        save(noise_metric_table(noise, "minority_recall", "MinorityRecall"),
             "表58_噪声鲁棒性_最小类召回随SNR_锁定配置")

        # The per-SNR degradation ranking is exported by
        # scripts/finalize_aligned_reporting.py as 表69 (with the low-SNR paired
        # tests in 表70). It is computed here only for the console summary so the
        # paper directory keeps exactly one ranking table.
        grid = [c for c in cols if c != "clean"]
        rows = []
        for key in ORDER:
            g = noise[noise.model == key]
            if g.empty:
                continue
            clean = 100 * g[g.snr_db == "clean"].accuracy.mean()
            row = {"模型": CN[key], "model_key": key, "无噪准确率(%)": round(clean, 4)}
            drops = []
            for c in grid:
                acc = 100 * g[g.snr_db == c].accuracy.mean()
                drop = clean - acc
                drops.append(drop)
                row[f"{int(c)}dB_下降(个百分点)"] = round(drop, 4)
            accs = [100 * g[g.snr_db == c].accuracy.mean() for c in grid]
            # mean accuracy across the whole grid = one comparable robustness number
            row["全网格平均准确率(%)"] = round(float(np.mean(accs)), 4)
            row["最大下降(个百分点)"] = round(float(np.max(drops)), 4)
            row["5dB准确率(%)"] = round(
                100 * g[g.snr_db == min(grid)].accuracy.mean(), 4)
            row["鲁棒性排序依据"] = "全网格平均准确率越高越鲁棒"
            rows.append(row)
        degradation = pd.DataFrame(rows).sort_values(
            "全网格平均准确率(%)", ascending=False).reset_index(drop=True)
        degradation.insert(0, "鲁棒性排名", np.arange(1, len(degradation) + 1))
        print("\n=== 噪声退化排名（仅控制台；论文表为表69）===")
        print(degradation[["鲁棒性排名", "模型", "全网格平均准确率(%)",
                           "最大下降(个百分点)"]].to_string(index=False))

        detail = noise.copy()
        detail.insert(0, "模型", detail["model"].map(CN))
        for c in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                  "minority_recall"):
            if c in detail.columns:
                detail[c] = (100 * detail[c]).round(4)
        save(detail, "表60_噪声鲁棒性逐种子原始记录_锁定配置")

        print("\n=== 噪声鲁棒性（准确率，%）===")
        show = noise_metric_table(noise, "accuracy", "Accuracy")
        keep = ["模型"] + [c for c in show.columns if c.endswith("_均值(%)")]
        print(show[keep].to_string(index=False))
    else:
        print(".. noise study not ready")

    abl = load(ABL)
    if abl is not None:
        metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                   "cohen_kappa"]
        rows = []
        for model in ("di_emstgat", "ab_emstgat"):
            sub = abl[abl.model == model]
            if sub.empty:
                continue
            full = sub[sub.variant == "full"].set_index("seed").sort_index()
            if full.empty:
                continue
            for variant in sub.variant.unique():
                g = sub[sub.variant == variant].set_index("seed").sort_index()
                if g.empty:
                    continue
                row = {
                    "模型": CN[model], "model_key": model,
                    "变体": g.variant_cn.iloc[0], "variant_key": variant,
                    "移除组件": ("—" if variant == "full"
                             else g.variant_cn.iloc[0].replace("去除", "")),
                    "参数量": int(g.param_count.iloc[0]),
                    "种子数": int(len(g)),
                }
                for m in metrics:
                    row[f"{m}_均值(%)"] = round(100 * g[m].mean(), 4)
                    row[f"{m}_标准差(%)"] = round(
                        100 * g[m].std(ddof=1) if len(g) > 1 else 0.0, 4)
                if variant == "full":
                    row.update({"准确率变化(个百分点)": 0.0,
                                "宏F1变化(个百分点)": 0.0,
                                "参数量变化": 0, "参数量变化(%)": 0.0,
                                "配对t检验p": np.nan, "Wilcoxon p": np.nan,
                                "变体更差的种子": "—", "结论": "基准（完整模型）"})
                else:
                    common = g.index.intersection(full.index)
                    d_acc = (g.loc[common, "accuracy"]
                             - full.loc[common, "accuracy"]).to_numpy(float)
                    d_f1 = (g.loc[common, "f1_macro"]
                            - full.loc[common, "f1_macro"]).to_numpy(float)
                    if len(common) >= 2 and not np.allclose(d_acc, 0):
                        pt = float(ttest_rel(g.loc[common, "accuracy"],
                                             full.loc[common, "accuracy"]).pvalue)
                        try:
                            pw = float(wilcoxon(d_acc).pvalue)
                        except ValueError:
                            pw = np.nan
                    else:
                        pt, pw = 1.0, 1.0
                    delta = 100 * float(d_acc.mean())
                    dparam = int(g.param_count.iloc[0] - full.param_count.iloc[0])
                    if delta < -0.2 and pt < 0.05:
                        verdict = "组件有效（移除后显著掉点）"
                    elif delta < -0.2:
                        verdict = "组件可能有效（掉点但未达显著）"
                    elif abs(delta) <= 0.2:
                        verdict = "贡献不明显（在种子波动内）"
                    else:
                        verdict = "移除后反而更好（该组件未挣回参数）"
                    row.update({
                        "准确率变化(个百分点)": round(delta, 4),
                        "宏F1变化(个百分点)": round(100 * float(d_f1.mean()), 4),
                        "参数量变化": dparam,
                        "参数量变化(%)": round(
                            100.0 * dparam / int(full.param_count.iloc[0]), 3),
                        "配对t检验p": round(pt, 6),
                        "Wilcoxon p": (round(pw, 6) if np.isfinite(pw) else np.nan),
                        "变体更差的种子": f"{int((d_acc < 0).sum())}/{len(d_acc)}",
                        "结论": verdict,
                    })
                rows.append(row)
        summary = pd.DataFrame(rows)
        # Per-condition variant statistics are exported by
        # scripts/finalize_aligned_reporting.py as 表71 (three conditions in one
        # table) plus 表72; this single-condition view stays console-only.

        # 表62 is the THREE-condition per-seed record table (285 rows) written by
        # scripts/export_aligned_results.py. This script only loads the clean
        # condition, so writing here would truncate it to 95 rows.
        detail = abl.copy()
        detail.insert(0, "模型", detail["model"].map(CN))
        for c in metrics:
            detail[c] = (100 * detail[c]).round(4)
        print(f"clean-condition ablation records loaded: {len(detail)} (表62 is written by export_aligned_results)")

        f1_cols = [c for c in abl.columns if c.startswith("f1_")
                   and c not in {"f1_macro", "f1_weighted"}]
        if f1_cols:
            rows = []
            for model in ("di_emstgat", "ab_emstgat"):
                sub = abl[abl.model == model]
                if sub.empty:
                    continue
                for variant in sub.variant.unique():
                    g = sub[sub.variant == variant]
                    row = {"模型": CN[model], "变体": g.variant_cn.iloc[0]}
                    for c in f1_cols:
                        cls = c[3:]
                        row[f"{CLASS_CN.get(cls, cls)}_F1均值(%)"] = round(
                            100 * g[c].mean(), 4)
                    rows.append(row)
            # Per-class F1 per variant is recoverable from the retained per-seed
            # record table 表62; not exported as a separate paper table.
            print(pd.DataFrame(rows).to_string(index=False))

        contract = {
            "noise_manifest": (json.loads((NOISE / "manifest.json").read_text(
                encoding="utf-8")) if (NOISE / "manifest.json").exists() else None),
            "ablation_manifest": (json.loads((ABL / "manifest.json").read_text(
                encoding="utf-8")) if (ABL / "manifest.json").exists() else None),
            "noise_protocol": ("train once on clean data, then evaluate the same "
                               "fitted model on clean input and on the full "
                               "40->5 dB SNR grid; noise is added to evaluation "
                               "inputs only, so this measures deployment-time "
                               "sensor degradation, not augmentation"),
            "ablation_protocol": ("exactly one component removed per variant; "
                                  "dataset, split, seeds, capacity, epochs, "
                                  "optimizer and boundary loss all held fixed; "
                                  "the full model is re-run inside the study so "
                                  "it shares the variants' code path"),
            "statistics": ("paired t-test and Wilcoxon over the shared seeds, "
                           "because the split is seed-determined"),
            "outer_test_evaluated": True,
        }
        (OUT / "噪声与消融实验契约_锁定配置_v2.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n=== 消融实验（准确率变化，个百分点）===")
        show = summary[["模型", "变体", "参数量", "accuracy_均值(%)",
                        "准确率变化(个百分点)", "配对t检验p", "变体更差的种子",
                        "结论"]]
        print(show.to_string(index=False))
    else:
        print(".. ablation study not ready")

    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
