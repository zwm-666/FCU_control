"""Re-export historical noise/ablation result tables from preserved raw studies.

This script does not train or alter any raw study root. It reconstructs the
paper-side historical tables from their stored per-seed JSON so the user's
"preserve experiment results" requirement is literal, while the current aligned
DesignB_plus_v1 tables remain separate and are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/studies")
OUT = Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS = OUT / "表格分项归档" / "current_components"

NOISE = ROOT / "noise_physics3_designB"
ABL = {
    "干净条件": ROOT / "ablation_physics3_designB",
    "SNR=10dB": ROOT / "ablation_snr10_designB",
    "训练集20%": ROOT / "ablation_train20pct_designB",
}
CN = {
    "di_emstgat": "DI-EMSTGAT（本文）", "ab_emstgat": "AB-EMSTGAT（本文）",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "patchtst": "PatchTST",
    "mtgnn": "MTGNN", "tcn": "TCN", "transformer": "Transformer",
    "itransformer": "iTransformer", "gatv2": "GATv2",
}
ORDER = ["di_emstgat", "ab_emstgat", "patchtst", "mtgnn", "tcn", "xgboost",
         "lightgbm", "itransformer", "transformer", "gatv2"]
VARIANT_CN = {
    "full": "完整模型", "no_dcc": "去除膨胀因果卷积", "no_bigru": "去除双向GRU",
    "no_knn_graph": "去除时序KNN图", "no_self_attn": "去除多头自注意力",
    "no_attn_pool": "注意力池化替为均值池化", "no_skip_cls": "去除表格跳连分类器",
    "no_pos_emb": "去除位置编码", "no_boundary": "去除边界损失",
    "no_router_gate": "去除特征相关性门控",
}
VARIANTS = {
    "di_emstgat": ["full", "no_dcc", "no_bigru", "no_knn_graph", "no_self_attn",
                    "no_attn_pool", "no_skip_cls", "no_pos_emb", "no_boundary"],
    "ab_emstgat": ["full", "no_dcc", "no_bigru", "no_knn_graph", "no_self_attn",
                    "no_attn_pool", "no_skip_cls", "no_pos_emb", "no_boundary",
                    "no_router_gate"],
}


def records(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads((path / "per_seed_records.json").read_text(encoding="utf-8")))


def save(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    path = COMPONENTS / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return {"name": name, "rows": int(len(frame)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": int(path.stat().st_size)}


def noise_tables(frame: pd.DataFrame) -> list[dict[str, Any]]:
    outputs = []
    grid = sorted({float(x) for x in frame.snr_db if x != "clean"}, reverse=True)
    cols = ["clean", *grid]
    for metric, name in (("accuracy", "表34_噪声鲁棒性_准确率随SNR.csv"),
                         ("f1_macro", "表35_噪声鲁棒性_宏F1随SNR.csv"),
                         ("minority_recall", "表37_噪声鲁棒性_最小类召回随SNR.csv")):
        rows = []
        for model in ORDER:
            g = frame[frame.model == model]
            if g.empty:
                continue
            row = {"模型": CN[model], "model_key": model,
                   "种子数": int(g.seed.nunique())}
            for c in cols:
                sub = g[g.snr_db == c]
                tag = "无噪" if c == "clean" else f"{int(c)}dB"
                row[f"{tag}_均值(%)"] = round(100 * sub[metric].mean(), 4)
                row[f"{tag}_标准差(%)"] = round(
                    100 * sub[metric].std(ddof=1) if len(sub) > 1 else 0.0, 4)
            rows.append(row)
        outputs.append(save(pd.DataFrame(rows), name))

    rows = []
    for model in ORDER:
        g = frame[frame.model == model]
        if g.empty:
            continue
        clean = 100 * g[g.snr_db == "clean"].accuracy.mean()
        accs = [100 * g[g.snr_db == c].accuracy.mean() for c in grid]
        drops = [clean - x for x in accs]
        row = {"模型": CN[model], "model_key": model,
               "无噪准确率(%)": round(clean, 4),
               "全网格平均准确率(%)": round(float(np.mean(accs)), 4),
               "最大下降(个百分点)": round(float(max(drops)), 4),
               "5dB准确率(%)": round(accs[-1], 4)}
        for c, drop in zip(grid, drops):
            row[f"{int(c)}dB_下降(个百分点)"] = round(drop, 4)
        rows.append(row)
    degradation = pd.DataFrame(rows).sort_values(
        "全网格平均准确率(%)", ascending=False).reset_index(drop=True)
    degradation.insert(0, "鲁棒性排名", np.arange(1, len(degradation) + 1))
    outputs.append(save(degradation, "表36_噪声退化与鲁棒性排名.csv"))

    detail = frame.copy()
    detail.insert(0, "模型", detail.model.map(CN))
    for c in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
              "minority_recall"):
        detail[c] = (100 * detail[c]).round(4)
    outputs.append(save(detail, "表38_噪声鲁棒性逐种子原始记录.csv"))
    return outputs


def pair(a: pd.Series, b: pd.Series) -> tuple[float, float, float, int, int]:
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    if len(d) < 2 or np.allclose(d, 0):
        return (100 * float(d.mean()) if len(d) else 0.0, 1.0, 1.0,
                int((d < 0).sum()), len(d))
    pt = float(ttest_rel(a.loc[common], b.loc[common]).pvalue)
    try:
        pw = float(wilcoxon(d).pvalue)
    except ValueError:
        pw = np.nan
    return 100 * float(d.mean()), pt, pw, int((d < 0).sum()), len(d)


def ablation_clean_tables(frame: pd.DataFrame) -> list[dict[str, Any]]:
    outputs = []
    rows = []
    for model in ("di_emstgat", "ab_emstgat"):
        sub = frame[frame.model == model]
        full = sub[sub.variant == "full"].set_index("seed").sort_index()
        for variant in VARIANTS[model]:
            g = sub[sub.variant == variant].set_index("seed").sort_index()
            d, pt, pw, worse, n = pair(g.accuracy, full.accuracy)
            rows.append({
                "模型": CN[model], "model_key": model, "变体": VARIANT_CN[variant],
                "variant_key": variant, "参数量": int(g.param_count.iloc[0]),
                "种子数": n, "accuracy_均值(%)": 100 * g.accuracy.mean(),
                "accuracy_标准差(%)": 100 * g.accuracy.std(ddof=1),
                "准确率变化(个百分点)": d, "配对t检验p": pt,
                "Wilcoxon p": pw, "变体更差的种子": "—" if variant == "full" else f"{worse}/{n}",
                "结论": "基准" if variant == "full" else "历史单条件消融",
            })
    outputs.append(save(pd.DataFrame(rows), "表39_消融实验_组件贡献与显著性.csv"))
    detail = frame.copy()
    detail.insert(0, "模型", detail.model.map(CN))
    for c in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "cohen_kappa"):
        detail[c] = (100 * detail[c]).round(4)
    outputs.append(save(detail, "表40_消融实验逐种子原始记录.csv"))

    f1_cols = [c for c in frame.columns if c.startswith("f1_")
               and c not in {"f1_macro", "f1_weighted"}]
    f1_rows = []
    for model in ("di_emstgat", "ab_emstgat"):
        sub = frame[frame.model == model]
        for variant in sub.variant.unique():
            g = sub[sub.variant == variant]
            row = {"模型": CN[model], "变体": VARIANT_CN[variant], "variant_key": variant}
            for col in f1_cols:
                row[f"{col[3:]}_F1均值(%)"] = round(100 * g[col].mean(), 4)
            f1_rows.append(row)
    outputs.append(save(pd.DataFrame(f1_rows), "表41_消融实验_类别级F1.csv"))
    return outputs


def condition_tables(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    outputs = []
    rows = []
    for condition, frame in frames.items():
        for model in ("di_emstgat", "ab_emstgat"):
            g = frame[frame.model == model]
            full = g[g.variant == "full"].set_index("seed").accuracy.sort_index()
            for variant in VARIANTS[model]:
                v = g[g.variant == variant].set_index("seed").accuracy.sort_index()
                d, pt, pw, worse, n = pair(v, full)
                rows.append({"条件": condition, "模型": CN[model], "model_key": model,
                             "变体": VARIANT_CN[variant], "variant_key": variant,
                             "参数量": int(g[g.variant == variant].param_count.iloc[0]),
                             "完整模型Accuracy(%)": 100 * full.mean(),
                             "变体Accuracy(%)": 100 * v.mean(),
                             "变化(pp)": d, "配对t检验p": pt, "Wilcoxon p": pw,
                             "变体更差种子": "—" if variant == "full" else f"{worse}/{n}"})
    outputs.append(save(pd.DataFrame(rows), "表42_消融条件的区分能力对照.csv"))

    wide = []
    for model in ("di_emstgat", "ab_emstgat"):
        variants = [v for v in VARIANTS[model] if v not in {"full", "no_boundary"}]
        for variant in variants:
            row = {"模型": CN[model], "model_key": model,
                   "变体": VARIANT_CN[variant], "variant_key": variant}
            signs = []
            for condition, frame in frames.items():
                sub = frame[frame.model == model]
                full = sub[sub.variant == "full"].set_index("seed").accuracy.sort_index()
                v = sub[sub.variant == variant].set_index("seed").accuracy.sort_index()
                d, pt, _, _, n = pair(v, full)
                row[f"{condition}变化(pp)"] = d
                row[f"{condition}p"] = pt
                signs.append(int(np.sign(d)))
            nz = [x for x in signs if x]
            row["方向一致性"] = ("三条件一致：移除后更差" if nz and all(x < 0 for x in nz)
                             else "三条件一致：移除后更好" if nz and all(x > 0 for x in nz)
                             else "各条件均为零" if not nz else "条件间方向不一致，不能判定")
            wide.append(row)
    outputs.append(save(pd.DataFrame(wide), "表43_消融三条件对照与方向一致性.csv"))
    return outputs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    noise = records(NOISE)
    outputs.extend(noise_tables(noise))
    old_frames = {name: records(path) for name, path in ABL.items()}
    outputs.extend(ablation_clean_tables(old_frames["干净条件"]))
    outputs.extend(condition_tables(old_frames))
    audit = {
        "kind": "historical_result_table_reexport",
        "source_roots": {"noise": str(NOISE), "ablation": {k: str(v) for k, v in ABL.items()}},
        "source_record_counts": {"noise": len(noise), **{k: len(v) for k, v in old_frames.items()}},
        "outputs": outputs,
        "raw_study_roots_modified": False,
        "purpose": "Restore paper-side historical tables from preserved JSON; current aligned tables are separate.",
    }
    (OUT / "历史实验结果表保留重导出审计.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": len(outputs), "source_records": audit["source_record_counts"],
                      "audit": str(OUT / "历史实验结果表保留重导出审计.json")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
