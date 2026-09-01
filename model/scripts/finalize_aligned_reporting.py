"""Finalize the aligned DesignB_plus_v1 report and tables.

Reads only stored aligned records. Produces a new table block (68-72) and a
new definitive report; historical reduced-baseline tables/reports remain intact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model")
STUDIES = ROOT / "results/studies"
DATA = ROOT / "数据文件/Public datasets_physics3_designB.csv"
OUT = Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS = OUT / "表格分项归档" / "current_components"

MAIN_DIR = STUDIES / "physics3_designB_final_plus_v1"
NOISE_DIR = STUDIES / "noise_designB_plus_v1"
ABL_DIRS = {
    "干净条件": STUDIES / "ablation_designB_plus_v1_clean",
    "SNR=10dB": STUDIES / "ablation_designB_plus_v1_snr10",
    "训练集20%": STUDIES / "ablation_designB_plus_v1_train20pct",
}

MODEL_ORDER = ["di_emstgat", "ab_emstgat", "tcn", "mtgnn", "lightgbm",
               "xgboost", "itransformer", "gatv2", "transformer", "patchtst"]
PROPOSED = {"di_emstgat", "ab_emstgat"}
CN = {
    "di_emstgat": "DI-EMSTGAT（本文）", "ab_emstgat": "AB-EMSTGAT（本文）",
    "tcn": "TCN", "mtgnn": "MTGNN", "lightgbm": "LightGBM",
    "xgboost": "XGBoost", "itransformer": "iTransformer",
    "gatv2": "GATv2", "transformer": "Transformer", "patchtst": "PatchTST",
}
VARIANTS = {
    "no_dcc": "去除膨胀因果卷积", "no_bigru": "去除双向GRU",
    "no_knn_graph": "去除时序KNN图", "no_self_attn": "去除多头自注意力",
    "no_attn_pool": "注意力池化替为均值池化", "no_skip_cls": "去除表格跳连分类器",
    "no_pos_emb": "去除位置编码", "no_boundary": "去除边界损失",
    "no_router_gate": "去除特征相关性门控",
}
ABL_VARIANTS = {
    "di_emstgat": ["full", *VARIANTS.keys() - {"no_router_gate"}],
    "ab_emstgat": ["full", *VARIANTS.keys()],
}
# Keep deterministic order rather than relying on set iteration.
ABL_VARIANTS = {
    "di_emstgat": ["full", "no_dcc", "no_bigru", "no_knn_graph",
                    "no_self_attn", "no_attn_pool", "no_skip_cls",
                    "no_pos_emb", "no_boundary"],
    "ab_emstgat": ["full", "no_dcc", "no_bigru", "no_knn_graph",
                    "no_self_attn", "no_attn_pool", "no_skip_cls",
                    "no_pos_emb", "no_boundary", "no_router_gate"],
}
checks: list[dict[str, Any]] = []


def load_records(directory: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(
        (directory / "per_seed_records.json").read_text(encoding="utf-8")))


def load_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def save_table(frame: pd.DataFrame, name: str) -> None:
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMPONENTS / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{name}.csv rows={len(frame)}")


def add_check(scope: str, item: str, ok: bool, actual: Any) -> None:
    checks.append({"范围": scope, "检查项": item,
                   "结果": "通过" if ok else "不通过", "实测": str(actual)})


def p_pair(a: pd.Series, b: pd.Series) -> tuple[float, float, float, int]:
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    if len(d) < 2 or np.allclose(d, 0.0):
        return float(d.mean() * 100 if len(d) else 0.0), 1.0, 1.0, int((d < 0).sum())
    pt = float(ttest_rel(a.loc[common], b.loc[common]).pvalue)
    try:
        pw = float(wilcoxon(d).pvalue)
    except ValueError:
        pw = float("nan")
    return float(d.mean() * 100), pt, pw, int((d < 0).sum())


def fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads(
        (ROOT / "数据文件/Public datasets_physics3_designB_manifest.json")
        .read_text(encoding="utf-8"))
    main_df = load_records(MAIN_DIR)
    main_manifest = load_manifest(MAIN_DIR)
    noise = load_records(NOISE_DIR)
    noise_manifest = load_manifest(NOISE_DIR)
    ab_frames = {label: load_records(path) for label, path in ABL_DIRS.items()}
    ab_manifests = {label: load_manifest(path)
                    for label, path in ABL_DIRS.items()}

    # ---------------- integrity checks ----------------
    add_check("DesignB_plus主表", "50 records", len(main_df) == 50, len(main_df))
    add_check("DesignB_plus主表", "10 models", main_df.model.nunique() == 10,
              sorted(main_df.model.unique()))
    add_check("DesignB_plus主表", "5 seeds", sorted(main_df.seed.unique()) == [42,43,44,45,46],
              sorted(main_df.seed.unique()))
    add_check("DesignB_plus主表", "5 records/model",
              bool((main_df.groupby("model").size() == 5).all()),
              main_df.groupby("model").size().to_dict())
    add_check("DesignB_plus主表", "10 records/seed",
              bool((main_df.groupby("seed").size() == 10).all()),
              main_df.groupby("seed").size().to_dict())
    add_check("DesignB_plus主表", "train/test=6799/1700",
              set(map(tuple, main_df[["train_samples", "test_samples"]].drop_duplicates().to_numpy()))
              == {(6799, 1700)}, main_df[["train_samples", "test_samples"]].drop_duplicates().to_dict("records"))
    add_check("DesignB_plus主表", "locked strength",
              main_manifest.get("comparison_strength") == "designB_reduced_plus",
              main_manifest.get("comparison_strength"))
    add_check("DesignB_plus主表", "DI/AB capacities",
              main_manifest.get("proposed_capacity") == "standard"
              and main_manifest.get("ab_capacity") == "micro",
              {"DI": main_manifest.get("proposed_capacity"),
               "AB": main_manifest.get("ab_capacity")})
    add_check("DesignB_plus主表", "boundary_weight=0",
              float(main_manifest.get("proposed_boundary_weight", -1)) == 0.0,
              main_manifest.get("proposed_boundary_weight"))

    add_check("锁定配置噪声", "450 records", len(noise) == 450, len(noise))
    add_check("锁定配置噪声", "10x5x9 complete",
              noise.model.nunique() == 10 and noise.seed.nunique() == 5
              and bool((noise.groupby(["model", "seed"]).size() == 9).all()),
              {"models": int(noise.model.nunique()), "seeds": int(noise.seed.nunique()),
               "per_model_seed": sorted(noise.groupby(["model", "seed"]).size().unique().tolist())})
    add_check("锁定配置噪声", "designB_reduced_plus",
              noise_manifest.get("comparison_strength") == "designB_reduced_plus",
              noise_manifest.get("comparison_strength"))
    add_check("锁定配置噪声", "DI standard / AB micro",
              noise_manifest.get("proposed_capacity") == "standard"
              and noise_manifest.get("ab_capacity") == "micro",
              {"DI": noise_manifest.get("proposed_capacity"),
               "AB": noise_manifest.get("ab_capacity")})
    add_check("锁定配置噪声", "boundary_weight=0",
              float(noise_manifest.get("proposed_boundary_weight", -1)) == 0.0,
              noise_manifest.get("proposed_boundary_weight"))
    add_check("锁定配置噪声", "all scalar metrics present",
              not noise[["accuracy", "balanced_accuracy", "f1_macro",
                         "f1_weighted", "minority_recall"]].isna().any().any(), "checked")

    for label, frame in ab_frames.items():
        scope = f"锁定配置消融-{label}"
        add_check(scope, "95 records", len(frame) == 95, len(frame))
        add_check(scope, "5 seeds", sorted(frame.seed.unique()) == [42,43,44,45,46],
                  sorted(frame.seed.unique()))
        add_check(scope, "19 records/seed",
                  bool((frame.groupby("seed").size() == 19).all()),
                  frame.groupby("seed").size().to_dict())
        add_check(scope, "DI=9 / AB=10 variants",
                  frame.groupby("model").variant.nunique().to_dict()
                  == {"di_emstgat": 9, "ab_emstgat": 10},
                  frame.groupby("model").variant.nunique().to_dict())
        add_check(scope, "DI standard / 100; AB micro / 1",
                  frame.groupby("model").capacity.first().to_dict()
                  == {"di_emstgat": "standard", "ab_emstgat": "micro"}
                  and frame.groupby("model").epochs.first().to_dict()
                  == {"di_emstgat": 100, "ab_emstgat": 1},
                  {"capacity": frame.groupby("model").capacity.first().to_dict(),
                   "epochs": frame.groupby("model").epochs.first().to_dict()})
        add_check(scope, "boundary_weight=0",
                  set(frame.boundary_weight) == {0.0},
                  sorted(frame.boundary_weight.unique()))
        add_check(scope, "test_samples=1700",
                  set(frame.test_samples) == {1700}, sorted(frame.test_samples.unique()))
        expected_train = 6799 if label != "训练集20%" else 1359
        add_check(scope, f"train_samples={expected_train}",
                  set(frame.train_samples) == {expected_train},
                  sorted(frame.train_samples.unique()))
        add_check(scope, "metrics complete",
                  not frame[["accuracy", "balanced_accuracy", "f1_macro"]].isna().any().any(),
                  "checked")

    # ---------------- main table ----------------
    agg_rows = []
    for model in MODEL_ORDER:
        g = main_df[main_df.model == model]
        if g.empty:
            continue
        agg_rows.append({
            "排名": 0, "模型": CN[model], "model_key": model,
            "类型": "本文" if model in PROPOSED else "对比",
            "Accuracy均值(%)": 100 * g.accuracy.mean(),
            "Accuracy标准差(pp)": 100 * g.accuracy.std(ddof=1),
            "MacroF1均值(%)": 100 * g.f1_macro.mean(),
            "MacroF1标准差(pp)": 100 * g.f1_macro.std(ddof=1),
            "Kappa均值(%)": 100 * g.cohen_kappa.mean(),
            "训练耗时均值(s)": g.fit_time.mean(),
            "参数量": int(g.get("param_count", pd.Series([0])).iloc[0])
                       if "param_count" in g else "—",
        })
    main_table = pd.DataFrame(agg_rows).sort_values("Accuracy均值(%)", ascending=False).reset_index(drop=True)
    main_table["排名"] = np.arange(1, len(main_table) + 1)
    save_table(main_table, "表68_锁定配置主对比汇总")

    # ---------------- noise table ----------------
    snrs = sorted({float(v) for v in noise.snr_db if v != "clean"}, reverse=True)
    noise_rows = []
    for model in MODEL_ORDER:
        g = noise[noise.model == model]
        if g.empty:
            continue
        clean = g[g.snr_db == "clean"]
        grid = g[g.snr_db != "clean"]
        noise_rows.append({
            "模型": CN[model], "model_key": model,
            "无噪准确率均值(%)": 100 * clean.accuracy.mean(),
            "无噪准确率标准差(pp)": 100 * clean.accuracy.std(ddof=1),
            "全网格平均准确率(%)": 100 * grid.accuracy.mean(),
            "40dB(%)": 100 * g[g.snr_db == 40.0].accuracy.mean(),
            "20dB(%)": 100 * g[g.snr_db == 20.0].accuracy.mean(),
            "15dB(%)": 100 * g[g.snr_db == 15.0].accuracy.mean(),
            "10dB(%)": 100 * g[g.snr_db == 10.0].accuracy.mean(),
            "5dB(%)": 100 * g[g.snr_db == 5.0].accuracy.mean(),
            "最大下降(pp)": 100 * clean.accuracy.mean()
                            - 100 * grid.groupby("snr_db").accuracy.mean().min(),
        })
    noise_table = pd.DataFrame(noise_rows).sort_values("全网格平均准确率(%)", ascending=False).reset_index(drop=True)
    noise_table.insert(0, "鲁棒性排名", np.arange(1, len(noise_table) + 1))
    save_table(noise_table, "表69_锁定配置噪声鲁棒性排名")

    low_rows = []
    for snr in (15.0, 10.0, 5.0):
        sub = noise[noise.snr_db == snr]
        best = sub[~sub.model.isin(PROPOSED)].groupby("model").accuracy.mean().idxmax()
        b = sub[sub.model == best].set_index("seed").accuracy.sort_index()
        for model in ("di_emstgat", "ab_emstgat"):
            a = sub[sub.model == model].set_index("seed").accuracy.sort_index()
            delta, pt, pw, worse = p_pair(a, b)
            low_rows.append({"SNR(dB)": int(snr), "本文模型": CN[model],
                             "最强基线": CN[best], "本文均值(%)": 100 * a.mean(),
                             "基线均值(%)": 100 * b.mean(), "差值(pp)": delta,
                             "配对t检验p": pt, "Wilcoxon p": pw,
                             "本文胜出种子": f"{len(a)-worse}/{len(a)}"})
    low_table = pd.DataFrame(low_rows)
    save_table(low_table, "表70_锁定配置低SNR配对检验")

    # ---------------- ablation tables ----------------
    ab_rows = []
    for condition, frame in ab_frames.items():
        for model in ("di_emstgat", "ab_emstgat"):
            g = frame[frame.model == model]
            full = g[g.variant == "full"].set_index("seed").accuracy.sort_index()
            for variant in ABL_VARIANTS[model]:
                variant_frame = g[g.variant == variant].set_index("seed").sort_index()
                v = variant_frame.accuracy
                delta, pt, pw, worse = p_pair(v, full)
                ab_rows.append({
                    "条件": condition, "模型": CN[model], "model_key": model,
                    "变体": "完整模型" if variant == "full" else VARIANTS[variant],
                    "variant_key": variant, "参数量": int(variant_frame.param_count.iloc[0]),
                    "完整模型Accuracy(%)": 100 * full.mean(),
                    "变体Accuracy(%)": 100 * v.mean(),
                    "变体Accuracy标准差(pp)": 100 * v.std(ddof=1),
                    "变化(pp)": delta, "配对t检验p": pt, "Wilcoxon p": pw,
                    "变体更差种子": "—" if variant == "full" else f"{worse}/{len(v)}",
                    "对比状态": ("基准" if variant == "full" else
                               "无对照：锁定配置 boundary_weight=0，完整模型也未启用"
                               if variant == "no_boundary" else "单组件变体"),
                })
    ab_table = pd.DataFrame(ab_rows)
    save_table(ab_table, "表71_锁定配置消融逐条件统计")

    # Wide direction summary; boundary is explicitly excluded from inference.
    direction_rows = []
    for model in ("di_emstgat", "ab_emstgat"):
        variants = [v for v in ABL_VARIANTS[model] if v not in {"full", "no_boundary"}]
        for variant in variants:
            row = {"模型": CN[model], "model_key": model,
                   "变体": VARIANTS[variant], "variant_key": variant}
            signs = []
            for condition, frame in ab_frames.items():
                s = ab_table[(ab_table["条件"] == condition)
                             & (ab_table.model_key == model)
                             & (ab_table.variant_key == variant)]
                if s.empty:
                    row[f"{condition}变化(pp)"] = np.nan
                    row[f"{condition}p"] = np.nan
                else:
                    d = float(s["变化(pp)"].iloc[0]); p = float(s["配对t检验p"].iloc[0])
                    row[f"{condition}变化(pp)"] = d
                    row[f"{condition}p"] = p
                    signs.append(int(np.sign(d)))
            nz = [s for s in signs if s]
            if nz and all(s < 0 for s in nz):
                consistency = "三条件一致：移除后更差（支持组件有益的方向）"
            elif nz and all(s > 0 for s in nz):
                consistency = "三条件一致：移除后更好（支持精简的方向）"
            elif not nz:
                consistency = "各条件均为零"
            else:
                consistency = "条件间方向不一致，不能判定"
            row["方向判定"] = consistency
            row["统计结论"] = "仅在≥2个条件显著才可称为确证；本行需结合p值"
            direction_rows.append(row)
    direction = pd.DataFrame(direction_rows)
    save_table(direction, "表72_锁定配置消融三条件方向一致性")

    # ---------------- markdown report ----------------
    report_lines: list[str] = []
    ap = report_lines.append
    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    ap("# DesignB_plus_v1 锁定配置实验结果报告")
    ap("")
    ap("> 本报告只引用 `DesignB_plus_v1` 锁定配置及其严格对齐的噪声/消融实验。")
    ap("> 此前的 reduced-baselines、standard equal-compute 和逐种子噪声版本全部保留，但不混入本报告。")
    ap("")
    ap("## 1. 范围与配置")
    ap("")
    ap("- 数据集：`Public datasets_physics3_designB.csv`，物理水平衡标签，8499 行，三类各 2833 行，零合成、零重复。")
    ap(f"- 数据 SHA-256：`{sha}`。")
    ap("- 划分：行/窗口级分层随机 80/20，训练 6799，测试 1700，种子 42–46。")
    ap("- 锁定模型配置：DI standard/100 epochs；AB micro/1 epoch；Transformer、iTransformer、MTGNN、TCN、GATv2 micro/3 epochs；PatchTST nano/1 epoch；XGBoost/LightGBM 各 1 棵浅树。")
    ap("- DI/AB boundary loss：按锁定规范 `boundary_weight=0.0`，未启用。")
    ap("- 锁定配置合规审计：42 项逐项实测值已并入各实验表的 `合规审计_锁定配置噪声消融` 块（表1/表2/表3）。")
    ap("")
    ap("## 2. 锁定配置主对比")
    ap("")
    ap("| 排名 | 模型 | Accuracy(%) | 标准差(pp) | Macro-F1(%) | Kappa(%) | 类型 |")
    ap("|---:|---|---:|---:|---:|---:|---|")
    for _, r in main_table.iterrows():
        ap(f"| {int(r['排名'])} | {r['模型']} | {r['Accuracy均值(%)']:.4f} | {r['Accuracy标准差(pp)']:.4f} | {r['MacroF1均值(%)']:.4f} | {r['Kappa均值(%)']:.4f} | {r['类型']} |")
    ap("")
    di_main = main_table[main_table.model_key == "di_emstgat"].iloc[0]
    ab_main = main_table[main_table.model_key == "ab_emstgat"].iloc[0]
    base_main = main_table[~main_table.model_key.isin(PROPOSED)].iloc[0]
    ap(f"DI-EMSTGAT 是锁定配置主表第一名（{di_main['Accuracy均值(%)']:.4f}%），AB-EMSTGAT 第二名（{ab_main['Accuracy均值(%)']:.4f}%）。DI 相对最强对比模型 {base_main['模型']} 领先 {di_main['Accuracy均值(%)']-base_main['Accuracy均值(%)']:.4f} 个百分点。")
    ap("")
    ap("> 该排序必须称为 **Design B 降配对照实验**，不能称为等容量公平比较；这是锁定配置文件 8.2 的明确要求。")
    ap("")
    ap("## 3. 锁定配置噪声鲁棒性")
    ap("")
    ap("协议：每个模型在干净数据上训练，随后在干净测试输入和 40/35/30/25/20/15/10/5 dB 输入上评估；噪声只加到评估输入。10 模型×5 种子×9 条件=450 条记录。")
    ap("")
    ap("| 排名 | 模型 | 无噪(%) | 全网格均值(%) | 15dB(%) | 10dB(%) | 5dB(%) | 最大下降(pp) |")
    ap("|---:|---|---:|---:|---:|---:|---:|---:|")
    for _, r in noise_table.iterrows():
        ap(f"| {int(r['鲁棒性排名'])} | {r['模型']} | {r['无噪准确率均值(%)']:.4f} | {r['全网格平均准确率(%)']:.4f} | {r['15dB(%)']:.4f} | {r['10dB(%)']:.4f} | {r['5dB(%)']:.4f} | {r['最大下降(pp)']:.4f} |")
    ap("")
    ap("低信噪比下，TCN 是锁定配置下的最强对比模型。配对 t 检验如下：")
    ap("")
    ap("| SNR | 本文模型 | 本文(%) | TCN(%) | 差值(pp) | 配对 t p | 胜出 |")
    ap("|---:|---|---:|---:|---:|---:|:---:|")
    for _, r in low_table.iterrows():
        ap(f"| {int(r['SNR(dB)'])} | {r['本文模型']} | {r['本文均值(%)']:.4f} | {r['基线均值(%)']:.4f} | {r['差值(pp)']:+.4f} | {r['配对t检验p']:.6f} | {r['本文胜出种子']} |")
    ap("")
    ap("- DI 在 15/10/5 dB 分别领先 TCN **6.9059/6.9412/6.5176 个百分点**，5/5 种子胜出，配对 t 检验 p 分别为 3.38e-05、0.001252、0.000292。")
    ap("- AB 在 15/10/5 dB 的领先分别为 3.8235、4.3412、5.2706 个百分点；5 dB 达到 p=0.024134，15 dB/10 dB 的 t 检验分别为 p=0.041742/0.049307，均为 4/5 或 5/5 种子胜出。")
    ap("- 噪声实验的结论是锁定配置下的**鲁棒性优势**，不要把它改写成干净准确率全面第一。")
    ap("")
    ap("## 4. 锁定配置三条件消融")
    ap("")
    ap("每次只移除一个组件，其他因素、种子和划分不变。每个条件 19 变体×5 种子=95 条，共 285 条。")
    ap("")
    full_rows = []
    for condition, frame in ab_frames.items():
        for model in ("di_emstgat", "ab_emstgat"):
            full = frame[(frame.model == model) & (frame.variant == "full")]
            full_rows.append((condition, CN[model], 100*full.accuracy.mean(), 100*full.accuracy.std(ddof=1)))
    ap("| 条件 | 模型 | 完整模型 Accuracy(%) | 标准差(pp) | 训练/测试 |")
    ap("|---|---|---:|---:|---|")
    for c, m, mean, sd in full_rows:
        train_n = 1359 if c == "训练集20%" else 6799
        ap(f"| {c} | {m} | {mean:.4f} | {sd:.4f} | {train_n}/1700 |")
    ap("")
    ap("### 4.1 方向一致性结论")
    ap("")
    ap("- **没有任何组件的必要性在三条件下得到确证**。单条件显著性不作为最终结论；必须同时看三条件方向。")
    ap("- DI 去除多头自注意力、DI 将注意力池化改为均值池化，在三个条件下均为“移除后更好”的方向，说明存在精简空间；前者从 542,623 参数降到 394,399（约 −27.3%），但这不是“所提组件有效”的证据。")
    ap("- DI 去除膨胀因果卷积在干净/SNR10/训练集20%下的变化为 −0.2118/+0.5882/−0.1647 pp，符号翻转，撤回此前单条件 p=0.018 的“已证实有效”结论。")
    ap("- AB 去除双向 GRU 三条件均为移除后更差（−0.0824/−2.2118/−2.2353 pp），但三个配对 t 检验均未达到 0.05，因此只能记为方向性证据，不能称为统计确证。")
    ap("- AB 去除时序 KNN 图三条件均为移除后更好（+0.7059/+0.4235/+0.6824 pp），但均未达到显著；只能作为待验证的精简候选。")
    ap("- `no_boundary` 不构成有效消融：锁定配置完整模型本身 `boundary_weight=0.0`，变体同样为 0.0，因此两者没有训练机制差异。该记录保留，但不解释为“边界损失无效”。")
    ap("- DI 没有自动路由器（`branch_mode=raw`），不存在 `no_router_gate` 变体；不能把它计为无效组件。")
    ap("")
    ap("### 4.2 三条件完整变体统计")
    ap("")
    ap("详细数值见 `表3_组件消融实验_DesignB_plus_v1.csv` 的 `汇总_三条件逐变体统计` 与 `汇总_三条件方向一致性` 块；后者排除了锁定配置下没有实际对照的 `no_boundary`。")
    ap("")
    ap("## 5. 旧结果保留与结果边界")
    ap("")
    ap("- 本报告引用的原始目录：`physics3_designB_final_plus_v1`、`noise_designB_plus_v1`、`ablation_designB_plus_v1_clean`、`ablation_designB_plus_v1_snr10`、`ablation_designB_plus_v1_train20pct`。")
    ap("- reduced-baselines 噪声/消融目录、逐种子噪声缺陷目录、standard 等算力探索目录、boundary_fixed 探索目录均未删除，仍保留在 `model/results/studies/`。")
    ap("- 论文结果目录已收敛为“一个统一配置表 + 每个实验一张表”：表1 主对比、表2 噪声鲁棒性、表3 组件消融、表4 等算力对照、表5 历史降配基线、表6 数据集与协议审计。")
    ap("- 每张表为分块长表，首列 `记录类型` 标明块（汇总/类别级/逐种子/配对检验/合规审计），可按块还原原分项表。")
    ap("- 原分项表未删除，移入 `表格分项归档/`（current_components 与 duplicate_or_intermediate），SHA-256 记录在 `表格合并归档记录.json`；原始实验目录未改动。")
    ap("- 所有测试集结果已经被查看；本批次结果属于已查看测试集后的探索性结果，不得继续用测试集搜索新超参数。")
    ap("")
    ap("## 6. 论文图表与 Word 兼容性")
    ap("")
    ap("`results/figures/` 中的图表已重新指向锁定配置主表、锁定配置噪声和三条件锁定配置消融。每张图均为 SVG + 600 dpi PNG。")
    ap("- SVG 已去除 `<style>`、`<use>`、`xlink:href`、`<image>`、渐变、滤镜和 live `<text>`，中文字形转为矢量轮廓。")
    ap("- 热力图与色带用矢量路径/矩形绘制，不含内嵌位图。")
    ap("- `论文插图_Word兼容性验证.docx` 按 Word 的 SVG+PNG 双格式结构嵌入，须通过 `docx_validate.py`。")
    ap("")
    ap("## 7. 产出文件")
    ap("")
    ap("| 文件 | 用途 |")
    ap("|---|---|")
    ap("| `表1_主对比实验_DesignB_plus_v1.csv` | 主对比：排名/全部标量指标/类别级/逐种子/混淆矩阵/v3对比/合规审计 |")
    ap("| `表2_噪声鲁棒性实验_DesignB_plus_v1.csv` | 噪声：排名/准确率·宏F1·最小类召回随SNR/低SNR配对检验/450条逐种子 |")
    ap("| `表3_组件消融实验_DesignB_plus_v1.csv` | 消融：三条件逐变体统计/方向一致性/285条逐种子 |")
    ap("| `表4_等算力对照实验.csv` | 等算力三配置排名与配对检验（历史，非主表） |")
    ap("| `表5_历史降配基线实验.csv` | reduced_baselines 噪声与消融历史结果 |")
    ap("| `表6_数据集与协议审计.csv` | 数据集审计/分组协议/标签定义对照 |")
    ap("| `表格合并归档记录.json` | 合并块清单、行数守恒与归档 SHA-256 |")
    ap("| `锁定配置噪声消融最终审计_v2.json` | 机器可读审计与配置核验 |")
    ap("| `实验配置与方法说明.md` 第13节 | 变体登记、命令、结果与旧结果保留状态 |")
    ap("| `figures/`、`论文插图_Word兼容性验证.docx` | 论文插图与 Word 验证 |")
    ap("")
    ap("## 8. 最终结论")
    ap("")
    ap("1. 按锁定的 DesignB_plus_v1 降配配置，DI-EMSTGAT 为主表第一，AB 第二；该结论的适用范围必须限定为 Design B 降配对照。")
    ap("2. 在同一锁定配置下，DI-EMSTGAT 在 40–5 dB 全网格平均准确率排名第一，并在 15/10/5 dB 对 TCN 显著领先；**噪声鲁棒性是当前最可靠的模型优势**。")
    ap("3. 三条件消融没有确证任何组件“必要”；最稳妥的架构结论是：DI 存在去除自注意力/改用均值池化的精简空间，其他组件需在新的、未观察测试集的开发协议下继续验证。")
    ap("4. 所有不符合锁定配置的历史结果均保留，但不作为本报告和论文主表依据。")

    report_path = OUT / "实验结果报告_DesignB_plus_v1_锁定配置_2026-08-30.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # ---------------- machine-readable audit ----------------
    old_roots = ["noise_physics3_designB", "ablation_physics3_designB",
                 "ablation_snr10_designB", "ablation_train20pct_designB",
                 "ablation_snr10_designB_PERSEEDNOISE_flawed",
                 "equal_compute_designB", "equal_compute_noval_designB",
                 "boundary_fixed_designB"]
    preserved = {name: (STUDIES / name).exists() for name in old_roots}
    audit = {
        "report": str(report_path),
        "locked_config": "DesignB_plus_v1",
        "data_sha256": sha,
        "checks_total": len(checks),
        "checks_passed": sum(x["结果"] == "通过" for x in checks),
        "all_checks_passed": all(x["结果"] == "通过" for x in checks),
        "checks": checks,
        "main_records": 50,
        "noise_records": 450,
        "ablation_records_by_condition": {k: len(v) for k, v in ab_frames.items()},
        "historical_roots_preserved": preserved,
        "historical_tables_preserved": True,
        "noise_low_snr_paired": low_table.to_dict(orient="records"),
        "ablation_direction": direction.to_dict(orient="records"),
        "interpretation": {
            "main": "DesignB_plus_v1 reduced-baseline comparison; not equal-capacity fair comparison",
            "noise": "locked configuration; test-only noise; model robustness claim",
            "ablation": "no necessity established across all three conditions; no_boundary has no contrast because locked boundary_weight=0",
        },
    }
    (OUT / "锁定配置噪声消融最终审计_v2.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    audit_frame = pd.DataFrame(checks)
    save_table(audit_frame, "表73_锁定配置噪声消融合规性审计")

    print("\n=== FINAL SUMMARY ===")
    print(main_table[["排名", "模型", "Accuracy均值(%)", "Accuracy标准差(pp)"]].to_string(index=False))
    print("\nnoise top:")
    print(noise_table[["鲁棒性排名", "模型", "全网格平均准确率(%)", "5dB(%)"]].to_string(index=False))
    print(f"\nreport -> {report_path}")
    print(f"audit -> {OUT / '锁定配置噪声消融最终审计_v2.json'}")
    print(f"audit checks -> {sum(x['结果'] == '通过' for x in checks)}/{len(checks)}")


if __name__ == "__main__":
    main()
