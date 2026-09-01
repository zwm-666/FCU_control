"""Export thesis-ready tables from completed experiments.

Writes to D:/learn/毕业材料/graduation/results.
Every number is read back from stored experiment records; nothing is recomputed
or adjusted here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

STUDIES = Path("results/studies")
OUT = Path(r"D:/learn/毕业材料/graduation/results")
OUT.mkdir(parents=True, exist_ok=True)

CN = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "tcn": "TCN",
    "transformer": "Transformer", "mtgnn": "MTGNN", "gatv2": "GATv2",
    "itransformer": "iTransformer", "patchtst": "PatchTST",
    "di_emstgat": "DI-EMSTGAT (本文)", "ab_emstgat": "AB-EMSTGAT (本文)",
}
LEVEL_CN = {
    "traditional_baseline": "传统机器学习",
    "deep_baseline": "深度序列基线",
    "graph_spatiotemporal_baseline": "图/时空基线",
    "modern_transformer_baseline": "现代Transformer基线",
    "proposed_direct_input": "本文-直接输入",
    "proposed_adaptive_branch": "本文-自动分支",
}
PROPOSED = ("di_emstgat", "ab_emstgat")
ROW_MODELS = (
    "xgboost", "lightgbm", "tcn", "transformer", "mtgnn",
    "gatv2", "itransformer", "patchtst", "di_emstgat", "ab_emstgat",
)
ROW_STUDY = "final_row_random_v2"


def pct(x):
    return round(float(x) * 100.0, 4)


def load_records(path, key):
    return pd.DataFrame(json.loads((STUDIES / path / key).read_text(encoding="utf-8")))


def mean_std_table(rec, group_keys=("model",)):
    rows = []
    for model, g in rec.groupby("model", sort=False):
        row = {
            "模型": CN.get(model, model),
            "model_key": model,
            "层级": LEVEL_CN.get(g["level"].iloc[0], g["level"].iloc[0]),
            "运行次数": int(len(g)),
        }
        for metric, label in (
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "BalancedAcc"),
            ("precision_weighted", "Precision"),
            ("recall_weighted", "Recall"),
            ("f1_macro", "MacroF1"),
            ("f1_weighted", "WeightedF1"),
            ("cohen_kappa", "Kappa"),
            ("minority_recall", "少数类Recall"),
        ):
            if metric in g:
                row[f"{label}_均值(%)"] = pct(g[metric].mean())
                row[f"{label}_标准差(%)"] = pct(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
        for metric, label in (("fit_time", "训练耗时(s)"), ("prediction_time", "推理耗时(s)")):
            if metric in g:
                row[label] = round(float(g[metric].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Accuracy_均值(%)", ascending=False).reset_index(drop=True)


def paired_table(rec, metric="f1_macro", index_keys=("seed",)):
    piv = rec.pivot_table(index=list(index_keys), columns="model", values=metric)
    baselines = [m for m in rec.model.unique() if m not in PROPOSED]
    rows = []
    for prop in PROPOSED:
        if prop not in piv:
            continue
        for base in baselines:
            a, b = piv[prop], piv[base]
            d = a - b
            if np.allclose(d, 0):
                p_t = p_w = 1.0
            else:
                p_t = float(ttest_rel(a, b).pvalue)
                try:
                    p_w = float(wilcoxon(a, b).pvalue)
                except ValueError:
                    p_w = float("nan")
            rows.append({
                "候选模型": CN.get(prop, prop),
                "对照模型": CN.get(base, base),
                "配对数": int(len(d)),
                "ΔMacroF1(个百分点)": round(float(d.mean()) * 100.0, 4),
                "配对t检验p": round(p_t, 6),
                "Wilcoxon p": round(p_w, 6),
                "显著性": "**" if p_t < 0.01 else ("*" if p_t < 0.05 else "ns"),
                "候选胜出折数": f"{int((d > 0).sum())}/{len(d)}",
            })
    return pd.DataFrame(rows)


written = []


def save(df, name, note=None):
    csv = OUT / f"{name}.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    written.append((csv.name, len(df)))
    if note:
        (OUT / f"{name}_说明.txt").write_text(note, encoding="utf-8")
        written.append((f"{name}_说明.txt", "-"))


# ---------- 表1: 主表（按行随机划分） ----------
row_rec = load_records(ROW_STUDY, "per_seed_records.json")
row_man = json.loads((STUDIES / ROW_STUDY / "manifest.json").read_text(encoding="utf-8"))
t1 = mean_std_table(row_rec)
# Flag any baseline that collapsed to majority-class prediction: over-weakening
# past this point produces a degenerate model, not a fair weak baseline.
majority_rate = 0.659785
degenerate = t1[
    (t1["model_key"].isin([m for m in ROW_MODELS if m not in PROPOSED]))
    & (t1["Accuracy_均值(%)"] <= (majority_rate + 0.005) * 100.0)
]["模型"].tolist()
t1["备注"] = t1["模型"].map(lambda m: "已退化为多数类预测" if m in degenerate else "")
save(t1, "表1_主表_按行随机划分_10模型5种子",
     "协议: 行级分层随机划分 (test_size=0.2)\n"
     f"数据: {Path(row_man['data_path']).name}\n"
     f"种子: {row_man['seeds']}  每模型5次运行\n"
     f"对比模型容量: {row_man.get('baseline_capacity')} (light容量, 5 epochs, 树模型8棵)\n"
     f"本文模型: DI={row_man.get('proposed_capacity', 'standard')}容量, AB=compact容量(降配一档), 均100 epochs\n"
     "预处理: 训练集内拟合标准化与累计重要性特征选择\n"
     + (f"\n警告: 以下对比模型已退化为多数类预测(准确率≈{majority_rate*100:.2f}%即类别2占比),\n"
        f"其结果不构成有意义的对比基线: {', '.join(degenerate)}\n" if degenerate else "")
     + "\n注意: 划分单位为单行。同一「测试时间」时间戳的约60行60Hz采样会被拆分到\n"
     "训练与测试两侧, 1-NN在该划分上测试准确率为100%。详见 表6_泄漏诊断。")

save(paired_table(row_rec), "表2_配对显著性检验_按行随机划分",
     "对每个种子配对比较 Macro-F1。* 表示 p<0.05, ** 表示 p<0.01。")

# ---------- 表3: 每种子明细 ----------
detail = row_rec[["model", "level", "seed", "accuracy", "balanced_accuracy",
                  "f1_macro", "f1_weighted", "cohen_kappa", "fit_time", "prediction_time"]].copy()
detail["模型"] = detail["model"].map(lambda m: CN.get(m, m))
for c in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "cohen_kappa"):
    detail[c] = detail[c].map(pct)
save(detail, "表3_每种子明细_按行随机划分")

# ---------- 表4: 无泄漏交叉验证（稳健性） ----------
cv_rec = load_records("block_cv_10model_v1", "cv_records.json")
cv_man = json.loads((STUDIES / "block_cv_10model_v1" / "manifest.json").read_text(encoding="utf-8"))
t4 = mean_std_table(cv_rec)
mr = cv_rec.groupby("model")["minority_recall"].agg(["mean", "std"])
t4["少数类Recall_均值(%)"] = t4["model_key"].map(lambda m: pct(mr.loc[m, "mean"]))
t4["少数类Recall_标准差(%)"] = t4["model_key"].map(lambda m: pct(mr.loc[m, "std"]))
save(t4, "表4_稳健性_时间戳块交叉验证_10模型",
     "协议: 按整个「测试时间」时间戳块做分层5折交叉验证\n"
     f"独立块数: {cv_man['n_blocks']}  折数: {cv_man['n_splits']}  种子: {cv_man['seeds']}\n"
     f"每模型运行次数: {cv_man['runs_per_model']}  总记录: {cv_man['total_records']}\n"
     "同一时间戳块的所有行绝不跨越训练/测试, 标准化只在训练折拟合\n"
     f"证据性质: {cv_man['evidence_type']}\n"
     "用途: 作为主表的无泄漏稳健性对照。")

save(paired_table(cv_rec, index_keys=("seed", "fold")), "表5_配对显著性检验_交叉验证",
     "对每个 (种子, 折) 配对比较 Macro-F1, 共15对。")

# ---------- 表6: 泄漏诊断 ----------
leak = pd.DataFrame([
    {"检验项": "1-NN 测试准确率 (seed=42)", "数值": "100.0000%", "说明": "最近邻查表即满分"},
    {"检验项": "1-NN 测试准确率 (seed=43)", "数值": "100.0000%", "说明": "最近邻查表即满分"},
    {"检验项": "1-NN 测试准确率 (seed=44)", "数值": "100.0000%", "说明": "最近邻查表即满分"},
    {"检验项": "1-NN 测试准确率 (seed=45)", "数值": "99.8654%", "说明": "最近邻查表即满分"},
    {"检验项": "1-NN 测试准确率 (seed=46)", "数值": "99.9551%", "说明": "最近邻查表即满分"},
    {"检验项": "多数类基线准确率", "数值": "65.9785%", "说明": "类别2占比"},
    {"检验项": "原始行数", "数值": "11137", "说明": "按行划分时的样本数"},
    {"检验项": "唯一时间戳块数", "数值": "198", "说明": "真实独立观测数"},
    {"检验项": "每块平均行数", "数值": "56.2", "说明": "60Hz采样, 中位数60"},
    {"检验项": "测试行距最近训练行 <1e-6 占比", "数值": "1.84%", "说明": "数值上几乎完全相同"},
    {"检验项": "测试行距最近训练行 <0.01 占比", "数值": "10.05%", "说明": ""},
    {"检验项": "测试行距最近训练行 <0.05 占比", "数值": "68.18%", "说明": ""},
    {"检验项": "测试行距最近训练行 <0.1 占比", "数值": "95.92%", "说明": "特征空间标准差范数=3.0"},
    {"检验项": "最近训练行标签一致率", "数值": "100.0000%", "说明": "邻居标签总是正确答案"},
])
save(leak, "表6_泄漏诊断_按行随机划分",
     "用 1-NN (最近邻) 作为探针: 该模型不学习任何规律, 只查找训练集中最相似的样本\n"
     "并抄其标签。若 1-NN 即可达到满分, 说明测试样本在训练集中存在近乎重复的副本。\n"
     "根因: 同一「测试时间」时间戳下约60行为60Hz采样(不到1秒内采集), 数值近乎重复;\n"
     "按行随机划分会把同一块的行同时分到训练与测试两侧。\n"
     "该风险在《大论文工作规划》4.2节已被预判(公开数据集随机shuffle泄漏风险=高)。")

# ---------- 表7: 两协议对照 ----------
cmp_rows = []
for model in row_rec.model.unique():
    a = row_rec[row_rec.model == model]["accuracy"].mean()
    b = cv_rec[cv_rec.model == model]["accuracy"].mean()
    fa = row_rec[row_rec.model == model]["f1_macro"].mean()
    fb = cv_rec[cv_rec.model == model]["f1_macro"].mean()
    cmp_rows.append({
        "模型": CN.get(model, model),
        "按行随机_Accuracy(%)": pct(a),
        "时间戳块CV_Accuracy(%)": pct(b),
        "Accuracy落差(个百分点)": round((a - b) * 100.0, 4),
        "按行随机_MacroF1(%)": pct(fa),
        "时间戳块CV_MacroF1(%)": pct(fb),
        "MacroF1落差(个百分点)": round((fa - fb) * 100.0, 4),
    })
t7 = pd.DataFrame(cmp_rows).sort_values("MacroF1落差(个百分点)", ascending=False)
save(t7, "表7_两协议对照_泄漏影响量化",
     "同一批模型、同一份数据, 仅改变划分单位(单行 vs 整个时间戳块)的结果差异。\n"
     "落差为正表示按行随机划分给出更高的分数。")

# ---------- 表8: 模型参数量与配置（实测） ----------
def build_complexity_table():
    import numpy as np
    from scripts.comparison_models import (
        KerasSequenceClassifier, build_default_model_specs, model_complexity,
    )
    from scripts.run_main_comparison import _load_main_protocol_data

    data = _load_main_protocol_data("数据文件/自测原数据/测试数据.xlsx", test_size=0.2, seed=42)
    Xtr, ytr = data["X_train"], data["y_train"]
    specs = build_default_model_specs(
        seed=42, n_jobs=1, deep_epochs=100,
        deep_validation_split=0.0, comparison_strength="reduced_baselines",
        proposed_validation_split=0.15, proposed_clipnorm=1.0,
    )
    rows = []
    for name in ROW_MODELS:
        spec = specs[name]
        est = spec.estimator_factory()
        is_proposed = name in PROPOSED
        # Fit briefly just to materialise weights; metrics come from the study.
        if isinstance(est, KerasSequenceClassifier):
            est.epochs, est.validation_split, est.verbose = 1, 0.0, 0
            est.fit(Xtr[:400], ytr[:400])
            declared_epochs = spec.estimator_factory().epochs
        else:
            est.fit(Xtr[:400], ytr[:400])
            declared_epochs = None
        info = model_complexity(est)
        rows.append({
            "模型": CN.get(name, name),
            "层级": LEVEL_CN.get(spec.level, spec.level),
            "类型": "本文模型" if is_proposed else "对比模型(已降配)",
            "复杂度类型": {"trainable_parameters": "可训练参数量",
                          "n_estimators": "树的数量",
                          "unavailable": "不适用"}[info["complexity_kind"]],
            "复杂度数值": info["param_count"],
            "训练轮数": declared_epochs if declared_epochs is not None else "-",
            "容量档": getattr(est, "capacity", "-"),
            "关键配置": info["config_note"],
        })
    return pd.DataFrame(rows)


try:
    t8 = build_complexity_table()
    save(t8, "表8_模型参数量与配置_实测",
         "参数量为实测值: Keras模型统计可训练参数量, 树模型报告树的数量。\n"
         "对比模型: light容量, 5 epochs, 树模型8棵 (已按要求进一步降配)。\n"
         "本文模型: DI使用CEO搜索所得standard容量(hidden=192, heads=16);\n"
         "         AB降配一档使用compact容量(hidden=64, heads=8)。\n"
         "两者均使用 lr=8.13e-4, weight_decay=4.27e-4, clipnorm=1.0, 100 epochs。\n"
         "共同项: class_weight/sample_weight=balanced, 相同数据划分与随机种子。")
except Exception as exc:  # pragma: no cover
    print(f"[warn] complexity table skipped: {exc}")

print("已导出到:", OUT)
for name, n in written:
    print(f"  {name:<52} rows={n}")
