"""Export the completed public improved candidate/control study for the thesis.

All metric values are read from stored per-seed JSON records.  The parameter
counts are materialised from the current factories with a one-epoch fit only to
build weights; they are not used to select a model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import train_test_split

from scripts.comparison_models import (
    KerasSequenceClassifier,
    model_complexity,
    build_default_model_specs,
)
from scripts.run_main_comparison import _load_main_protocol_data

ROOT = Path("results/studies")
CAND_ROOT = ROOT / "public_improved_v2"
CTRL_ROOT = ROOT / "public_matched_control_v2_no_delete"
OLD_ROOT = ROOT / "public_10model_5seed_v1"
OUT = Path(r"D:/learn/毕业材料/graduation/results")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = (
    "di_emstgat", "ab_emstgat", "xgboost", "lightgbm", "patchtst", "mtgnn",
    "tcn", "transformer", "itransformer", "gatv2",
)
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
CLASS_NAMES = ["Flooding", "Membrane_Drying", "Normal", "Thermal_Management_Fault"]
PROPOSED = {"di_emstgat", "ab_emstgat"}


def load_records(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "per_seed_records.json").read_text(encoding="utf-8"))


def load_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return round(float(value) * 100.0, 4)


def save(df: pd.DataFrame, name: str) -> None:
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{name}.csv rows={len(df)}")


candidate = load_records(CAND_ROOT)
control = load_records(CTRL_ROOT)
old = load_records(OLD_ROOT)
cand = pd.DataFrame(candidate)
ctrl = pd.DataFrame(control)
oldf = pd.DataFrame(old)

# ---------- latest candidate aggregate: every stored scalar metric ----------
metric_defs = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "BalancedAcc"),
    ("precision_weighted", "Precision_weighted"),
    ("recall_weighted", "Recall_weighted"),
    ("f1_macro", "MacroF1"),
    ("f1_weighted", "F1_weighted"),
    ("cohen_kappa", "CohenKappa"),
]
rows = []
for model in MODEL_ORDER:
    g = cand[cand.model == model]
    if g.empty:
        continue
    row = {
        "模型": CN[model],
        "model_key": model,
        "层级": LEVEL_CN.get(g.level.iloc[0], g.level.iloc[0]),
        "运行次数": int(len(g)),
    }
    for metric, label in metric_defs:
        row[f"{label}_均值(%)"] = pct(g[metric].mean())
        row[f"{label}_标准差(%)"] = pct(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
    row["训练耗时均值(s)"] = round(float(g.fit_time.mean()), 4)
    row["训练耗时标准差(s)"] = round(float(g.fit_time.std(ddof=1)), 4)
    row["推理耗时均值(s)"] = round(float(g.prediction_time.mean()), 4)
    row["推理耗时标准差(s)"] = round(float(g.prediction_time.std(ddof=1)), 4)
    rows.append(row)
latest_summary = pd.DataFrame(rows)
save(latest_summary, "表15_公开数据集改进版主表_10模型5种子")

# ---------- per-seed scalar metrics ----------
scalar = [
    "model", "level", "seed", "accuracy", "balanced_accuracy",
    "precision_weighted", "recall_weighted", "f1_macro", "f1_weighted",
    "cohen_kappa", "fit_time", "prediction_time", "feature_count",
    "train_samples", "test_samples",
]
detail = cand[scalar].copy()
detail.insert(0, "模型", detail["model"].map(CN))
detail.insert(1, "层级中文", detail["level"].map(lambda x: LEVEL_CN.get(x, x)))
for metric, _ in metric_defs:
    detail[metric] = detail[metric].map(pct)
save(detail, "表16_公开数据集改进版逐种子全部指标")

# ---------- class-level report and confusion matrices ----------
class_rows: list[dict[str, Any]] = []
cm_rows: list[dict[str, Any]] = []
for record in candidate:
    report = record["classification_report"]
    matrix = record["confusion_matrix"]
    for class_name in CLASS_NAMES:
        values = report[class_name]
        class_rows.append({
            "模型": CN[record["model"]],
            "model_key": record["model"],
            "seed": int(record["seed"]),
            "类别": class_name,
            "Precision(%)": pct(values["precision"]),
            "Recall(%)": pct(values["recall"]),
            "F1(%)": pct(values["f1-score"]),
            "支持数": int(values["support"]),
        })
    cm_rows.append({
        "模型": CN[record["model"]],
        "model_key": record["model"],
        "seed": int(record["seed"]),
        "混淆矩阵": json.dumps(matrix, ensure_ascii=False),
    })
class_detail = pd.DataFrame(class_rows)
save(class_detail, "表17_公开数据集改进版逐种子类别指标")
save(pd.DataFrame(cm_rows), "表18_公开数据集改进版混淆矩阵")

class_agg = []
for (model, class_name), g in class_detail.groupby(["model_key", "类别"], sort=False):
    class_agg.append({
        "模型": CN[model],
        "model_key": model,
        "类别": class_name,
        "Precision均值(%)": round(float(g["Precision(%)"].mean()), 4),
        "Precision标准差(%)": round(float(g["Precision(%)"].std(ddof=1)), 4),
        "Recall均值(%)": round(float(g["Recall(%)"].mean()), 4),
        "Recall标准差(%)": round(float(g["Recall(%)"].std(ddof=1)), 4),
        "F1均值(%)": round(float(g["F1(%)"].mean()), 4),
        "F1标准差(%)": round(float(g["F1(%)"].std(ddof=1)), 4),
        "支持数(每种子)": int(g["支持数"].iloc[0]),
    })
save(pd.DataFrame(class_agg), "表19_公开数据集改进版类别级指标")

# ---------- complexity/config table from the actual current factories ----------
data = _load_main_protocol_data("数据文件/Public datasets.csv", test_size=0.2, seed=42)
X, y = data["X_train"], data["y_train"]
X_probe, _, y_probe, _ = train_test_split(
    X, y, train_size=min(400, len(y)), random_state=42, stratify=y
)
specs = build_default_model_specs(
    seed=42,
    n_jobs=1,
    deep_epochs=100,
    deep_validation_split=0.0,
    comparison_strength="reduced_baselines",
    proposed_validation_split=0.15,
    proposed_clipnorm=1.0,
    proposed_capacity="standard",
    ab_capacity="compact",
    proposed_boundary_weight=0.05,
    proposed_boundary_margin=0.05,
)
architecture_notes = {
    "xgboost": "15棵树; max_depth=2; lr=0.05; min_child_weight=10; reg_lambda=5; subsample=0.7; colsample=0.7",
    "lightgbm": "6棵树; num_leaves=4; max_depth=2; min_child_samples=30; lr=0.05; subsample=0.7; colsample=0.7",
    "tcn": "light: Conv1D=32×2, dilation=(1,2), dense=32, dropout=0.35, epochs=3",
    "transformer": "light: dense=32, heads=2, key_dim=8, dense=32, dropout=0.35, epochs=5",
    "mtgnn": "light: variate/node dense=32, temporal Conv1D=32×2, dense=32, epochs=3",
    "gatv2": "light: additive graph attention, node dense=32, dense=32, epochs=5",
    "itransformer": "light: variate-axis attention, heads=2, key_dim=8, dense=32, epochs=5",
    "patchtst": "light: patch_size=3, heads=2, key_dim=8, dense=32, epochs=3",
    "di_emstgat": "standard: hidden=192, heads=16, dropout=0.206, knn_k=6, dilation=4, lr=8.13e-4, wd=4.27e-4, boundary=0.05",
    "ab_emstgat": "compact: hidden=64, heads=8, dropout=0.30, knn_k=6, dilation=4, lr=8.13e-4, wd=4.27e-4, boundary=0.05, pair=(1↔3)",
}
complexity_rows = []
for model in MODEL_ORDER:
    spec = specs[model]
    estimator = spec.estimator_factory()
    if isinstance(estimator, KerasSequenceClassifier):
        declared_epochs = estimator.epochs
        estimator.epochs = 1
        estimator.validation_split = 0.0
        estimator.verbose = 0
        estimator.fit(X_probe, y_probe)
    else:
        declared_epochs = "-"
        estimator.fit(X_probe, y_probe)
    info = model_complexity(estimator)
    complexity_rows.append({
        "模型": CN[model],
        "model_key": model,
        "层级": LEVEL_CN.get(spec.level, spec.level),
        "类型": "本文模型" if model in PROPOSED else "对比模型（按要求降配）",
        "复杂度类型": "可训练参数量" if info["complexity_kind"] == "trainable_parameters" else "树的数量",
        "复杂度数值": int(info["param_count"]),
        "容量档": getattr(estimator, "capacity", "-") if model not in {"xgboost", "lightgbm"} else "-",
        "正式训练轮数": declared_epochs,
        "关键配置": architecture_notes[model],
    })
save(pd.DataFrame(complexity_rows), "表20_公开数据集改进版参数量与配置")

# ---------- matched candidate vs no-boundary control ----------
def paired_rows(metric: str) -> list[dict[str, Any]]:
    rows = []
    for model in ("di_emstgat", "ab_emstgat"):
        a = cand[cand.model == model].set_index("seed")[metric].sort_index()
        b = ctrl[ctrl.model == model].set_index("seed")[metric].sort_index()
        delta = a - b
        pt = 1.0 if np.allclose(delta, 0.0) else float(ttest_rel(a, b).pvalue)
        try:
            pw = 1.0 if np.allclose(delta, 0.0) else float(wilcoxon(delta).pvalue)
        except ValueError:
            pw = float("nan")
        rows.append({
            "模型": CN[model],
            "model_key": model,
            "指标": metric,
            "匹配种子数": int(len(delta)),
            "候选均值(%)": pct(a.mean()),
            "控制均值(%)": pct(b.mean()),
            "候选-控制差值(个百分点)": round(float(delta.mean()) * 100, 4),
            "配对t检验p": round(pt, 6),
            "Wilcoxon p": round(pw, 6),
            "候选胜出种子": f"{int((delta > 0).sum())}/{len(delta)}",
        })
    return rows

matched = []
for metric, _ in metric_defs:
    matched.extend(paired_rows(metric))
save(pd.DataFrame(matched), "表21_公开数据集改进候选_vs_严格匹配控制")

# ---------- candidate versus historical public v1 (descriptive only) ----------
change_rows = []
for model in MODEL_ORDER:
    a = cand[cand.model == model].set_index("seed").sort_index()
    b = oldf[oldf.model == model].set_index("seed").sort_index()
    if a.empty or b.empty:
        continue
    for metric, label in (("accuracy", "Accuracy"), ("f1_macro", "MacroF1"), ("f1_weighted", "F1_weighted")):
        delta = a[metric] - b[metric]
        change_rows.append({
            "模型": CN[model],
            "model_key": model,
            "指标": label,
            "旧public_v1均值(%)": pct(b[metric].mean()),
            "改进版均值(%)": pct(a[metric].mean()),
            "变化(个百分点)": round(float(delta.mean()) * 100, 4),
            "备注": "描述性变化；配置也发生改变，不能视为单变量因果效果",
        })
save(pd.DataFrame(change_rows), "表22_公开数据集改进版_vs_历史结果")

# ---------- machine-readable contract ----------
contract = {
    "candidate_root": str(CAND_ROOT),
    "matched_control_root": str(CTRL_ROOT),
    "historical_root": str(OLD_ROOT),
    "candidate_records": len(candidate),
    "control_records": len(control),
    "historical_records": len(old),
    "candidate_manifest": load_manifest(CAND_ROOT),
    "control_manifest": load_manifest(CTRL_ROOT),
    "model_order": list(MODEL_ORDER),
    "classes": CLASS_NAMES,
    "selection_status": "exploratory_after_outer_test_observation",
    "matched_control_only_difference": "proposed_boundary_weight 0.05 -> 0.0; all other candidate/control settings matched",
    "baseline_policy": {
        "tcn_mtgnn_patchtst_epochs": 3,
        "lightgbm_estimators": 6,
        "xgboost_estimators": 15,
        "other_deep_baseline_epochs": 5,
    },
    "notes": [
        "Candidate test metrics were observed; do not use them to tune another candidate.",
        "Historical public_v1 comparison is descriptive because proposed capacity changed.",
        "Parameter counts are materialised from current factories with a one-epoch probe fit.",
    ],
}
(OUT / "公开数据集改进版_实验契约与审计.json").write_text(
    json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "公开数据集改进版_说明.txt").write_text(
    "公开数据集改进版结果说明\n\n"
    "数据: Public datasets.csv; 4类; 分层随机80/20; 5种子(42,43,44,45,46); 训练/测试=7076/1770。\n"
    "候选: DI standard, AB compact; lr=8.13e-4, wd=4.27e-4, clipnorm=1.0, 100 epochs;\n"
    "       针对 Membrane_Drying(1) <-> Thermal_Management_Fault(3) 的双向边界损失 weight=0.05, margin=0.05。\n"
    "控制: 与候选完全匹配，仅 boundary_weight=0.0; 结果保存于独立目录。\n"
    "基线: TCN/MTGNN/PatchTST=3 epochs; Transformer/GATv2/iTransformer=5 epochs; LightGBM=6 trees; XGBoost=15 trees。\n"
    "重要: 候选和控制均已观察外部测试集，因此后续不要利用该测试结果继续搜索；表21给出因果匹配比较，表22仅作历史描述。\n",
    encoding="utf-8",
)
print("OUT", OUT)
