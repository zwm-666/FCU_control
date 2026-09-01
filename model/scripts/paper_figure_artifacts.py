"""Build and plot source artifacts for manuscript-style experiment figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = 1
DEFAULT_MODEL_NAME = "CEO-QAAdamW-EMSTGAT"
PALETTE = [
    "#2A6FBB",
    "#E07A3F",
    "#4E9F50",
    "#9B59B6",
    "#C44E52",
    "#4C72B0",
    "#8172B3",
    "#64B5CD",
]
FIGURE_TEXT = {
    "feature_index": "特征序号",
    "standardized_signal": "标准化特征值",
    "raw_feature_title": "典型故障特征信号",
    "contribution": "贡献率",
    "contribution_ratio": "贡献率",
    "cumulative": "累计贡献率",
    "cumulative_ratio": "累计贡献率",
    "reduced_distribution_title": "降维后样本分布",
    "score": "指标值",
    "accuracy": "准确率",
    "weighted_f1": "加权 F1",
    "kappa": "Kappa 系数",
    "comparison_metrics_title": "准确率 / F1 / Kappa 指标对比",
    "computation_time": "计算时间（秒）",
    "time_accuracy_title": "计算时间与准确率权衡",
    "convergence_title": "优化算法收敛曲线",
    "iteration_epoch": "迭代次数 / 训练轮次",
    "fitness_loss": "适应度 / 损失",
    "validation_loss": "验证损失",
    "training_loss": "训练损失",
    "noise_axis": "信噪比 / 噪声等级",
    "noise_title": "噪声鲁棒性",
    "module_ablation": "模块消融实验",
    "kfold_accuracy": "K 折交叉验证准确率",
    "average_accuracy": "平均故障诊断准确率",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _safe_class_names(y: np.ndarray, class_names: Optional[Sequence[str]]) -> List[str]:
    if class_names:
        return [str(name) for name in class_names]
    max_label = int(np.max(y)) if len(y) else -1
    return [f"Class {idx}" for idx in range(max_label + 1)]


def _probability_records(y_proba: Optional[np.ndarray], class_names: Sequence[str]) -> List[Dict[str, Any]]:
    if y_proba is None:
        return []
    proba = np.asarray(y_proba, dtype=float)
    records = []
    for row_idx, row in enumerate(proba):
        record = {"sample_index": int(row_idx)}
        for class_idx, value in enumerate(row):
            label = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
            record[f"prob_{label}"] = float(value)
        records.append(record)
    return records


def build_feature_curve_records(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    records: List[Dict[str, Any]] = []
    for class_id in sorted(np.unique(y).tolist()):
        mask = y == class_id
        class_label = class_names[int(class_id)] if int(class_id) < len(class_names) else str(class_id)
        values = X[mask]
        mean_curve = np.mean(values, axis=0)
        std_curve = np.std(values, axis=0)
        records.append(
            {
                "class_id": int(class_id),
                "class_name": class_label,
                "sample_count": int(mask.sum()),
                "feature_names": [str(name) for name in feature_names],
                "mean": mean_curve.tolist(),
                "std": std_curve.tolist(),
            }
        )
    return records


def build_reduction_artifacts(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    n_components: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}")
    component_count = int(max(1, min(n_components, X.shape[0], X.shape[1])))
    X_scaled = StandardScaler().fit_transform(X)

    try:
        reducer = KernelPCA(
            n_components=component_count,
            kernel="rbf",
            gamma=None,
            eigen_solver="auto",
            random_state=random_state,
        )
        coords = reducer.fit_transform(X_scaled)
        eigenvalues = np.asarray(getattr(reducer, "eigenvalues_", []), dtype=float)
        method = "KPCA-rbf"
        if eigenvalues.size < component_count or float(np.sum(eigenvalues)) <= 0:
            raise ValueError("KernelPCA did not expose usable eigenvalues")
        ratios = eigenvalues[:component_count] / np.sum(eigenvalues)
    except Exception:
        reducer = PCA(n_components=component_count, random_state=random_state)
        coords = reducer.fit_transform(X_scaled)
        ratios = np.asarray(reducer.explained_variance_ratio_, dtype=float)
        eigenvalues = np.asarray(reducer.explained_variance_, dtype=float)
        method = "PCA-fallback"

    contribution = []
    cumulative = 0.0
    for idx in range(component_count):
        ratio = float(ratios[idx]) if idx < len(ratios) else 0.0
        cumulative += ratio
        contribution.append(
            {
                "component": f"PC{idx + 1}",
                "contribution": ratio,
                "cumulative_contribution": float(min(cumulative, 1.0)),
                "eigenvalue": float(eigenvalues[idx]) if idx < len(eigenvalues) else None,
                "method": method,
            }
        )

    reduced_samples = []
    for sample_idx, row in enumerate(coords):
        class_id = int(y[sample_idx])
        record = {
            "sample_index": int(sample_idx),
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
        }
        for component_idx in range(component_count):
            record[f"PC{component_idx + 1}"] = float(row[component_idx])
        reduced_samples.append(record)

    return {
        "kpca_contribution": contribution,
        "reduced_samples": reduced_samples,
        "reduction_method": method,
    }


def build_prediction_records(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_names: Sequence[str],
) -> List[Dict[str, Any]]:
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    prob_records = _probability_records(y_proba, class_names)
    records = []
    for idx, (true_id, pred_id) in enumerate(zip(y_test, y_pred)):
        true_int = int(true_id)
        pred_int = int(pred_id)
        record = {
            "sample_index": int(idx),
            "true_id": true_int,
            "predicted_id": pred_int,
            "true_name": class_names[true_int] if true_int < len(class_names) else str(true_int),
            "predicted_name": class_names[pred_int] if pred_int < len(class_names) else str(pred_int),
            "correct": bool(true_int == pred_int),
        }
        if idx < len(prob_records):
            record.update({k: v for k, v in prob_records[idx].items() if k != "sample_index"})
        records.append(record)
    return records


def build_training_figure_artifacts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    feature_names: Sequence[str],
    class_names: Optional[Sequence[str]],
    history: Optional[Dict[str, Sequence[float]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    random_state: int = 42,
) -> Dict[str, Any]:
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    names = _safe_class_names(y_all, class_names)
    X_all = np.vstack([np.asarray(X_train), np.asarray(X_test)])

    reduction = build_reduction_artifacts(
        X_all,
        y_all,
        names,
        n_components=3,
        random_state=random_state,
    )
    labels = list(range(len(names)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized_cm = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

    merged_metrics = dict(metrics or {})
    merged_metrics.setdefault("accuracy", float(np.mean(np.asarray(y_test) == np.asarray(y_pred))))
    merged_metrics.setdefault("cohen_kappa", float(cohen_kappa_score(y_test, y_pred, labels=labels)))

    return _jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "model_name": model_name,
            "feature_names": [str(name) for name in feature_names],
            "class_names": names,
            "raw_feature_curves": build_feature_curve_records(X_all, y_all, feature_names, names),
            "kpca_contribution": reduction["kpca_contribution"],
            "reduced_samples": reduction["reduced_samples"],
            "reduction_method": reduction["reduction_method"],
            "predictions": build_prediction_records(y_test, y_pred, y_proba, names),
            "confusion_matrix": cm,
            "confusion_matrix_normalized": normalized_cm,
            "training_history": history or {},
            "metrics": merged_metrics,
        }
    )


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> Optional[str]:
    rows = list(rows)
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _curve_long_rows(curves: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for curve in curves:
        for idx, feature in enumerate(curve["feature_names"]):
            rows.append(
                {
                    "class_id": curve["class_id"],
                    "class_name": curve["class_name"],
                    "feature_index": idx,
                    "feature_name": feature,
                    "mean": curve["mean"][idx],
                    "std": curve["std"][idx],
                    "sample_count": curve["sample_count"],
                }
            )
    return rows


def _history_rows(history: Dict[str, Sequence[float]]) -> List[Dict[str, Any]]:
    max_len = max((len(values) for values in history.values()), default=0)
    rows = []
    for idx in range(max_len):
        row = {"epoch": idx + 1}
        for key, values in history.items():
            if idx < len(values):
                row[key] = float(values[idx])
        rows.append(row)
    return rows


def save_training_figure_artifacts(artifacts: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "paper_figure_data.json"
    json_path.write_text(
        json.dumps(_jsonable(artifacts), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paths: Dict[str, Optional[str]] = {
        "json": str(json_path),
        "feature_curves_csv": _write_csv(output / "paper_feature_curves.csv", _curve_long_rows(artifacts.get("raw_feature_curves", []))),
        "kpca_contribution_csv": _write_csv(output / "paper_kpca_contribution.csv", artifacts.get("kpca_contribution", [])),
        "reduced_samples_csv": _write_csv(output / "paper_reduced_samples.csv", artifacts.get("reduced_samples", [])),
        "predictions_csv": _write_csv(output / "paper_predictions.csv", artifacts.get("predictions", [])),
        "training_history_csv": _write_csv(output / "paper_training_history.csv", _history_rows(artifacts.get("training_history", {}))),
    }
    return {key: value for key, value in paths.items() if value is not None}


def load_artifacts(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial Unicode MS",
                "DejaVu Sans",
                "Arial",
                "Helvetica",
            ],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_raw_feature_curves(artifacts: Dict[str, Any], output_path: str) -> str:
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for idx, curve in enumerate(artifacts.get("raw_feature_curves", [])):
        x = np.arange(len(curve["mean"]))
        mean = np.asarray(curve["mean"], dtype=float)
        std = np.asarray(curve["std"], dtype=float)
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(x, mean, color=color, label=curve["class_name"], linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel(FIGURE_TEXT["feature_index"])
    ax.set_ylabel(FIGURE_TEXT["standardized_signal"])
    ax.set_title(FIGURE_TEXT["raw_feature_title"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    return _save(fig, Path(output_path))


def plot_kpca_contribution(artifacts: Dict[str, Any], output_path: str) -> str:
    rows = artifacts.get("kpca_contribution", [])
    _apply_pub_style()
    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    labels = [row["component"] for row in rows]
    x = np.arange(len(labels))
    contribution = [row["contribution"] for row in rows]
    cumulative = [row["cumulative_contribution"] for row in rows]
    ax1.bar(x, contribution, color="#5B8DB8", label=FIGURE_TEXT["contribution"])
    ax1.set_ylabel(FIGURE_TEXT["contribution_ratio"])
    ax1.set_ylim(0, max(1.0, max(cumulative or [1.0]) * 1.05))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, marker="o", color="#C44E52", label=FIGURE_TEXT["cumulative"])
    ax2.set_ylabel(FIGURE_TEXT["cumulative_ratio"])
    ax2.set_ylim(0, 1.05)
    ax1.set_title(f"{artifacts.get('reduction_method', 'KPCA')} {FIGURE_TEXT['contribution']}")
    return _save(fig, Path(output_path))


def plot_reduced_sample_distribution(artifacts: Dict[str, Any], output_path: str) -> str:
    df = pd.DataFrame(artifacts.get("reduced_samples", []))
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    if not df.empty:
        for idx, (label, group) in enumerate(df.groupby("class_name")):
            ax.scatter(group["PC1"], group["PC2"], s=24, alpha=0.78, label=label, color=PALETTE[idx % len(PALETTE)])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(FIGURE_TEXT["reduced_distribution_title"])
    ax.grid(alpha=0.2)
    ax.legend()
    return _save(fig, Path(output_path))


def _load_json_optional(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return load_artifacts(str(candidate))


def plot_comparison_confusion_matrices(comparison_payload: Dict[str, Any], output_path: str) -> str:
    results = comparison_payload.get("results", [])
    class_names = comparison_payload.get("metadata", {}).get("class_names") or []
    cols = min(3, max(1, len(results)))
    rows = int(np.ceil(len(results) / cols))
    _apply_pub_style()
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, result in zip(axes.ravel(), results):
        ax.axis("on")
        cm = np.asarray(result.get("confusion_matrix", []), dtype=float)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(result.get("model", "model"), fontsize=9)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return _save(fig, Path(output_path))


def plot_comparison_metrics(comparison_payload: Dict[str, Any], output_path: str) -> str:
    results = sorted(comparison_payload.get("results", []), key=lambda row: row.get("accuracy", 0), reverse=True)
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.45 * len(results) + 1.8)))
    labels = [row.get("model", "model") for row in results]
    y = np.arange(len(labels))
    width = 0.24
    ax.barh(y - width, [row.get("accuracy", 0) for row in results], width, label=FIGURE_TEXT["accuracy"], color="#4C72B0")
    ax.barh(y, [row.get("f1_weighted", row.get("f1_score", 0)) for row in results], width, label=FIGURE_TEXT["weighted_f1"], color="#55A868")
    ax.barh(y + width, [row.get("cohen_kappa", 0) for row in results], width, label=FIGURE_TEXT["kappa"], color="#C44E52")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(FIGURE_TEXT["score"])
    ax.set_title(FIGURE_TEXT["comparison_metrics_title"])
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, Path(output_path))


def plot_time_accuracy_tradeoff(comparison_payload: Dict[str, Any], output_path: str) -> str:
    results = comparison_payload.get("results", [])
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for idx, row in enumerate(results):
        total_time = float(row.get("fit_time", 0.0)) + float(row.get("prediction_time", 0.0))
        ax.scatter(total_time, row.get("accuracy", 0), s=60, color=PALETTE[idx % len(PALETTE)])
        ax.text(total_time, row.get("accuracy", 0), f" {row.get('model', '')}", fontsize=8, va="center")
    ax.set_xlabel(FIGURE_TEXT["computation_time"])
    ax.set_ylabel(FIGURE_TEXT["accuracy"])
    ax.set_title(FIGURE_TEXT["time_accuracy_title"])
    ax.grid(alpha=0.25)
    return _save(fig, Path(output_path))


def plot_convergence(artifacts: Dict[str, Any], output_path: str) -> Optional[str]:
    history = artifacts.get("optimization_convergence") or artifacts.get("training_history", {})
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    if isinstance(history, list):
        for idx, item in enumerate(history):
            values = item.get("fitness", item.get("values", []))
            ax.plot(np.arange(1, len(values) + 1), values, label=item.get("name", f"算法 {idx + 1}"))
    elif isinstance(history, dict):
        if "val_loss" in history:
            ax.plot(np.arange(1, len(history["val_loss"]) + 1), history["val_loss"], label=FIGURE_TEXT["validation_loss"])
        if "loss" in history:
            ax.plot(np.arange(1, len(history["loss"]) + 1), history["loss"], label=FIGURE_TEXT["training_loss"])
    ax.set_xlabel(FIGURE_TEXT["iteration_epoch"])
    ax.set_ylabel(FIGURE_TEXT["fitness_loss"])
    ax.set_title(FIGURE_TEXT["convergence_title"])
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(fig, Path(output_path))


def plot_metric_table(payload: Dict[str, Any], output_path: str, title: str, name_key: str = "model") -> Optional[str]:
    rows = payload.get("results") or payload.get("ablation") or payload.get("noise") or payload.get("cv_results") or []
    if not rows:
        return None
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.45 * len(rows) + 1.5)))
    labels = [str(row.get(name_key, row.get("name", row.get("variant", idx)))) for idx, row in enumerate(rows)]
    y = np.arange(len(labels))
    width = 0.24
    ax.barh(y - width, [row.get("accuracy", 0) for row in rows], width, label=FIGURE_TEXT["accuracy"])
    ax.barh(y, [row.get("f1_score", row.get("f1_weighted", 0)) for row in rows], width, label=FIGURE_TEXT["weighted_f1"])
    ax.barh(y + width, [row.get("cohen_kappa", row.get("kappa", 0)) for row in rows], width, label=FIGURE_TEXT["kappa"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(FIGURE_TEXT["score"])
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, Path(output_path))


def plot_noise_accuracy(payload: Dict[str, Any], output_path: str) -> Optional[str]:
    rows = payload.get("results") or payload.get("noise") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    level_col = "snr" if "snr" in df.columns else "noise_level"
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for idx, (model, group) in enumerate(df.groupby(df.get("model", pd.Series(["model"] * len(df))))):
        group = group.sort_values(level_col)
        ax.plot(group[level_col], group["accuracy"], marker="o", label=str(model), color=PALETTE[idx % len(PALETTE)])
    ax.set_xlabel(FIGURE_TEXT["noise_axis"])
    ax.set_ylabel(FIGURE_TEXT["accuracy"])
    ax.set_title(FIGURE_TEXT["noise_title"])
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(fig, Path(output_path))


def generate_requested_figures(
    training_artifact_json: str,
    output_dir: str,
    comparison_json: Optional[str] = None,
    ablation_json: Optional[str] = None,
    noise_json: Optional[str] = None,
    cv_json: Optional[str] = None,
) -> Dict[str, str]:
    artifacts = load_artifacts(training_artifact_json)
    output = Path(output_dir)
    paths: Dict[str, Optional[str]] = {
        "raw_feature_curves": plot_raw_feature_curves(artifacts, output / "01_raw_feature_curves.png"),
        "kpca_contribution": plot_kpca_contribution(artifacts, output / "02_kpca_contribution.png"),
        "reduced_sample_distribution": plot_reduced_sample_distribution(artifacts, output / "03_reduced_sample_distribution.png"),
        "optimization_convergence": plot_convergence(artifacts, output / "06_optimization_convergence.png"),
    }

    comparison_payload = _load_json_optional(comparison_json)
    if comparison_payload:
        paths.update(
            {
                "comparison_confusion_matrices": plot_comparison_confusion_matrices(comparison_payload, output / "04_comparison_confusion_matrices.png"),
                "comparison_metrics": plot_comparison_metrics(comparison_payload, output / "05_comparison_metrics.png"),
                "time_accuracy_tradeoff": plot_time_accuracy_tradeoff(comparison_payload, output / "07_time_accuracy_tradeoff.png"),
            }
        )

    ablation_payload = _load_json_optional(ablation_json)
    if ablation_payload:
        paths["ablation_metrics"] = plot_metric_table(ablation_payload, output / "08_ablation_metrics.png", FIGURE_TEXT["module_ablation"])

    noise_payload = _load_json_optional(noise_json)
    if noise_payload:
        paths["noise_accuracy"] = plot_noise_accuracy(noise_payload, output / "09_noise_accuracy.png")

    cv_payload = _load_json_optional(cv_json)
    if cv_payload:
        paths["kfold_accuracy"] = plot_metric_table(cv_payload, output / "11_kfold_accuracy.png", FIGURE_TEXT["kfold_accuracy"])
        paths["average_accuracy"] = plot_metric_table(cv_payload, output / "12_average_accuracy.png", FIGURE_TEXT["average_accuracy"])

    return {key: value for key, value in paths.items() if value}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate manuscript figure panels from saved experiment artifacts.")
    parser.add_argument("--training-artifacts", required=True, help="Path to paper_figure_data.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated paper figures.")
    parser.add_argument("--comparison-json", default=None, help="Optional comparison_results.json.")
    parser.add_argument("--ablation-json", default=None, help="Optional ablation result JSON.")
    parser.add_argument("--noise-json", default=None, help="Optional noise experiment JSON.")
    parser.add_argument("--cv-json", default=None, help="Optional K-fold/CV result JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = generate_requested_figures(
        training_artifact_json=args.training_artifacts,
        output_dir=args.output_dir,
        comparison_json=args.comparison_json,
        ablation_json=args.ablation_json,
        noise_json=args.noise_json,
        cv_json=args.cv_json,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
