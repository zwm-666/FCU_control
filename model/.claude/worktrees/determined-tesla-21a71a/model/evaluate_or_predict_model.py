"""离线评估/预测脚本：
- 有标签数据：输出分类评估指标
- 无标签数据：输出预测类别与置信度
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import CUSTOM_OBJECTS
from preprocess_utils import (
    decode_label_ids,
    encode_labels_with_meta,
    load_preprocess_meta,
    load_tabular_data,
    transform_features_with_meta,
)


def _resolve_label_column(columns, user_label_col: str = "") -> str:
    """解析标签列名。"""
    if user_label_col:
        if user_label_col not in columns:
            raise ValueError(f"指定标签列不存在: {user_label_col}")
        return user_label_col

    common_candidates = ["State_Label", "label", "Label", "state_label"]
    for name in common_candidates:
        if name in columns:
            return name

    return ""


def evaluate_mode(
    df: pd.DataFrame, model, meta: dict, label_col: str, output_path: str
):
    """有标签评估模式。"""
    feature_df = df.drop(columns=[label_col])
    X = transform_features_with_meta(feature_df, meta)
    y_true = encode_labels_with_meta(df[label_col].values, meta)

    pred_prob = model.predict(X, verbose=0)
    pred = np.argmax(pred_prob, axis=1)

    metrics = {
        "samples": int(X.shape[0]),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, average="weighted", zero_division=0)
        ),
        "f1_score": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("评估完成")
    print(f"输出: {output_path}")
    print(
        f"Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
        f"Rec={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
    )
    print("=" * 60)


def predict_mode(df: pd.DataFrame, model, meta: dict, output_path: str):
    """无标签预测模式。"""
    X = transform_features_with_meta(df, meta)
    pred_prob = model.predict(X, verbose=0)
    pred = np.argmax(pred_prob, axis=1)
    labels = decode_label_ids(pred, meta)
    confidence = pred_prob.max(axis=1)

    result_df = df.copy()
    result_df["pred_label"] = labels
    result_df["pred_class_id"] = pred.astype(int)
    result_df["confidence"] = confidence.astype(float)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_ext = os.path.splitext(output_path)[1].lower()
    if file_ext == ".csv":
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif file_ext in {".xlsx", ".xls"}:
        result_df.to_excel(output_path, index=False)
    else:
        raise ValueError("预测输出仅支持 .csv / .xlsx / .xls")

    print("=" * 60)
    print("预测完成")
    print(f"输出: {output_path}")
    print(f"样本数: {len(result_df)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="离线评估/预测脚本")
    parser.add_argument("--data-path", required=True, help="输入数据路径（CSV/XLSX）")
    parser.add_argument("--model-path", required=True, help="模型路径（.keras）")
    parser.add_argument(
        "--meta-path", required=True, help="预处理元数据路径（preprocess_meta.json）"
    )
    parser.add_argument(
        "--mode", required=True, choices=["evaluate", "predict"], help="运行模式"
    )
    parser.add_argument("--label-col", default="", help="标签列名（evaluate 模式可选）")
    parser.add_argument("--output", default="", help="输出路径")
    args = parser.parse_args()

    df = load_tabular_data(args.data_path)
    meta = load_preprocess_meta(args.meta_path)

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )

    if args.mode == "evaluate":
        label_col = _resolve_label_column(df.columns, args.label_col)
        if not label_col:
            raise ValueError(
                "evaluate 模式需要标签列，请通过 --label-col 指定或使用常见标签列名"
            )

        output_path = args.output or os.path.join(
            os.path.dirname(os.path.abspath(args.model_path)),
            "offline_eval_metrics.json",
        )
        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        evaluate_mode(
            df=df, model=model, meta=meta, label_col=label_col, output_path=output_path
        )
        return

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.model_path)), "offline_predictions.csv"
    )
    predict_mode(df=df, model=model, meta=meta, output_path=output_path)


if __name__ == "__main__":
    main()
