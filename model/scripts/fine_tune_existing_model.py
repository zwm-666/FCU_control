"""继续训练脚本：基于已有 .keras 模型做增量训练。"""

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from core.model import CUSTOM_OBJECTS
from core.preprocess_utils import (
    encode_labels_with_meta,
    load_preprocess_meta,
    load_tabular_data,
    transform_features_with_meta,
)


def _resolve_label_column(columns, user_label_col: str = "") -> str:
    """解析标签列名：优先命令行指定，否则尝试常见列名。"""
    if user_label_col:
        if user_label_col not in columns:
            raise ValueError(f"指定标签列不存在: {user_label_col}")
        return user_label_col

    common_candidates = ["State_Label", "label", "Label", "state_label"]
    for name in common_candidates:
        if name in columns:
            return name

    # 兜底：默认最后一列
    return str(columns[-1])


def load_finetune_dataset(
    data_path: str, meta_path: str, label_col: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """加载并按 preprocess_meta 规则变换数据，返回 (X, y)。"""
    df = load_tabular_data(data_path)
    meta = load_preprocess_meta(meta_path)

    y_col = _resolve_label_column(df.columns, user_label_col=label_col)
    if y_col not in df.columns:
        raise ValueError(f"标签列不存在: {y_col}")

    feature_df = df.drop(columns=[y_col])
    X = transform_features_with_meta(feature_df, meta)
    y = encode_labels_with_meta(df[y_col].values, meta)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="继续训练已有 EnhancedMSTGAT 模型")
    parser.add_argument(
        "--data-path", required=True, help="带标签数据文件路径（CSV/XLSX）"
    )
    parser.add_argument("--model-path", required=True, help="已有 .keras 模型路径")
    parser.add_argument("--meta-path", required=True, help="preprocess_meta.json 路径")
    parser.add_argument("--output-model", default="", help="继续训练后模型输出路径")
    parser.add_argument("--output-metrics", default="", help="验证指标输出 JSON 路径")
    parser.add_argument("--label-col", default="", help="标签列名（可选）")
    parser.add_argument("--epochs", type=int, default=20, help="继续训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--val-size", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X, y = load_finetune_dataset(
        args.data_path, args.meta_path, label_col=args.label_col
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=y,
        random_state=args.seed,
    )

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )

    # 继续训练使用较小学习率，避免破坏原有收敛状态
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    pred_prob = model.predict(X_val, verbose=0)
    pred = np.argmax(pred_prob, axis=1)

    metrics = {
        "samples": int(X.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "accuracy": float(accuracy_score(y_val, pred)),
        "precision": float(
            precision_score(y_val, pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_val, pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_val, pred, average="weighted", zero_division=0)),
    }

    # 默认输出到原模型同目录
    output_model = args.output_model
    if not output_model:
        model_dir = os.path.dirname(os.path.abspath(args.model_path))
        output_model = os.path.join(model_dir, "ceo_qaadam_emstgat_finetuned.keras")

    os.makedirs(os.path.dirname(os.path.abspath(output_model)), exist_ok=True)
    model.save(output_model)

    output_metrics = args.output_metrics
    if not output_metrics:
        output_metrics = os.path.join(
            os.path.dirname(os.path.abspath(output_model)), "finetune_metrics.json"
        )

    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("继续训练完成")
    print(f"模型输出: {output_model}")
    print(f"指标输出: {output_metrics}")
    print(
        f"Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
        f"Rec={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
