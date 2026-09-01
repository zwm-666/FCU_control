"""基于已保存模型产物执行离线 SHAP 分析。"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from core.preprocess_utils import (
    load_preprocess_meta,
    prepare_training_dataframe,
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

    return str(columns[-1])


def load_shap_inputs_for_saved_split(
    data_path: str,
    meta_path: str,
    test_size: float = 0.3,
    seed: int = 42,
    label_col: str = "",
) -> Dict[str, object]:
    """按保存模型对应的比例重建 train/test 划分，并用已保存 meta 做特征变换。"""
    meta = load_preprocess_meta(meta_path)
    cleaned_df, prepared_label_col = prepare_training_dataframe(data_path)

    resolved_label_col = (
        label_col
        if label_col
        else str(meta.get("label_col") or prepared_label_col or _resolve_label_column(cleaned_df.columns))
    )
    if resolved_label_col not in cleaned_df.columns:
        raise ValueError(f"标签列不存在: {resolved_label_col}")

    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=test_size,
        stratify=cleaned_df[resolved_label_col],
        random_state=seed,
    )

    X_train = transform_features_with_meta(
        train_df.drop(columns=[resolved_label_col]),
        meta,
    )
    X_test = transform_features_with_meta(
        test_df.drop(columns=[resolved_label_col]),
        meta,
    )

    return {
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "feature_names": [str(x) for x in meta["selected_features"]],
        "label_col": resolved_label_col,
    }


def run_saved_model_shap(
    data_path: str,
    model_path: str,
    meta_path: str,
    output_dir: str,
    test_size: float = 0.3,
    seed: int = 42,
    label_col: str = "",
):
    """对已保存模型执行离线 SHAP 分析。"""
    import tensorflow as tf

    from core.model import CUSTOM_OBJECTS, generate_shap_artifacts

    payload = load_shap_inputs_for_saved_split(
        data_path=data_path,
        meta_path=meta_path,
        test_size=test_size,
        seed=seed,
        label_col=label_col,
    )
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
    return generate_shap_artifacts(
        model=model,
        X_train=payload["X_train"],
        X_test=payload["X_test"],
        feature_names=payload["feature_names"],
        output_dir=output_dir,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description="基于已保存 70/30 模型执行 SHAP 分析")
    parser.add_argument("--data-path", default=os.path.join("数据文件", "测试数据.xlsx"), help="原始数据路径")
    parser.add_argument(
        "--model-path",
        default=os.path.join("results_testdata_70_30", "ceo_qaadam_emstgat.keras"),
        help="已保存模型路径",
    )
    parser.add_argument(
        "--meta-path",
        default=os.path.join("results_testdata_70_30", "preprocess_meta.json"),
        help="已保存预处理元数据路径",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("results_testdata_70_30", "shap_offline"),
        help="SHAP 输出目录",
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="模型对应测试比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--label-col", default="", help="标签列名，可选")
    args = parser.parse_args()

    result = run_saved_model_shap(
        data_path=args.data_path,
        model_path=args.model_path,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        label_col=args.label_col,
    )

    print("=" * 60)
    print("离线 SHAP 分析完成")
    print(f"状态: {result['status']}")
    print(f"输出目录: {args.output_dir}")
    print(f"元数据: {result['meta_path']}")
    if "summary_path" in result:
        print(f"summary: {result['summary_path']}")
    if "bar_path" in result:
        print(f"bar: {result['bar_path']}")
    if "importance_path" in result:
        print(f"importance: {result['importance_path']}")
    if result.get("detail"):
        print(f"说明: {result['detail']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
