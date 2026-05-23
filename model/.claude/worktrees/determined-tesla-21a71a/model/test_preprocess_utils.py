"""preprocess_utils 回归测试。

覆盖三类核心能力：
1) 训练预处理中的去功率/保电流策略与元数据生成；
2) preprocess_meta 的读写一致性与特征变换复用；
3) 标签编码/解码与类别映射一致性。
"""

import json

import numpy as np
import pandas as pd

from preprocess_utils import (
    decode_label_ids,
    encode_labels_with_meta,
    load_preprocess_meta,
    save_preprocess_meta,
    train_preprocess_and_select,
    transform_features_with_meta,
)


def test_train_preprocess_and_select_generates_meta_and_enforces_policy(tmp_path):
    """验证训练侧会移除功率列、保留电流列，并产出完整 preprocess_meta。"""
    df = pd.DataFrame(
        {
            "stackVoltage": [0.50, 0.51, 0.49, 0.52, 0.48, 0.53, 0.47, 0.54],
            "stackCurrent": [100, 110, 95, 105, 102, 111, 97, 109],
            "PW": [50, 56, 46, 55, 49, 59, 45, 58],
            "airFlow": [6.1, 6.2, 6.0, 6.3, 6.1, 6.4, 6.0, 6.2],
            "State_Label": [
                "Normal",
                "Flooding",
                "Normal",
                "Flooding",
                "Normal",
                "Flooding",
                "Normal",
                "Flooding",
            ],
        }
    )
    data_path = tmp_path / "toy.csv"
    df.to_csv(data_path, index=False)

    data = train_preprocess_and_select(str(data_path), test_size=0.25, seed=7)

    assert "PW" not in data["feature_names"]
    assert any("current" in name.lower() for name in data["feature_names"])
    assert "preprocess_meta" in data

    meta = data["preprocess_meta"]
    assert "selected_features" in meta
    assert "scaler" in meta and "mean" in meta["scaler"] and "scale" in meta["scaler"]
    assert "removed_power_columns" in meta
    assert "retained_current_columns" in meta


def test_preprocess_meta_roundtrip_and_transform(tmp_path):
    """验证 preprocess_meta 可回写，并能稳定复用到推理特征变换。"""
    meta = {
        "selected_features": ["stackVoltage", "stackCurrent"],
        "scaler": {"mean": [1.0, 10.0], "scale": [2.0, 5.0]},
        "class_names": ["Normal", "Flooding"],
        "label_classes": ["Normal", "Flooding"],
        "fill_values": {"stackVoltage": 1.0, "stackCurrent": 10.0},
        "outlier_bounds": {"stackVoltage": [0.0, 3.0], "stackCurrent": [0.0, 20.0]},
        "removed_power_columns": ["PW"],
        "retained_current_columns": ["stackCurrent"],
        "label_col": "State_Label",
    }

    meta_path = tmp_path / "preprocess_meta.json"
    save_preprocess_meta(meta, str(meta_path))
    loaded = load_preprocess_meta(str(meta_path))

    assert loaded["selected_features"] == ["stackVoltage", "stackCurrent"]

    df = pd.DataFrame(
        {
            "stackVoltage": [1.0, np.nan],
            "stackCurrent": [10.0, 30.0],
            "unused": [123, 456],
        }
    )
    X = transform_features_with_meta(df, loaded)

    assert X.shape == (2, 2)
    # 第一行正好等于均值，标准化后应接近 0
    assert np.allclose(X[0], np.array([0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_label_encode_decode_with_meta():
    """验证标签名称与类别 id 的双向映射在 meta 下保持一致。"""
    meta = {
        "selected_features": ["a"],
        "scaler": {"mean": [0.0], "scale": [1.0]},
        "class_names": ["Normal", "Flooding"],
        "label_classes": ["Normal", "Flooding"],
        "fill_values": {"a": 0.0},
        "outlier_bounds": {"a": [-1.0, 1.0]},
        "removed_power_columns": [],
        "retained_current_columns": ["stackCurrent"],
    }

    y = encode_labels_with_meta(np.array(["Normal", "Flooding", "Normal"]), meta)
    assert y.tolist() == [0, 1, 0]

    names = decode_label_ids(np.array([1, 0, 1]), meta)
    assert names == ["Flooding", "Normal", "Flooding"]
