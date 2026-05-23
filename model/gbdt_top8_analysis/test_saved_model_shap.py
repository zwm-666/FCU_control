"""saved-model SHAP 脚本测试。"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from generate_saved_model_shap import load_shap_inputs_for_saved_split
from preprocess_utils import save_preprocess_meta


def _make_temp_dir() -> Path:
    return Path(tempfile.mkdtemp(dir="."))


def test_load_shap_inputs_for_saved_split_uses_saved_meta_and_requested_ratio():
    temp_dir = _make_temp_dir()
    data_path = temp_dir / "toy.csv"
    df = pd.DataFrame(
        {
            "stackVoltage": [0.50, 0.51, 0.49, 0.52, 0.48, 0.53, 0.47, 0.54, 0.46, 0.55],
            "stackCurrent": [100, 110, 95, 105, 102, 111, 97, 109, 96, 112],
            "PW": [50, 56, 46, 55, 49, 59, 45, 58, 44, 60],
            "airFlow": [6.1, 6.2, 6.0, 6.3, 6.1, 6.4, 6.0, 6.2, 5.9, 6.5],
            "State_Label": [
                "Normal",
                "Flooding",
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
    df.to_csv(data_path, index=False)

    meta = {
        "selected_features": ["stackVoltage", "stackCurrent"],
        "scaler": {"mean": [0.5, 100.0], "scale": [0.1, 10.0]},
        "class_names": ["Normal", "Flooding"],
        "label_classes": ["Flooding", "Normal"],
        "fill_values": {"stackVoltage": 0.5, "stackCurrent": 100.0},
        "outlier_bounds": {"stackVoltage": [0.0, 1.0], "stackCurrent": [0.0, 200.0]},
        "removed_power_columns": ["PW"],
        "retained_current_columns": ["stackCurrent"],
        "label_col": "State_Label",
    }
    meta_path = temp_dir / "preprocess_meta.json"
    save_preprocess_meta(meta, str(meta_path))

    payload = load_shap_inputs_for_saved_split(
        data_path=str(data_path),
        meta_path=str(meta_path),
        test_size=0.3,
        seed=42,
        label_col="",
    )

    assert payload["feature_names"] == ["stackVoltage", "stackCurrent"]
    assert payload["X_train"].shape == (7, 2)
    assert payload["X_test"].shape == (3, 2)
    assert payload["label_col"] == "State_Label"
    assert np.issubdtype(payload["X_train"].dtype, np.floating)
    assert np.issubdtype(payload["X_test"].dtype, np.floating)
