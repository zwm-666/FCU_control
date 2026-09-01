"""通用数据预处理工具：训练/微调/离线评估共用。"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ================== 基础工具 ==================
def load_tabular_data(data_path: str) -> pd.DataFrame:
    """按扩展名读取 CSV 或 Excel。"""
    file_ext = os.path.splitext(data_path)[1].lower()
    if file_ext == ".csv":
        return pd.read_csv(data_path)
    if file_ext in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)
    raise ValueError(f"不支持的数据文件格式: {file_ext}")


def normalize_feature_name(name: str) -> str:
    """标准化特征名，便于中英文关键词识别。"""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(name).strip().lower())


def identify_power_and_current_columns(
    feature_names: List[str],
) -> Tuple[List[str], List[str]]:
    """识别功率列与电流列。"""
    power_columns: List[str] = []
    current_columns: List[str] = []

    power_keywords = ("功率", "power", "stackpower", "pw")
    current_keywords = ("电流", "current", "stackcurrent", "iwrite", "currenta")
    current_exclude_keywords = ("aircurrent", "currentlimit")

    for feature_name in feature_names:
        normalized = normalize_feature_name(feature_name)
        if any(keyword in normalized for keyword in power_keywords):
            power_columns.append(feature_name)
            continue

        if any(keyword in normalized for keyword in current_keywords) and not any(
            keyword in normalized for keyword in current_exclude_keywords
        ):
            current_columns.append(feature_name)

    return power_columns, current_columns


def assert_feature_policy(feature_names: List[str]):
    """约束：训练输入不能含功率列，且至少保留一个电流列。"""
    power_columns, current_columns = identify_power_and_current_columns(feature_names)
    if power_columns:
        raise ValueError(f"训练输入中不能包含功率列: {power_columns}")
    if not current_columns:
        raise ValueError("训练输入中必须至少保留一个电流列")


def ensure_current_columns_selected(
    feature_names: List[str],
    selected_idx: np.ndarray,
    importances: np.ndarray,
    current_columns: List[str],
) -> np.ndarray:
    """若筛选后没有电流列，则按重要性补回所有电流列。"""
    if not current_columns:
        return np.asarray(selected_idx, dtype=int)

    current_idx = [
        idx for idx, name in enumerate(feature_names) if name in current_columns
    ]
    selected_set = {int(idx) for idx in np.asarray(selected_idx, dtype=int).tolist()}

    if any(idx in selected_set for idx in current_idx):
        return np.asarray(
            sorted(selected_set, key=lambda idx: (-importances[idx], idx)), dtype=int
        )

    selected_set.update(current_idx)
    return np.asarray(
        sorted(selected_set, key=lambda idx: (-importances[idx], idx)), dtype=int
    )


def fit_missing_values(
    train_df: pd.DataFrame, label_col: Optional[str] = None
) -> Dict[str, Any]:
    """仅基于训练集拟合缺失值规则。"""
    fill_values: Dict[str, Any] = {}
    for col in train_df.columns:
        if label_col and col == label_col:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            fill_values[col] = float(train_df[col].median())
        else:
            mode = train_df[col].mode(dropna=True)
            fill_values[col] = mode.iloc[0] if not mode.empty else ""
    return fill_values


def apply_missing_values(df: pd.DataFrame, fill_values: Dict[str, Any]) -> pd.DataFrame:
    """把拟合好的缺失值规则应用到数据。"""
    df = df.copy()
    for col, fill_value in fill_values.items():
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(fill_value)
    return df


def fit_outlier_bounds(
    train_df: pd.DataFrame,
    label_col: Optional[str] = None,
    threshold: float = 3.0,
) -> Dict[str, Tuple[float, float]]:
    """仅基于训练集拟合异常值裁剪边界。"""
    bounds: Dict[str, Tuple[float, float]] = {}
    numeric_cols = train_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if label_col and col == label_col:
            continue
        mean = float(train_df[col].mean())
        std = float(train_df[col].std())
        if np.isnan(std) or std < 1e-8:
            bounds[col] = (mean, mean)
        else:
            margin = threshold * std
            bounds[col] = (mean - margin, mean + margin)
    return bounds


def apply_outlier_bounds(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """把拟合好的异常值边界应用到数据。"""
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """移除重复行。"""
    return df.drop_duplicates(keep="first")


def get_class_names(raw_labels: np.ndarray, encoded_labels: np.ndarray) -> List[str]:
    """根据编码结果回收类别名（按类别 id 顺序）。"""
    class_names: List[str] = []
    for class_id in np.unique(encoded_labels):
        first_idx = int(np.where(encoded_labels == class_id)[0][0])
        class_names.append(str(raw_labels[first_idx]))
    return class_names


def prepare_training_dataframe(
    data_path: str,
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """执行与划分比例无关的通用清洗，返回可复用 DataFrame。"""
    df = load_tabular_data(data_path)
    cols_to_remove = ["State", "state", "tsec"]
    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])

    datetime_cols = [
        col for col in df.columns[:-1] if pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    if datetime_cols:
        df = df.drop(columns=datetime_cols)

    if label_col is None:
        label_col = str(df.columns[-1])
    if label_col not in df.columns:
        raise ValueError(f"标签列不存在: {label_col}")

    df = remove_duplicates(df)
    return df, str(label_col)


def train_preprocess_and_select_from_df(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    label_col: Optional[str] = None,
    importance_threshold: float = 0.95,
) -> Dict[str, Any]:
    """基于已清洗 DataFrame 拟合预处理与特征选择。"""
    df = df.copy()
    if label_col is None:
        label_col = str(df.columns[-1])
    if label_col not in df.columns:
        raise ValueError(f"标签列不存在: {label_col}")

    feature_columns = [c for c in df.columns if c != label_col]
    power_columns, current_columns = identify_power_and_current_columns(feature_columns)

    if power_columns:
        df = df.drop(columns=power_columns)
    if not current_columns:
        raise ValueError("未识别到电流列，无法满足保留电流特征的训练要求")

    assert_feature_policy([c for c in df.columns if c != label_col])
    current_columns = [c for c in current_columns if c in df.columns and c != label_col]
    if not current_columns:
        raise ValueError("移除功率列后未保留任何电流列，无法继续训练")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed,
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    fill_values = fit_missing_values(train_df, label_col=label_col)
    train_df = apply_missing_values(train_df, fill_values)
    test_df = apply_missing_values(test_df, fill_values)

    outlier_bounds = fit_outlier_bounds(train_df, label_col=label_col)
    train_df = apply_outlier_bounds(train_df, outlier_bounds)
    test_df = apply_outlier_bounds(test_df, outlier_bounds)

    feature_names = [c for c in train_df.columns if c != label_col]
    X_train = train_df[feature_names].values.astype(np.float32)
    X_test = test_df[feature_names].values.astype(np.float32)
    y_train_raw = train_df[label_col].values
    y_test_raw = test_df[label_col].values

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    class_names = get_class_names(y_train_raw, y_train)
    label_classes = [str(item) for item in le.classes_.tolist()]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed)
    gb.fit(X_train, y_train)
    importances = gb.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[sorted_idx])
    cumsum_norm = cumsum / cumsum[-1]
    n_selected = np.searchsorted(cumsum_norm, importance_threshold) + 1
    selected_idx = sorted_idx[:n_selected]
    selected_idx = ensure_current_columns_selected(
        feature_names=feature_names,
        selected_idx=selected_idx,
        importances=importances,
        current_columns=current_columns,
    )

    selected_features = [feature_names[i] for i in selected_idx]
    selected_importances = importances[selected_idx]
    assert_feature_policy(selected_features)

    X_train = X_train[:, selected_idx]
    X_test = X_test[:, selected_idx]

    scaler_selected = {
        "mean": scaler.mean_[selected_idx].astype(float).tolist(),
        "scale": scaler.scale_[selected_idx].astype(float).tolist(),
    }

    fill_values_selected = {
        k: fill_values[k] for k in selected_features if k in fill_values
    }
    outlier_bounds_selected = {
        k: [float(outlier_bounds[k][0]), float(outlier_bounds[k][1])]
        for k in selected_features
        if k in outlier_bounds
    }

    preprocess_meta = {
        "selected_features": selected_features,
        "scaler": scaler_selected,
        "class_names": [str(x) for x in class_names],
        "label_classes": label_classes,
        "fill_values": fill_values_selected,
        "outlier_bounds": outlier_bounds_selected,
        "removed_power_columns": [str(x) for x in power_columns],
        "retained_current_columns": [str(x) for x in current_columns],
        "label_col": str(label_col),
    }

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.astype(np.int32),
        "y_test": y_test.astype(np.int32),
        "num_classes": len(np.unique(y_train)),
        "class_names": [str(x) for x in class_names],
        "label_classes": label_classes,
        "feature_names": selected_features,
        "feature_importances": selected_importances,
        "scaler": scaler,
        "fill_values": fill_values_selected,
        "outlier_bounds": outlier_bounds_selected,
        "removed_power_columns": [str(x) for x in power_columns],
        "retained_current_columns": [str(x) for x in current_columns],
        "preprocess_meta": preprocess_meta,
    }


# ================== 训练侧预处理（拟合规则） ==================
def train_preprocess_and_select(
    data_path: str,
    test_size: float = 0.2,
    seed: int = 42,
    label_col: Optional[str] = None,
    importance_threshold: float = 0.95,
) -> Dict[str, Any]:
    """训练场景：拟合缺失值/异常值/标准化/特征选择，并返回可复用元数据。"""
    df, resolved_label_col = prepare_training_dataframe(data_path, label_col=label_col)
    return train_preprocess_and_select_from_df(
        df=df,
        test_size=test_size,
        seed=seed,
        label_col=resolved_label_col,
        importance_threshold=importance_threshold,
    )


# ================== 元数据读写 ==================
def _json_safe(obj: Any) -> Any:
    """把 numpy 标量/数组转换为 JSON 兼容类型。"""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_preprocess_meta(meta: Dict[str, Any], save_path: str) -> None:
    """保存 preprocess_meta.json。"""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2, ensure_ascii=False)


def load_preprocess_meta(meta_path: str) -> Dict[str, Any]:
    """读取 preprocess_meta.json 并做基础字段校验。"""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_fields = [
        "selected_features",
        "scaler",
        "class_names",
        "fill_values",
        "outlier_bounds",
        "removed_power_columns",
        "retained_current_columns",
    ]
    for field in required_fields:
        if field not in meta:
            raise ValueError(f"preprocess_meta 缺少字段: {field}")

    scaler = meta.get("scaler", {})
    if "mean" not in scaler or "scale" not in scaler:
        raise ValueError("preprocess_meta.scaler 缺少 mean/scale")

    return meta


# ================== 推理/评估侧复用 ==================
def transform_features_with_meta(
    feature_df: pd.DataFrame, meta: Dict[str, Any]
) -> np.ndarray:
    """按 preprocess_meta 对新数据做同构变换，输出可直接喂给模型的特征矩阵。"""
    selected_features = [str(x) for x in meta["selected_features"]]
    df = feature_df.copy()

    for col in selected_features:
        if col not in df.columns:
            df[col] = np.nan

    df = df[selected_features]

    fill_values = meta.get("fill_values", {})
    for col in selected_features:
        if df[col].isnull().any():
            if col in fill_values:
                df[col] = df[col].fillna(fill_values[col])
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = df[col].fillna("")

    outlier_bounds = meta.get("outlier_bounds", {})
    for col in selected_features:
        if col in outlier_bounds:
            lower, upper = outlier_bounds[col]
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(
                lower=float(lower), upper=float(upper)
            )

    X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

    scaler = meta["scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    safe_scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)

    if mean.shape[0] != X.shape[1] or safe_scale.shape[0] != X.shape[1]:
        raise ValueError("preprocess_meta 中 scaler 维度与 selected_features 不一致")

    return (X - mean) / safe_scale


def encode_labels_with_meta(labels: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """使用 preprocess_meta 中的类别定义把标签编码为模型 id。"""
    label_classes = meta.get("label_classes") or meta.get("class_names")
    mapping = {str(name): idx for idx, name in enumerate(label_classes)}

    encoded: List[int] = []
    unknown_labels: List[str] = []
    for item in labels:
        key = str(item)
        if key not in mapping:
            unknown_labels.append(key)
            continue
        encoded.append(mapping[key])

    if unknown_labels:
        uniq = sorted(set(unknown_labels))
        raise ValueError(f"发现未在训练类别中的标签: {uniq}")

    return np.asarray(encoded, dtype=np.int32)


def decode_label_ids(label_ids: np.ndarray, meta: Dict[str, Any]) -> List[str]:
    """把模型输出类别 id 还原为类别名。"""
    names = [str(x) for x in (meta.get("class_names") or [])]
    result: List[str] = []
    for idx in label_ids:
        i = int(idx)
        if 0 <= i < len(names):
            result.append(names[i])
        else:
            result.append(str(i))
    return result
