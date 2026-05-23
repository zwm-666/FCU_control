from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = BASE_DIR / "数据文件" / "水淹和膜干故障测试数据"
RAW_ROOT = Path(r"D:\learn\研究生文件\燃料电池数据集\海亿\交流阻抗资料")

SHARED_VALUE_COLUMNS = ["电堆总电压", "电堆总电流", "电堆功率"]

EXTRA_FEATURE_COLUMNS = [
    "FC系统入口高压",
    "进堆氢压",
    "出堆氢压",
    "进堆空压",
    "FC空压机出口压力",
    "进堆空温",
    "出堆空温",
    "进堆水温",
    "出堆水温",
    "氢气循环泵FK",
    "比例阀反馈",
    "purge时间",
    "离心机速度FK",
    "离心机功率",
    "三通阀开度FK",
    "高压水泵转速FK",
    "FC空压机出口温度",
    "FC换热器入口温度",
]

DERIVED_FEATURE_SPECS = [
    ("氢压差", "进堆氢压", "出堆氢压"),
    ("空压差", "进堆空压", "FC空压机出口压力"),
    ("空温升", "出堆空温", "进堆空温"),
    ("水温升", "出堆水温", "进堆水温"),
]

SOURCE_MAP = {
    "正常": {
        "target_file": TARGET_DIR / "正常.xlsx",
        "raw_files": [
            RAW_ROOT / "2月5号测试数据" / "正常测试数据" / "2月5号 正常测试数据20260205.xlsx",
        ],
    },
    "过湿": {
        "target_file": TARGET_DIR / "过湿.xlsx",
        "raw_files": [
            RAW_ROOT / "2月3号测试数据" / "过湿测试" / "2月3号过湿测试数据.xlsx",
            RAW_ROOT / "2月4号测试数据" / "过湿1" / "2月4号过湿测试数据.xlsx",
            RAW_ROOT / "2月4号测试数据" / "过湿2" / "2月4号过湿测试2.xlsx",
            RAW_ROOT / "2月5号测试数据" / "过湿测试" / "2月5号过湿测试数据20260205.xlsx",
        ],
    },
    "过干": {
        "target_file": TARGET_DIR / "过干.xlsx",
        "raw_files": [
            RAW_ROOT / "2月5号测试数据" / "过干测试" / "过干1" / "2月5号过干测试1.xlsx",
            RAW_ROOT / "2月5号测试数据" / "过干测试" / "过干2" / "2月5号过干测试2.xlsx",
            RAW_ROOT / "2月6号测试数据" / "过干1" / "过干测试数据1.xlsx",
            RAW_ROOT / "2月6号测试数据" / "过干2" / "过干测试数据2.xlsx",
            RAW_ROOT / "2月6号测试数据" / "过干3" / "2月6号过干测试数据3.xlsx",
            RAW_ROOT / "2月7号测试数据" / "过干1" / "2月7号过干测试数据1.xlsx",
            RAW_ROOT / "2月7号测试数据" / "过干2" / "2月7号过干测试数据2.xlsx",
        ],
    },
}


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series)


def _prepare_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    keyed["测试时间"] = _ensure_datetime(keyed["测试时间"])
    keyed["测试分钟"] = keyed["测试时间"].dt.floor("min")
    for col in SHARED_VALUE_COLUMNS:
        keyed[f"{col}_匹配值"] = keyed[col].astype(float).round(3)
    keyed["匹配组序号"] = keyed.groupby(
        ["测试分钟"] + [f"{col}_匹配值" for col in SHARED_VALUE_COLUMNS]
    ).cumcount()
    return keyed


def _distance_matrix(target_block: pd.DataFrame, raw_block: pd.DataFrame) -> np.ndarray:
    target_values = target_block[[f"{col}_匹配值" for col in SHARED_VALUE_COLUMNS]].to_numpy()
    raw_values = raw_block[[f"{col}_匹配值" for col in SHARED_VALUE_COLUMNS]].to_numpy()
    diff = target_values[:, None, :] - raw_values[None, :, :]
    return np.sum(diff * diff, axis=2)


def _iter_unmatched_minutes(aligned: pd.DataFrame) -> Iterable[pd.Timestamp]:
    minutes = aligned.loc[aligned["raw_row_id"].isna(), "测试分钟"]
    return minutes.dropna().drop_duplicates().tolist()


def align_target_with_raw(target_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    target = _prepare_keys(target_df).reset_index(drop=True)
    raw = _prepare_keys(raw_df).reset_index(drop=True)
    raw["raw_row_id"] = raw.index

    feature_columns = [col for col in EXTRA_FEATURE_COLUMNS if col in raw.columns]
    exact_cols = ["测试分钟"] + [f"{col}_匹配值" for col in SHARED_VALUE_COLUMNS] + ["匹配组序号"]
    aligned = target.merge(raw[exact_cols + ["raw_row_id"] + feature_columns], on=exact_cols, how="left")

    used_raw_ids = set(aligned["raw_row_id"].dropna().astype(int).tolist())

    for minute in _iter_unmatched_minutes(aligned):
        target_block = aligned[(aligned["测试分钟"] == minute) & (aligned["raw_row_id"].isna())].copy()
        raw_block = raw[(raw["测试分钟"] == minute) & (~raw["raw_row_id"].isin(used_raw_ids))].copy()
        if target_block.empty or raw_block.empty:
            continue

        distances = _distance_matrix(target_block, raw_block)
        available_target = list(range(len(target_block)))
        available_raw = list(range(len(raw_block)))
        assignments: list[tuple[int, int]] = []

        while available_target and available_raw:
            best_pair = None
            best_distance = None
            for ti in available_target:
                for ri in available_raw:
                    distance = float(distances[ti, ri])
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_pair = (ti, ri)
            if best_pair is None:
                break
            assignments.append(best_pair)
            available_target.remove(best_pair[0])
            available_raw.remove(best_pair[1])

        for target_pos, raw_pos in assignments:
            aligned_index = target_block.index[target_pos]
            raw_row = raw_block.iloc[raw_pos]
            aligned.at[aligned_index, "raw_row_id"] = int(raw_row["raw_row_id"])
            for col in feature_columns:
                aligned.at[aligned_index, col] = raw_row[col]
            used_raw_ids.add(int(raw_row["raw_row_id"]))

    # 某些分钟内，EIS 汇总行数会比原始秒级日志更多；此时允许复用同一分钟内最接近的原始记录。
    for minute in _iter_unmatched_minutes(aligned):
        target_block = aligned[(aligned["测试分钟"] == minute) & (aligned["raw_row_id"].isna())].copy()
        raw_block = raw[raw["测试分钟"] == minute].copy()
        if target_block.empty or raw_block.empty:
            continue

        distances = _distance_matrix(target_block, raw_block)
        for target_pos in range(len(target_block)):
            raw_pos = int(np.argmin(distances[target_pos]))
            aligned_index = target_block.index[target_pos]
            raw_row = raw_block.iloc[raw_pos]
            aligned.at[aligned_index, "raw_row_id"] = int(raw_row["raw_row_id"])
            for col in feature_columns:
                aligned.at[aligned_index, col] = raw_row[col]

    return aligned


def build_augmented_dataframe(target_df: pd.DataFrame, raw_df: pd.DataFrame, label: str) -> pd.DataFrame:
    aligned = align_target_with_raw(target_df, raw_df)
    augmented = target_df.copy().reset_index(drop=True)

    augmented["测试时间"] = _ensure_datetime(augmented["测试时间"])
    augmented["label"] = label
    augmented["raw_row_id"] = aligned["raw_row_id"]

    for col in EXTRA_FEATURE_COLUMNS:
        augmented[col] = aligned[col] if col in aligned.columns else np.nan

    for new_col, left_col, right_col in DERIVED_FEATURE_SPECS:
        if left_col in augmented.columns and right_col in augmented.columns:
            augmented[new_col] = augmented[left_col] - augmented[right_col]

    return augmented


def _load_raw_concat(raw_files: list[Path]) -> pd.DataFrame:
    frames = []
    for raw_file in raw_files:
        df = pd.read_excel(raw_file).copy()
        df = df.assign(source_file=raw_file.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def augment_all_fault_files() -> dict[str, Path]:
    output_paths: dict[str, Path] = {}
    combined_frames = []

    for label, config in SOURCE_MAP.items():
        target_df = pd.read_excel(config["target_file"])
        raw_df = _load_raw_concat(config["raw_files"])
        augmented = build_augmented_dataframe(target_df, raw_df, label=label)

        output_path = config["target_file"].with_name(f"{config['target_file'].stem}_补充特征.xlsx")
        augmented.to_excel(output_path, index=False)
        output_paths[label] = output_path
        combined_frames.append(augmented)

    combined_path = TARGET_DIR / "水淹和膜干故障测试数据_补充特征汇总.xlsx"
    pd.concat(combined_frames, ignore_index=True).to_excel(combined_path, index=False)
    output_paths["汇总"] = combined_path
    return output_paths


if __name__ == "__main__":
    paths = augment_all_fault_files()
    for name, path in paths.items():
        print(f"{name}: {path}")
