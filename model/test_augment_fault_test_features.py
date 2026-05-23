from pathlib import Path
import sys
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment_fault_test_features import (
    DERIVED_FEATURE_SPECS,
    EXTRA_FEATURE_COLUMNS,
    align_target_with_raw,
    build_augmented_dataframe,
)


class AugmentFaultTestFeaturesTest(unittest.TestCase):
    def test_align_target_with_raw_prefers_exact_match_then_falls_back_within_minute(self):
        target = pd.DataFrame(
            {
                "测试时间": pd.to_datetime(
                    [
                        "2026-02-05 13:54:00",
                        "2026-02-05 13:54:00",
                        "2026-02-05 13:54:00",
                    ]
                ),
                "电堆总电压": [100.0, 100.0, 101.0],
                "电堆总电流": [50.0, 50.0, 51.1],
                "电堆功率": [5.0, 5.0, 5.2],
            }
        )
        raw = pd.DataFrame(
            {
                "测试时间": pd.to_datetime(
                    [
                        "2026-02-05 13:54:01",
                        "2026-02-05 13:54:02",
                        "2026-02-05 13:54:03",
                    ]
                ),
                "电堆总电压": [100.0, 100.0, 101.0],
                "电堆总电流": [50.0, 50.0, 51.0],
                "电堆功率": [5.0, 5.0, 5.1],
                "氢气循环泵FK": [4001, 4002, 4003],
            }
        )

        aligned = align_target_with_raw(target, raw)

        self.assertEqual(aligned["氢气循环泵FK"].tolist(), [4001, 4002, 4003])
        self.assertTrue(aligned["raw_row_id"].notna().all())

    def test_build_augmented_dataframe_adds_selected_and_derived_features(self):
        target = pd.DataFrame(
            {
                "测试时间": pd.to_datetime(["2026-02-05 13:54:00"]),
                "总阻抗": [51.01],
                "平均阻抗": [0.21],
                "最高阻抗": [0.298],
                "次高阻抗": [0.291],
                "最低阻抗": [0.153],
                "次低阻抗": [0.164],
                "标准差": [0.028],
                "EIS电阻实部": [0.057],
                "EIS电阻虚部": [0.015],
                "电堆总电压": [195.5],
                "电堆总电流": [30.6],
                "电堆功率": [5.9823],
            }
        )
        raw = pd.DataFrame(
            {
                "测试时间": pd.to_datetime(["2026-02-05 13:54:08"]),
                "电堆总电压": [195.5],
                "电堆总电流": [30.6],
                "电堆功率": [5.9823],
                "FC系统入口高压": [11.5],
                "进堆氢压": [0.95],
                "出堆氢压": [0.82],
                "进堆空压": [0.74],
                "FC空压机出口压力": [0.77],
                "进堆空温": [63.0],
                "出堆空温": [69.0],
                "进堆水温": [71.0],
                "出堆水温": [75.5],
                "氢气循环泵FK": [4999],
                "比例阀反馈": [38.6],
                "purge时间": [8],
                "离心机速度FK": [68888],
                "离心机功率": [3.4],
                "三通阀开度FK": [24],
                "高压水泵转速FK": [4000],
                "FC空压机出口温度": [70.0],
                "FC换热器入口温度": [41.2],
            }
        )

        augmented = build_augmented_dataframe(target, raw, label="正常")

        self.assertTrue(set(EXTRA_FEATURE_COLUMNS).issubset(augmented.columns))
        self.assertAlmostEqual(augmented.loc[0, "氢压差"], 0.13, places=6)
        self.assertAlmostEqual(augmented.loc[0, "空压差"], -0.03, places=6)
        self.assertAlmostEqual(augmented.loc[0, "空温升"], 6.0, places=6)
        self.assertAlmostEqual(augmented.loc[0, "水温升"], 4.5, places=6)
        self.assertEqual(augmented.loc[0, "label"], "正常")
        self.assertGreaterEqual(len(DERIVED_FEATURE_SPECS), 4)


if __name__ == "__main__":
    unittest.main()
