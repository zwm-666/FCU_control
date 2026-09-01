"""Remove superseded thesis tables without touching raw study artifacts.

SUPERSEDED: the current curation entry point is
``scripts/curate_thesis_result_tables.py``, which owns the authoritative
keep/delete lists and writes ``论文结果表清理记录_统一配置表.json``. This script
is kept because its cleanup record (``结果表格清理记录_DesignB_plus_v1.json``)
documents the first-generation cleanup; its retain list is derived from the
current curation script so the two can never disagree.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.curate_thesis_result_tables import KEEP_TABLES, UNIFIED_CONFIG

RESULTS = Path(r"D:/learn/毕业材料/graduation/results")

DELETE_NAMES = []

# Historical deletion names are retained below for audit/reference only.
LEGACY_DELETE_NAMES = [
    # First-generation row-random / public four-class tables.
    "表1_主表_按行随机划分_10模型5种子.csv",
    "表2_配对显著性检验_按行随机划分.csv",
    "表3_每种子明细_按行随机划分.csv",
    "表4_稳健性_时间戳块交叉验证_10模型.csv",
    "表5_配对显著性检验_交叉验证.csv",
    "表6_泄漏诊断_按行随机划分.csv",
    "表7_两协议对照_泄漏影响量化.csv",
    "表8_模型参数量与配置_实测.csv",
    "表8_模型配置对照.csv",
    "表9_公开数据集主表_10模型5种子.csv",
    "表10_公开数据集逐种子指标.csv",
    "表11_公开数据集类别级指标.csv",
    "表12_公开数据集逐种子类别级指标.csv",
    "表13_公开数据集混淆矩阵.csv",
    "表14_公开数据集实验配置.csv",
    "表15_公开数据集改进版主表_10模型5种子.csv",
    "表16_公开数据集改进版逐种子全部指标.csv",
    "表17_公开数据集改进版逐种子类别指标.csv",
    "表18_公开数据集改进版混淆矩阵.csv",
    "表19_公开数据集改进版类别级指标.csv",
    "表20_公开数据集改进版参数量与配置.csv",
    "表21_公开数据集改进候选_vs_严格匹配控制.csv",
    "表22_公开数据集改进版_vs_历史结果.csv",
    # Superseded Design-B main-comparison tables.
    "表26_物理判据数据集主表_10模型5种子.csv",
    "表27_物理判据逐种子全部指标.csv",
    "表28_物理判据逐种子类别指标.csv",
    "表29_物理判据混淆矩阵.csv",
    "表30_物理判据类别级指标.csv",
    "表32_物理判据_本文模型_vs_最强基线.csv",
    "表30_DesignB最终主表_10模型5种子.csv",
    "表31_DesignB逐种子全部指标.csv",
    "表32_DesignB逐种子类别指标.csv",
    "表33_DesignB混淆矩阵.csv",
    "表34_DesignB类别级指标.csv",
    "表35_DesignB_DI强于AB及基线95阈值审计.csv",
    "表36_DesignB参数量与配置.csv",
]

DELETE_NOTES = []

LEGACY_DELETE_NOTES = [
    "表1_主表_按行随机划分_10模型5种子_说明.txt",
    "表2_配对显著性检验_按行随机划分_说明.txt",
    "表4_稳健性_时间戳块交叉验证_10模型_说明.txt",
    "表5_配对显著性检验_交叉验证_说明.txt",
    "表6_泄漏诊断_按行随机划分_说明.txt",
    "表7_两协议对照_泄漏影响量化_说明.txt",
    "表8_模型参数量与配置_实测_说明.txt",
    "表8_模型配置对照_说明.txt",
    "表9-14_公开数据集_说明.txt",
]

RETAIN_NAMES = ([UNIFIED_CONFIG] + sorted(KEEP_TABLES - {UNIFIED_CONFIG})
                 if UNIFIED_CONFIG in KEEP_TABLES
                 else sorted(KEEP_TABLES))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    removed = []
    missing = []
    for name in DELETE_NAMES + DELETE_NOTES:
        path = RESULTS / name
        if not path.exists():
            missing.append(name)
            continue
        removed.append({
            "name": name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "kind": "table" if name.endswith(".csv") else "table_note",
        })
        path.unlink()

    retained_missing = [name for name in RETAIN_NAMES if not (RESULTS / name).exists()]
    remaining_tables = sorted(path.name for path in RESULTS.glob("表*.csv"))
    unexpected_current = [name for name in RETAIN_NAMES if name.endswith(".csv") and name not in remaining_tables]
    record = {
        "cleanup_time_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "When a newer experiment is accepted as the current main result, delete superseded main-result tables; retain raw study directories and independent sensitivity/audit tables.",
        "current_main_variant": "DesignB_plus_v1",
        "current_main_table_family": "表37-表44（DesignB_plus）",
        "removed_count": len(removed),
        "removed": removed,
        "missing_requested_deletions": missing,
        "retained_current_tables": RETAIN_NAMES,
        "retained_missing": retained_missing,
        "remaining_table_files": remaining_tables,
        "verification": {
            "removed_files_absent": all(not (RESULTS / item["name"]).exists() for item in removed),
            "current_tables_present": not unexpected_current and not retained_missing,
            "raw_study_artifacts_touched": False,
        },
    }
    manifest = RESULTS / "结果表格清理记录_DesignB_plus_v1.json"
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "removed_count": record["removed_count"],
        "removed_tables": [item["name"] for item in removed if item["kind"] == "table"],
        "removed_notes": [item["name"] for item in removed if item["kind"] == "table_note"],
        "missing_requested_deletions": missing,
        "retained_missing": retained_missing,
        "remaining_table_count": len(remaining_tables),
        "manifest": str(manifest),
        "verification": record["verification"],
    }, ensure_ascii=False, indent=2))
    return 0 if not retained_missing and record["verification"]["removed_files_absent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
