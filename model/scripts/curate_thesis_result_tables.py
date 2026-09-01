"""Curate thesis results: one unified config table plus experiment result tables.

Dry-run by default. Use --apply only after reviewing the printed plan.
Never touches model/results/studies raw artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(r"D:/learn/毕业材料/graduation/results")
UNIFIED_CONFIG = "模型统一配置表.csv"
UNIFIED_CONFIG_NOTE = "模型统一配置表_说明.txt"

# User requirement: retain all experiment results. The current paper directory
# may contain multiple historical table families; none is deleted by this policy.
# The explicit list below remains as a human-readable current set, while the
# preserve-all guard in plan() protects every existing CSV.
#
# Layout since 2026-08-31 (see 实验配置与方法说明.md §15-§16): the top level holds
# ONLY the tables the thesis body cites. Per-seed records and audits live in
# 附录表/; process-only experiments and the merged per-experiment tables live in
# 表格分项归档/. Built by scripts/build_paper_tables.py.
KEEP_TABLES = {
    # Tables cited by the thesis body (outline sections 4.1-4.4.6).
    "论文表1_不同模型故障诊断性能对比.csv",
    "论文表2_不同模型噪声鲁棒性分析.csv",
    "论文表3_分支结构消融实验.csv",
    "论文表4_数据集与预处理.csv",
    # The one and only configuration table.
    UNIFIED_CONFIG,
}

# User instruction for this project: preserve all experiment results. The
# static list above documents the intended current set; this union prevents a
# later rerun from silently deleting any historical CSV that was not yet named
# when the list was written.
if RESULTS.exists():
    KEEP_TABLES.update(path.name for path in RESULTS.glob("*.csv"))

# Kept for backward-compatible imports only. The active policy is non-destructive:
# no existing result CSV or note is deleted. Raw study directories are also
# untouched.
EXPLICIT_DELETE = set()

# Historical note names are preserved too; this set is intentionally empty.
DELETE_NOTE_NAMES = set()

# Legacy delete list retained in source control as documentation is no longer
# active.
LEGACY_EXPLICIT_DELETE = {
    # First-generation row-random/public exports.
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
    # Superseded DesignB main comparison and parameter tables.
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
    "表43_DesignB_plus约束审计.csv",
    "表44_DesignB_plus参数量与配置.csv",
    "表47_锁定配置合规性审计.csv",
    # Superseded noise tables and superseded/duplicate locked-config fragments.
    # 表48-52 are byte-identical duplicates of 表56-60 (same noise study), and
    # 表59/64/65 were superseded by 表69/70; 表61/66/67 by 表71/72.
    "表34_噪声鲁棒性_准确率随SNR.csv",
    "表35_噪声鲁棒性_宏F1随SNR.csv",
    "表36_噪声退化与鲁棒性排名.csv",
    "表37_噪声鲁棒性_最小类召回随SNR.csv",
    "表38_噪声鲁棒性逐种子原始记录.csv",
    "表48_噪声鲁棒性_准确率随SNR_锁定配置.csv",
    "表49_噪声鲁棒性_宏F1随SNR_锁定配置.csv",
    "表50_噪声鲁棒性_最小类召回随SNR_锁定配置.csv",
    "表51_噪声退化与鲁棒性排名_锁定配置.csv",
    "表52_噪声鲁棒性逐种子原始记录_锁定配置.csv",
    "表59_噪声退化与鲁棒性排名_锁定配置.csv",
    "表64_锁定配置噪声鲁棒性排名.csv",
    "表65_锁定配置噪声低SNR配对检验.csv",
    "表53_消融实验_组件贡献与显著性_锁定配置.csv",
    "表54_消融实验逐种子原始记录_锁定配置.csv",
    "表55_消融实验_类别级F1_锁定配置.csv",
    # Reduced-baseline ablation tables, superseded by the locked-config ablation.
    "表39_消融实验_组件贡献与显著性.csv",
    "表40_消融实验逐种子原始记录.csv",
    "表41_消融实验_类别级F1.csv",
    "表42_消融条件的区分能力对照.csv",
    "表43_消融三条件对照与方向一致性.csv",
    "表61_消融实验_各条件组件贡献_锁定配置.csv",
    "表63_消融三条件方向一致性_锁定配置.csv",
    "表66_锁定配置消融逐条件统计.csv",
    "表67_锁定配置消融三条件方向一致性.csv",
}

# Legacy note-delete list retained only for provenance documentation.
LEGACY_DELETE_NOTE_NAMES = {
    "表1_主表_按行随机划分_10模型5种子_说明.txt",
    "表2_配对显著性检验_按行随机划分_说明.txt",
    "表4_稳健性_时间戳块交叉验证_10模型_说明.txt",
    "表5_配对显著性检验_交叉验证_说明.txt",
    "表6_泄漏诊断_按行随机划分_说明.txt",
    "表7_两协议对照_泄漏影响量化_说明.txt",
    "表8_模型参数量与配置_实测_说明.txt",
    "表8_模型配置对照_说明.txt",
    "表9-14_公开数据集_说明.txt",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan() -> tuple[list[Path], list[Path], list[str]]:
    remove_tables = []
    remove_notes = []
    unexpected = []
    for path in sorted(RESULTS.glob("*.csv")):
        # Non-destructive preservation policy: every existing result CSV stays.
        if path.name not in KEEP_TABLES:
            unexpected.append(path.name)
    # Notes are also preserved; do not schedule deletions.
    remove_tables.clear()
    remove_notes.clear()
    return remove_tables, remove_notes, unexpected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform deletion; without it only print the plan")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    remove_tables, remove_notes, unexpected = plan()
    missing_keep = sorted(name for name in KEEP_TABLES if not (RESULTS / name).exists())
    print(json.dumps({
        "mode": "apply" if args.apply else "dry_run",
        "keep_tables": sorted(KEEP_TABLES),
        "remove_tables": [path.name for path in remove_tables],
        "remove_notes": [path.name for path in remove_notes],
        "unexpected_csv_not_removed": unexpected,
        "missing_keep_tables": missing_keep,
        "raw_study_artifacts_touched": False,
    }, ensure_ascii=False, indent=2))
    if not args.apply:
        return 0
    if missing_keep:
        raise SystemExit(f"Refusing cleanup: missing keep tables: {missing_keep}")

    removed = []
    for path in remove_tables + remove_notes:
        removed.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "kind": "table" if path.suffix.lower() == ".csv" else "table_note",
        })
        path.unlink()

    remaining_tables = sorted(path.name for path in RESULTS.glob("*.csv"))
    forbidden_remaining = sorted(
        name for name in (EXPLICIT_DELETE | DELETE_NOTE_NAMES) if (RESULTS / name).exists()
    )
    record = {
        "cleanup_time_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "Keep one unified model configuration table and all distinct thesis experiment result tables; remove superseded duplicates and pure compliance/configuration tables.",
        "current_configuration_table": UNIFIED_CONFIG,
        "kept_table_count": len(remaining_tables),
        "kept_tables": remaining_tables,
        "removed_count": len(removed),
        "removed": removed,
        "verification": {
            "unified_config_present": (RESULTS / UNIFIED_CONFIG).exists(),
            "removed_files_absent": not forbidden_remaining,
            "remaining_tables_are_allowlisted": set(remaining_tables).issubset(KEEP_TABLES),
            "raw_study_artifacts_touched": False,
        },
    }
    manifest = RESULTS / "论文结果表清理记录_统一配置表.json"
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "mode": "applied",
        "removed_count": len(removed),
        "remaining_table_count": len(remaining_tables),
        "forbidden_remaining": forbidden_remaining,
        "manifest": str(manifest),
        "verification": record["verification"],
    }, ensure_ascii=False, indent=2))
    checks = record["verification"]
    return 0 if checks["unified_config_present"] and checks["removed_files_absent"] and checks["remaining_tables_are_allowlisted"] and not checks["raw_study_artifacts_touched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
