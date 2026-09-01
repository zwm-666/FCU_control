"""Consolidate the thesis result tables to ONE table per experiment.

Motivation: the paper directory accumulated 48 CSVs because tables were created
per METRIC KIND (summary / per-seed / per-class / confusion / paired test) rather
than per EXPERIMENT. Table numbers even collide across experiments (表37 exists
both as the main-comparison table and as a noise minority-recall table).

This script writes one block-structured long table per experiment and MOVES the
component tables into ``表格分项归档/``. Nothing is deleted, matching the
non-destructive retention policy in 实验配置与方法说明.md §14.3, and no raw study
artifact under ``model/results/studies/`` is touched.

Block layout: the first column ``记录类型`` names the block; the column set is the
union of all block columns, summary blocks first. Any original component table is
recoverable by filtering on ``记录类型``.

Dry-run by default; pass --apply to write and move.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS = Path(r"D:/learn/毕业材料/graduation/results")
ARCHIVE = RESULTS / "表格分项归档"
CURRENT_ARCHIVE = ARCHIVE / "current_components"
DUP_ARCHIVE = ARCHIVE / "duplicate_or_intermediate"
UNIFIED_CONFIG = "模型统一配置表.csv"
MANIFEST = RESULTS / "表格合并归档记录.json"

# 表73 is a cross-experiment compliance audit keyed by its 范围 column; each
# experiment table takes the rows whose 范围 starts with one of these prefixes.
AUDIT_TABLE = "表73_锁定配置噪声消融合规性审计.csv"
AUDIT_SCOPE_PREFIX = {
    "表1_主对比实验_DesignB_plus_v1.csv": ("DesignB_plus主表",),
    "表2_噪声鲁棒性实验_DesignB_plus_v1.csv": ("锁定配置噪声",),
    "表3_组件消融实验_DesignB_plus_v1.csv": ("锁定配置消融",),
}

# target file -> ordered list of (block name, component csv)
CONSOLIDATION: dict[str, list[tuple[str, str]]] = {
    "表1_主对比实验_DesignB_plus_v1.csv": [
        ("汇总_锁定配置排名", "表68_锁定配置主对比汇总.csv"),
        ("汇总_全部标量指标", "表37_DesignB_plus主表_后五增强.csv"),
        ("类别级指标_跨种子聚合", "表41_DesignB_plus类别级指标.csv"),
        ("逐种子_全部指标", "表38_DesignB_plus逐种子全部指标.csv"),
        ("逐种子_类别级指标", "表39_DesignB_plus逐种子类别指标.csv"),
        ("逐种子_混淆矩阵", "表40_DesignB_plus混淆矩阵.csv"),
        ("配对检验_v3对比plus", "表42_DesignB_v3_vs_plus逐模型配对变化.csv"),
        ("合规审计_锁定配置29项", "表47_锁定配置合规性审计.csv"),
    ],
    "表2_噪声鲁棒性实验_DesignB_plus_v1.csv": [
        ("汇总_鲁棒性排名", "表69_锁定配置噪声鲁棒性排名.csv"),
        ("汇总_准确率随SNR", "表56_噪声鲁棒性_准确率随SNR_锁定配置.csv"),
        ("汇总_宏F1随SNR", "表57_噪声鲁棒性_宏F1随SNR_锁定配置.csv"),
        ("汇总_最小类召回随SNR", "表58_噪声鲁棒性_最小类召回随SNR_锁定配置.csv"),
        ("配对检验_低SNR", "表70_锁定配置低SNR配对检验.csv"),
        ("逐种子_原始记录", "表60_噪声鲁棒性逐种子原始记录_锁定配置.csv"),
    ],
    "表3_组件消融实验_DesignB_plus_v1.csv": [
        ("汇总_三条件逐变体统计", "表71_锁定配置消融逐条件统计.csv"),
        ("汇总_三条件方向一致性", "表72_锁定配置消融三条件方向一致性.csv"),
        ("逐种子_原始记录", "表62_消融实验逐种子原始记录_锁定配置.csv"),
    ],
    "表4_等算力对照实验.csv": [
        ("汇总_三配置排名", "表44_等算力对比_三配置排名.csv"),
        ("配对检验_本文对比最强基线", "表45_各配置下本文与最强基线的配对检验.csv"),
    ],
    "表5_历史降配基线实验.csv": [
        ("噪声_汇总_准确率随SNR", "表34_噪声鲁棒性_准确率随SNR.csv"),
        ("噪声_汇总_宏F1随SNR", "表35_噪声鲁棒性_宏F1随SNR.csv"),
        ("噪声_汇总_最小类召回随SNR", "表37_噪声鲁棒性_最小类召回随SNR.csv"),
        ("噪声_汇总_退化与排名", "表36_噪声退化与鲁棒性排名.csv"),
        ("噪声_逐种子_原始记录", "表38_噪声鲁棒性逐种子原始记录.csv"),
        ("消融_汇总_组件贡献", "表39_消融实验_组件贡献与显著性.csv"),
        ("消融_汇总_类别级F1", "表41_消融实验_类别级F1.csv"),
        ("消融_汇总_三条件区分能力", "表42_消融条件的区分能力对照.csv"),
        ("消融_汇总_三条件方向一致性", "表43_消融三条件对照与方向一致性.csv"),
        ("消融_逐种子_原始记录", "表40_消融实验逐种子原始记录.csv"),
    ],
    "表6_数据集与协议审计.csv": [
        ("数据集_独立审计", "表23_物理判据数据集独立审计.csv"),
        ("数据集_来源与构成", "表33_数据集来源与构成对照.csv"),
        ("协议_按块分组诚实基线", "表24_按块分组诚实基线对照.csv"),
        ("协议_episode纯度与时间结构", "表25_episode纯度与标签时间结构.csv"),
        ("标签定义_物理判据对比旧合成版", "表31_物理判据_vs_旧合成版_配对对照.csv"),
    ],
}

# Byte-identical duplicates of the retained tables, and intermediate versions
# superseded by 表69-72. Archived separately so the top level stays readable.
DUPLICATE_OR_INTERMEDIATE = [
    "表48_噪声鲁棒性_准确率随SNR_锁定配置.csv",
    "表49_噪声鲁棒性_宏F1随SNR_锁定配置.csv",
    "表50_噪声鲁棒性_最小类召回随SNR_锁定配置.csv",
    "表51_噪声退化与鲁棒性排名_锁定配置.csv",
    "表52_噪声鲁棒性逐种子原始记录_锁定配置.csv",
    "表59_噪声退化与鲁棒性排名_锁定配置.csv",
    "表61_消融实验_各条件组件贡献_锁定配置.csv",
    "表63_消融三条件方向一致性_锁定配置.csv",
    "表64_锁定配置噪声鲁棒性排名.csv",
    "表65_锁定配置噪声低SNR配对检验.csv",
    "表66_锁定配置消融逐条件统计.csv",
    "表67_锁定配置消融三条件方向一致性.csv",
]

FINAL_TOP_LEVEL = {UNIFIED_CONFIG, *CONSOLIDATION}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def locate(name: str) -> Path | None:
    """Find a component table at the top level or in either archive dir.

    Exporters write component tables straight into 表格分项归档/current_components,
    so a rerun of this script finds them there; a legacy top-level copy still wins
    so the very first consolidation works before the exporters were redirected.
    """
    for candidate in (RESULTS / name, CURRENT_ARCHIVE / name, DUP_ARCHIVE / name):
        if candidate.exists():
            return candidate
    return None


def audit_rows_for(target: str) -> tuple[list[str], list[dict[str, str]]]:
    """Slice 表73 by 范围 so each experiment carries its own audit rows."""
    prefixes = AUDIT_SCOPE_PREFIX.get(target)
    path = locate(AUDIT_TABLE)
    if not prefixes or path is None:
        return [], []
    fields, rows = read_rows(path)
    kept = [r for r in rows if any(str(r.get("范围", "")).startswith(p) for p in prefixes)]
    return fields, kept


def build(target: str, blocks: list[tuple[str, str]]) -> tuple[list[str], list[dict[str, str]], dict[str, int], list[str]]:
    """Return (fieldnames, rows, per-block row counts, missing components)."""
    ordered_fields: list[str] = ["记录类型", "来源分项表"]
    collected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    missing: list[str] = []

    for block, component in blocks:
        path = locate(component)
        if path is None:
            missing.append(component)
            continue
        fields, rows = read_rows(path)
        for field in fields:
            if field not in ordered_fields:
                ordered_fields.append(field)
        for row in rows:
            item = {"记录类型": block, "来源分项表": component}
            item.update(row)
            collected.append(item)
        counts[block] = len(rows)

    audit_fields, audit = audit_rows_for(target)
    if audit:
        for field in audit_fields:
            if field not in ordered_fields:
                ordered_fields.append(field)
        for row in audit:
            item = {"记录类型": "合规审计_锁定配置噪声消融", "来源分项表": AUDIT_TABLE}
            item.update(row)
            collected.append(item)
        counts["合规审计_锁定配置噪声消融"] = len(audit)

    return ordered_fields, collected, counts, missing


def write_table(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write consolidated tables and move components; without it only print the plan")
    args = parser.parse_args()

    plan: dict[str, Any] = {"mode": "apply" if args.apply else "dry_run", "targets": {}}
    built: dict[str, tuple[list[str], list[dict[str, str]], dict[str, int]]] = {}
    all_missing: list[str] = []

    for target, blocks in CONSOLIDATION.items():
        fields, rows, counts, missing = build(target, blocks)
        built[target] = (fields, rows, counts)
        all_missing.extend(missing)
        plan["targets"][target] = {
            "blocks": counts,
            "total_rows": len(rows),
            "columns": len(fields),
            "missing_components": missing,
        }

    components = [c for blocks in CONSOLIDATION.values() for _, c in blocks]
    plan["archive_current"] = sorted(set(components) | {AUDIT_TABLE})
    plan["archive_duplicate_or_intermediate"] = [
        name for name in DUPLICATE_OR_INTERMEDIATE if (RESULTS / name).exists()
    ]
    plan["components_read_from_archive"] = sorted(
        name for name in plan["archive_current"]
        if not (RESULTS / name).exists() and locate(name) is not None
    )
    existing = {p.name for p in RESULTS.glob("*.csv")}
    accounted = set(plan["archive_current"]) | set(DUPLICATE_OR_INTERMEDIATE) | FINAL_TOP_LEVEL
    plan["unaccounted_csv"] = sorted(existing - accounted)
    plan["missing_components"] = sorted(set(all_missing))
    plan["raw_study_artifacts_touched"] = False
    plan["deletes"] = "none; components are MOVED into 表格分项归档/"

    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if not args.apply:
        return 0
    if plan["missing_components"]:
        raise SystemExit(f"Refusing: missing components {plan['missing_components']}")
    if plan["unaccounted_csv"]:
        raise SystemExit(f"Refusing: unaccounted CSVs {plan['unaccounted_csv']}")

    CURRENT_ARCHIVE.mkdir(parents=True, exist_ok=True)
    DUP_ARCHIVE.mkdir(parents=True, exist_ok=True)

    for target, (fields, rows, _counts) in built.items():
        write_table(RESULTS / target, fields, rows)

    moved: list[dict[str, Any]] = []
    for name in plan["archive_current"]:
        src = RESULTS / name
        if not src.exists():
            continue
        record = {"name": name, "bytes": src.stat().st_size, "sha256": sha256(src),
                  "archive": "current_components"}
        shutil.move(str(src), str(CURRENT_ARCHIVE / name))
        moved.append(record)
    for name in plan["archive_duplicate_or_intermediate"]:
        src = RESULTS / name
        if not src.exists():
            continue
        record = {"name": name, "bytes": src.stat().st_size, "sha256": sha256(src),
                  "archive": "duplicate_or_intermediate"}
        shutil.move(str(src), str(DUP_ARCHIVE / name))
        moved.append(record)

    remaining = sorted(p.name for p in RESULTS.glob("*.csv"))
    # Record the FULL archive contents, not only what this run moved, so a rerun
    # (when everything is already archived) does not overwrite the provenance of
    # the first consolidation with an empty list.
    archive_inventory = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
         "archive": path.parent.name}
        for path in sorted(ARCHIVE.rglob("*.csv"))
    ]
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "One table per experiment at the top level; component tables archived, never deleted.",
        "top_level_tables": remaining,
        "consolidated": {
            target: {"blocks": counts, "total_rows": len(rows),
                     "sha256": sha256(RESULTS / target)}
            for target, (_f, rows, counts) in built.items()
        },
        "moved_this_run": moved,
        "archive_inventory": archive_inventory,
        "archive_dirs": {"current": str(CURRENT_ARCHIVE), "duplicate": str(DUP_ARCHIVE)},
        "deleted": [],
        "raw_study_artifacts_touched": False,
        "verification": {
            "top_level_is_exactly_expected": set(remaining) == FINAL_TOP_LEVEL,
            "archive_component_count": len(archive_inventory),
            "row_conservation": {
                target: sum(counts.values()) == len(rows)
                for target, (_f, rows, counts) in built.items()
            },
        },
    }
    MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "mode": "applied",
        "top_level_count": len(remaining),
        "top_level": remaining,
        "archived_count": len(moved),
        "manifest": str(MANIFEST),
        "verification": record["verification"],
    }, ensure_ascii=False, indent=2))

    ok = (record["verification"]["top_level_is_exactly_expected"]
          and all(record["verification"]["row_conservation"].values()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
