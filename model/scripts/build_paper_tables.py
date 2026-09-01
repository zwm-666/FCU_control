"""Build the tables the thesis actually cites, plus appendix and archive layers.

Motivation: the paper directory held 6 experiment tables with 32 blocks / 2077
rows, but the thesis outline (曾维明-毕业论文提纲) only cites 12 of those blocks.
Process-only results (equal-compute, historical reduced_baselines, compliance
self-audits, label-version comparison) must not appear as thesis results at all —
the locked spec §8.2 explicitly forbids using equal-compute as a fair comparison.

Three layers, nothing deleted:

  top level      论文表1-4  — blocks cited by the thesis body
  results/附录表/ 附录A/B    — per-seed records and audits (transparency, not body)
  表格分项归档/    过程性实验/ — process-only experiment tables, summarised in prose

Dry-run by default; --apply writes and moves.
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
APPENDIX = RESULTS / "附录表"
ARCHIVE = RESULTS / "表格分项归档"
PROCESS_ARCHIVE = ARCHIVE / "过程性实验"
# The consolidated per-experiment tables are the INPUT to this script; they are
# not process-only, so they get their own archive dir and stay re-readable.
MERGED_ARCHIVE = ARCHIVE / "合并实验表"
UNIFIED_CONFIG = "模型统一配置表.csv"
MANIFEST = RESULTS / "论文表格分层记录.json"

# Source experiment tables produced by scripts/consolidate_thesis_tables.py.
SRC_MAIN = "表1_主对比实验_DesignB_plus_v1.csv"
SRC_NOISE = "表2_噪声鲁棒性实验_DesignB_plus_v1.csv"
SRC_ABLATION = "表3_组件消融实验_DesignB_plus_v1.csv"
SRC_EQUAL = "表4_等算力对照实验.csv"
SRC_HISTORICAL = "表5_历史降配基线实验.csv"
SRC_DATASET = "表6_数据集与协议审计.csv"

# ---------------------------------------------------------------- thesis body
# target -> ordered [(new block name, source table, source block, outline ref)]
PAPER_TABLES: dict[str, list[tuple[str, str, str, str]]] = {
    "论文表1_不同模型故障诊断性能对比.csv": [
        ("模型排名与核心指标", SRC_MAIN, "汇总_锁定配置排名", "4.4.5"),
        ("全部标量指标", SRC_MAIN, "汇总_全部标量指标", "4.4.5"),
        ("类别级指标", SRC_MAIN, "类别级指标_跨种子聚合", "4.4.2"),
        ("最优模型混淆矩阵", SRC_MAIN, "逐种子_混淆矩阵", "4.4.2"),
    ],
    "论文表2_不同模型噪声鲁棒性分析.csv": [
        ("鲁棒性排名", SRC_NOISE, "汇总_鲁棒性排名", "4.4.6"),
        ("准确率随SNR", SRC_NOISE, "汇总_准确率随SNR", "4.4.6"),
        ("宏F1随SNR", SRC_NOISE, "汇总_宏F1随SNR", "4.4.6"),
        ("低SNR配对检验", SRC_NOISE, "配对检验_低SNR", "4.4.6"),
    ],
    "论文表3_分支结构消融实验.csv": [
        ("三条件逐变体统计", SRC_ABLATION, "汇总_三条件逐变体统计", "4.4.3"),
        ("三条件方向一致性", SRC_ABLATION, "汇总_三条件方向一致性", "4.4.3"),
    ],
    "论文表4_数据集与预处理.csv": [
        ("数据集来源与构成", SRC_DATASET, "数据集_来源与构成", "4.1"),
        ("数据集独立审计", SRC_DATASET, "数据集_独立审计", "4.2"),
    ],
}

# The confusion-matrix block holds all 10 models x 5 seeds; the body only shows
# the best model's matrix, so keep just those rows.
BEST_MODEL_KEY = "di_emstgat"
CONFUSION_BLOCK = "最优模型混淆矩阵"

# ------------------------------------------------------------------- appendix
APPENDIX_TABLES: dict[str, list[tuple[str, str, str]]] = {
    "附录A_逐种子原始记录.csv": [
        ("主对比_逐种子全部指标", SRC_MAIN, "逐种子_全部指标"),
        ("主对比_逐种子类别级指标", SRC_MAIN, "逐种子_类别级指标"),
        ("主对比_逐种子混淆矩阵", SRC_MAIN, "逐种子_混淆矩阵"),
        ("噪声_逐种子原始记录", SRC_NOISE, "逐种子_原始记录"),
        ("消融_逐种子原始记录", SRC_ABLATION, "逐种子_原始记录"),
    ],
    "附录B_协议与合规审计.csv": [
        ("合规审计_锁定配置主表", SRC_MAIN, "合规审计_锁定配置29项"),
        ("合规审计_主表范围", SRC_MAIN, "合规审计_锁定配置噪声消融"),
        ("合规审计_噪声范围", SRC_NOISE, "合规审计_锁定配置噪声消融"),
        ("合规审计_消融范围", SRC_ABLATION, "合规审计_锁定配置噪声消融"),
        ("协议_按块分组诚实基线", SRC_DATASET, "协议_按块分组诚实基线"),
        ("协议_episode纯度与时间结构", SRC_DATASET, "协议_episode纯度与时间结构"),
        ("噪声_最小类召回随SNR", SRC_NOISE, "汇总_最小类召回随SNR"),
        ("主对比_内部调参对照", SRC_MAIN, "配对检验_v3对比plus"),
    ],
}

# ------------------------------------------------------- process-only results
# Whole tables that must not be cited as thesis results. Archived and summarised.
PROCESS_ONLY_TABLES = [SRC_EQUAL, SRC_HISTORICAL]
# Blocks inside 表6 that are process-only (dataset version comparison).
PROCESS_ONLY_BLOCKS = [(SRC_DATASET, "标签定义_物理判据对比旧合成版")]

PROCESS_SUMMARY = {
    "等算力对照实验": {
        "source_table": SRC_EQUAL,
        "raw_root": "model/results/studies/equal_compute_designB",
        "why_not_in_thesis": "锁定规范 §8.2 明确要求非本文模型使用降配预算；等算力配置违反该规范，不得作为主表或公平比较依据",
        "one_line": "等算力探索显示树模型在标准预算下可超过本文模型，但该配置违反锁定规范，仅作过程记录",
    },
    "历史降配基线实验": {
        "source_table": SRC_HISTORICAL,
        "raw_root": "model/results/studies/noise_physics3_designB, ablation_physics3_designB, ablation_snr10_designB, ablation_train20pct_designB",
        "why_not_in_thesis": "使用 comparison_strength=reduced_baselines，基线预算与锁定主表 designB_reduced_plus 不一致",
        "one_line": "早期 reduced_baselines 预算下的噪声与消融结果，与锁定主表预算不一致，已被锁定配置结果取代",
    },
    "标签定义版本对照": {
        "source_table": SRC_DATASET,
        "source_block": "标签定义_物理判据对比旧合成版",
        "raw_root": "model/results/studies/legacy_synthetic_3class_control",
        "why_not_in_thesis": "数据集版本变更的过程对照，不是模型性能结论",
        "one_line": "物理水平衡判据标签与旧合成标签的配对对照，用于说明数据集版本变更的影响，非模型结论",
    },
}

# Outline sections with no result yet. Registered honestly, never faked.
KNOWN_GAPS = {
    "4.4.4 优化前后模型的故障诊断结果对比": {
        "needed": "QAAdam / CEO 超参寻优前后的受控对照",
        "status": "结果不存在；锁定配置未跑该对照",
        "action": "登记为待补实验，需新建预注册变体与新输出目录",
    },
    "4.3 分支特征提取效果分析": {
        "needed": "4.3.1 DI vs AB 受控对比；4.3.2 三阶段 t-SNE；4.3.3 路由热图与分支负载",
        "status": "4.3.1 可由论文表1 的 DI/AB 数值支撑；t-SNE 与路由热图产物不存在",
        "action": "t-SNE 与路由热图登记为待补产物",
    },
}

FINAL_TOP_LEVEL = {UNIFIED_CONFIG, *PAPER_TABLES}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(name: str) -> Path | None:
    """Find a consolidated experiment table at the top level or in the archive.

    After --apply the sources live in 表格分项归档/, so this keeps the script
    idempotent: rerunning it rebuilds the same layers instead of silently
    producing empty tables.
    """
    for candidate in (RESULTS / name, MERGED_ARCHIVE / name, PROCESS_ARCHIVE / name):
        if candidate.exists():
            return candidate
    return None


def read_blocks(name: str) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    """Return (fieldnames, {block: rows}) for a consolidated experiment table."""
    path = locate(name)
    if path is None:
        return [], {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            grouped.setdefault(row["记录类型"], []).append(row)
    return fields, grouped


def drop_empty_columns(fields: list[str], rows: list[dict[str, str]]) -> list[str]:
    """Keep only columns that carry a value somewhere — merged blocks leave many
    all-empty columns behind, which makes a paper table unreadable."""
    used = {field for field in fields
            if any(str(row.get(field, "")).strip() for row in rows)}
    return [field for field in fields if field in used]


def assemble(spec: list[tuple[str, ...]], with_outline: bool) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
    ordered: list[str] = ["记录类型"]
    if with_outline:
        ordered.append("提纲小节")
    ordered.append("来源分项表")
    collected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    cache: dict[str, tuple[list[str], dict[str, list[dict[str, str]]]]] = {}

    for entry in spec:
        if with_outline:
            block, src, src_block, outline = entry
        else:
            block, src, src_block = entry
            outline = ""
        if src not in cache:
            cache[src] = read_blocks(src)
        fields, grouped = cache[src]
        rows = grouped.get(src_block, [])
        if block == CONFUSION_BLOCK:
            rows = [r for r in rows if r.get("model_key") == BEST_MODEL_KEY]
        for field in fields:
            if field not in ordered and field not in {"记录类型", "来源分项表"}:
                ordered.append(field)
        for row in rows:
            item = {"记录类型": block}
            if with_outline:
                item["提纲小节"] = outline
            item["来源分项表"] = row.get("来源分项表", src)
            for key, value in row.items():
                if key not in {"记录类型", "来源分项表", "提纲小节"}:
                    item[key] = value
            collected.append(item)
        counts[block] = len(rows)

    kept = drop_empty_columns(ordered, collected)
    return kept, collected, counts


def write_table(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the layered tables and archive process-only results")
    args = parser.parse_args()

    built_paper = {name: assemble(spec, True) for name, spec in PAPER_TABLES.items()}
    built_appendix = {name: assemble(spec, False) for name, spec in APPENDIX_TABLES.items()}

    source_rows = 0
    for name in (SRC_MAIN, SRC_NOISE, SRC_ABLATION, SRC_EQUAL, SRC_HISTORICAL, SRC_DATASET):
        _f, grouped = read_blocks(name)
        source_rows += sum(len(v) for v in grouped.values())

    paper_rows = sum(len(rows) for _f, rows, _c in built_paper.values())
    appendix_rows = sum(len(rows) for _f, rows, _c in built_appendix.values())

    plan: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry_run",
        "source_total_rows": source_rows,
        "paper_tables": {
            name: {"blocks": counts, "rows": len(rows), "columns": len(fields)}
            for name, (fields, rows, counts) in built_paper.items()
        },
        "appendix_tables": {
            name: {"blocks": counts, "rows": len(rows), "columns": len(fields)}
            for name, (fields, rows, counts) in built_appendix.items()
        },
        "paper_rows": paper_rows,
        "appendix_rows": appendix_rows,
        "process_only_tables": PROCESS_ONLY_TABLES,
        "process_only_blocks": [f"{t}::{b}" for t, b in PROCESS_ONLY_BLOCKS],
        "known_gaps": sorted(KNOWN_GAPS),
        "deletes": "none; process-only tables are MOVED into 表格分项归档/过程性实验/",
        "raw_study_artifacts_touched": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if not args.apply:
        return 0

    for name, (fields, rows, _counts) in built_paper.items():
        write_table(RESULTS / name, fields, rows)
    for name, (fields, rows, _counts) in built_appendix.items():
        write_table(APPENDIX / name, fields, rows)

    # Move the inputs out of the top level: process-only tables to 过程性实验/,
    # the consolidated experiment tables to 合并实验表/. Nothing is deleted.
    PROCESS_ARCHIVE.mkdir(parents=True, exist_ok=True)
    MERGED_ARCHIVE.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, Any]] = []
    for name in (SRC_MAIN, SRC_NOISE, SRC_ABLATION, SRC_DATASET, *PROCESS_ONLY_TABLES):
        src = RESULTS / name
        if not src.exists():
            continue
        process_only = name in PROCESS_ONLY_TABLES
        target_dir = PROCESS_ARCHIVE if process_only else MERGED_ARCHIVE
        record = {"name": name, "bytes": src.stat().st_size, "sha256": sha256(src),
                  "process_only": process_only, "archive": target_dir.name}
        shutil.move(str(src), str(target_dir / name))
        moved.append(record)

    remaining = sorted(p.name for p in RESULTS.glob("*.csv"))
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "rule": ("Top level holds only the tables the thesis body cites; per-seed "
                 "records and audits go to 附录表/; process-only experiments are "
                 "archived and summarised in prose. Nothing is deleted."),
        "top_level_tables": remaining,
        "appendix_tables": sorted(p.name for p in APPENDIX.glob("*.csv")),
        "paper_tables": {
            name: {"blocks": counts, "rows": len(rows), "sha256": sha256(RESULTS / name),
                   "outline_sections": sorted({e[3] for e in PAPER_TABLES[name]})}
            for name, (_f, rows, counts) in built_paper.items()
        },
        "appendix_detail": {
            name: {"blocks": counts, "rows": len(rows), "sha256": sha256(APPENDIX / name)}
            for name, (_f, rows, counts) in built_appendix.items()
        },
        "row_accounting": {
            "source_total": source_rows,
            "paper": paper_rows,
            "appendix": appendix_rows,
            "process_only_and_duplicated": source_rows - paper_rows - appendix_rows,
            "note": ("paper + appendix < source because process-only blocks are "
                     "archived, and the confusion-matrix block appears in full in "
                     "附录A but is trimmed to the best model in 论文表1"),
        },
        "process_only_summary": PROCESS_SUMMARY,
        "known_gaps": KNOWN_GAPS,
        "archived_this_run": moved,
        "archive_dirs": {"components": str(ARCHIVE / "current_components"),
                         "duplicates": str(ARCHIVE / "duplicate_or_intermediate"),
                         "merged_experiment_tables": str(MERGED_ARCHIVE),
                         "process_only": str(PROCESS_ARCHIVE)},
        "deleted": [],
        "raw_study_artifacts_touched": False,
        "verification": {
            "top_level_is_exactly_expected": set(remaining) == FINAL_TOP_LEVEL,
            "paper_row_conservation": {
                name: sum(counts.values()) == len(rows)
                for name, (_f, rows, counts) in built_paper.items()
            },
            "appendix_row_conservation": {
                name: sum(counts.values()) == len(rows)
                for name, (_f, rows, counts) in built_appendix.items()
            },
        },
    }
    MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "mode": "applied",
        "top_level": remaining,
        "appendix": record["appendix_tables"],
        "paper_rows": paper_rows,
        "appendix_rows": appendix_rows,
        "archived_this_run": [m["name"] for m in moved],
        "manifest": str(MANIFEST),
        "verification": record["verification"],
    }, ensure_ascii=False, indent=2))

    ok = (record["verification"]["top_level_is_exactly_expected"]
          and all(record["verification"]["paper_row_conservation"].values())
          and all(record["verification"]["appendix_row_conservation"].values()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
