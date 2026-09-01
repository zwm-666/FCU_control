"""Audit a locked Design B run against 实验配置与方法说明.md.

The spec file is the authority. This script checks a run that already exists
against every constraint written there, so the paper tables are built from a
verified run rather than from assumption.

Two registered variants (`--variant`), differing ONLY in the AB budget:
  DesignB_plus_v1   AB = micro / 1 epoch   -> DI wins the main table (historical)
  DesignB_branch_v1 AB = compact / 100 ep  -> AB wins (adaptive branch primary)
Both remain auditable; the historical run is never re-interpreted under the new
expectations, and vice versa.

Checks (spec section in brackets):
  [2.1] dataset path, row count, class balance
  [2.3] split protocol, train/test sizes, seeds, 50 records, 10 models
  [3.1] capacity/epoch budget per model, incl. the variant's AB budget,
        PatchTST=nano/1, XGBoost/LightGBM = 1 shallow tree
  [3.2] DI standard/100 epochs, boundary_weight = 0.0
  [5.1] every required metric present on every record
  [5.4] the variant's verified per-seed inequalities and baseline ceiling
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ROOT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model")
DATA = MODEL_ROOT / "数据文件/Public datasets_physics3_designB.csv"
DATA_MANIFEST = MODEL_ROOT / "数据文件/Public datasets_physics3_designB_manifest.json"
OUT = Path(r"D:/learn/毕业材料/graduation/results")
# Component tables land in the archive dir; scripts/consolidate_thesis_tables.py
# merges them into the per-experiment tables at the top level (说明 §15).
COMPONENTS = OUT / "表格分项归档" / "current_components"

EXPECTED_BY_VARIANT = {
    # Historical locked config: AB deliberately de-tuned to micro/1 epoch.
    "DesignB_plus_v1": {
        "run": "physics3_designB_final_plus_v1",
        "records": 50, "models": 10, "seeds": [42, 43, 44, 45, 46],
        "train": 6799, "test": 1700,
        "di_accuracy": 98.4235, "ab_accuracy": 94.6471,
        "baseline_max_accuracy": 95.0,
        "baseline_best_single_seed": 92.4118,
        "comparison_strength": "designB_reduced_plus",
        "proposed_capacity": "standard", "ab_capacity": "micro",
        "ab_epochs": 1,
        "boundary_weight": 0.0,
        "primary_model": "di_emstgat",
    },
    # Branch-primary variant: AB gets compact/100 so the adaptive branch (the
    # thesis contribution) is not handicapped relative to DI. Baselines and DI
    # are byte-identical to DesignB_plus_v1.
    "DesignB_branch_v1": {
        "run": "physics3_designB_branch_v1",
        "records": 50, "models": 10, "seeds": [42, 43, 44, 45, 46],
        "train": 6799, "test": 1700,
        "di_accuracy": 98.4235, "ab_accuracy": 98.8824,
        "baseline_max_accuracy": 95.0,
        "baseline_best_single_seed": 92.4118,
        "comparison_strength": "designB_reduced_plus",
        "proposed_capacity": "standard", "ab_capacity": "compact",
        "ab_epochs": 100,
        "boundary_weight": 0.0,
        "primary_model": "ab_emstgat",
    },
}

PROPOSED = {"di_emstgat", "ab_emstgat"}
REQUIRED_METRICS = [
    "accuracy", "balanced_accuracy", "precision_weighted", "recall_weighted",
    "f1_macro", "f1_weighted", "cohen_kappa", "fit_time", "prediction_time",
    "feature_count", "train_samples", "test_samples",
]

checks: list[dict] = []


def check(section: str, name: str, ok: bool, detail: str) -> None:
    checks.append({"规范条目": section, "检查项": name,
                   "结果": "通过" if ok else "不通过", "实测": detail})
    print(f"  [{'OK ' if ok else 'FAIL'}] {section} {name}: {detail}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="DesignB_branch_v1",
                        choices=sorted(EXPECTED_BY_VARIANT),
                        help="Which registered variant's expectations to assert.")
    args = parser.parse_args(argv)
    EXPECTED = EXPECTED_BY_VARIANT[args.variant]
    variant = args.variant
    run = MODEL_ROOT / "results/studies" / EXPECTED["run"]
    checks.clear()

    print("=" * 78)
    print(f"审计变体：{variant}   运行目录：{EXPECTED['run']}")
    print(f"AB 预算：{EXPECTED['ab_capacity']}/{EXPECTED['ab_epochs']}ep   "
          f"主模型：{EXPECTED['primary_model']}")
    print("=" * 78)

    records = json.loads((run / "per_seed_records.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    print("=" * 78)
    print("A. 数据集 [规范 2.1]")
    print("=" * 78)
    data = pd.read_csv(DATA)
    dm = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    check("2.1", "数据文件存在", DATA.exists(), str(DATA.name))
    check("2.1", "SHA-256 与 manifest 一致", sha == dm.get("output_sha256"),
          f"{sha[:16]}... vs manifest {str(dm.get('output_sha256'))[:16]}...")
    check("2.1", "总样本数 = 8499", len(data) == 8499, str(len(data)))
    counts = data["State_Label"].value_counts().to_dict()
    check("2.1", "三类各 2833", all(counts.get(c) == 2833 for c in
          ("Normal", "Flooding", "Membrane_Drying")), json.dumps(counts, ensure_ascii=False))
    check("2.1", "不含 Thermal_Management_Fault",
          "Thermal_Management_Fault" not in counts, "已排除")
    check("2.1", "合成样本数 = 0",
          sum(dm.get("synthetic_counts", {}).values()) == 0,
          str(sum(dm.get("synthetic_counts", {}).values())))

    print("\n" + "=" * 78)
    print("B. 划分与记录完整性 [规范 2.3 / 5.4]")
    print("=" * 78)
    check("2.3", "记录总数 = 50", len(frame) == EXPECTED["records"], str(len(frame)))
    check("2.3", "模型数 = 10", frame.model.nunique() == EXPECTED["models"],
          str(frame.model.nunique()))
    check("2.3", "种子 = 42-46",
          sorted(frame.seed.unique().tolist()) == EXPECTED["seeds"],
          str(sorted(frame.seed.unique().tolist())))
    per_model = frame.groupby("model").size()
    check("2.3", "每模型恰好 5 条", bool((per_model == 5).all()),
          f"min={per_model.min()} max={per_model.max()}")
    per_seed = frame.groupby("seed").size()
    check("2.3", "每种子恰好 10 条", bool((per_seed == 10).all()),
          f"min={per_seed.min()} max={per_seed.max()}")
    check("2.3", "训练样本 = 6799",
          bool((frame.train_samples == EXPECTED["train"]).all()),
          str(sorted(frame.train_samples.unique().tolist())))
    check("2.3", "测试样本 = 1700",
          bool((frame.test_samples == EXPECTED["test"]).all()),
          str(sorted(frame.test_samples.unique().tolist())))
    check("2.3", "划分协议为行级分层随机",
          "stratified" in str(manifest.get("split_protocol", "")).lower(),
          str(manifest.get("split_protocol"))[:60])

    print("\n" + "=" * 78)
    print("C. 锁定模型配置 [规范 3.1 / 3.2 / 3.3]")
    print("=" * 78)
    check("3.1", "comparison_strength = designB_reduced_plus",
          manifest.get("comparison_strength") == EXPECTED["comparison_strength"],
          str(manifest.get("comparison_strength")))
    check("3.2", "DI capacity = standard",
          manifest.get("proposed_capacity") == EXPECTED["proposed_capacity"],
          str(manifest.get("proposed_capacity")))
    check("3.3", f"AB capacity = {EXPECTED['ab_capacity']}",
          manifest.get("ab_capacity") == EXPECTED["ab_capacity"],
          str(manifest.get("ab_capacity")))
    # Historical runs predate the ab_epochs manifest field; derive it then.
    ab_epochs_actual = manifest.get("ab_epochs")
    if ab_epochs_actual is None:
        ab_epochs_actual = (
            1 if manifest.get("comparison_strength") in {"designB_reduced", "designB_reduced_plus"}
            else manifest.get("deep_epochs")
        )
    check("3.3", f"AB epochs = {EXPECTED['ab_epochs']}",
          int(ab_epochs_actual) == EXPECTED["ab_epochs"],
          f"{ab_epochs_actual}" + ("（由 comparison_strength 推导，manifest 未记录）"
                                   if manifest.get("ab_epochs") is None else ""))
    check("3.2", "deep_epochs = 100", manifest.get("deep_epochs") == 100,
          str(manifest.get("deep_epochs")))
    check("3.2", "boundary_weight = 0.0",
          float(manifest.get("proposed_boundary_weight", -1)) == EXPECTED["boundary_weight"],
          str(manifest.get("proposed_boundary_weight")))

    print("\n" + "=" * 78)
    print("D. 必须保存的指标 [规范 5.1]")
    print("=" * 78)
    missing = [m for m in REQUIRED_METRICS if m not in frame.columns]
    check("5.1", "标量指标齐全", not missing,
          "全部存在" if not missing else f"缺少 {missing}")
    has_report = all("classification_report" in r for r in records)
    has_cm = all("confusion_matrix" in r for r in records)
    check("5.1", "每条含类别级报告", has_report, "是" if has_report else "否")
    check("5.1", "每条含混淆矩阵", has_cm, "是" if has_cm else "否")
    nan_cols = [m for m in REQUIRED_METRICS if m in frame.columns
                and frame[m].isna().any()]
    check("5.1", "无缺失值", not nan_cols,
          "无" if not nan_cols else f"含NaN: {nan_cols}")

    print("\n" + "=" * 78)
    print("E. 已验证约束 [规范 5.4]")
    print("=" * 78)
    di = frame[frame.model == "di_emstgat"].set_index("seed").sort_index()
    ab = frame[frame.model == "ab_emstgat"].set_index("seed").sort_index()
    di_acc, ab_acc = 100 * di.accuracy.mean(), 100 * ab.accuracy.mean()
    check("5.4", f"DI 平均 Accuracy = {EXPECTED['di_accuracy']}%",
          abs(di_acc - EXPECTED["di_accuracy"]) < 1e-3, f"{di_acc:.4f}%")
    check("5.4", f"AB 平均 Accuracy = {EXPECTED['ab_accuracy']}%",
          abs(ab_acc - EXPECTED["ab_accuracy"]) < 1e-3, f"{ab_acc:.4f}%")
    # Which proposed variant leads is a property of the AB budget, so assert it
    # per variant instead of hardcoding DI as the winner.
    primary = EXPECTED["primary_model"]
    lead, trail = (ab, di) if primary == "ab_emstgat" else (di, ab)
    lead_name, trail_name = ("AB", "DI") if primary == "ab_emstgat" else ("DI", "AB")
    win_acc = int((lead.accuracy.values > trail.accuracy.values).sum())
    win_f1 = int((lead.f1_macro.values > trail.f1_macro.values).sum())
    check("5.4", f"主模型为 {lead_name}（均值更高）",
          lead.accuracy.mean() > trail.accuracy.mean(),
          f"{lead_name} {100 * lead.accuracy.mean():.4f}% vs "
          f"{trail_name} {100 * trail.accuracy.mean():.4f}%")
    check("5.4", f"{lead_name} 在 Accuracy 上的逐种子胜场（>=3/5）",
          win_acc >= 3, f"{win_acc}/5")
    check("5.4", f"{lead_name} 在 Macro-F1 上的逐种子胜场（>=3/5）",
          win_f1 >= 3, f"{win_f1}/5")
    base = frame[~frame.model.isin(PROPOSED)]
    worst = 100 * base.accuracy.max()
    check("5.4", "所有对比模型每个种子 Accuracy <= 95%",
          worst <= EXPECTED["baseline_max_accuracy"], f"最高 {worst:.4f}%")
    check("5.4", f"对比模型最高单种子 = {EXPECTED['baseline_best_single_seed']}%",
          abs(worst - EXPECTED["baseline_best_single_seed"]) < 1e-3,
          f"{worst:.4f}%")

    print("\n" + "=" * 78)
    print("F. 主表（锁定配置实测）")
    print("=" * 78)
    agg = frame.groupby("model").agg(
        acc=("accuracy", "mean"), sd=("accuracy", "std"),
        f1=("f1_macro", "mean"), kappa=("cohen_kappa", "mean"),
        fit=("fit_time", "mean")).sort_values("acc", ascending=False)
    agg[["acc", "sd", "f1", "kappa"]] *= 100
    agg["类型"] = ["本文" if m in PROPOSED else "对比" for m in agg.index]
    print(agg.round(4).to_string())

    failed = [c for c in checks if c["结果"] == "不通过"]
    print("\n" + "=" * 78)
    print(f"审计结果（{variant}）：{len(checks) - len(failed)}/{len(checks)} 项通过")
    print("=" * 78)
    if failed:
        for c in failed:
            print(f"  不通过 -> {c['规范条目']} {c['检查项']}: {c['实测']}")
    else:
        print(f"  {variant} 与规范文件完全一致，可用于论文。")

    OUT.mkdir(parents=True, exist_ok=True)
    COMPONENTS.mkdir(parents=True, exist_ok=True)
    # Per-variant filenames so auditing one variant never overwrites another's
    # record (spec 7.4: never overwrite historical evidence).
    suffix = "" if variant == "DesignB_plus_v1" else f"_{variant}"
    pd.DataFrame(checks).to_csv(
        COMPONENTS / f"表47_锁定配置合规性审计{suffix}.csv",
        index=False, encoding="utf-8-sig")
    (OUT / f"锁定配置合规性审计{suffix}.json").write_text(json.dumps({
        "spec_file": "实验配置与方法说明.md",
        "locked_config": variant,
        "run_dir": str(run),
        "ab_budget": f"{EXPECTED['ab_capacity']}/{EXPECTED['ab_epochs']}ep",
        "primary_model": EXPECTED["primary_model"],
        "data_sha256": sha,
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed),
        "all_passed": not failed,
        "failed": failed,
        "note": ("等算力实验 (comparison_strength=standard) 违反规范 8.2，"
                 "其结果不得作为主表；仅可作为独立变体记录，且必须使用"
                 "新变体名与新目录"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
