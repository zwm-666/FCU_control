"""Build the single configuration table covering every retained experiment.

The table is intentionally normalized: model rows describe per-model settings;
experiment rows describe common protocol settings for supporting analyses.
All values come from existing manifests/result tables or are explicitly marked
as not recorded by the originating manifest. No training is performed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

OUT = Path(r"D:/learn/毕业材料/graduation/results")
MODEL_RESULTS = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/studies")
DATA_PATH = r"D:/my_project/h2-fcu-modern-dashboard/model/数据文件/Public datasets_physics3_designB.csv"
DATASET = "Public datasets_physics3_designB.csv"
CONFIG_PATH = OUT / "模型统一配置表.csv"

FIELDS = [
    "configuration_version", "record_type", "experiment_name", "experiment_kind", "status",
    "dataset", "data_path", "label_col", "split_protocol", "test_size", "seeds",
    "train_samples", "test_samples", "models", "condition", "eval_snr_db", "train_fraction",
    "noise_seed", "snr_grid_db", "model", "model_key", "variant", "role", "capacity",
    "complexity_kind", "complexity_value", "epochs", "learning_rate", "weight_decay",
    "validation_split", "boundary_weight", "boundary_margin", "optimizer",
    "augmentation_or_degradation", "key_configuration", "result_root", "result_tables",
    "source_manifest",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def val(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def base_row(**kwargs: Any) -> dict[str, str]:
    row = {field: "" for field in FIELDS}
    row.update({key: val(value) for key, value in kwargs.items() if key in row})
    return row


def manifest_common(manifest: dict[str, Any]) -> dict[str, Any]:
    data_path = manifest.get("data_path", DATA_PATH)
    return {
        "dataset": DATASET,
        "data_path": data_path,
        "label_col": manifest.get("label_col", "State_Label"),
        "split_protocol": manifest.get("split_protocol", "row/window-level stratified random split"),
        "test_size": manifest.get("test_size", 0.2),
        "seeds": manifest.get("seeds", [42, 43, 44, 45, 46]),
        "train_samples": manifest.get("train_samples", "6799"),
        "test_samples": manifest.get("test_samples", "1700"),
        "models": manifest.get("models", []),
        "source_manifest": str((MODEL_RESULTS / "physics3_designB_final_plus_v1" / "manifest.json").resolve()),
    }


def main_rows() -> list[dict[str, str]]:
    """Read the already-created main config rows, regardless of old/new schema."""
    source = CONFIG_PATH
    if not source.exists():
        raise FileNotFoundError(source)
    old = csv_rows(source)
    rows: list[dict[str, str]] = []
    for item in old:
        # If this script is rerun, keep only the current main model rows and
        # rebuild the complete table from the manifests below.
        if item.get("experiment_name") and item.get("experiment_name") != "DesignB_plus_v1":
            continue
        key = item.get("model_key", "")
        if not key:
            continue
        rows.append(base_row(
            configuration_version="DesignB_plus_v1",
            record_type="model",
            experiment_name="DesignB_plus_v1",
            experiment_kind="main_comparison",
            status="current_locked",
            dataset=item.get("dataset", DATASET),
            data_path=DATA_PATH,
            label_col=item.get("label_col", "State_Label"),
            split_protocol=item.get("split_protocol", "row/window-level stratified random split"),
            test_size=item.get("test_size", 0.2),
            seeds=item.get("seeds", "42,43,44,45,46"),
            train_samples=item.get("train_samples", 6799),
            test_samples=item.get("test_samples", 1700),
            models="10",
            model=item.get("model", ""),
            model_key=key,
            variant="full",
            role=item.get("role", ""),
            capacity=item.get("capacity", ""),
            complexity_kind=item.get("complexity_kind", ""),
            complexity_value=item.get("complexity_value", ""),
            epochs=item.get("epochs", ""),
            learning_rate=item.get("learning_rate", ""),
            weight_decay=item.get("weight_decay", ""),
            validation_split=item.get("validation_split", ""),
            boundary_weight=item.get("boundary_weight", ""),
            boundary_margin=item.get("boundary_margin", ""),
            optimizer="Adam / AdamW-compatible weight decay for proposed models",
            augmentation_or_degradation="none; clean evaluation",
            key_configuration=item.get("key_configuration", ""),
            result_root="results/studies/physics3_designB_final_plus_v1",
            result_tables="论文表1_不同模型故障诊断性能对比.csv;附录A_逐种子原始记录.csv;附录B_协议与合规审计.csv",
            source_manifest=str((MODEL_RESULTS / "physics3_designB_final_plus_v1" / "manifest.json").resolve()),
        ))
    if len(rows) != 10:
        raise AssertionError(f"Expected 10 DesignB_plus model rows, got {len(rows)}")
    return rows


def add_experiment_rows(rows: list[dict[str, str]], name: str, kind: str, status: str, manifest_path: Path, tables: str, **extra: Any) -> None:
    manifest = read_json(manifest_path)
    common = manifest_common(manifest)
    common.update(extra)
    common.update({
        "configuration_version": name,
        "record_type": "experiment",
        "experiment_name": name,
        "experiment_kind": kind,
        "status": status,
        "result_root": str(manifest_path.parent.resolve()),
        "result_tables": tables,
        "source_manifest": str(manifest_path.resolve()),
    })
    rows.append(base_row(**common))


def add_noise_rows(rows: list[dict[str, str]]) -> None:
    """Register both the current aligned noise result and its historical root."""
    entries = [
        (
            "noise_physics3_designB",
            "historical_result",
            # Historical reduced_baselines noise tables are consolidated into
            # 表5 (blocks prefixed 噪声_*); the raw study root stays the source.
            "表格分项归档/过程性实验/表5_历史降配基线实验.csv（过程性，不进论文）",
            "历史 reduced_baselines 噪声结果；保留用于过程审计，不作为当前锁定配置论文依据",
        ),
        (
            "noise_designB_plus_v1",
            "current_locked_result",
            "论文表2_不同模型噪声鲁棒性分析.csv;附录A_逐种子原始记录.csv;附录B_协议与合规审计.csv",
            "当前 DesignB_plus_v1 锁定配置 clean + 40/35/30/25/20/15/10/5 dB；噪声只加到评估输入",
        ),
    ]
    for dirname, status, tables, key_configuration in entries:
        path = MODEL_RESULTS / dirname / "manifest.json"
        if not path.exists():
            continue
        manifest = read_json(path)
        common = manifest_common(manifest)
        common.update({
            "configuration_version": ("DesignB_plus_v1" if dirname == "noise_designB_plus_v1"
                                      else dirname),
            "record_type": "experiment",
            "experiment_name": dirname,
            "experiment_kind": "noise_robustness",
            "status": status,
            "condition": "clean + evaluation-only Gaussian noise",
            "eval_snr_db": "clean,40,35,30,25,20,15,10,5",
            "snr_grid_db": manifest.get("snr_grid_db", [40, 35, 30, 25, 20, 15, 10, 5]),
            "augmentation_or_degradation": manifest.get(
                "noise_applied_to", "evaluation inputs only; training remains clean"),
            "result_root": str(path.parent.resolve()),
            "result_tables": tables,
            "source_manifest": str(path.resolve()),
            "key_configuration": key_configuration,
        })
        rows.append(base_row(**common))


def add_ablation_rows(rows: list[dict[str, str]]) -> None:
    """Register both historical and current lock-aligned ablations.

    The old reduced-baselines roots are deliberately retained as historical
    records. The current roots are separate records with the exact
    DesignB_plus_v1 budgets (DI standard/100, AB micro/1, boundary=0.0), so the
    unified table never silently presents an old budget as the current one.
    """
    variants_by_model = {
        "di_emstgat": ["full", "no_dcc", "no_bigru", "no_knn_graph", "no_self_attn", "no_attn_pool", "no_skip_cls", "no_pos_emb", "no_boundary"],
        "ab_emstgat": ["full", "no_dcc", "no_bigru", "no_knn_graph", "no_self_attn", "no_attn_pool", "no_skip_cls", "no_pos_emb", "no_boundary", "no_router_gate"],
    }
    # dirname, condition, eval_snr, train_fraction, status, result tables
    # Historical reduced-baseline ablation roots stay registered for provenance,
    # but their paper tables were removed as superseded, so they point at the
    # raw study root instead of a deleted CSV.
    # Historical reduced_baselines ablation tables are consolidated into 表5
    # (blocks prefixed 消融_*).
    HIST_TABLES = "表格分项归档/过程性实验/表5_历史降配基线实验.csv（过程性，不进论文）"
    ALIGNED_TABLES = "论文表3_分支结构消融实验.csv;附录A_逐种子原始记录.csv;附录B_协议与合规审计.csv"
    conditions = [
        ("ablation_physics3_designB", "clean", "clean", "1.0", "historical_result", HIST_TABLES),
        ("ablation_snr10_designB", "snr10dB", "10.0", "1.0", "historical_result", HIST_TABLES),
        ("ablation_train20pct_designB", "clean_train20pct", "clean", "0.2", "historical_result", HIST_TABLES),
        ("ablation_designB_plus_v1_clean", "clean", "clean", "1.0", "current_locked_result", ALIGNED_TABLES),
        ("ablation_designB_plus_v1_snr10", "snr10dB", "10.0", "1.0", "current_locked_result", ALIGNED_TABLES),
        ("ablation_designB_plus_v1_train20pct", "clean_train20pct", "clean", "0.2", "current_locked_result", ALIGNED_TABLES),
    ]
    for dirname, condition, snr, fraction, status, tables in conditions:
        path = MODEL_RESULTS / dirname / "manifest.json"
        if not path.exists():
            continue
        manifest = read_json(path)
        is_aligned = dirname.startswith("ablation_designB_plus_v1_")
        # The aligned runner records per-model values explicitly. Historical
        # manifests may not, so fall back to their legacy common fields.
        capacity_map = manifest.get("capacity", {})
        epochs_map = manifest.get("epochs_per_model", {})
        common = manifest_common(manifest)
        common.update({
            "configuration_version": "DesignB_plus_v1" if is_aligned else dirname,
            "experiment_name": dirname,
            "experiment_kind": "component_ablation",
            "status": status,
            "condition": condition,
            "eval_snr_db": snr,
            "train_fraction": fraction,
            "train_samples": 1359 if fraction == "0.2" else 6799,
            "test_samples": 1700,
            "noise_seed": manifest.get("noise_seed", ""),
            "models": "di_emstgat,ab_emstgat",
            "result_root": str(path.parent.resolve()),
            "result_tables": tables,
            "source_manifest": str(path.resolve()),
            "learning_rate": manifest.get("learning_rate", ""),
            "weight_decay": manifest.get("weight_decay", ""),
            "boundary_weight": manifest.get("boundary_weight", ""),
            "boundary_margin": manifest.get("boundary_margin", ""),
            "validation_split": 0.15,
            "optimizer": "Adam / AdamW-compatible weight decay",
            "augmentation_or_degradation": manifest.get(
                "degradation_rationale",
                "one component removed; all other settings held fixed"),
        })
        for model, variants in variants_by_model.items():
            for variant in variants:
                spec = manifest.get("variants", {}).get(variant, {})
                flags = spec.get("flags", {})
                override = spec.get("boundary_weight_override", "")
                capacity = capacity_map.get(model, "")
                epochs = epochs_map.get(model, manifest.get("epochs", ""))
                row = dict(common)
                row.update({
                    "record_type": "model_variant",
                    "model": model,
                    "model_key": model,
                    "variant": variant,
                    "role": "本文模型",
                    "capacity": capacity,
                    "epochs": epochs,
                    "boundary_weight": (override if override != ""
                                        else manifest.get("boundary_weight", "")),
                    "key_configuration": (
                        f"flags={json.dumps(flags, ensure_ascii=False, separators=(',', ':'))}; "
                        f"boundary_weight_override={override}"),
                })
                rows.append(base_row(**row))


def add_equal_compute_rows(rows: list[dict[str, str]]) -> None:
    """Register every retained equal-compute exploration separately.

    These runs are not paper-main results, but the user explicitly asked that
    experiment results be preserved. Keeping both manifests in the unified
    table makes their provenance visible without mixing them into the locked
    DesignB_plus_v1 rows.
    """
    entries = [
        ("equal_compute_designB", "historical_result_not_main",
         "等算力探索（本文仍使用15%内部验证，非锁定主表）"),
        ("equal_compute_noval_designB", "historical_result_not_main",
         "等算力探索（去除本文模型内部验证切分，非锁定主表）"),
    ]
    for dirname, status, key_configuration in entries:
        path = MODEL_RESULTS / dirname / "manifest.json"
        if not path.exists():
            continue
        manifest = read_json(path)
        common = manifest_common(manifest)
        common.update({
            "configuration_version": dirname,
            "experiment_name": dirname,
            "experiment_kind": "equal_compute_comparison",
            "status": status,
            "result_root": str(path.parent.resolve()),
            # Process-only: locked spec §8.2 forbids citing equal-compute as a fair
        # comparison, so it has no thesis table; archived for traceability.
        "result_tables": "表格分项归档/过程性实验/表4_等算力对照实验.csv（过程性，不进论文）",
            "source_manifest": str(path.resolve()),
            "epochs": manifest.get("deep_epochs", ""),
            "learning_rate": "proposed=0.000813; baseline=0.001（由工厂默认/记录推导）",
            "weight_decay": "proposed=0.000427; baseline=无",
            "validation_split": manifest.get("proposed_validation_split", ""),
            "boundary_weight": manifest.get("proposed_boundary_weight", ""),
            "boundary_margin": manifest.get("proposed_boundary_margin", ""),
            "optimizer": "equal-compute comparison",
            "augmentation_or_degradation": "none; clean evaluation",
            "key_configuration": key_configuration,
        })
        for model in manifest.get("models", []):
            is_prop = model in {"di_emstgat", "ab_emstgat"}
            row = dict(common)
            row.update({
                "record_type": "model",
                "model": model,
                "model_key": model,
                "variant": "full",
                "role": "本文模型" if is_prop else "对比模型",
                "capacity": (manifest.get("ab_capacity") if model == "ab_emstgat"
                             else manifest.get("proposed_capacity") if is_prop
                             else "standard"),
                "key_configuration": key_configuration,
            })
            rows.append(base_row(**row))

    # Boundary-pair exploration has its own manifest and no model-comparison
    # table in the current paper directory; register it as an experiment row.
    boundary = MODEL_RESULTS / "boundary_fixed_designB" / "manifest.json"
    if boundary.exists():
        manifest = read_json(boundary)
        common = manifest_common(manifest)
        common.update({
            "configuration_version": "boundary_fixed_designB",
            "record_type": "experiment",
            "experiment_name": "boundary_fixed_designB",
            "experiment_kind": "boundary_loss_candidate",
            "status": "historical_result_not_main",
            "result_root": str(boundary.parent.resolve()),
            "result_tables": "",
            "source_manifest": str(boundary.resolve()),
            "key_configuration": "boundary_pairs=(0,2),(2,0); exploratory candidate; not mixed into locked main result",
        })
        rows.append(base_row(**common))


def add_supporting_rows(rows: list[dict[str, str]]) -> None:
    # These are supporting result experiments retained in the paper directory;
    # they have no per-model training configuration, so use experiment rows.
    support = [
        # All four supporting audits now share 表6; the 记录类型 column inside it
        # separates 数据集_独立审计 / 数据集_来源与构成 / 协议_* / 标签定义_*.
        ("physics_dataset_audit", "dataset_audit", "supporting", "论文表4_数据集与预处理.csv", "物理判据数据集_独立审计.json"),
        # Grouped-protocol audit is transparency material, not a body result.
        ("episode_grouped_audit", "grouped_protocol_audit", "supporting", "附录B_协议与合规审计.csv", "物理判据_episode纯度分析.json"),
        # Label-version comparison is process-only (dataset revision evidence).
        ("label_definition_comparison", "dataset_comparison", "supporting", "表格分项归档/合并实验表/表6_数据集与协议审计.csv（过程性，不进论文）", "物理判据实验契约与审计.json"),
        ("dataset_composition_audit", "dataset_audit", "supporting", "论文表4_数据集与预处理.csv", "物理判据实验契约与审计.json"),
    ]
    for name, kind, status, tables, manifest_name in support:
        manifest_path = OUT / manifest_name
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if isinstance(manifest, dict):
                data_path = manifest.get("data_path", DATA_PATH)
                label_col = manifest.get("label_col", "State_Label")
            else:
                # Some supporting audits are JSON arrays rather than manifests.
                data_path = DATA_PATH
                label_col = "State_Label"
        else:
            data_path = DATA_PATH
            label_col = "State_Label"
        rows.append(base_row(
            configuration_version=name,
            record_type="experiment",
            experiment_name=name,
            experiment_kind=kind,
            status=status,
            dataset=DATASET,
            data_path=data_path,
            label_col="State_Label",
            split_protocol="见对应实验审计文件",
            models="",
            key_configuration="支撑性实验，无新增模型训练配置",
            result_root="D:/learn/毕业材料/graduation/results",
            result_tables=tables,
            source_manifest=str(manifest_path.resolve()),
        ))


def main() -> int:
    if not OUT.exists():
        OUT.mkdir(parents=True)
    rows = main_rows()
    add_noise_rows(rows)
    add_ablation_rows(rows)
    add_equal_compute_rows(rows)
    add_supporting_rows(rows)
    with CONFIG_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    note = OUT / "模型统一配置表_说明.txt"
    note.write_text(
        "模型统一配置表.csv 是论文结果目录中唯一的配置表。\n"
        "它集中记录 DesignB_plus 主对比、噪声鲁棒性、组件消融、等算力对比和支撑性审计实验。\n"
        "record_type=model/model_variant 表示模型或模型变体配置；record_type=experiment 表示没有新增训练模型的支撑性实验。\n"
        "历史 manifest 未记录的参数明确标为‘manifest未记录’，不使用当前配置冒充历史配置。\n"
        "当前主配置为 DesignB_plus_v1；主结果及各独立实验结果表分别保留在本目录。\n"
        "结果总目录：D:/learn/毕业材料/graduation/results\n"
        "原始实验目录：D:/my_project/h2-fcu-modern-dashboard/model/results/studies\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "path": str(CONFIG_PATH),
        "rows": len(rows),
        "record_types": {kind: sum(row["record_type"] == kind for row in rows) for kind in sorted({row["record_type"] for row in rows})},
        "experiments": sorted({row["experiment_name"] for row in rows}),
        "models_in_main": sum(row["experiment_name"] == "DesignB_plus_v1" and row["record_type"] == "model" for row in rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
