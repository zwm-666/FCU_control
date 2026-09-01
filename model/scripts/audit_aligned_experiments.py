"""Audit all aligned DesignB_plus_v1 satellite experiments.

The governing spec is 实验配置与方法说明.md. This audit is read-only and
checks the newly re-run roots without touching historical roots:

  noise_designB_plus_v1                 10 models x 5 seeds x 9 SNR conditions
  ablation_designB_plus_v1_clean        19 variants x 5 seeds
  ablation_designB_plus_v1_snr10        19 variants x 5 seeds
  ablation_designB_plus_v1_train20pct   19 variants x 5 seeds

It also computes the final aligned metrics and paired statistics used by the
report. No values are copied from console output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model")
STUDIES = ROOT / "results/studies"
DATA = ROOT / "数据文件/Public datasets_physics3_designB.csv"
DATA_MANIFEST = ROOT / "数据文件/Public datasets_physics3_designB_manifest.json"
# 表64-67 were superseded by 表69-72 (finalize_aligned_reporting.py); this
# read-only audit now writes its tables outside the paper directory.
OUT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/derived_superseded/aligned_audit_64_67")

MODELS = ["di_emstgat", "ab_emstgat", "tcn", "mtgnn", "lightgbm",
          "xgboost", "itransformer", "gatv2", "transformer", "patchtst"]
PROPOSED = {"di_emstgat", "ab_emstgat"}
ABL_VARIANTS = {
    "di_emstgat": ["full", "no_dcc", "no_bigru", "no_knn_graph",
                    "no_self_attn", "no_attn_pool", "no_skip_cls",
                    "no_pos_emb", "no_boundary"],
    "ab_emstgat": ["full", "no_dcc", "no_bigru", "no_knn_graph",
                    "no_self_attn", "no_attn_pool", "no_skip_cls",
                    "no_pos_emb", "no_boundary", "no_router_gate"],
}
CN = {
    "di_emstgat": "DI-EMSTGAT（本文）", "ab_emstgat": "AB-EMSTGAT（本文）",
    "tcn": "TCN", "mtgnn": "MTGNN", "lightgbm": "LightGBM",
    "xgboost": "XGBoost", "itransformer": "iTransformer",
    "gatv2": "GATv2", "transformer": "Transformer", "patchtst": "PatchTST",
}
VARIANT_CN = {
    "full": "完整模型", "no_dcc": "去除膨胀因果卷积", "no_bigru": "去除双向GRU",
    "no_knn_graph": "去除时序KNN图", "no_self_attn": "去除多头自注意力",
    "no_attn_pool": "注意力池化替为均值池化", "no_skip_cls": "去除表格跳连分类器",
    "no_pos_emb": "去除位置编码", "no_boundary": "去除边界损失",
    "no_router_gate": "去除特征相关性门控",
}

checks: list[dict[str, Any]] = []


def add_check(scope: str, item: str, ok: bool, actual: Any) -> None:
    row = {"范围": scope, "检查项": item, "结果": "通过" if ok else "不通过",
           "实测": str(actual)}
    checks.append(row)
    print(f"  [{'OK ' if ok else 'FAIL'}] {scope} {item}: {actual}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_noise() -> tuple[pd.DataFrame, dict]:
    name = "noise_designB_plus_v1"
    root = STUDIES / name
    records = read_json(root / "per_seed_records.json")
    frame = pd.DataFrame(records)
    manifest = read_json(root / "manifest.json")
    print("\n" + "=" * 84)
    print("A. aligned noise study")
    print("=" * 84)
    add_check(name, "records=450", len(frame) == 450, len(frame))
    add_check(name, "10 models", set(frame.model) == set(MODELS), sorted(frame.model.unique()))
    add_check(name, "5 seeds", sorted(frame.seed.unique().tolist()) == [42,43,44,45,46],
              sorted(frame.seed.unique().tolist()))
    snrs = sorted({x for x in frame.snr_db.unique() if x != "clean"})
    add_check(name, "SNR grid", snrs == [5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0], snrs)
    per_model_seed = frame.groupby(["model", "seed"]).size()
    add_check(name, "each model/seed has 9 conditions", bool((per_model_seed == 9).all()),
              f"min={per_model_seed.min()} max={per_model_seed.max()}")
    add_check(name, "comparison_strength=designB_reduced_plus",
              manifest.get("comparison_strength") == "designB_reduced_plus",
              manifest.get("comparison_strength"))
    add_check(name, "proposed_capacity=standard", manifest.get("proposed_capacity") == "standard",
              manifest.get("proposed_capacity"))
    add_check(name, "ab_capacity=micro", manifest.get("ab_capacity") == "micro",
              manifest.get("ab_capacity"))
    add_check(name, "boundary_weight=0", float(manifest.get("proposed_boundary_weight", -1)) == 0.0,
              manifest.get("proposed_boundary_weight"))
    add_check(name, "no missing scalar metrics", not frame[["accuracy", "balanced_accuracy",
              "f1_macro", "f1_weighted", "minority_recall"]].isna().any().any(), "checked")

    rows = []
    for model in MODELS:
        g = frame[frame.model == model]
        clean = g[g.snr_db == "clean"]
        grid = g[g.snr_db != "clean"]
        by_snr = grid.groupby("snr_db").accuracy.mean().sort_index(ascending=False)
        clean_acc = float(clean.accuracy.mean())
        rows.append({
            "模型": CN[model], "model_key": model,
            "无噪准确率(%)": 100 * clean_acc,
            "无噪标准差(pp)": 100 * clean.accuracy.std(ddof=1),
            "全网格平均准确率(%)": 100 * grid.accuracy.mean(),
            "5dB准确率(%)": 100 * by_snr.loc[5.0],
            "10dB准确率(%)": 100 * by_snr.loc[10.0],
            "15dB准确率(%)": 100 * by_snr.loc[15.0],
            "最大下降(pp)": clean_acc * 100 - 100 * by_snr.min(),
        })
    summary = pd.DataFrame(rows).sort_values("全网格平均准确率(%)", ascending=False).reset_index(drop=True)
    summary.insert(0, "鲁棒性排名", np.arange(1, len(summary) + 1))
    print("\nnoise ranking:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # strongest baseline at low SNR and paired tests against DI/AB
    print("\nlow-SNR proposed-vs-best-baseline paired tests:")
    paired_rows = []
    for snr in (15.0, 10.0, 5.0):
        sub = frame[frame.snr_db == snr]
        best = sub[~sub.model.isin(PROPOSED)].groupby("model").accuracy.mean().idxmax()
        b = sub[sub.model == best].set_index("seed").accuracy.sort_index()
        for model in ("di_emstgat", "ab_emstgat"):
            a = sub[sub.model == model].set_index("seed").accuracy.sort_index()
            d = (a - b).to_numpy(float)
            pt = 1.0 if np.allclose(d, 0) else float(ttest_rel(a, b).pvalue)
            try:
                pw = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
            except ValueError:
                pw = np.nan
            row = {"SNR": int(snr), "本文模型": CN[model], "最强基线": CN[best],
                   "本文均值(%)": 100 * a.mean(), "基线均值(%)": 100 * b.mean(),
                   "差值(pp)": 100 * d.mean(), "配对t_p": pt, "Wilcoxon_p": pw,
                   "胜出种子": f"{int((d > 0).sum())}/{len(d)}"}
            paired_rows.append(row)
            print(row)
    return summary, {"manifest": manifest, "paired_low_snr": paired_rows}


def audit_ablation(name: str, dirname: str, expected_condition: str, expected_fraction: float,
                   expected_snr: Any) -> pd.DataFrame:
    root = STUDIES / dirname
    records = read_json(root / "per_seed_records.json")
    frame = pd.DataFrame(records)
    print("\n" + "=" * 84)
    print(f"B. aligned ablation: {name}")
    print("=" * 84)
    expected_records = 19 * 5
    add_check(name, f"records={expected_records}", len(frame) == expected_records, len(frame))
    add_check(name, "condition", set(frame.condition) == {expected_condition}, sorted(frame.condition.unique()))
    add_check(name, "train_fraction", set(frame.train_fraction) == {expected_fraction},
              sorted(frame.train_fraction.unique()))
    add_check(name, "eval_snr_db", set(frame.eval_snr_db) == {expected_snr},
              sorted(frame.eval_snr_db.unique(), key=str))
    add_check(name, "seeds=42..46", sorted(frame.seed.unique().tolist()) == [42,43,44,45,46],
              sorted(frame.seed.unique().tolist()))
    model_variant = frame.groupby("model").variant.nunique().to_dict()
    add_check(name, "DI=9 variants, AB=10 variants",
              model_variant == {"di_emstgat": 9, "ab_emstgat": 10}, model_variant)
    per_seed = frame.groupby("seed").size()
    add_check(name, "19 records per seed", bool((per_seed == 19).all()), per_seed.to_dict())
    capacities = frame.groupby("model").capacity.first().to_dict()
    expected_capacities = {"di_emstgat": "standard", "ab_emstgat": "micro"}
    add_check(name, "capacities", capacities == expected_capacities,
              capacities)
    epochs = frame.groupby("model").epochs.first().to_dict()
    expected_epochs = {"di_emstgat": 100, "ab_emstgat": 1}
    add_check(name, "epochs", epochs == expected_epochs, epochs)
    add_check(name, "boundary_weight=0", set(frame.boundary_weight) == {0.0},
              sorted(frame.boundary_weight.unique()))
    add_check(name, "test_samples=1700", set(frame.test_samples) == {1700},
              sorted(frame.test_samples.unique()))
    expected_train = 6799 if expected_fraction == 1.0 else 1359
    add_check(name, f"train_samples={expected_train}", set(frame.train_samples) == {expected_train},
              sorted(frame.train_samples.unique()))
    add_check(name, "no missing metrics", not frame[["accuracy","f1_macro","balanced_accuracy"]].isna().any().any(), "checked")

    rows = []
    for model in ("di_emstgat", "ab_emstgat"):
        g = frame[frame.model == model]
        full = g[g.variant == "full"].set_index("seed").sort_index()
        for variant in ABL_VARIANTS[model]:
            v = g[g.variant == variant].set_index("seed").sort_index()
            if v.empty:
                continue
            row = {"条件": name, "condition_key": expected_condition,
                   "模型": CN[model], "model_key": model,
                   "变体": VARIANT_CN[variant], "variant_key": variant,
                   "参数量": int(v.param_count.iloc[0]),
                   "变体准确率均值(%)": 100 * v.accuracy.mean(),
                   "变体准确率标准差(pp)": 100 * v.accuracy.std(ddof=1),
                   "完整模型准确率均值(%)": 100 * full.accuracy.mean()}
            if variant == "full":
                row.update({"变化(pp)": 0.0, "配对t_p": np.nan, "Wilcoxon_p": np.nan,
                            "变体更差种子": "—", "方向": "基准"})
            else:
                common = v.index.intersection(full.index)
                d = (v.loc[common, "accuracy"] - full.loc[common, "accuracy"]).to_numpy(float)
                pt = 1.0 if np.allclose(d, 0) else float(ttest_rel(v.loc[common,"accuracy"], full.loc[common,"accuracy"]).pvalue)
                try:
                    pw = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
                except ValueError:
                    pw = np.nan
                row.update({"变化(pp)": 100 * d.mean(), "配对t_p": pt,
                            "Wilcoxon_p": pw, "变体更差种子": f"{int((d < 0).sum())}/{len(d)}",
                            "方向": "移除后更差" if d.mean() < 0 else "移除后更好" if d.mean() > 0 else "相同"})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    noise_summary, noise_extra = audit_noise()
    clean = audit_ablation("干净条件", "ablation_designB_plus_v1_clean", "clean", 1.0, "clean")
    snr10 = audit_ablation("SNR=10dB", "ablation_designB_plus_v1_snr10", "snr10dB", 1.0, 10.0)
    train20 = audit_ablation("训练集20%", "ablation_designB_plus_v1_train20pct", "clean_train20pct", 0.2, "clean")

    # cross-condition direction table
    print("\n" + "=" * 84)
    print("C. cross-condition direction consistency")
    print("=" * 84)
    all_frames = {"干净": clean, "SNR10": snr10, "训练集20%": train20}
    rows = []
    for model in ("di_emstgat", "ab_emstgat"):
        variants = [v for v in ABL_VARIANTS[model] if v != "full"]
        for variant in variants:
            r = {"模型": CN[model], "model_key": model,
                 "变体": VARIANT_CN[variant], "variant_key": variant}
            signs = []
            for label, f in all_frames.items():
                z = f[(f.model_key == model) & (f.variant_key == variant)]
                if z.empty:
                    r[f"{label}_变化(pp)"] = np.nan
                    r[f"{label}_p"] = np.nan
                else:
                    d = float(z["变化(pp)"].iloc[0]); p = z["配对t_p"].iloc[0]
                    r[f"{label}_变化(pp)"] = d; r[f"{label}_p"] = p
                    signs.append(int(np.sign(d)))
            nz = [s for s in signs if s]
            r["方向一致性"] = ("一致：移除后更差" if nz and all(s < 0 for s in nz)
                             else "一致：移除后更好" if nz and all(s > 0 for s in nz)
                             else "全部相同/未生效" if not nz else "不一致")
            rows.append(r)
    cross = pd.DataFrame(rows)
    print(cross.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    noise_summary.to_csv(OUT / "表64_锁定配置噪声鲁棒性排名.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(noise_extra["paired_low_snr"]).to_csv(
        OUT / "表65_锁定配置噪声低SNR配对检验.csv", index=False, encoding="utf-8-sig")
    ab = pd.concat([clean, snr10, train20], ignore_index=True)
    ab.to_csv(OUT / "表66_锁定配置消融逐条件统计.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(OUT / "表67_锁定配置消融三条件方向一致性.csv", index=False, encoding="utf-8-sig")

    report = {
        "checks": checks,
        "check_total": len(checks),
        "check_passed": sum(x["结果"] == "通过" for x in checks),
        "all_passed": all(x["结果"] == "通过" for x in checks),
        "noise_summary": noise_summary.to_dict(orient="records"),
        "noise_low_snr": noise_extra["paired_low_snr"],
        "ablation_summary": ab.to_dict(orient="records"),
        "ablation_direction": cross.to_dict(orient="records"),
        "historical_roots_preserved": True,
    }
    (OUT / "锁定配置噪声消融最终审计.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nFINAL AUDIT: {report['check_passed']}/{report['check_total']} checks passed")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
