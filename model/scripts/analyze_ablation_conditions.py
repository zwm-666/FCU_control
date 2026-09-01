"""Cross-condition ablation analysis: which evaluation condition can resolve components?

THE PROBLEM THIS ANSWERS
  On clean data the models sit at ~98.9% and miss only ~19 of 1700 test rows,
  while seed-to-seed noise alone is +-6.6 rows. A paired test over 5 seeds
  therefore cannot resolve any single component. Two degradation conditions were
  run to break that ceiling:

    clean        full training data, clean evaluation
    snr10dB      clean training data, evaluation at SNR = 10 dB (FIXED draw)
    train20pct   20% stratified training data, clean evaluation

  A first SNR attempt keyed the noise draw on the training seed, which folded
  pure evaluation noise into the across-seed spread (measured: 0.51 pp, 32% of
  the observed spread) and destroyed discriminative power. That run is kept as
  ..._PERSEEDNOISE_flawed for comparison and is reported, not hidden.

WHAT THIS SCRIPT REPORTS
  1. Discriminability per condition: seed sd relative to the error budget, and
     the smallest effect a 5-seed paired test could detect there.
  2. Per-variant results in every condition side by side.
  3. Direction consistency across conditions -- the honest criterion. A
     component whose effect keeps the same sign in all three conditions is far
     more credible than one significant in a single condition, because three
     conditions is effectively three independent replications of the question.
  4. A verdict per component that never overstates: significance in one
     condition alone is called "conditional", not "established".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

STUDIES = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/studies")
# 表42/表43 came from the reduced-baseline ablation roots and were superseded
# by the locked-config 表71/表72; output stays outside the paper directory.
OUT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model/results/derived_superseded/reduced_ablation_42_43")
N_TEST = 1700

CONDITIONS = [
    ("clean", "干净条件", "ablation_physics3_designB"),
    ("snr10dB", "SNR=10dB", "ablation_snr10_designB"),
    ("train20pct", "训练集20%", "ablation_train20pct_designB"),
]
FLAWED = ("snr10dB_perseed_noise", "SNR10(有缺陷:逐种子噪声)",
          "ablation_snr10_designB_PERSEEDNOISE_flawed")

CN = {"di_emstgat": "DI-EMSTGAT", "ab_emstgat": "AB-EMSTGAT"}
VARIANT_CN = {
    "full": "完整模型", "no_dcc": "去除膨胀因果卷积", "no_bigru": "去除双向GRU",
    "no_knn_graph": "去除时序KNN图", "no_self_attn": "去除多头自注意力",
    "no_attn_pool": "注意力池化替为均值池化", "no_skip_cls": "去除表格跳连分类器",
    "no_pos_emb": "去除位置编码", "no_boundary": "去除边界损失",
    "no_router_gate": "去除特征相关性门控",
}
# t critical for a two-sided paired test, n=5 (df=4)
T_CRIT_N5 = 2.776


def load(dirname: str) -> Optional[pd.DataFrame]:
    path = STUDIES / dirname / "per_seed_records.json"
    if not path.exists():
        return None
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return frame if len(frame) else None


def paired(frame: pd.DataFrame, model: str, variant: str) -> Optional[dict]:
    full = frame[(frame.model == model) & (frame.variant == "full")]
    g = frame[(frame.model == model) & (frame.variant == variant)]
    if full.empty or g.empty:
        return None
    a = g.set_index("seed").accuracy.sort_index()
    b = full.set_index("seed").accuracy.sort_index()
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return None
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    if np.allclose(d, 0.0):
        return {"delta_pp": 0.0, "p": 1.0, "p_wilcoxon": 1.0, "effect": 0.0,
                "worse": 0, "n": len(d), "identical": True}
    sd = d.std(ddof=1)
    try:
        pw = float(wilcoxon(d).pvalue)
    except ValueError:
        pw = float("nan")
    return {"delta_pp": 100 * float(d.mean()),
            "p": float(ttest_rel(a.loc[common], b.loc[common]).pvalue),
            "p_wilcoxon": pw,
            "effect": float(abs(d.mean()) / sd) if sd > 0 else 0.0,
            "worse": int((d < 0).sum()), "n": len(d), "identical": False}


def main() -> None:
    loaded: List[tuple] = []
    for key, label, dirname in CONDITIONS:
        frame = load(dirname)
        if frame is None:
            print(f".. condition {key} not available ({dirname})")
            continue
        seeds = sorted(frame.seed.unique())
        # Expected rows per seed = sum of variants per model, NOT variant.nunique():
        # DI has 9 variants (no router to ablate) and AB has 10, so a complete
        # seed contributes 19 rows, not 10.
        expected_per_seed = int(
            frame.groupby("model").variant.nunique().sum())
        per_seed = frame.groupby("seed").size()
        complete = bool((per_seed == expected_per_seed).all() and len(seeds) >= 5)
        print(f"loaded {key:12s} {len(frame):3d} records, seeds={seeds}, "
              f"{expected_per_seed}/seed expected"
              f"{'' if complete else '  (INCOMPLETE)'}")
        loaded.append((key, label, frame, complete))

    if not loaded:
        print("no conditions available")
        return

    # ---------- 1. discriminability of each condition ----------
    rows = []
    for key, label, frame, complete in loaded:
        for model in ("di_emstgat", "ab_emstgat"):
            f = frame[(frame.model == model) & (frame.variant == "full")]
            if f.empty:
                continue
            acc, sd = f.accuracy.mean(), f.accuracy.std(ddof=1)
            err_rows = (1 - acc) * N_TEST
            sd_rows = sd * N_TEST
            # minimum detectable effect for a paired t-test at n=5
            mde_pp = 100 * T_CRIT_N5 * sd / np.sqrt(len(f)) if len(f) > 1 else np.nan
            rows.append({
                "条件": label, "condition": key, "模型": CN[model],
                "完整模型准确率(%)": round(100 * acc, 4),
                "平均错误行数": round(err_rows, 1),
                "种子标准差(pp)": round(100 * sd, 4),
                "标准差折合行数": round(sd_rows, 1),
                "标准差/错误数": round(sd_rows / err_rows, 3) if err_rows else np.nan,
                "最小可检出效应(pp)": round(mde_pp, 4),
                "样本数": int(len(f)),
                "数据完整": bool(complete),
            })
    disc = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    disc.to_csv(OUT / "表42_消融条件的区分能力对照.csv", index=False,
                encoding="utf-8-sig")
    print("\n=== 1. 各条件的区分能力（标准差/错误数越小越好）===")
    print(disc[["条件", "模型", "完整模型准确率(%)", "平均错误行数",
                "种子标准差(pp)", "标准差/错误数", "最小可检出效应(pp)"]]
          .to_string(index=False))

    # the flawed run, for the methodological record
    flawed = load(FLAWED[2])
    if flawed is not None:
        frows = []
        for model in ("di_emstgat", "ab_emstgat"):
            f = flawed[(flawed.model == model) & (flawed.variant == "full")]
            if f.empty:
                continue
            acc, sd = f.accuracy.mean(), f.accuracy.std(ddof=1)
            frows.append({"条件": FLAWED[1], "模型": CN[model],
                          "完整模型准确率(%)": round(100 * acc, 4),
                          "种子标准差(pp)": round(100 * sd, 4),
                          "标准差/错误数": round(sd * N_TEST / ((1 - acc) * N_TEST), 3)})
        if frows:
            print("\n-- 对照：逐种子噪声版（有缺陷，保留供方法学说明）--")
            print(pd.DataFrame(frows).to_string(index=False))

    # ---------- 2 & 3. per-variant across conditions ----------
    variants = [v for v in VARIANT_CN if v != "full"]
    rows = []
    for model in ("di_emstgat", "ab_emstgat"):
        for variant in variants:
            entry = {"模型": CN[model], "model_key": model,
                     "变体": VARIANT_CN[variant], "variant_key": variant}
            signs, sig_conditions, present = [], [], 0
            for key, label, frame, complete in loaded:
                res = paired(frame, model, variant)
                if res is None:
                    entry[f"{label}_变化(pp)"] = np.nan
                    entry[f"{label}_p"] = np.nan
                    continue
                present += 1
                entry[f"{label}_变化(pp)"] = round(res["delta_pp"], 4)
                entry[f"{label}_p"] = round(res["p"], 4)
                entry[f"{label}_效应量"] = round(res["effect"], 3)
                if res["identical"]:
                    signs.append(0)
                else:
                    signs.append(int(np.sign(res["delta_pp"])))
                    if res["p"] < 0.05:
                        sig_conditions.append(label)
            entry["出现条件数"] = present
            nonzero = [s for s in signs if s != 0]
            if present == 0:
                # e.g. no_router_gate on DI-EMSTGAT: DI runs in 'raw' branch mode
                # and has no router at all, so the variant was never built. This
                # is NOT APPLICABLE, not "component had no effect".
                entry["方向一致性"] = "不适用（该模型无此组件）"
                entry["一致方向"] = "—"
            elif not nonzero:
                entry["方向一致性"] = "全部为零（组件未生效）"
                entry["一致方向"] = "—"
            elif all(s > 0 for s in nonzero):
                entry["方向一致性"] = f"一致（{len(nonzero)}/{len(nonzero)} 移除后更好）"
                entry["一致方向"] = "移除更好"
            elif all(s < 0 for s in nonzero):
                entry["方向一致性"] = f"一致（{len(nonzero)}/{len(nonzero)} 移除后更差）"
                entry["一致方向"] = "组件有效"
            else:
                entry["方向一致性"] = "不一致（条件间符号相反）"
                entry["一致方向"] = "—"
            entry["显著条件"] = "、".join(sig_conditions) if sig_conditions else "无"
            entry["显著条件数"] = len(sig_conditions)

            # verdict that does not overstate
            consistent = entry["一致方向"]
            n_sig = len(sig_conditions)
            if present == 0:
                verdict = "不适用（该模型不含此组件）"
            elif entry["方向一致性"].startswith("全部为零"):
                verdict = "组件未生效（各条件下预测完全相同）"
            elif consistent == "组件有效" and n_sig >= 2:
                verdict = "组件有效：方向一致且多条件显著"
            elif consistent == "组件有效" and n_sig == 1:
                verdict = "组件可能有效：方向一致，仅单条件显著"
            elif consistent == "组件有效":
                verdict = "倾向有效：方向一致但均未达显著"
            elif consistent == "移除更好" and n_sig >= 2:
                verdict = "该组件未挣回代价：方向一致且多条件显著"
            elif consistent == "移除更好" and n_sig == 1:
                verdict = "该组件可能冗余：方向一致，仅单条件显著"
            elif consistent == "移除更好":
                verdict = "倾向冗余：方向一致但均未达显著"
            else:
                verdict = "无法判定：条件间方向相反"
            entry["综合判定"] = verdict
            rows.append(entry)

    cross = pd.DataFrame(rows)
    cross.to_csv(OUT / "表43_消融三条件对照与方向一致性.csv", index=False,
                 encoding="utf-8-sig")

    print("\n=== 2. 三条件对照（准确率变化，个百分点）===")
    cols = ["模型", "变体"]
    for _, label, _, _ in loaded:
        cols += [f"{label}_变化(pp)", f"{label}_p"]
    print(cross[[c for c in cols if c in cross.columns]].to_string(index=False))

    print("\n=== 3. 方向一致性与综合判定 ===")
    print(cross[["模型", "变体", "方向一致性", "显著条件数", "综合判定"]]
          .to_string(index=False))

    # ---------- 4. what can actually be claimed ----------
    print("\n=== 4. 可写入论文的结论 ===")
    established = cross[cross.综合判定.str.startswith("组件有效")]
    redundant = cross[cross.综合判定.str.contains("未挣回代价|可能冗余")]
    inactive = cross[cross.综合判定.str.startswith("组件未生效")]
    not_applicable = cross[cross.综合判定.str.startswith("不适用")]
    print(f"  证据较强的有效组件 : {len(established)}")
    for _, r in established.iterrows():
        print(f"    {r['模型']} {r['变体']}  ({r['综合判定']})")
    print(f"  证据较强的冗余组件 : {len(redundant)}")
    for _, r in redundant.iterrows():
        print(f"    {r['模型']} {r['变体']}  ({r['综合判定']})")
    print(f"  未生效（实现问题） : {len(inactive)}")
    for _, r in inactive.iterrows():
        print(f"    {r['模型']} {r['变体']}")
    if len(not_applicable):
        print(f"  不适用（模型无此组件）: {len(not_applicable)}")
        for _, r in not_applicable.iterrows():
            print(f"    {r['模型']} {r['变体']}")

    # parameter savings from consistently redundant components
    base = loaded[0][2]
    print("\n=== 5. 精简空间（基于干净条件参数量）===")
    for model in ("di_emstgat", "ab_emstgat"):
        full_p = base[(base.model == model) & (base.variant == "full")]
        if full_p.empty:
            continue
        fp = int(full_p.param_count.iloc[0])
        cand = cross[(cross.model_key == model)
                     & cross.综合判定.str.contains("未挣回代价|可能冗余|倾向冗余")]
        if cand.empty:
            print(f"  {CN[model]}: 无一致冗余组件")
            continue
        print(f"  {CN[model]} 完整参数量 {fp:,}")
        for _, r in cand.iterrows():
            g = base[(base.model == model) & (base.variant == r["variant_key"])]
            if g.empty:
                continue
            vp = int(g.param_count.iloc[0])
            if vp < fp:
                print(f"    去除{r['变体'].replace('去除','')}: {vp:,} "
                      f"({100*(vp-fp)/fp:+.1f}%)  {r['综合判定']}")

    manifest = {
        "conditions": [{"key": k, "label": l, "records": int(len(f)),
                        "complete": bool(c)} for k, l, f, c in loaded],
        "flawed_run_kept": FLAWED[2],
        "flawed_run_reason": (
            "the evaluation noise draw was keyed on the training seed, which "
            "injected 0.51 pp of pure evaluation variance (32% of the observed "
            "across-seed spread) into the paired comparison and masked component "
            "effects; kept for the methodological record, not deleted"),
        "n_test_rows": N_TEST,
        "t_crit_n5_two_sided": T_CRIT_N5,
        "interpretation_rule": (
            "direction consistency across conditions is the primary criterion; "
            "significance in a single condition is reported as conditional, "
            "never as established"),
    }
    (OUT / "消融三条件分析契约.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
