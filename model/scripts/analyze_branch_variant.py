"""Statistics for the DesignB_branch_v1 main comparison (read-only).

Reports, in one pass so no partial number can be narrated as a finding:
  * DI/baseline bit-identity against the DesignB_plus_v1 locked run
    (DI and all baselines are configured identically; only AB changed)
  * AB vs DI paired t + Wilcoxon + per-seed delta vector + win count
  * AB vs strongest baseline, same treatment
  * per-model mean +- sample sd ranking
No training, no writes.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDIES = PROJECT_ROOT / "results" / "studies"
NEW = "physics3_designB_branch_detuned_v1"
OLD = "physics3_designB_final_plus_v1"
PROPOSED = ("di_emstgat", "ab_emstgat")
# With AB as the primary model, DI is de-tuned onto the comparison side, so its
# per-seed values legitimately differ from the locked run. Only the eight true
# baselines must stay bit-identical.
IDENTITY_EXEMPT = {"ab_emstgat", "di_emstgat"}


def records(name: str) -> list[dict]:
    return json.loads((STUDIES / name / "per_seed_records.json").read_text(encoding="utf-8"))


def by_model(recs: list[dict], key: str = "accuracy") -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for row in recs:
        out[row["model"]][int(row["seed"])] = float(row[key])
    return out


def paired(a: dict[int, float], b: dict[int, float], label: str) -> dict:
    seeds = sorted(set(a) & set(b))
    va = np.array([a[s] for s in seeds])
    vb = np.array([b[s] for s in seeds])
    delta_pp = (va - vb) * 100.0
    t_res = stats.ttest_rel(va, vb)
    try:
        w_p = float(stats.wilcoxon(va, vb).pvalue)
    except ValueError:
        w_p = float("nan")
    sd = float(delta_pp.std(ddof=1)) if len(delta_pp) > 1 else 0.0
    return {
        "comparison": label,
        "seeds": seeds,
        "per_seed_delta_pp": [round(float(x), 4) for x in delta_pp],
        "mean_delta_pp": round(float(delta_pp.mean()), 4),
        "sd_delta_pp": round(sd, 4),
        "paired_t_p": float(t_res.pvalue),
        "wilcoxon_p": w_p,
        "wins": f"{int((delta_pp > 0).sum())}/{len(delta_pp)}",
        "effect_size": round(float(abs(delta_pp.mean()) / sd), 4) if sd > 0 else None,
    }


def main() -> int:
    new = records(NEW)
    old = records(OLD)
    N = by_model(new)
    O = by_model(old)

    print("=" * 78)
    print("A. 内部一致性：8 个基线配置未变，逐种子必须逐位一致")
    print("   （DI 本轮降配为对比模型，AB 提为主模型，两者本就应该变化）")
    print("=" * 78)
    identical_all = True
    for model in sorted(N):
        if model in IDENTITY_EXEMPT:
            continue
        same = all(
            abs(N[model][s] - O[model][s]) < 1e-12 for s in sorted(set(N[model]) & set(O[model]))
        )
        identical_all &= same
        print(f"  {model:14s} identical={same}")
    for model in sorted(IDENTITY_EXEMPT):
        old_mean = np.mean(list(O[model].values())) * 100
        new_mean = np.mean(list(N[model].values())) * 100
        print(f"  {model:14s} 预期变化: {old_mean:.4f}% -> {new_mean:.4f}% "
              f"({new_mean - old_mean:+.4f} pp)")
    print(f"\n  结论：{'8 个基线全部逐位一致' if identical_all else '基线存在非预期差异，必须先排查'}")

    print("\n" + "=" * 78)
    print("B. 排名（5 种子 mean ± 样本 sd，单位 %）")
    print("=" * 78)
    ranking = sorted(
        ((m, np.mean(list(v.values())) * 100, np.std(list(v.values()), ddof=1) * 100)
         for m, v in N.items()),
        key=lambda x: -x[1],
    )
    for i, (m, mu, sd) in enumerate(ranking, 1):
        tag = "  <-- 本文" if m in PROPOSED else ""
        print(f"  {i:2d}. {m:14s} {mu:8.4f} ± {sd:6.4f}{tag}")

    strongest_baseline = next(m for m, _, _ in ranking if m not in PROPOSED)
    # DI is now on the comparison side, so also report the strongest model that
    # is not the primary proposed one — that is what AB must beat to lead.
    strongest_non_primary = next(m for m, _, _ in ranking if m != "ab_emstgat")
    print(f"\n  最强非本文基线: {strongest_baseline}   "
          f"AB 之外最强模型: {strongest_non_primary}")

    print("\n" + "=" * 78)
    print("C. 配对检验（同一种子配对，双侧）")
    print("=" * 78)
    tests = [
        paired(N["ab_emstgat"], N["di_emstgat"], "AB(compact/100) - DI(降配 light/1)"),
        paired(N["ab_emstgat"], N[strongest_baseline], f"AB - {strongest_baseline}(最强基线)"),
        paired(N["di_emstgat"], N[strongest_baseline], f"DI(降配) - {strongest_baseline}(最强基线)"),
        paired(N["ab_emstgat"], O["ab_emstgat"], "AB(compact/100) - AB(micro/1, 旧锁定)"),
        paired(N["di_emstgat"], O["di_emstgat"], "DI(降配 light/1) - DI(standard/100, 旧锁定)"),
    ]
    for t in tests:
        print(f"\n  {t['comparison']}")
        print(f"    逐种子差值(pp): {t['per_seed_delta_pp']}")
        print(f"    mean±sd(pp)   : {t['mean_delta_pp']:+.4f} ± {t['sd_delta_pp']:.4f}")
        print(f"    配对 t p      : {t['paired_t_p']:.6f}")
        print(f"    Wilcoxon p    : {t['wilcoxon_p']:.6f}")
        print(f"    胜场          : {t['wins']}   效应量: {t['effect_size']}")

    print("\n" + "=" * 78)
    print("D. 最小可检出效应（分支主表能分辨多大差异）")
    print("=" * 78)
    n_test = int(new[0]["test_samples"])
    for m in PROPOSED:
        vals = np.array(list(N[m].values())) * 100
        seed_sd = float(vals.std(ddof=1))
        n = len(vals)
        t_crit = float(stats.t.ppf(0.975, n - 1))
        mde = t_crit / np.sqrt(n) * seed_sd
        print(f"  {m:14s} seed_sd={seed_sd:.4f} pp  MDE≈{mde:.4f} pp "
              f"({mde * n_test / 100:.1f} 行 / {n_test} 行测试集)")

    print("\n" + "=" * 78)
    print("E. Macro-F1 交叉验证同一结论")
    print("=" * 78)
    NF = by_model(new, "f1_macro")
    f1 = paired(NF["ab_emstgat"], NF["di_emstgat"], "AB - DI (Macro-F1)")
    print(f"  逐种子差值(pp): {f1['per_seed_delta_pp']}")
    print(f"  mean±sd(pp)   : {f1['mean_delta_pp']:+.4f} ± {f1['sd_delta_pp']:.4f}  "
          f"t p={f1['paired_t_p']:.6f}  胜场={f1['wins']}")

    out = {
        "run": NEW,
        "baseline_run_for_identity_check": OLD,
        "baselines_bit_identical": bool(identical_all),
        "identity_exempt_by_design": sorted(IDENTITY_EXEMPT),
        "records": len(new),
        "test_samples": n_test,
        "ranking_accuracy_pct": [
            {"rank": i, "model": m, "mean": round(mu, 6), "sd": round(sd, 6)}
            for i, (m, mu, sd) in enumerate(ranking, 1)
        ],
        "strongest_baseline": strongest_baseline,
        "strongest_non_primary_model": strongest_non_primary,
        "paired_tests_accuracy": tests,
        "paired_test_macro_f1_ab_vs_di": f1,
    }
    dest = STUDIES / NEW / "branch_variant_statistics.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
