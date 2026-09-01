"""Statistics for the DesignB_branch_v1 noise-robustness rerun (read-only).

AB-EMSTGAT (adaptive branch) is the primary model here; DI is de-tuned onto the
comparison side. Reports the SNR grid, the robustness ranking, and paired tests
of AB against the strongest non-proposed baseline at low SNR, all in one pass so
no partial number gets narrated as a finding.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDIES = PROJECT_ROOT / "results" / "studies"
RUN = "noise_designB_branch_v1"
OLD_RUN = "noise_designB_plus_v1"
PROPOSED = ("ab_emstgat", "di_emstgat")
PRIMARY = "ab_emstgat"
LOW_SNR = (15.0, 10.0, 5.0)
CN = {
    "ab_emstgat": "AB-EMSTGAT（本文）", "di_emstgat": "DI-EMSTGAT（降配对照）",
    "tcn": "TCN", "mtgnn": "MTGNN", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "itransformer": "iTransformer", "gatv2": "GATv2",
    "transformer": "Transformer", "patchtst": "PatchTST",
}


def load(run: str) -> pd.DataFrame:
    recs = json.loads((STUDIES / run / "per_seed_records.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(recs)
    frame["snr_key"] = frame["snr_db"].astype(str)
    return frame


def cell(frame: pd.DataFrame, model: str, snr: str, metric: str = "accuracy") -> dict[int, float]:
    sub = frame[(frame.model == model) & (frame.snr_key == snr)]
    return {int(r.seed): float(getattr(r, metric)) for r in sub.itertuples()}


def paired(a: dict[int, float], b: dict[int, float]) -> dict:
    seeds = sorted(set(a) & set(b))
    va = np.array([a[s] for s in seeds])
    vb = np.array([b[s] for s in seeds])
    delta = (va - vb) * 100
    try:
        w = float(stats.wilcoxon(va, vb).pvalue)
    except ValueError:
        w = float("nan")
    return {
        "per_seed_delta_pp": [round(float(x), 4) for x in delta],
        "mean_delta_pp": round(float(delta.mean()), 4),
        "sd_delta_pp": round(float(delta.std(ddof=1)), 4) if len(delta) > 1 else 0.0,
        "paired_t_p": float(stats.ttest_rel(va, vb).pvalue),
        "wilcoxon_p": w,
        "wins": f"{int((delta > 0).sum())}/{len(delta)}",
    }


def main() -> int:
    frame = load(RUN)
    grid = ["clean"] + [f"{v}" for v in (40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0)]
    models = [m for m, _ in sorted(
        ((m, frame[frame.model == m].accuracy.mean()) for m in frame.model.unique()),
        key=lambda x: -x[1])]

    print("=" * 96)
    print(f"A. 完整性：{len(frame)} 条记录 "
          f"({frame.model.nunique()} 模型 × {frame.seed.nunique()} 种子 × {frame.snr_key.nunique()} 档)")
    print("=" * 96)
    counts = frame.groupby(["model", "seed"]).size()
    print(f"  每 model/seed 条数: min={counts.min()} max={counts.max()}   "
          f"种子={sorted(frame.seed.unique())}")

    print("\n" + "=" * 96)
    print("B. 准确率随 SNR（5 种子均值 %）")
    print("=" * 96)
    header = f"{'模型':22s}" + "".join(f"{g:>9s}" for g in grid)
    print(header)
    rows = []
    for m in models:
        vals = [np.mean(list(cell(frame, m, g).values())) * 100 for g in grid]
        full_mean = float(np.mean(vals))
        drop = float(vals[0] - min(vals))
        rows.append({"model": m, "model_cn": CN.get(m, m), "clean": vals[0],
                     "grid_mean": full_mean, "max_drop_pp": drop,
                     **{f"snr_{g}": v for g, v in zip(grid, vals)}})
        tag = " *" if m in PROPOSED else "  "
        print(f"{CN.get(m, m):22s}" + "".join(f"{v:9.4f}" for v in vals) + tag)

    ranking = sorted(rows, key=lambda r: -r["grid_mean"])
    print("\n" + "=" * 96)
    print("C. 鲁棒性排名（全网格平均准确率）")
    print("=" * 96)
    for i, r in enumerate(ranking, 1):
        print(f"  {i:2d}. {r['model_cn']:22s} 全网格={r['grid_mean']:8.4f}%  "
              f"无噪={r['clean']:8.4f}%  最大下降={r['max_drop_pp']:6.4f} pp")

    strongest = next(r["model"] for r in ranking if r["model"] not in PROPOSED)
    print(f"\n  最强非本文基线：{CN.get(strongest, strongest)}")

    print("\n" + "=" * 96)
    print(f"D. 低 SNR 配对检验：{CN[PRIMARY]} vs 最强基线 / vs DI(降配)")
    print("=" * 96)
    tests = []
    for snr in LOW_SNR:
        key = f"{snr}"
        for opponent in (strongest, "di_emstgat"):
            t = paired(cell(frame, PRIMARY, key), cell(frame, opponent, key))
            t.update({"snr_db": snr, "model": PRIMARY, "opponent": opponent,
                      "primary_mean_pct": round(float(np.mean(list(cell(frame, PRIMARY, key).values())) * 100), 4),
                      "opponent_mean_pct": round(float(np.mean(list(cell(frame, opponent, key).values())) * 100), 4)})
            tests.append(t)
            print(f"\n  {snr:>5.1f} dB  AB {t['primary_mean_pct']:.4f}% vs "
                  f"{CN.get(opponent, opponent)} {t['opponent_mean_pct']:.4f}%")
            print(f"           逐种子差值(pp): {t['per_seed_delta_pp']}")
            print(f"           mean±sd={t['mean_delta_pp']:+.4f}±{t['sd_delta_pp']:.4f}  "
                  f"t p={t['paired_t_p']:.6f}  Wilcoxon p={t['wilcoxon_p']:.6f}  "
                  f"胜场={t['wins']}")

    print("\n" + "=" * 96)
    print("E. 与旧锁定噪声运行（AB micro/1, DI standard/100）对照")
    print("=" * 96)
    try:
        old = load(OLD_RUN)
        for m in PROPOSED:
            for key in ("clean", "15.0", "10.0", "5.0"):
                n = np.mean(list(cell(frame, m, key).values())) * 100
                o = np.mean(list(cell(old, m, key).values())) * 100
                print(f"  {CN[m]:22s} {key:>6s}  旧 {o:8.4f}% -> 新 {n:8.4f}%  ({n - o:+.4f} pp)")
        baselines_same = all(
            abs(np.mean(list(cell(frame, m, g).values()))
                - np.mean(list(cell(old, m, g).values()))) < 1e-12
            for m in models if m not in PROPOSED for g in grid)
        print(f"\n  8 个基线逐档均值与旧运行一致: {baselines_same}")
    except FileNotFoundError:
        baselines_same = None
        print("  旧运行不存在，跳过对照")

    out = {
        "run": RUN,
        "records": int(len(frame)),
        "primary_model": PRIMARY,
        "config": "DI light/1ep (de-tuned comparison), AB compact/100ep (primary)",
        "snr_grid": grid,
        "per_model": rows,
        "robustness_ranking": [
            {"rank": i, "model": r["model"], "grid_mean": round(r["grid_mean"], 6),
             "clean": round(r["clean"], 6), "max_drop_pp": round(r["max_drop_pp"], 6)}
            for i, r in enumerate(ranking, 1)],
        "strongest_baseline": strongest,
        "low_snr_paired_tests": tests,
        "baselines_match_previous_noise_run": baselines_same,
    }
    dest = STUDIES / RUN / "noise_branch_statistics.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
