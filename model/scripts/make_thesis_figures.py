"""Generate every thesis figure as a Word-safe SVG from stored results.

Reads only saved JSON/CSV result files -- trains nothing, so figures always
match the reported numbers and can be regenerated cheaply.

Figures
  fig1  main comparison: 10 models, accuracy + macro F1, mean with std bars
  fig2  noise robustness: accuracy vs SNR, clean -> 5 dB
  fig3  noise degradation: accuracy drop relative to clean, per model
  fig4  ablation: per-component accuracy delta for DI and AB
  fig5  ablation parameter cost vs accuracy delta
  fig6  confusion matrix of the best model (counts + row-normalised)
  fig7  per-class F1 by model
  fig8  honest-baseline gap: row-random vs episode-grouped split
  fig9  dataset provenance: synthetic/duplicate composition
  fig10 physics criterion: water activity per class with the two thresholds
  fig11 noise robustness heatmap (model x SNR)
  fig12 seed-level spread of the proposed models vs baselines

Every file passes validate_word_svg(): no <style>, no <use>, no xlink, no
class=, no gradients/filters, explicit px width/height plus viewBox, and all
text converted to vector outlines so no font is needed to open it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.thesis_svg import (
    FULL_W, HALF_W, PALETTE, apply_thesis_style, hatch_for, new_fig,
    save_word_svg, style_series, vector_colorbar, vector_heatmap,
)

MODEL_ROOT = Path(r"D:/my_project/h2-fcu-modern-dashboard/model")
STUDIES = MODEL_ROOT / "results/studies"
DATA_DIR = MODEL_ROOT / "数据文件"
OUT_RESULTS = Path(r"D:/learn/毕业材料/graduation/results")
OUT_FIG = OUT_RESULTS / "figures"

# Locked configuration DesignB_branch_v1: the paper's primary model is
# AB-EMSTGAT (compact/100ep); DI-EMSTGAT (light/1ep) is the direct-input
# comparison model. Baselines keep the restricted budget required by spec 8.2.
# Figures must be built from these runs, not from the equal-compute exploration
# and not from the withdrawn DesignB_plus_v1 tables.
MAIN = STUDIES / "physics3_designB_branch_detuned_v1"
CTRL = STUDIES / "legacy_synthetic_3class_control"
NOISE = STUDIES / "noise_designB_branch_v1"
ABL = STUDIES / "ablation_designB_branch_v1_clean"
ABL_CONDITIONS = [
    ("干净条件", STUDIES / "ablation_designB_branch_v1_clean"),
    ("SNR=10dB", STUDIES / "ablation_designB_branch_v1_snr10"),
    ("训练集20%", STUDIES / "ablation_designB_branch_v1_train20pct"),
]

CN = {
    "di_emstgat": "DI-EMSTGAT", "ab_emstgat": "AB-EMSTGAT",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "patchtst": "PatchTST",
    "mtgnn": "MTGNN", "tcn": "TCN", "transformer": "Transformer",
    "itransformer": "iTransformer", "gatv2": "GATv2",
}
ORDER = ["di_emstgat", "ab_emstgat", "patchtst", "mtgnn", "tcn", "xgboost",
         "lightgbm", "itransformer", "transformer", "gatv2"]
PROPOSED = {"di_emstgat", "ab_emstgat"}
CLASS_CN = {"Flooding": "水淹", "Membrane_Drying": "膜干", "Normal": "正常"}

reports: List[Dict[str, object]] = []


def emit(fig, name: str) -> None:
    info = save_word_svg(fig, OUT_FIG / f"{name}.svg")
    reports.append({"figure": name, **info})
    print(f"  {name}.svg  {info['bytes']:>7d} B  paths={info['paths']:>4d} "
          f"uses_flattened={info['uses_flattened']:>4d} text={info['text_elements']}")


def load_records(root: Path) -> Optional[pd.DataFrame]:
    path = root / "per_seed_records.json"
    if not path.exists():
        return None
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def bar_label(ax, bars, values, fmt="{:.2f}", dy=0.15, fontsize=6.5):
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() + dy),
                    ha="center", va="bottom", fontsize=fontsize)


# ---------------------------------------------------------------------------
def fig_main_comparison(main: pd.DataFrame) -> None:
    """fig1: accuracy and macro F1 for all 10 models, std as error bars."""
    rows = []
    for key in ORDER:
        g = main[main.model == key]
        if g.empty:
            continue
        rows.append({
            "key": key, "name": CN[key],
            "acc": 100 * g.accuracy.mean(), "acc_sd": 100 * g.accuracy.std(ddof=1),
            "f1": 100 * g.f1_macro.mean(), "f1_sd": 100 * g.f1_macro.std(ddof=1),
        })
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    width = 0.38

    fig, ax = new_fig(15.5, 7.4)
    b1 = ax.bar(x - width / 2, frame.acc, width, yerr=frame.acc_sd, capsize=2.5,
                label="准确率 Accuracy", color=PALETTE[0], edgecolor="black",
                linewidth=0.6, error_kw={"linewidth": 0.7})
    b2 = ax.bar(x + width / 2, frame.f1, width, yerr=frame.f1_sd, capsize=2.5,
                label="宏平均F1 Macro-F1", color=PALETTE[1], hatch="///",
                edgecolor="black", linewidth=0.6, error_kw={"linewidth": 0.7})
    for i, key in enumerate(frame.key):
        if key in PROPOSED:
            ax.axvspan(i - 0.5, i + 0.5, color="#000000", alpha=0.05, zorder=0)
    bar_label(ax, b1, frame.acc.to_numpy())
    bar_label(ax, b2, frame.f1.to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(frame.name, rotation=22, ha="right")
    ax.set_ylabel("指标数值 (%)")
    ax.set_ylim(78, 101.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("物理判据数据集上10种方法的测试集性能（5种子均值±标准差，灰底为本文方法）")
    emit(fig, "图1_主对比_10模型准确率与宏F1")


def fig_noise_curves(noise: pd.DataFrame) -> None:
    """fig2: accuracy vs SNR. fig3: degradation relative to clean."""
    order = [k for k in ORDER if k in set(noise.model)]
    grid = sorted({float(v) for v in noise.snr_db if v != "clean"}, reverse=True)
    labels = ["无噪"] + [f"{int(v)}" for v in grid]
    xs = np.arange(len(labels))

    fig, ax = new_fig(15.5, 7.6)
    for i, key in enumerate(order):
        g = noise[noise.model == key]
        means, sds = [], []
        for tag in ["clean"] + grid:
            sel = g[g.snr_db == tag] if tag == "clean" else g[g.snr_db == tag]
            means.append(100 * sel.accuracy.mean())
            sds.append(100 * sel.accuracy.std(ddof=1) if len(sel) > 1 else 0.0)
        style = style_series(i)
        lw = 2.0 if key in PROPOSED else 1.1
        ms = 5.0 if key in PROPOSED else 3.4
        ax.errorbar(xs, means, yerr=sds, label=CN[key], linewidth=lw,
                    markersize=ms, capsize=2, elinewidth=0.6, **style)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("测试输入信噪比 SNR (dB)")
    ax.set_ylabel("准确率 (%)")
    ax.legend(ncol=3, loc="lower left", fontsize=6.6)
    ax.set_title("噪声鲁棒性：仅对测试输入加噪，训练数据保持干净（5种子均值±标准差）")
    emit(fig, "图2_噪声鲁棒性_准确率随SNR变化")

    fig, ax = new_fig(15.5, 7.0)
    for i, key in enumerate(order):
        g = noise[noise.model == key]
        clean = 100 * g[g.snr_db == "clean"].accuracy.mean()
        drops = [clean - 100 * g[g.snr_db == v].accuracy.mean() for v in grid]
        style = style_series(i)
        lw = 2.0 if key in PROPOSED else 1.1
        ax.plot(np.arange(len(grid)), drops, label=CN[key], linewidth=lw,
                markersize=5.0 if key in PROPOSED else 3.4, **style)
    ax.set_xticks(np.arange(len(grid)))
    ax.set_xticklabels([f"{int(v)}" for v in grid])
    ax.set_xlabel("测试输入信噪比 SNR (dB)")
    ax.set_ylabel("相对无噪的准确率下降 (个百分点)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(ncol=3, loc="upper left", fontsize=6.6)
    ax.set_title("噪声退化幅度：数值越低越鲁棒")
    emit(fig, "图3_噪声退化幅度")


def fig_noise_heatmap(noise: pd.DataFrame) -> None:
    """fig11: model x SNR accuracy heatmap with printed values."""
    order = [k for k in ORDER if k in set(noise.model)]
    grid = sorted({float(v) for v in noise.snr_db if v != "clean"}, reverse=True)
    cols = ["clean"] + grid
    matrix = np.array([[100 * noise[(noise.model == k) & (noise.snr_db == c)].accuracy.mean()
                        for c in cols] for k in order])

    fig, ax = new_fig(15.5, 7.8)
    # vector cells, never a rasterised imshow (see thesis_svg.vector_heatmap)
    im = vector_heatmap(ax, matrix, cmap="YlGnBu",
                        vmin=float(np.nanmin(matrix)), vmax=100.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["无噪"] + [f"{int(v)}" for v in grid])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([CN[k] for k in order])
    ax.set_xlabel("测试输入信噪比 SNR (dB)")
    mid = (np.nanmax(matrix) + np.nanmin(matrix)) / 2
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if np.isfinite(matrix[r, c]):
                ax.text(c, r, f"{matrix[r, c]:.1f}", ha="center", va="center",
                        fontsize=6.0,
                        color="white" if matrix[r, c] < mid else "black")
    ax.grid(False)
    # vector colourbar: fig.colorbar() would embed a raster gradient strip
    vector_colorbar(fig, ax, "YlGnBu", float(np.nanmin(matrix)), 100.0,
                    label="准确率 (%)")
    ax.set_title("噪声鲁棒性热力图（5种子均值准确率，%）")
    emit(fig, "图11_噪声鲁棒性热力图")


def fig_ablation_conditions() -> None:
    """fig13: the same ablation under three conditions, with direction consistency.

    A single-condition ablation figure is misleading here: the clean-condition
    p=0.018 for DCC flips sign under the other two conditions. Showing all three
    lets the reader see consistency (or its absence) directly.
    """
    frames = []
    for label, root in ABL_CONDITIONS:
        f = load_records(root)
        if f is not None and len(f):
            frames.append((label, f))
    if len(frames) < 2:
        return

    variants = [v for v in frames[0][1].variant.unique() if v != "full"]
    label_of = {v: frames[0][1][frames[0][1].variant == v].variant_cn.iloc[0]
                for v in variants}

    for model, tag in (("di_emstgat", "DI-EMSTGAT"), ("ab_emstgat", "AB-EMSTGAT")):
        if model not in set(frames[0][1].model):
            continue
        fig, ax = new_fig(15.5, 7.8)
        x = np.arange(len(variants))
        width = 0.8 / len(frames)
        for i, (cond, f) in enumerate(frames):
            sub = f[f.model == model]
            full = sub[sub.variant == "full"].set_index("seed").accuracy
            deltas, errs = [], []
            for v in variants:
                g = sub[sub.variant == v].set_index("seed").accuracy
                if g.empty:
                    deltas.append(np.nan); errs.append(0.0); continue
                c = g.index.intersection(full.index)
                d = 100 * (g.loc[c] - full.loc[c])
                deltas.append(d.mean())
                errs.append(d.std(ddof=1) if len(d) > 1 else 0.0)
            ax.bar(x + (i - (len(frames) - 1) / 2) * width, deltas, width,
                   yerr=errs, capsize=2, label=cond, color=PALETTE[i],
                   hatch=hatch_for(i), edgecolor="black", linewidth=0.6,
                   error_kw={"linewidth": 0.7})
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([label_of[v] for v in variants], rotation=24, ha="right")
        ax.set_ylabel("相对完整模型的准确率变化 (个百分点)")
        ax.legend(loc="lower right", title="评估条件", fontsize=6.8,
                  title_fontsize=6.8)
        ax.set_title(f"{tag} 消融实验三条件对照（负值=该组件有效；"
                     f"条件间符号一致才可信）")
        emit(fig, f"图13_消融三条件对照_{tag}")


def fig_ablation(abl: pd.DataFrame, n_seeds: int = 5) -> None:
    """fig4: per-component accuracy delta. fig5: parameter cost vs delta."""
    # A partial study must be visibly labelled on the figure itself; a caption
    # in a log is not enough to stop it reaching the thesis.
    provisional = "" if n_seeds >= 5 else f"【未完成：仅{n_seeds}/5种子，结果暂定】"
    for model, tag in (("di_emstgat", "DI-EMSTGAT"), ("ab_emstgat", "AB-EMSTGAT")):
        if model not in set(abl.model):
            continue
    models = [m for m in ("di_emstgat", "ab_emstgat") if m in set(abl.model)]

    variants = [v for v in abl.variant.unique() if v != "full"]
    label_of = {v: abl[abl.variant == v].variant_cn.iloc[0] for v in variants}

    fig, ax = new_fig(15.5, 7.6)
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(variants))
    for i, model in enumerate(models):
        sub = abl[abl.model == model]
        full = sub[sub.variant == "full"].set_index("seed").accuracy
        deltas, errs = [], []
        for v in variants:
            g = sub[sub.variant == v].set_index("seed").accuracy
            if g.empty:
                deltas.append(np.nan)
                errs.append(0.0)
                continue
            common = g.index.intersection(full.index)
            d = 100 * (g.loc[common] - full.loc[common])
            deltas.append(d.mean())
            errs.append(d.std(ddof=1) if len(d) > 1 else 0.0)
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, deltas, width,
                      yerr=errs, capsize=2, label=CN[model],
                      color=PALETTE[i], hatch=hatch_for(i), edgecolor="black",
                      linewidth=0.6, error_kw={"linewidth": 0.7})
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([label_of[v] for v in variants], rotation=24, ha="right")
    ax.set_ylabel("相对完整模型的准确率变化 (个百分点)")
    ax.legend(loc="lower right")
    ax.set_title(f"消融实验：每次仅移除一个组件（{n_seeds}种子均值±标准差，"
                 f"负值表示该组件有效）{provisional}")
    emit(fig, "图4_消融实验_组件贡献")

    fig, ax = new_fig(15.5, 7.2)
    for i, model in enumerate(models):
        sub = abl[abl.model == model]
        full = sub[sub.variant == "full"]
        full_params = int(full.param_count.iloc[0])
        full_acc = full.set_index("seed").accuracy
        xs, ys, names = [], [], []
        for v in variants:
            g = sub[sub.variant == v]
            if g.empty:
                continue
            common = g.set_index("seed").index.intersection(full_acc.index)
            d = 100 * (g.set_index("seed").accuracy.loc[common] - full_acc.loc[common]).mean()
            xs.append(100.0 * (int(g.param_count.iloc[0]) - full_params) / full_params)
            ys.append(d)
            names.append(label_of[v])
        style = style_series(i)
        ax.scatter(xs, ys, s=34, label=CN[model], color=style["color"],
                   marker=style["marker"], edgecolor="black", linewidth=0.5,
                   zorder=3)
        for xv, yv, nm in zip(xs, ys, names):
            ax.annotate(nm, (xv, yv), fontsize=5.8,
                        textcoords="offset points", xytext=(3, 3))
    ax.axhline(0, color="black", linewidth=0.9)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("参数量相对变化 (%)")
    ax.set_ylabel("准确率变化 (个百分点)")
    ax.legend(loc="lower left")
    ax.set_title(f"组件的参数代价与精度收益（左下象限＝该组件省参但掉点）{provisional}")
    emit(fig, "图5_消融_参数代价与精度收益")


def fig_confusion(main_records: List[dict]) -> None:
    """fig6: counts + row-normalised confusion matrix of the best model."""
    frame = pd.DataFrame(main_records)
    best = frame.groupby("model").accuracy.mean().idxmax()
    sub = [r for r in main_records if r["model"] == best]
    pick = max(sub, key=lambda r: r["accuracy"])
    matrix = np.array(pick["confusion_matrix"], dtype=float)
    names = list(pick["classification_report"].keys())
    classes = [n for n in names if n in CLASS_CN]
    ticks = [f"{CLASS_CN[c]}\n{c}" for c in classes]

    fig, axes = new_fig(15.5, 6.6, ncols=2)
    for ax, (data, title, fmt, vmax) in zip(axes, [
        (matrix, f"混淆矩阵（计数）", "{:.0f}", matrix.max()),
        (100 * matrix / matrix.sum(axis=1, keepdims=True),
         "混淆矩阵（按真实类归一化，%）", "{:.1f}", 100.0),
    ]):
        im = vector_heatmap(ax, data, cmap="Blues", vmin=0.0, vmax=float(vmax))
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(ticks, fontsize=6.4)
        ax.set_yticklabels(ticks, fontsize=6.4)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(title, fontsize=8.4)
        ax.grid(False)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ax.text(c, r, fmt.format(data[r, c]), ha="center", va="center",
                        fontsize=7.0,
                        color="white" if data[r, c] > 0.55 * vmax else "black")
    fig.suptitle(f"最优模型 {CN[best]}（seed={pick['seed']}，"
                 f"准确率{100*pick['accuracy']:.2f}%）的测试集混淆矩阵", fontsize=8.8)
    fig.tight_layout()
    emit(fig, "图6_最优模型混淆矩阵")


def fig_per_class(main_records: List[dict]) -> None:
    """fig7: per-class F1 for every model."""
    rows = []
    for rec in main_records:
        rep = rec["classification_report"]
        for cls in CLASS_CN:
            if cls in rep:
                rows.append({"model": rec["model"], "cls": cls,
                             "f1": rep[cls]["f1-score"]})
    frame = pd.DataFrame(rows)
    order = [k for k in ORDER if k in set(frame.model)]
    classes = list(CLASS_CN)
    x = np.arange(len(order))
    width = 0.8 / len(classes)

    fig, ax = new_fig(15.5, 7.2)
    for i, cls in enumerate(classes):
        means = [100 * frame[(frame.model == k) & (frame.cls == cls)].f1.mean()
                 for k in order]
        sds = [100 * frame[(frame.model == k) & (frame.cls == cls)].f1.std(ddof=1)
               for k in order]
        ax.bar(x + (i - (len(classes) - 1) / 2) * width, means, width, yerr=sds,
               capsize=2, label=f"{CLASS_CN[cls]} {cls}", color=PALETTE[i],
               hatch=hatch_for(i), edgecolor="black", linewidth=0.6,
               error_kw={"linewidth": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels([CN[k] for k in order], rotation=22, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(70, 101.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="lower left")
    ax.set_title("各方法的类别级F1（5种子均值±标准差）")
    emit(fig, "图7_类别级F1对比")


def fig_honest_baseline() -> None:
    """fig8: row-random vs episode-grouped accuracy, the leakage gap."""
    path = OUT_RESULTS / "物理判据_分组划分诚实基线.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    name_cn = {"physics_designB": "物理判据\n方案B(定负载)",
               "physics_designA": "物理判据\n方案A(自由负载)",
               "legacy_synthetic": "旧合成版\n3class",
               "published_4class": "原始公开\n4类"}
    rows = []
    for r in data:
        grouped = r.get("episode_grouped_split")
        rows.append({
            "name": name_cn.get(r["dataset"], r["dataset"]),
            "row": 100 * r["row_random_split"]["rf200"]["accuracy_mean"],
            "grp": 100 * grouped["rf200"]["accuracy_mean"] if grouped else np.nan,
            "grp_sd": 100 * grouped["rf200"]["accuracy_std"] if grouped else 0.0,
        })
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    width = 0.38

    fig, ax = new_fig(15.5, 7.0)
    b1 = ax.bar(x - width / 2, frame.row, width, label="按行随机划分（存在近邻泄漏）",
                color=PALETTE[5], edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, frame.grp, width, yerr=frame.grp_sd, capsize=2.5,
                label="按录制段分组划分（诚实基线）", color=PALETTE[1],
                hatch="///", edgecolor="black", linewidth=0.6,
                error_kw={"linewidth": 0.7})
    bar_label(ax, b1, frame.row.to_numpy(), dy=0.6)
    bar_label(ax, b2, np.nan_to_num(frame.grp.to_numpy()), dy=0.6)
    for i, (a, b) in enumerate(zip(frame.row, frame.grp)):
        if np.isfinite(b):
            ax.annotate(f"↓{a-b:.1f}", (i, max(a, b) + 5.0), ha="center",
                        fontsize=6.6)
    ax.set_xticks(x)
    ax.set_xticklabels(frame.name, fontsize=6.8)
    ax.set_ylabel("随机森林准确率 (%)")
    ax.set_ylim(0, 118)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=6.8)
    ax.set_title("按行随机划分会显著虚高：同一数据同一模型的两种划分对照")
    emit(fig, "图8_按行与按块划分的诚实基线差异")


def fig_dataset_provenance() -> None:
    """fig9: real vs synthetic vs duplicated composition per dataset."""
    specs = [
        ("物理判据方案B\n（本文）", DATA_DIR / "Public datasets_physics3_designB.csv",
         DATA_DIR / "Public datasets_physics3_designB_manifest.json"),
        ("物理判据方案A", DATA_DIR / "Public datasets_physics3_designA.csv",
         DATA_DIR / "Public datasets_physics3_designA_manifest.json"),
        ("旧合成版\n3class_12000", DATA_DIR / "Public datasets_3class_12000.csv",
         DATA_DIR / "Public datasets_3class_12000_manifest.json"),
        ("原始公开\n4类", DATA_DIR / "Public datasets.csv", None),
    ]
    rows = []
    for label, csv, man in specs:
        if not csv.exists():
            continue
        frame = pd.read_csv(csv)
        feats = [c for c in frame.columns
                 if c not in {"State", "State_Label", "tsec"}]
        live = [c for c in feats if frame[c].nunique() > 1]
        dup = int(frame[live].round(9).duplicated().sum())
        syn = 0
        if man and man.exists():
            counts = json.loads(man.read_text(encoding="utf-8")).get(
                "synthetic_counts", {})
            syn = int(sum(counts.values())) if counts else 0
        rows.append({"name": label, "total": len(frame), "syn": syn, "dup": dup,
                     "real": len(frame) - syn - dup})
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))

    fig, ax = new_fig(15.5, 7.0)
    ax.bar(x, frame.real, 0.6, label="真实且唯一的行", color=PALETTE[2],
           edgecolor="black", linewidth=0.6)
    ax.bar(x, frame.syn, 0.6, bottom=frame.real, label="KNN插值合成行",
           color=PALETTE[1], hatch="xxx", edgecolor="black", linewidth=0.6)
    ax.bar(x, frame.dup, 0.6, bottom=frame.real + frame.syn, label="重复行",
           color=PALETTE[6], hatch="...", edgecolor="black", linewidth=0.6)
    for i, r in frame.iterrows():
        bad = r.syn + r.dup
        note = "全部真实" if bad == 0 else f"非真实/重复 {bad} 行（{100*bad/r.total:.1f}%）"
        ax.annotate(note, (i, r.total + 180), ha="center", fontsize=6.4)
    ax.set_xticks(x)
    ax.set_xticklabels(frame.name, fontsize=6.8)
    ax.set_ylabel("样本行数")
    ax.set_ylim(0, frame.total.max() * 1.18)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.8)
    ax.set_title("数据集构成对照：本文物理判据数据集零合成、零重复")
    emit(fig, "图9_数据集构成对照")


def fig_water_activity() -> None:
    """fig10: the physical criterion itself -- a_w per class with thresholds."""
    csv = DATA_DIR / "Public datasets_physics3_designB.csv"
    if not csv.exists():
        return
    import sys
    sys.path.insert(0, str(MODEL_ROOT))
    from scripts.build_physics_3class import cathode_water_balance

    frame = pd.read_csv(csv)
    phys = cathode_water_balance(frame)
    frame = frame.assign(a_w=phys["a_w"].to_numpy(),
                         rh_out=phys["RH_out_at_cell"].to_numpy())
    classes = ["Membrane_Drying", "Normal", "Flooding"]

    fig, axes = new_fig(15.5, 6.8, ncols=2)
    ax = axes[0]
    data = [frame.loc[frame.State_Label == c, "a_w"].to_numpy() for c in classes]
    parts = ax.boxplot(data, patch_artist=True, widths=0.55,
                       medianprops={"color": "black", "linewidth": 1.1},
                       flierprops={"marker": ".", "markersize": 1.6,
                                   "markerfacecolor": "#555555",
                                   "markeredgecolor": "none"})
    for i, box in enumerate(parts["boxes"]):
        box.set_facecolor(PALETTE[i])
        box.set_edgecolor("black")
        box.set_linewidth(0.6)
        box.set_hatch(hatch_for(i))
    ax.axhline(1.00, color="#B22222", linewidth=1.1, linestyle=(0, (5, 2)))
    ax.axhline(0.70, color="#1F4E79", linewidth=1.1, linestyle=(0, (2, 2)))
    ax.annotate("aw = 1.00 饱和线（液态水析出）", (0.52, 1.02),
                fontsize=6.2, color="#B22222")
    ax.annotate("aw = 0.70 脱水线（质子导率下降）", (0.52, 0.62),
                fontsize=6.2, color="#1F4E79")
    ax.set_xticklabels([f"{CLASS_CN[c]}\n{c}" for c in classes], fontsize=6.6)
    ax.set_ylabel("阴极通道平均水活度 aw")
    ax.set_title("水活度判据的类间分离", fontsize=8.4)

    ax = axes[1]
    for i, c in enumerate(classes):
        vals = frame.loc[frame.State_Label == c, "rh_out"].to_numpy()
        ax.hist(vals, bins=46, histtype="stepfilled", alpha=1.0,
                label=f"{CLASS_CN[c]} {c}", color=PALETTE[i],
                hatch=hatch_for(i), edgecolor="black", linewidth=0.4)
    ax.axvline(100.0, color="#B22222", linewidth=1.1, linestyle=(0, (5, 2)))
    ax.annotate("RH=100%", (101, ax.get_ylim()[1] * 0.86), fontsize=6.2,
                color="#B22222")
    ax.set_xlabel("阴极出口相对湿度（按电池温度换算，%）")
    ax.set_ylabel("样本数")
    ax.legend(fontsize=6.4, loc="upper right")
    ax.set_title("出口湿度分布", fontsize=8.4)
    fig.suptitle("第一性原理阴极水平衡判据：阈值为热力学常数，非数据分位数",
                 fontsize=8.8)
    fig.tight_layout()
    emit(fig, "图10_水活度物理判据")


def fig_seed_spread(main: pd.DataFrame) -> None:
    """fig12: per-seed accuracy spread; stability, not just the mean."""
    order = [k for k in ORDER if k in set(main.model)]
    fig, ax = new_fig(15.5, 7.0)
    data = [100 * main[main.model == k].accuracy.to_numpy() for k in order]
    parts = ax.boxplot(data, patch_artist=True, widths=0.55,
                       medianprops={"color": "black", "linewidth": 1.0},
                       flierprops={"marker": "o", "markersize": 2.4,
                                   "markerfacecolor": "none",
                                   "markeredgecolor": "black"})
    for i, (box, key) in enumerate(zip(parts["boxes"], order)):
        box.set_facecolor(PALETTE[0] if key in PROPOSED else "#BBBBBB")
        box.set_edgecolor("black")
        box.set_linewidth(0.6)
        if key in PROPOSED:
            box.set_hatch("///")
    for i, values in enumerate(data, start=1):
        ax.scatter(np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values)),
                   values, s=8, color="black", zorder=4, linewidth=0)
    ax.set_xticklabels([CN[k] for k in order], rotation=22, ha="right")
    ax.set_ylabel("准确率 (%)")
    ax.set_axisbelow(True)
    ax.set_title("逐种子准确率分布（黑点为单次运行，深色为本文方法）")
    emit(fig, "图12_逐种子准确率分布")


def main() -> None:
    apply_thesis_style()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    print(f"figures -> {OUT_FIG}")

    main_df = load_records(MAIN)
    if main_df is not None:
        recs = json.loads((MAIN / "per_seed_records.json").read_text(encoding="utf-8"))
        fig_main_comparison(main_df)
        fig_confusion(recs)
        fig_per_class(recs)
        fig_seed_spread(main_df)
    else:
        print("  !! main study missing")

    noise_df = load_records(NOISE)
    if noise_df is not None and len(noise_df):
        fig_noise_curves(noise_df)
        fig_noise_heatmap(noise_df)
    else:
        print("  .. noise study not ready yet")

    abl_df = load_records(ABL)
    if abl_df is not None and len(abl_df):
        n_seeds = int(abl_df.seed.nunique())
        if n_seeds < 5:
            print(f"  !! ablation has only {n_seeds}/5 seeds -- figures 4/5 are "
                  f"PROVISIONAL, do not put them in the thesis yet")
        fig_ablation(abl_df, n_seeds=n_seeds)
        fig_ablation_conditions()
    else:
        print("  .. ablation study not ready yet")

    fig_honest_baseline()
    fig_dataset_provenance()
    fig_water_activity()

    manifest = {
        "figure_dir": str(OUT_FIG),
        "count": len(reports),
        "word_compatibility": {
            "text_rendering": "all glyphs converted to vector outlines "
                              "(svg.fonttype=path); no font required to open, "
                              "so CJK can never fall back to tofu boxes",
            "known_trade_off": "outlined text is not selectable/searchable in "
                               "Word; chosen because a missing glyph is fatal "
                               "and a non-searchable label is not",
            "removed_constructs": ["<style> CSS", "class= attributes",
                                   "<use>/xlink:href references", "<metadata>"],
            "root_sizing": "explicit px width/height plus viewBox",
            "greyscale_safe": "series differ by colour AND dash AND marker; "
                              "bars also differ by hatch",
            "png_twin_dpi": 600,
            "insert_path": "Word: 插入 -> 图片 -> 此设备，选择 .svg 文件",
        },
        "figures": reports,
    }
    (OUT_RESULTS / "图表清单与Word兼容性说明.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(reports)} figures written; all passed Word-safety validation")


if __name__ == "__main__":
    main()
