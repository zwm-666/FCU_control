# -*- coding: utf-8 -*-
"""
用模型实际选定的特征分析 过干新增.xlsx 数据，
与水淹和膜干故障测试数据中每一类(正常/过干/过湿)数据的相关性，
并对过干新增数据进行分类。

步骤：
1. 加载测试数据.xlsx，从 preprocess_meta.json 读取模型实际选用特征
2. 加载水淹和膜干故障测试数据（正常、过干、过湿）
3. 用选定特征分析 过干新增.xlsx 与各类数据的相关性
4. 用训练好的GBDT模型对过干新增数据进行分类预测
5. 输出可视化和分析报告
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(r"C:\Users\86191\Desktop\h2-fcu-modern-dashboard\model")
FAULT_DIR = BASE / "数据文件" / "水淹和膜干故障测试数据"
OUTPUT_DIR = BASE / "gbdt_top10_analysis"
BASE1 = Path(r"C:\Users\86191\Desktop\h2-fcu-modern-dashboard\model\数据文件")
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUTPUT_DIR / "analysis_log.txt"
log_lines = []

def log(msg=""):
    log_lines.append(str(msg))
    try:
        print(msg)
    except:
        pass

def save_log():
    LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")


# ============================================================
# 1. 加载训练数据，从 preprocess_meta.json 读取模型实际选用特征
# ============================================================
log("=" * 70)
log("步骤1: 加载测试数据，读取模型实际选用特征")
log("=" * 70)

df_train = pd.read_excel(str(BASE / "测试数据.xlsx"))
log(f"测试数据.xlsx: {df_train.shape}")

# 从 preprocess_meta.json 读取模型实际选用的特征
META_PATH = BASE / "results_testdata_60_40" / "preprocess_meta.json"
with open(str(META_PATH), encoding="utf-8") as f:
    preprocess_meta = json.load(f)
selected_features = preprocess_meta["selected_features"]
N_FEAT = len(selected_features)
log(f"从 {META_PATH.name} 读取模型选用特征 ({N_FEAT}个): {selected_features}")

# 移除非特征列
label_col = "类型"
datetime_cols = [c for c in df_train.columns if pd.api.types.is_datetime64_any_dtype(df_train[c])]
drop_cols = datetime_cols + ["电堆功率"]  # 移除功率(训练策略)和时间列
drop_cols = [c for c in drop_cols if c in df_train.columns]
df_train_clean = df_train.drop(columns=drop_cols)
log(f"移除列: {drop_cols}")

feature_cols = [c for c in df_train_clean.columns if c != label_col]
X_all = df_train_clean[feature_cols].values.astype(np.float32)
y_all = df_train_clean[label_col].values

# 处理缺失值
X_all = np.nan_to_num(X_all, nan=0.0)

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y_all)
class_names = [str(c) for c in le.classes_]
log(f"类别: {class_names} -> {dict(zip(class_names, [int((y_encoded==i).sum()) for i in range(len(class_names))]))}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# 划分训练/测试集
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# 训练GBDT（用全部特征，获取重要性排名）
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_tr, y_tr)

train_acc = gb.score(X_tr, y_tr)
test_acc = gb.score(X_te, y_te)
log(f"GBDT训练精度: {train_acc:.4f}, 测试精度: {test_acc:.4f}")

# 获取选定特征的重要性
importances = gb.feature_importances_
selected_idx = [feature_cols.index(f) for f in selected_features]
selected_importances = importances[selected_idx]

log(f"\n模型选定的 {N_FEAT} 个特征 (按重要性排序):")
rank_order = np.argsort(selected_importances)[::-1]
top_features_sorted = [selected_features[i] for i in rank_order]
top_importances_sorted = selected_importances[rank_order]
for i, (feat, imp) in enumerate(zip(top_features_sorted, top_importances_sorted)):
    log(f"  {i+1:2d}. {feat:12s}  重要性={imp:.6f}")

# ============================================================
# 可视化1: 模型选定特征重要性柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(N_FEAT-1, -1, -1), top_importances_sorted, color=plt.cm.viridis(np.linspace(0.2, 0.9, N_FEAT)))
ax.set_yticks(range(N_FEAT-1, -1, -1))
ax.set_yticklabels(top_features_sorted)
ax.set_xlabel("特征重要性")
ax.set_title(f"GBDT 模型选定 {N_FEAT} 个特征重要性排名", fontsize=14, fontweight="bold")
for bar, val in zip(bars, top_importances_sorted):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f"{val:.4f}", va="center")
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "01_selected_feature_importance.png"), dpi=150)
plt.close()
log(f"图1: {N_FEAT}个选定特征重要性 -> 01_selected_feature_importance.png")


# ============================================================
# 2. 加载各类数据
# ============================================================
log("\n" + "=" * 70)
log("步骤2: 加载水淹和膜干故障测试数据 + 过干新增数据")
log("=" * 70)

df_normal = pd.read_excel(str(FAULT_DIR / "正常.xlsx"))
df_dry = pd.read_excel(str(FAULT_DIR / "过干.xlsx"))
df_wet = pd.read_excel(str(FAULT_DIR / "过湿.xlsx"))
df_new = pd.read_excel(str(BASE1 / "过干新增.xlsx"))

log(f"正常数据: {df_normal.shape}")
log(f"过干数据: {df_dry.shape}")
log(f"过湿数据: {df_wet.shape}")
log(f"过干新增数据: {df_new.shape}")

# 找出公共特征列（选定特征中在所有数据集中都存在的）
all_dfs = {"正常": df_normal, "过干": df_dry, "过湿": df_wet, "过干新增": df_new}
available_features = []
for feat in selected_features:
    available = all([feat in df.columns for df in all_dfs.values()])
    if available:
        available_features.append(feat)
    else:
        missing_in = [name for name, df in all_dfs.items() if feat not in df.columns]
        log(f"  WARNING: '{feat}' 缺失于 {missing_in}")

TOP_FEATURES = available_features
log(f"\n最终分析特征 ({len(TOP_FEATURES)}个): {TOP_FEATURES}")


# ============================================================
# 3. 各类数据选定特征统计描述
# ============================================================
log("\n" + "=" * 70)
log(f"步骤3: 各类数据选定{len(TOP_FEATURES)}个特征统计描述")
log("=" * 70)

stats_summary = {}
for name, df in all_dfs.items():
    desc = df[TOP_FEATURES].describe().T
    stats_summary[name] = desc
    log(f"\n--- {name} ---")
    log(desc[["mean", "std", "min", "max"]].to_string())


# ============================================================
# 4. 相关性分析：过干新增 vs 各类数据
# ============================================================
log("\n" + "=" * 70)
log("步骤4: 过干新增数据与各类故障数据的相关性分析")
log("=" * 70)

def compute_feature_correlation(df1, df2, features):
    """计算两组数据在指定特征上的相关性指标"""
    results = {}
    for feat in features:
        v1 = df1[feat].dropna().values
        v2 = df2[feat].dropna().values
        
        # 均值差异
        mean_diff = abs(np.mean(v1) - np.mean(v2))
        
        # KS检验 (分布相似性)
        ks_stat, ks_pval = stats.ks_2samp(v1, v2)
        
        # 均值分布重叠度 (用均值和标准差估算)
        m1, s1 = np.mean(v1), max(np.std(v1), 1e-8)
        m2, s2 = np.mean(v2), max(np.std(v2), 1e-8)
        
        # Bhattacharyya距离（近似正态）
        bhatt = 0.25 * np.log(0.25 * (s1**2/s2**2 + s2**2/s1**2 + 2)) + 0.25 * ((m1-m2)**2 / (s1**2 + s2**2))
        
        # Cohen's d (效应量)
        pooled_std = np.sqrt((s1**2 + s2**2) / 2)
        cohens_d = abs(m1 - m2) / pooled_std if pooled_std > 1e-8 else 0
        
        results[feat] = {
            "mean_diff": mean_diff,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "bhatt_dist": bhatt,
            "cohens_d": cohens_d,
            "mean_new": np.mean(v1),
            "std_new": np.std(v1),
            "mean_ref": np.mean(v2),
            "std_ref": np.std(v2),
        }
    return results

correlation_results = {}
for name in ["正常", "过干", "过湿"]:
    corr = compute_feature_correlation(df_new, all_dfs[name], TOP_FEATURES)
    correlation_results[name] = corr
    
    log(f"\n--- 过干新增 vs {name} ---")
    log(f"{'特征':12s} {'均值差':>10s} {'KS统计量':>10s} {'KS p值':>10s} {'Bhatt距离':>10s} {'Cohen_d':>10s}")
    for feat in TOP_FEATURES:
        r = corr[feat]
        log(f"{feat:12s} {r['mean_diff']:10.4f} {r['ks_stat']:10.4f} {r['ks_pval']:10.4f} {r['bhatt_dist']:10.4f} {r['cohens_d']:10.4f}")


# ============================================================
# 可视化2: 相关性热力图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ["ks_stat", "bhatt_dist", "cohens_d"]
metric_names = ["KS统计量\n(越小越相似)", "Bhattacharyya距离\n(越小越相似)", "Cohen's d效应量\n(越小越相似)"]

for ax, metric, mname in zip(axes, metrics, metric_names):
    data = []
    for ref_name in ["正常", "过干", "过湿"]:
        row = [correlation_results[ref_name][feat][metric] for feat in TOP_FEATURES]
        data.append(row)
    data = np.array(data)
    
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(TOP_FEATURES)))
    ax.set_xticklabels(TOP_FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["正常", "过干", "过湿"])
    ax.set_title(mname, fontsize=11)
    
    for i in range(3):
        for j in range(len(TOP_FEATURES)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=7,
                   color="white" if data[i,j] > data.max()*0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("过干新增数据 vs 各类故障数据 — 特征分布差异", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "02_correlation_heatmap.png"), dpi=150)
plt.close()
log("\n图2: 相关性热力图 -> 02_correlation_heatmap.png")


# ============================================================
# 可视化3: 选定特征分布对比箱线图
# ============================================================
n_features = len(TOP_FEATURES)
n_cols = min(n_features, 4)
n_rows = (n_features + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = np.array(axes).flatten() if n_features > 1 else [axes]

colors = {"正常": "#2ecc71", "过干": "#e74c3c", "过湿": "#3498db", "过干新增": "#f39c12"}

for idx, feat in enumerate(TOP_FEATURES):
    ax = axes[idx]
    box_data = []
    labels = []
    for name in ["正常", "过干", "过湿", "过干新增"]:
        vals = all_dfs[name][feat].dropna().values
        box_data.append(vals)
        labels.append(name)
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6)
    for patch, name in zip(bp["boxes"], labels):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

# 隐藏多余子图
for idx in range(n_features, len(axes)):
    axes[idx].set_visible(False)
plt.suptitle(f"选定{n_features}个特征分布对比（正常 vs 过干 vs 过湿 vs 过干新增）", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "03_feature_distribution_boxplot.png"), dpi=150)
plt.close()
log("图3: 特征分布箱线图 -> 03_feature_distribution_boxplot.png")


# ============================================================
# 可视化4: 特征密度分布对比
# ============================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = np.array(axes).flatten() if n_features > 1 else [axes]

for idx, feat in enumerate(TOP_FEATURES):
    ax = axes[idx]
    for name in ["正常", "过干", "过湿", "过干新增"]:
        vals = all_dfs[name][feat].dropna().values
        if len(vals) > 2:
            try:
                kde = stats.gaussian_kde(vals)
                x_range = np.linspace(vals.min() - 0.1*abs(vals.min()), vals.max() + 0.1*abs(vals.max()), 200)
                ax.plot(x_range, kde(x_range), label=name, color=colors[name], linewidth=2)
                ax.fill_between(x_range, kde(x_range), alpha=0.15, color=colors[name])
            except:
                pass
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# 隐藏多余子图
for idx in range(n_features, len(axes)):
    axes[idx].set_visible(False)
plt.suptitle(f"选定{n_features}个特征核密度分布对比", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "04_feature_density_comparison.png"), dpi=150)
plt.close()
log("图4: 特征密度分布 -> 04_feature_density_comparison.png")


# ============================================================
# 5. 用GBDT对过干新增数据分类预测
# ============================================================
log("\n" + "=" * 70)
log(f"步骤5: 用选定{len(TOP_FEATURES)}个特征GBDT模型对过干新增数据分类预测")
log("=" * 70)

# 用选定特征重新训练GBDT
sel_idx_in_features = [feature_cols.index(f) for f in TOP_FEATURES]
X_tr_sel = X_tr[:, sel_idx_in_features]
X_te_sel = X_te[:, sel_idx_in_features]

gb_sel = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb_sel.fit(X_tr_sel, y_tr)

train_acc_sel = gb_sel.score(X_tr_sel, y_tr)
test_acc_sel = gb_sel.score(X_te_sel, y_te)
log(f"选定{len(TOP_FEATURES)}特征GBDT: 训练精度={train_acc_sel:.4f}, 测试精度={test_acc_sel:.4f}")

# 在测试集上的分类报告
y_pred_test = gb_sel.predict(X_te_sel)
log("\n测试集分类报告:")
class_labels = {0: "正常", 1: "过干", 2: "过湿"}
target_names = [class_labels.get(int(c), str(c)) for c in le.classes_]
log(classification_report(y_te, y_pred_test, target_names=target_names))

# 对过干新增数据预测
X_new = df_new[TOP_FEATURES].values.astype(np.float32)
X_new = np.nan_to_num(X_new, nan=0.0)

# 使用与训练对应位置的scaler参数
scaler_mean = scaler.mean_[sel_idx_in_features]
scaler_scale = scaler.scale_[sel_idx_in_features]
X_new_scaled = (X_new - scaler_mean) / np.where(np.abs(scaler_scale) < 1e-8, 1.0, scaler_scale)

y_new_pred = gb_sel.predict(X_new_scaled)
y_new_proba = gb_sel.predict_proba(X_new_scaled)

# 统计分类结果
pred_counts = pd.Series(y_new_pred).value_counts().sort_index()
log(f"\n过干新增数据分类结果 (共{len(y_new_pred)}条):")
for cls_id, count in pred_counts.items():
    pct = count / len(y_new_pred) * 100
    cls_name = class_labels.get(cls_id, str(cls_id))
    log(f"  {cls_name} (类别{cls_id}): {count} 条 ({pct:.1f}%)")

# 平均概率
avg_proba = y_new_proba.mean(axis=0)
log(f"\n过干新增数据平均预测概率:")
for i, name in enumerate(target_names):
    log(f"  {name}: {avg_proba[i]:.4f}")


# ============================================================
# 6. 对水淹和膜干故障测试数据也进行预测
# ============================================================
log("\n" + "=" * 70)
log("步骤6: 对水淹和膜干故障测试数据进行分类预测")
log("=" * 70)

for name, df in [("正常", df_normal), ("过干", df_dry), ("过湿", df_wet)]:
    X_ref = df[TOP_FEATURES].values.astype(np.float32)
    X_ref = np.nan_to_num(X_ref, nan=0.0)
    X_ref_scaled = (X_ref - scaler_mean) / np.where(np.abs(scaler_scale) < 1e-8, 1.0, scaler_scale)
    
    y_ref_pred = gb_sel.predict(X_ref_scaled)
    y_ref_proba = gb_sel.predict_proba(X_ref_scaled)
    
    ref_counts = pd.Series(y_ref_pred).value_counts().sort_index()
    log(f"\n--- {name}数据 预测结果 ({len(df)}条) ---")
    for cls_id, count in ref_counts.items():
        pct = count / len(y_ref_pred) * 100
        cls_name = class_labels.get(cls_id, str(cls_id))
        log(f"  {cls_name}: {count} 条 ({pct:.1f}%)")
    
    avg_p = y_ref_proba.mean(axis=0)
    log(f"  平均概率: 正常={avg_p[0]:.4f}, 过干={avg_p[1]:.4f}, 过湿={avg_p[2]:.4f}")




# ============================================================
# 7. 综合相关性评分
# ============================================================
log("\n" + "=" * 70)
log("步骤7: 综合相关性评分")
log("=" * 70)

# 计算过干新增数据与各类数据的综合相似度
similarity_scores = {}
for ref_name in ["正常", "过干", "过湿"]:
    corr = correlation_results[ref_name]
    
    # 综合评分 (KS统计量越小越相似，取倒数)
    avg_ks = np.mean([corr[f]["ks_stat"] for f in TOP_FEATURES])
    avg_bhatt = np.mean([corr[f]["bhatt_dist"] for f in TOP_FEATURES])
    avg_cohens = np.mean([corr[f]["cohens_d"] for f in TOP_FEATURES])
    
    # 相似度 = 1 / (1 + 距离)
    sim_ks = 1 / (1 + avg_ks)
    sim_bhatt = 1 / (1 + avg_bhatt)
    sim_cohens = 1 / (1 + avg_cohens)
    
    # 加权平均
    overall_sim = 0.3 * sim_ks + 0.4 * sim_bhatt + 0.3 * sim_cohens
    
    similarity_scores[ref_name] = {
        "avg_ks": avg_ks,
        "avg_bhatt": avg_bhatt,
        "avg_cohens_d": avg_cohens,
        "sim_ks": sim_ks,
        "sim_bhatt": sim_bhatt,
        "sim_cohens": sim_cohens,
        "overall_similarity": overall_sim,
    }
    
    log(f"\n--- 过干新增 vs {ref_name} ---")
    log(f"  平均KS统计量: {avg_ks:.4f} (相似度: {sim_ks:.4f})")
    log(f"  平均Bhatt距离: {avg_bhatt:.4f} (相似度: {sim_bhatt:.4f})")
    log(f"  平均Cohen's d: {avg_cohens:.4f} (相似度: {sim_cohens:.4f})")
    log(f"  ★ 综合相似度: {overall_sim:.4f}")

# 找最相似的类别
best_match = max(similarity_scores, key=lambda k: similarity_scores[k]["overall_similarity"])
log(f"\n{'='*50}")
log(f"★★★ 过干新增数据与 [{best_match}] 类数据最为相似 ★★★")
log(f"    综合相似度: {similarity_scores[best_match]['overall_similarity']:.4f}")
log(f"{'='*50}")


# ============================================================
# 可视化7: 综合相似度雷达图
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

categories = ["KS相似度", "Bhatt相似度", "Cohen's d相似度"]
n_cats = len(categories)
angles = [n / n_cats * 2 * np.pi for n in range(n_cats)]
angles += angles[:1]

radar_colors = {"正常": "#2ecc71", "过干": "#e74c3c", "过湿": "#3498db"}

for ref_name in ["正常", "过干", "过湿"]:
    s = similarity_scores[ref_name]
    values = [s["sim_ks"], s["sim_bhatt"], s["sim_cohens"]]
    values += values[:1]
    
    ax.plot(angles, values, "o-", linewidth=2, label=f"vs {ref_name}", color=radar_colors[ref_name])
    ax.fill(angles, values, alpha=0.15, color=radar_colors[ref_name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("过干新增数据与各类故障数据的相似度\n", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "07_similarity_radar.png"), dpi=150)
plt.close()
log("\n图7: 相似度雷达图 -> 07_similarity_radar.png")


# ============================================================
# 8. 保存详细结果到Excel
# ============================================================
log("\n" + "=" * 70)
log("步骤8: 保存结果")
log("=" * 70)

# 为过干新增数据加上分类结果
df_new_result = df_new.copy()
df_new_result["GBDT预测类别"] = y_new_pred
df_new_result["GBDT预测类别名"] = [class_labels.get(int(p), str(p)) for p in y_new_pred]
df_new_result["P(正常)"] = y_new_proba[:, 0]
df_new_result["P(过干)"] = y_new_proba[:, 1]
df_new_result["P(过湿)"] = y_new_proba[:, 2]

result_path = OUTPUT_DIR / "过干新增_GBDT_选定特征分类结果.xlsx"
with pd.ExcelWriter(str(result_path), engine="openpyxl") as writer:
    df_new_result.to_excel(writer, sheet_name="分类结果", index=False)
    
    # 相关性分析结果
    corr_rows = []
    for ref_name in ["正常", "过干", "过湿"]:
        for feat in TOP_FEATURES:
            r = correlation_results[ref_name][feat]
            corr_rows.append({
                "参考类别": ref_name,
                "特征": feat,
                "过干新增_均值": r["mean_new"],
                "过干新增_标准差": r["std_new"],
                f"{ref_name}_均值": r["mean_ref"],
                f"{ref_name}_标准差": r["std_ref"],
                "均值差": r["mean_diff"],
                "KS统计量": r["ks_stat"],
                "KS_p值": r["ks_pval"],
                "Bhatt距离": r["bhatt_dist"],
                "Cohen_d": r["cohens_d"],
            })
    pd.DataFrame(corr_rows).to_excel(writer, sheet_name="相关性分析", index=False)
    
    # 综合评分
    sim_rows = []
    for ref_name in ["正常", "过干", "过湿"]:
        s = similarity_scores[ref_name]
        sim_rows.append({
            "参考类别": ref_name,
            "平均KS统计量": s["avg_ks"],
            "平均Bhatt距离": s["avg_bhatt"],
            "平均Cohen_d": s["avg_cohens_d"],
            "KS相似度": s["sim_ks"],
            "Bhatt相似度": s["sim_bhatt"],
            "Cohen_d相似度": s["sim_cohens"],
            "综合相似度": s["overall_similarity"],
        })
    pd.DataFrame(sim_rows).to_excel(writer, sheet_name="综合相似度", index=False)
    
    # 选定特征信息
    feat_info = pd.DataFrame({
        "排名": range(1, len(TOP_FEATURES)+1),
        "特征名": TOP_FEATURES,
        "GBDT重要性": [importances[feature_cols.index(f)] for f in TOP_FEATURES],
    })
    feat_info.to_excel(writer, sheet_name="选定特征", index=False)

log(f"结果已保存: {result_path}")


# ============================================================
# 9. 综合结论
# ============================================================
log("\n" + "=" * 70)
log("分析结论")
log("=" * 70)

log(f"\n1. GBDT模型使用选定{len(TOP_FEATURES)}个特征的测试精度: {test_acc_sel:.4f}")
log(f"\n2. 选定特征 ({len(TOP_FEATURES)}个): {', '.join(TOP_FEATURES)}")
log(f"\n3. 过干新增数据分类结果:")
for cls_id in range(3):
    count = (y_new_pred == cls_id).sum()
    pct = count / len(y_new_pred) * 100
    log(f"   - {class_labels.get(cls_id, str(cls_id))}: {count} 条 ({pct:.1f}%)")

log(f"\n4. 综合相似度排名:")
sorted_sim = sorted(similarity_scores.items(), key=lambda x: x[1]["overall_similarity"], reverse=True)
for rank, (name, s) in enumerate(sorted_sim, 1):
    log(f"   {rank}. vs {name}: {s['overall_similarity']:.4f}")

log(f"\n5. 结论: 过干新增数据在选定{len(TOP_FEATURES)}个特征维度上与 [{best_match}] 类数据最为相似。")

save_log()
log(f"\n完整日志已保存: {LOG_FILE}")
log("分析完成！")
