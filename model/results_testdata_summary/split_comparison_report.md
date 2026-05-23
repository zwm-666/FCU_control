# 燃料电池诊断模型 —— 训练/测试集比例对比测试报告（当前代码实现）

- 模型：**CEO-QAAdamW-EnhancedMSTGAT**
- 数据集：`model/测试数据.xlsx`（11137 × 230）
- 入口：`model/model.py`（`run_multi_split_experiments`）
- 预处理：`model/preprocess_utils.py`（共享 `prepare_training_dataframe` + 划分后拟合型 `train_preprocess_and_select_from_df`）
- 运行命令：`python model.py`
- 统一配置：`epochs=100`，`budget=10`，`seed=42`，`skip_ceo=True`，`use_gpu=True`（本机无 GPU，自动回退 CPU）
- 运行时间：2026-04-17 17:15 – 18:10，单机 CPU（Python 3.9，TensorFlow 2.19）

---

## 1. 测试目的

在完全一致的模型结构、超参、随机种子和数据清洗管线下，评估三种训练/测试集比例（8:2、7:3、6:4）对模型诊断性能、训练时间和推理时间的影响，作为当前代码版本的基线。

## 2. 测试过程

1. `model.py` 的 `__main__` 先对 `测试数据.xlsx` 做一次性通用清洗（`prepare_training_dataframe`），生成共享 DataFrame 与 `label_col`，避免三组重复解析。
2. 依次调用 `run_training_pipeline`，仅改变 `test_size = 0.2 / 0.3 / 0.4`：
   - 每组都进入 `train_preprocess_and_select_from_df`，按划分后的训练集拟合 `StandardScaler` + GBDT 特征筛选，再应用到测试集。
   - 加载同一组固定最优超参（已在代码中硬编码，对应旧 `results.json` 的 `best_params`）。
   - 训练 EnhancedMSTGAT，`EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)`。
3. 每组产出：模型 `ceo_qaadam_emstgat.keras`、训练曲线 `training_history.png`、混淆矩阵 `confusion_matrix.png` / `confusion_matrix_normalized.png`、每类指标 `per_class_metrics.png`、特征重要性 `feature_importance.png`、预处理元信息 `preprocess_meta.json`、`results.json`。
4. 三组完成后写入 `results_testdata_summary/split_metrics_summary.json`，并生成横向对比图 `split_metrics_comparison.png`。
5. 运行日志保留在 `run_model.log`，可追溯每组 EarlyStopping 停在第几个 epoch。

## 3. 统一超参（三组一致）

| 超参 | 值 |
| --- | --- |
| learning_rate | 8.127e-4 |
| weight_decay | 4.266e-4 |
| hidden_units | 192 |
| attention_heads | 16 |
| max_sequence_length | 15 |
| knn_top_k | 6 |
| dropout_rate | 0.2056 |
| initial_noise_factor | 0.0242 |
| dilation_rate | 4 |
| use_batch_norm | False |
| embedding_dim | 160 |

## 4. 结果汇总（当前代码实测）

| 划分 | Accuracy | Precision | Recall | F1-Score | 训练耗时 (s) | 推理耗时 (s) | 筛选特征数 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **80 : 20** | **0.99955** | **0.99955** | **0.99955** | **0.99955** | 1242.17 | 7.47 | 9 |
| 70 : 30 | 0.99850 | 0.99851 | 0.99850 | 0.99850 | 1399.74 | 6.95 | 8 |
| 60 : 40 | 0.99753 | 0.99754 | 0.99753 | 0.99752 | 519.59 | 14.94 | 8 |

原始 JSON：`results_testdata_{80_20,70_30,60_40}/results.json`；汇总：`results_testdata_summary/split_metrics_summary.json`。

### 4.1 与上一版 (2026-03-30) 结果的对比

同一份 `测试数据.xlsx` + 同一 seed=42 的情况下，当前代码实测精度与上一版留存的 JSON 存在细微差异，主要源于 `model.py` 对 `prepare_training_dataframe`、`train_preprocess_and_select_from_df` 的重写，以及 `shap` 特征轴选择、模型序列化等改动：

| 划分 | Accuracy (旧 2026-03-30) | Accuracy (新 2026-04-17) | Δ |
| :--: | :--: | :--: | :--: |
| 80:20 | 0.999551 | 0.999551 | **一致** |
| 70:30 | 0.998504 | 0.998504 | **一致** |
| 60:40 | 0.998204 | **0.997531** | **−0.00067**（略降） |

> 80:20 / 70:30 两组在当前代码下完全复现了旧数字；60:40 的 `f1` 从 0.9982 跌到 0.9975，差距 ≈ 0.0007，来自划分后特征筛选 / 图构造的微小变动。

## 5. 结果分析

1. **精度随训练样本增加单调提升**：80:20 > 70:30 > 60:40，差距 < 0.003，三组均 ≥ 99.75%，模型在当前数据规模下已进入饱和区。
2. **训练耗时受数据量与早停共同影响**：80:20 训练 1242 s（约 32 epoch 后早停），70:30 反而最长 1399 s（训练 epoch 更多 + 单 epoch 稍慢），60:40 训练集最小，早停较早，519 s 收敛。
3. **推理耗时随测试集体量上升**：80:20 测试集 2228 样本预测 7.5 s；60:40 测试集 ≈ 4455 样本预测 14.9 s，基本线性（约 3.3 ms/样本）。
4. **特征筛选数量差异**：80:20 筛出 9 维特征，70:30/60:40 筛出 8 维 —— 划分后 GBDT 重要性阈值的轻微偏移所致。这是 60:40 精度略降的直接原因之一（信息面更窄）。
5. **混淆矩阵 / 每类指标**（各目录 `confusion_matrix*.png` / `per_class_metrics.png`）：三组在三类标签上均无明显偏置，没有出现某类误判集中。
6. **SHAP**：本机未装 `shap` 包，三份 `shap_meta.json` 均为 `status=failed, detail="No module named 'shap'"`。不影响精度指标；如需特征归因，单独 `pip install shap`，用 `generate_saved_model_shap.py` 对保存的 `.keras` 离线补做即可。

## 6. 结论与建议

- **推荐上线划分：80:20**。在当前代码实现下精度最高且与旧数字一致，可直接沿用。
- **70:30 为折中方案**，精度仅下降 ≈ 0.001，可作为需要更多验证样本时的选项。
- **60:40 不推荐**：当前代码下特征筛选退化、精度下降更明显（Δ≈0.002 相对 80:20），且测试集增大使推理耗时成倍上升。
- 模型对训练样本量已近饱和，后续收益应来自：
  - 扩充原始数据（尤其少数类）；
  - 稳定化划分后特征筛选（例如把特征选择放到 `prepare_training_dataframe` 里，三组共用一套特征）；
  - 安装 `shap` 做归因分析，定位剩余误分类样本。

## 7. 结果留存清单

```
model/
├── results_testdata_80_20/                 # 8:2 全量产物（模型、曲线、矩阵等）
│   └── results.json                        # acc=0.99955, f1=0.99955, train=1242s
├── results_testdata_70_30/                 # 7:3 全量产物
│   └── results.json                        # acc=0.99850, f1=0.99850, train=1400s
├── results_testdata_60_40/                 # 6:4 全量产物
│   └── results.json                        # acc=0.99753, f1=0.99752, train=520s
├── results_testdata_summary/
│   ├── split_metrics_summary.json          # 三组指标汇总
│   ├── split_metrics_comparison.png        # 横向对比图
│   └── split_comparison_report.md          # 本报告
└── run_model.log                           # 三组训练完整日志
```

> 复现命令（当前代码的唯一入口）：
> ```bash
> cd D:/h2-fcu-modern-dashboard/model
> python model.py
> ```
