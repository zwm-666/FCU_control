# EIS 训练与量化使用说明

## 1. 宽表 EIS 数据
每一行一个样本，列中直接包含 EIS 特征，例如：

- 标签列：`状态`

运行示例：

```bash
python model_eis_quant.py \
  --mode eis \
  --data-path 你的EIS数据.xlsx \
  --label-col 状态 \
  --test-size 0.2 \
  --epochs 100 \
  --skip-ceo \
  --output-dir results_eis
```


脚本会自动尝试转成宽表；也可以手动指定列名：

```bash
python model_eis_quant.py \
  --mode eis \
  --data-path 你的EIS长表.xlsx \
  --label-col 状态 \
  --sample-id-col sample_id \
  --frequency-col frequency \
  --zreal-col zreal \
  --zimag-col zimag \
  --phase-col phase \
  --output-dir results_eis_long
```

## 3. 标签映射
如果标签是数字编码，比如：
- `0` = 正常
- `1` = 过干
- `2` = 过湿

可以这样传入：

```bash
python model_eis_quant.py \
  --mode eis \
  --data-path 你的EIS数据.xlsx \
  --label-col label \
  --label-map '{"0":"正常","1":"过干","2":"过湿"}'
```

## 4. 主要输出文件
训练结束后会在输出目录生成：
- `results.json`：总体训练指标
- `eis_test_quantification.csv`：测试集逐样本量化结果
- `eis_quantization_report.json`：量化规则与类别统计
- `eis_quantization_report.txt`：面向人工阅读的规则摘要
- `feature_importance.png`、`shap_bar.png` 等解释性文件

## 5. 量化含义
- `state_score_0_100`
  - 0：极湿端
  - 50：正常中心
  - 100：极干端
- `dry_wet_index`
  - -100：偏湿 / 过湿
  - 0：正常
  - +100：偏干 / 过干
- `normal_score`
  - 越高表示越接近正常工况

## 6. 说明
`eis_quantization_report` 中的阈值来自当前训练数据统计，是“数据驱动阈值”，适合当前数据集/台架/传感条件，不应直接当作普适物理常数。

## 7. 传统/联合对比模型
现有深度模型可与 `comparison_models.py` 中的对比模型做横向比较。默认对比集只保留一个传统增强树基线，并加入更接近时序/联合建模的深度组合模型：

- 传统基线：`xgboost`
- 深度联合模型：`cnn_lstm`、`mcnn`、`resnet_lstm`、`transformer_gru`、`cnn_transformer`

运行完整默认对比：

```bash
python comparison_models.py \
  --data 测试数据.csv \
  --models default \
  --test-size 0.2 \
  --deep-epochs 30 \
  --output-dir results_comparison_models
```

如果只想先快速烟测，可限制样本数：

```bash
python comparison_models.py \
  --data 测试数据.csv \
  --models default \
  --max-train-samples 120 \
  --max-test-samples 60 \
  --no-feature-selection \
  --deep-epochs 1 \
  --output-dir results_comparison_smoke_default
```

输出包括 `comparison_results.json`、`comparison_summary.csv` 和 `comparison_metrics.png`。
