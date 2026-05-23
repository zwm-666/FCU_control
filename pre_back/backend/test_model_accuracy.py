"""
回放测试脚本：使用原始训练数据集验证 Dashboard 中 EnhancedMSTGAT 模型的预测准确率

用法：
    python test_model_accuracy.py

功能：
    1. 加载原始数据集 fuel_cell_labeled_dataset_improved.csv
    2. 执行与训练时一致的预处理流程（移除列、标准化、特征选择）
    3. 用 OnlineDiagnosis.predict() 逐条预测
    4. 同时用原始训练流水线（直接模型推理）做对比
    5. 输出分类报告和混淆矩阵
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# 确保可以导入 diagnosis 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ================== 配置 ==================
# 数据集路径
DATASET_PATH = r"D:\python\pythonProject3\数据集2\fuel_cell_labeled_dataset_improved.csv"

# diagnosis.py 中 CAN 数据到模型特征的映射
# 数据集中的原始特征名 -> diagnosis.py extract_features 使用的 CAN 字段名
DATASET_TO_CAN_MAPPING = {
    "U_totV": "stackVoltage",
    "T_Stack_inlet": "stackTemp",
    "PW": "stackPower",
    "RH_H2": "h2Concentration",
    "T_Air_inlet": "airInletTemp",
    "T_3": "ambientTemp",
    "m_Air": "airFlow",
}

# 数据集标签 -> diagnosis.py 标签映射
LABEL_MAPPING = {
    "Normal": "normal",
    "Flooding": "flooding",
    "Membrane_Drying": "membrane_drying",
    "Thermal_Management_Fault": "thermal_issue",
}

FEATURE_COLS = ["U_totV", "T_Stack_inlet", "PW", "RH_H2", "T_Air_inlet", "T_3", "m_Air"]


def build_machine_state(row: pd.Series) -> dict:
    """将数据集的一行转换为 MachineState 字典格式"""
    return {
        "power": {
            "stackVoltage": float(row.get("U_totV", 0)),
            "stackCurrent": float(row.get("i_write", 0)),
            "stackPower": float(row.get("PW", 0)),
            "dcfOutVoltage": 0,
            "dcfOutCurrent": 0,
        },
        "sensors": {
            "stackTemp": float(row.get("T_Stack_inlet", 25)),
            "ambientTemp": float(row.get("T_3", 25)),
            "h2Concentration": float(row.get("RH_H2", 0)),
            "airInletTemp": float(row.get("T_Air_inlet", 30.4)),
            "airFlow": float(row.get("m_Air", 6.18)),
            "h2InletPressure": float(row.get("P_Air_inlet", 0)),
            "h2CylinderPressure": 0,
        },
        "io": {
            "faultCode": 0,
        },
        "status": {
            "heartbeat": 0,
            "state": 2,
            "faultLevel": 0,
        },
    }


def test_via_diagnosis_module():
    """
    测试方式一：通过 OnlineDiagnosis.predict() 测试
    这是 Dashboard 实际使用的路径
    """
    from diagnosis import OnlineDiagnosis

    print("=" * 70)
    print("测试方式：通过 OnlineDiagnosis.predict() (Dashboard 实际调用路径)")
    print("=" * 70)

    # 初始化诊断器
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    diag = OnlineDiagnosis(model_dir=model_dir)
    status = diag.get_status()
    print(f"诊断器状态: {status}")

    if not status["is_trained"]:
        print("❌ 模型未加载，无法测试！")
        return

    # 加载数据集
    print(f"\n加载数据集: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"数据集形状: {df.shape}")

    label_col = "State_Label"
    if label_col not in df.columns:
        print(f"❌ 找不到标签列 '{label_col}'")
        return

    # 逐条预测
    y_true = []
    y_pred = []
    pred_times = []
    total = len(df)

    print(f"\n开始逐条预测 (共 {total} 条)...")
    start_time = time.time()

    for idx in range(total):
        row = df.iloc[idx]

        # 构造 MachineState
        machine_state = build_machine_state(row)

        # 真实标签
        true_label = LABEL_MAPPING.get(row[label_col], "unknown")
        y_true.append(true_label)

        # 预测
        t0 = time.time()
        result = diag.predict(machine_state)
        pred_times.append(time.time() - t0)

        pred_label = result["label"]
        y_pred.append(pred_label)

        # 进度报告
        if (idx + 1) % 1000 == 0 or idx == total - 1:
            correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
            acc = correct / len(y_true) * 100
            elapsed = time.time() - start_time
            print(f"  进度: {idx+1}/{total} | 当前准确率: {acc:.2f}% | 耗时: {elapsed:.1f}s")

    total_time = time.time() - start_time

    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

    all_labels = ["normal", "flooding", "membrane_drying", "thermal_issue"]
    all_labels_cn = ["正常", "水淹", "膜干燥", "热管理异常"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', labels=all_labels, zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', labels=all_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', labels=all_labels, zero_division=0)

    print(f"\n{'=' * 70}")
    print(f"测试结果总结")
    print(f"{'=' * 70}")
    print(f"  总样本数:     {total}")
    print(f"  准确率:       {acc:.4f} ({acc*100:.2f}%)")
    print(f"  精确率:       {prec:.4f}")
    print(f"  召回率:       {rec:.4f}")
    print(f"  F1 分数:      {f1:.4f}")
    print(f"  总耗时:       {total_time:.2f}s")
    print(f"  平均预测时间: {np.mean(pred_times)*1000:.2f}ms/条")

    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, labels=all_labels,
                                target_names=all_labels_cn, zero_division=0))

    print(f"混淆矩阵 (行=真实, 列=预测):")
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    # 美化输出
    header = "            " + "  ".join(f"{n:>8}" for n in all_labels_cn)
    print(header)
    for i, row_label in enumerate(all_labels_cn):
        row_str = "  ".join(f"{v:>8}" for v in cm[i])
        print(f"  {row_label:<8} {row_str}")

    print(f"\n{'=' * 70}")

    return acc


def test_via_direct_model():
    """
    测试方式二：使用训练流水线中一致的预处理 + 直接模型推理
    这会重现训练时的评估流程
    """
    print("\n" + "=" * 70)
    print("测试方式：训练流水线一致的预处理 + 直接模型推理")
    print("=" * 70)

    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return

    # 加载数据集
    print(f"加载数据集: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # 与训练流水线一致：移除 State, state, tsec 列
    for col in ['State', 'state', 'tsec']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 分离特征和标签
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"标签编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    num_classes = len(np.unique(y_encoded))
    print(f"特征数: {X.shape[1]}, 类别数: {num_classes}")

    # 划分数据集（与训练时一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 特征选择（与训练时一致：梯度提升重要性 >= 95%）
    print("计算特征重要性（与训练时一致）...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    importances = gb.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[sorted_idx])
    cumsum_norm = cumsum / cumsum[-1]
    n_selected = np.searchsorted(cumsum_norm, 0.95) + 1
    selected_idx = sorted_idx[:n_selected]

    X_train_selected = X_train_scaled[:, selected_idx]
    X_test_selected = X_test_scaled[:, selected_idx]

    print(f"选择 {len(selected_idx)}/{X.shape[1]} 个特征")

    # 加载模型
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ceo_qaadam_emstgat.keras")
    print(f"加载模型: {model_path}")

    from diagnosis import CUSTOM_OBJECTS
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    print(f"模型参数量: {model.count_params():,}")

    # 推理
    print("开始推理...")
    t0 = time.time()
    y_pred_prob = model.predict(X_test_selected, verbose=0)
    pred_time = time.time() - t0
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n测试结果 (训练流水线预处理):")
    print(f"  测试样本数:   {len(y_test)}")
    print(f"  准确率:       {acc:.4f} ({acc*100:.2f}%)")
    print(f"  推理耗时:     {pred_time:.2f}s")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    return acc


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"❌ 数据集文件不存在: {DATASET_PATH}")
        print("请修改脚本中的 DATASET_PATH 变量为正确的数据集路径")
        sys.exit(1)

    print("=" * 70)
    print("EnhancedMSTGAT 模型回放测试")
    print(f"数据集: {DATASET_PATH}")
    print("=" * 70)

    # 测试一：通过 Dashboard 实际调用路径
    acc1 = test_via_diagnosis_module()

    # 测试二：通过训练流水线一致的预处理
    acc2 = test_via_direct_model()

    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    if acc1 is not None:
        print(f"  Dashboard 调用路径准确率:  {acc1:.4f} ({acc1*100:.2f}%)")
    if acc2 is not None:
        print(f"  训练流水线预处理准确率:    {acc2:.4f} ({acc2*100:.2f}%)")
    print()
    print("说明:")
    print("  - '训练流水线预处理' 使用与训练时完全一致的标准化和特征选择，")
    print("    该结果应与 results.json 中的 98.3% 接近")
    print("  - 'Dashboard 调用路径' 使用 diagnosis.py 中的固定标准化参数，")
    print("    由于标准化参数不同，准确率可能会有偏差")
    print("=" * 70)
