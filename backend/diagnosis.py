"""
Online Fault Diagnosis Module for Hydrogen Fuel Cell System
Supports incremental learning with user feedback
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import joblib

# Try to import sklearn, fallback gracefully if not available
try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, diagnosis will be limited")

logger = logging.getLogger(__name__)

# 诊断标签定义
DIAGNOSIS_LABELS = {
    0: "normal",           # 正常
    1: "flooding",         # 水淹
    2: "membrane_drying",  # 膜干燥
    3: "thermal_issue"     # 热管理问题
}

DIAGNOSIS_LABELS_CN = {
    "normal": "正常",
    "flooding": "水淹故障",
    "membrane_drying": "膜干燥",
    "thermal_issue": "热管理异常"
}

# 特征映射：CAN数据字段 -> 模型特征
# 基于用户提供的7个特征：T_Stack_inlet, U_totV, T_3, RH_H2, PW, i_write, P_Air_inlet
FEATURE_MAPPING = {
    "T_Stack_inlet": "stackTemp",        # 电堆入口温度
    "U_totV": "stackVoltage",            # 总电压
    "T_3": "ambientTemp",                # 温度3 (用环境温度代替)
    "RH_H2": "h2Concentration",          # 氢气相对湿度 (用浓度代替)
    "PW": "stackPower",                  # 功率
    "i_write": "stackCurrent",           # 电流
    "P_Air_inlet": "h2InletPressure"     # 空气入口压力 (用氢气入口压力代替)
}

FEATURE_ORDER = ["T_Stack_inlet", "U_totV", "T_3", "RH_H2", "PW", "i_write", "P_Air_inlet"]


class OnlineDiagnosis:
    """
    在线故障诊断器
    - 支持增量学习 (partial_fit)
    - 自动特征标准化
    - 模型持久化
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "diagnosis_model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.data_path = os.path.join(model_dir, "training_data.json")
        
        # 确保目录存在
        os.makedirs(model_dir, exist_ok=True)
        
        # 初始化模型和标准化器
        self.model: Optional[SGDClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.training_samples: List[Dict] = []
        self.is_trained = False
        self.sample_count = 0
        
        # 加载或初始化
        self._initialize()
        
    def _initialize(self):
        """初始化或加载模型"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, diagnosis disabled")
            return
            
        # 尝试加载现有模型
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                logger.info(f"✓ 已加载诊断模型: {self.model_path}")
                
                # 加载训练数据计数
                if os.path.exists(self.data_path):
                    with open(self.data_path, 'r') as f:
                        data = json.load(f)
                        self.sample_count = data.get("sample_count", 0)
                        self.training_samples = data.get("samples", [])[-100:]  # 只保留最近100条
                        
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                self._create_new_model()
        else:
            self._create_new_model()
            
    def _create_new_model(self):
        """创建新的SGD分类器"""
        if not SKLEARN_AVAILABLE:
            return
            
        # SGDClassifier 支持 partial_fit 增量学习
        self.model = SGDClassifier(
            loss='log_loss',  # 逻辑回归，支持概率输出
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True
        )
        
        # 标准化器
        self.scaler = StandardScaler()
        
        # 初始化标准化器的默认参数（基于典型燃料电池数据范围）
        # 这些值会随着数据积累而更新
        default_means = np.array([50.0, 48.0, 25.0, 50.0, 2.0, 40.0, 0.3])  # 7个特征
        default_stds = np.array([20.0, 10.0, 10.0, 20.0, 1.0, 20.0, 0.1])
        
        self.scaler.mean_ = default_means
        self.scaler.scale_ = default_stds
        self.scaler.var_ = default_stds ** 2
        self.scaler.n_features_in_ = 7
        self.scaler.n_samples_seen_ = 1
        
        self.is_trained = False
        self.sample_count = 0
        logger.info("✓ 创建新的在线诊断模型 (未训练)")
        
    def extract_features(self, machine_state: Dict) -> Optional[np.ndarray]:
        """
        从机器状态提取特征向量
        
        Args:
            machine_state: MachineState字典
            
        Returns:
            特征向量 shape (7,)
        """
        try:
            power = machine_state.get("power", {})
            sensors = machine_state.get("sensors", {})
            
            features = [
                sensors.get("stackTemp", 25.0),           # T_Stack_inlet
                power.get("stackVoltage", 0.0),           # U_totV
                sensors.get("ambientTemp", 25.0),         # T_3
                sensors.get("h2Concentration", 0.0),      # RH_H2
                power.get("stackPower", 0.0),             # PW
                power.get("stackCurrent", 0.0),           # i_write
                sensors.get("h2InletPressure", 0.0)       # P_Air_inlet
            ]
            
            return np.array(features, dtype=np.float64)
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None
            
    def predict(self, machine_state: Dict) -> Dict:
        """
        预测故障类型
        
        Args:
            machine_state: MachineState字典
            
        Returns:
            {
                "label": "normal" | "flooding" | "membrane_drying" | "thermal_issue",
                "label_cn": "正常" | "水淹故障" | ...,
                "confidence": 0-100,
                "probabilities": {label: prob, ...},
                "is_trained": bool,
                "sample_count": int,
                "timestamp": int
            }
        """
        result = {
            "label": "normal",
            "label_cn": "正常",
            "confidence": 0.0,
            "probabilities": {},
            "is_trained": self.is_trained,
            "sample_count": self.sample_count,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        if not SKLEARN_AVAILABLE or self.model is None:
            return result
            
        # 提取特征
        features = self.extract_features(machine_state)
        if features is None:
            return result
            
        try:
            # 标准化
            X = features.reshape(1, -1)
            X_scaled = (X - self.scaler.mean_) / self.scaler.scale_
            
            if self.is_trained:
                # 预测概率
                probas = self.model.predict_proba(X_scaled)[0]
                pred_class = np.argmax(probas)
                confidence = float(probas[pred_class]) * 100
                
                label = DIAGNOSIS_LABELS.get(pred_class, "normal")
                result["label"] = label
                result["label_cn"] = DIAGNOSIS_LABELS_CN.get(label, "未知")
                result["confidence"] = round(confidence, 1)
                
                # 所有类别概率
                for i, prob in enumerate(probas):
                    lbl = DIAGNOSIS_LABELS.get(i, f"class_{i}")
                    result["probabilities"][lbl] = round(float(prob) * 100, 1)
            else:
                # 未训练时，基于简单规则给出初步诊断
                result = self._rule_based_diagnosis(machine_state, result)
                
        except Exception as e:
            logger.error(f"预测失败: {e}")
            # 回退到规则诊断
            result = self._rule_based_diagnosis(machine_state, result)
            
        return result
        
    def _rule_based_diagnosis(self, machine_state: Dict, result: Dict) -> Dict:
        """
        基于规则的简单诊断（当模型未训练时使用）
        """
        sensors = machine_state.get("sensors", {})
        power = machine_state.get("power", {})
        io_status = machine_state.get("io", {})
        
        temp = sensors.get("stackTemp", 25)
        voltage = power.get("stackVoltage", 0)
        current = power.get("stackCurrent", 0)
        fault_code = io_status.get("faultCode", 0)
        
        # 有故障码直接报故障
        if fault_code != 0:
            result["label"] = "thermal_issue"
            result["label_cn"] = "系统故障"
            result["confidence"] = 90.0
            return result
            
        # 温度过高
        if temp > 80:
            result["label"] = "thermal_issue"
            result["label_cn"] = "热管理异常"
            result["confidence"] = min(50 + (temp - 80) * 5, 95)
            return result
            
        # 电压异常低（可能膜干燥）
        if voltage > 0 and voltage < 20:
            result["label"] = "membrane_drying"
            result["label_cn"] = "膜干燥"
            result["confidence"] = 60.0
            return result
            
        # 正常
        result["label"] = "normal"
        result["label_cn"] = "正常"
        result["confidence"] = 70.0
        return result
        
    def add_feedback(self, machine_state: Dict, label: str) -> bool:
        """
        添加用户反馈标注，用于增量学习
        
        Args:
            machine_state: 当时的机器状态
            label: 用户标注的标签 ("normal", "flooding", "membrane_drying", "thermal_issue")
            
        Returns:
            是否成功添加
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            return False
            
        # 提取特征
        features = self.extract_features(machine_state)
        if features is None:
            return False
            
        # 标签转数字
        label_id = None
        for k, v in DIAGNOSIS_LABELS.items():
            if v == label:
                label_id = k
                break
                
        if label_id is None:
            logger.error(f"未知标签: {label}")
            return False
            
        try:
            # 更新标准化器（增量更新均值和方差）
            self._update_scaler(features)
            
            # 标准化
            X = features.reshape(1, -1)
            X_scaled = (X - self.scaler.mean_) / self.scaler.scale_
            y = np.array([label_id])
            
            # 增量训练
            all_classes = np.array(list(DIAGNOSIS_LABELS.keys()))
            self.model.partial_fit(X_scaled, y, classes=all_classes)
            
            self.is_trained = True
            self.sample_count += 1
            
            # 保存训练样本记录
            sample_record = {
                "timestamp": datetime.now().isoformat(),
                "features": features.tolist(),
                "label": label,
                "label_id": label_id
            }
            self.training_samples.append(sample_record)
            if len(self.training_samples) > 100:
                self.training_samples = self.training_samples[-100:]
                
            # 每10个样本保存一次模型
            if self.sample_count % 10 == 0:
                self.save_model()
                
            logger.info(f"✓ 增量学习完成, 标签: {label}, 样本总数: {self.sample_count}")
            return True
            
        except Exception as e:
            logger.error(f"增量学习失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _update_scaler(self, new_features: np.ndarray):
        """增量更新标准化器"""
        n = self.scaler.n_samples_seen_
        old_mean = self.scaler.mean_
        old_var = self.scaler.var_
        
        # Welford's online algorithm
        n_new = n + 1
        delta = new_features - old_mean
        new_mean = old_mean + delta / n_new
        delta2 = new_features - new_mean
        new_var = (old_var * n + delta * delta2) / n_new
        
        self.scaler.mean_ = new_mean
        self.scaler.var_ = new_var
        self.scaler.scale_ = np.sqrt(new_var + 1e-8)  # 防止除零
        self.scaler.n_samples_seen_ = n_new
        
    def save_model(self):
        """保存模型和标准化器"""
        if not SKLEARN_AVAILABLE or self.model is None:
            return
            
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # 保存训练数据记录
            with open(self.data_path, 'w') as f:
                json.dump({
                    "sample_count": self.sample_count,
                    "samples": self.training_samples,
                    "last_update": datetime.now().isoformat()
                }, f, indent=2)
                
            logger.info(f"✓ 模型已保存: {self.model_path}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            
    def get_status(self) -> Dict:
        """获取诊断器状态"""
        return {
            "available": SKLEARN_AVAILABLE,
            "is_trained": self.is_trained,
            "sample_count": self.sample_count,
            "model_path": self.model_path if os.path.exists(self.model_path) else None
        }
