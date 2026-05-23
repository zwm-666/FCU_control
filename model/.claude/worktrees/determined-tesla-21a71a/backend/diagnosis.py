"""
Online Fault Diagnosis Module for Hydrogen Fuel Cell System
Uses EnhancedMSTGAT (CEO-QAAdamW) deep learning model for fault classification
"""

import os
import json
import math
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ================== TensorFlow 导入 ==================
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TF 日志输出
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.layers import Dense, RepeatVector
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow 不可用，诊断功能将受限（仅规则诊断）")


# ================== 诊断标签定义 ==================
DIAGNOSIS_LABELS = {
    0: "flooding",         # 对应模型输出0
    1: "membrane_drying",  # 对应模型输出1
    2: "normal",           # 对应模型输出2
    3: "thermal_issue"     # 对应模型输出3
}

DIAGNOSIS_LABELS_CN = {
    "normal": "正常",
    "flooding": "水淹故障",
    "membrane_drying": "膜干燥",
    "thermal_issue": "热管理异常"
}

# 特征映射：CAN数据字段 -> 模型特征
# 基于 GBDT 自动选择出的 7 个重要特征：U_totV, T_Stack_inlet, PW, RH_H2, T_Air_inlet, T_3, m_Air
FEATURE_MAPPING = {
    "U_totV": "stackVoltage",            # 总电压
    "T_Stack_inlet": "stackTemp",        # 电堆入口温度
    "PW": "stackPower",                  # 功率
    "RH_H2": "h2Concentration",          # 氢气相对湿度 (用浓度代替)
    "T_Air_inlet": "airInletTemp",       # 空气入口温度 (缺失，默认30.4)
    "T_3": "ambientTemp",                # 温度3 (用环境温度代替)
    "m_Air": "airFlow"                   # 空气流量 (缺失，默认6.18)
}

FEATURE_ORDER = ["U_totV", "T_Stack_inlet", "PW", "RH_H2", "T_Air_inlet", "T_3", "m_Air"]


# ================== EnhancedMSTGAT 模型定义 ==================
# 从 ceo_qaadam_emstgat_trainer.py 提取的模型架构定义
# 加载 .keras 模型时需要这些自定义层

if TF_AVAILABLE:

    @dataclass
    class ModelConfig:
        """增强型MSTGAT模型配置"""
        input_shape: Tuple[int, ...]
        num_nodes: int
        num_classes: int
        hidden_units: int = 192
        attention_heads: int = 16
        dropout_rate: float = 0.206
        embedding_dim: int = 160
        max_sequence_length: int = 15
        use_batch_norm: bool = False
        activation: str = 'gelu'
        kernel_initializer: str = 'he_normal'
        graph_type: str = 'knn'
        knn_top_k: int = 6
        dilation_rate: int = 4

    class DilatedCausalConv(layers.Layer):
        """扩张因果卷积层"""

        def __init__(self, filters: int, kernel_size: int = 3, dilation_rate: int = 2,
                     dropout_rate: float = 0.2, use_batch_norm: bool = True, **kwargs):
            super().__init__(**kwargs)
            self.filters = filters
            self.kernel_size = kernel_size
            self.dilation_rate = dilation_rate
            self.dropout_rate = dropout_rate
            self.use_batch_norm_flag = use_batch_norm

            self.conv = layers.Conv1D(
                filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
                padding='causal', kernel_initializer='he_normal'
            )
            self.dropout = layers.Dropout(dropout_rate)
            if use_batch_norm:
                self.norm = layers.BatchNormalization()
            self.activation_layer = layers.Activation('gelu')
            self.use_skip = True

        def build(self, input_shape):
            if input_shape[-1] != self.conv.filters:
                self.use_skip = False
                self.proj = layers.Conv1D(self.conv.filters, 1)
            super().build(input_shape)

        def call(self, inputs, training=None):
            x = self.conv(inputs)
            if self.use_batch_norm_flag:
                x = self.norm(x, training=training)
            x = self.activation_layer(x)
            x = self.dropout(x, training=training)

            if self.use_skip:
                return x + inputs
            return x + self.proj(inputs)

        def get_config(self):
            config = super().get_config()
            config.update({
                'filters': self.filters, 'kernel_size': self.kernel_size,
                'dilation_rate': self.dilation_rate, 'dropout_rate': self.dropout_rate,
                'use_batch_norm': self.use_batch_norm_flag,
            })
            return config

    class KNNGraphBuilder(layers.Layer):
        """基于相似度的 KNN 图构建"""

        def __init__(self, top_k: int = 8, **kwargs):
            super().__init__(**kwargs)
            self.top_k = max(int(top_k), 1)

        @tf.function
        def call(self, inputs, training=None):
            B = tf.shape(inputs)[0]
            T = tf.shape(inputs)[1]

            x = tf.math.l2_normalize(inputs, axis=-1)
            sim = tf.matmul(x, x, transpose_b=True)

            k = tf.math.minimum(tf.cast(self.top_k, tf.int32), T)
            k = tf.math.maximum(k, 1)
            values, indices = tf.math.top_k(sim, k=k)

            neg_inf = tf.fill(tf.shape(sim), tf.constant(-1e9, dtype=sim.dtype))
            b_range = tf.range(B)[:, None, None]
            t_range = tf.range(T)[None, :, None]
            b_idx = tf.broadcast_to(b_range, tf.shape(indices))
            t_idx = tf.broadcast_to(t_range, tf.shape(indices))
            scatter_indices = tf.stack([b_idx, t_idx, indices], axis=-1)
            mask = tf.tensor_scatter_nd_update(neg_inf, tf.reshape(scatter_indices, [-1, 3]),
                                               tf.reshape(values, [-1]))

            adj = tf.nn.softmax(mask, axis=-1)
            return adj

        def get_config(self):
            config = super().get_config()
            config.update({'top_k': self.top_k})
            return config

    class MultiHeadAttention(layers.Layer):
        """多头注意力机制"""

        def __init__(self, num_heads: int = 8, hidden_units: int = 64,
                     dropout_rate: float = 0.2, activation: str = 'gelu', **kwargs):
            super().__init__(**kwargs)
            self.num_heads = num_heads
            self.hidden_units = hidden_units
            self.dropout_rate = dropout_rate
            self.activation = activation
            self.head_dim = hidden_units // num_heads

            self.query_dense = layers.Dense(hidden_units)
            self.key_dense = layers.Dense(hidden_units)
            self.value_dense = layers.Dense(hidden_units)
            self.combine_heads = layers.Dense(hidden_units)

            self.dropout = layers.Dropout(dropout_rate)
            self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

            self.ffn = tf.keras.Sequential([
                layers.Dense(hidden_units * 4, activation=activation),
                layers.Dropout(dropout_rate),
                layers.Dense(hidden_units)
            ])

        @tf.function
        def call(self, inputs, mask=None, training=None):
            batch_size = tf.shape(inputs)[0]

            query = self.query_dense(inputs)
            key = self.key_dense(inputs)
            value = self.value_dense(inputs)

            # Split heads
            query = tf.reshape(query, (batch_size, -1, self.num_heads, self.head_dim))
            query = tf.transpose(query, perm=[0, 2, 1, 3])
            key = tf.reshape(key, (batch_size, -1, self.num_heads, self.head_dim))
            key = tf.transpose(key, perm=[0, 2, 1, 3])
            value = tf.reshape(value, (batch_size, -1, self.num_heads, self.head_dim))
            value = tf.transpose(value, perm=[0, 2, 1, 3])

            # Attention
            logits = tf.matmul(query, key, transpose_b=True)
            scale = tf.cast(1.0 / tf.math.sqrt(float(self.head_dim)), logits.dtype)
            logits = logits * scale

            if mask is not None:
                logits += (mask * -1e9)

            attention_weights = tf.nn.softmax(logits, axis=-1)
            attention_weights = self.dropout(attention_weights, training=training)

            output = tf.matmul(attention_weights, value)
            output = tf.transpose(output, perm=[0, 2, 1, 3])
            output = tf.reshape(output, (batch_size, -1, self.num_heads * self.head_dim))

            attention_output = self.combine_heads(output)
            attention_output = self.layer_norm1(attention_output + inputs)

            ffn_output = self.ffn(attention_output)
            outputs = self.layer_norm2(ffn_output + attention_output)

            return outputs

        def get_config(self):
            config = super().get_config()
            config.update({
                'num_heads': self.num_heads, 'hidden_units': self.hidden_units,
                'dropout_rate': self.dropout_rate, 'activation': self.activation,
            })
            return config

    class EdgeAwareMultiHeadAttention(layers.Layer):
        """边感知多头注意力"""

        def __init__(self, num_heads: int = 8, hidden_units: int = 64,
                     dropout_rate: float = 0.2, activation: str = 'gelu', **kwargs):
            super().__init__(**kwargs)
            self.num_heads = num_heads
            self.hidden_units = hidden_units
            self.dropout_rate = dropout_rate
            self.activation = activation
            self.head_dim = hidden_units // num_heads

            self.query_dense = layers.Dense(hidden_units)
            self.key_dense = layers.Dense(hidden_units)
            self.value_dense = layers.Dense(hidden_units)
            self.out_dense = layers.Dense(hidden_units)

            self.dropout = layers.Dropout(dropout_rate)
            self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

            self.ffn = tf.keras.Sequential([
                layers.Dense(hidden_units * 4, activation=activation),
                layers.Dropout(dropout_rate),
                layers.Dense(hidden_units)
            ])

        @tf.function
        def call(self, inputs, adj_matrix, training=None):
            batch_size = tf.shape(inputs)[0]

            query = self.query_dense(inputs)
            key = self.key_dense(inputs)
            value = self.value_dense(inputs)

            # Split heads
            q = tf.reshape(query, (batch_size, -1, self.num_heads, self.head_dim))
            q = tf.transpose(q, perm=[0, 2, 1, 3])
            k = tf.reshape(key, (batch_size, -1, self.num_heads, self.head_dim))
            k = tf.transpose(k, perm=[0, 2, 1, 3])
            v = tf.reshape(value, (batch_size, -1, self.num_heads, self.head_dim))
            v = tf.transpose(v, perm=[0, 2, 1, 3])

            logits = tf.matmul(q, k, transpose_b=True)
            scale = tf.cast(1.0 / tf.math.sqrt(float(self.head_dim)), logits.dtype)
            logits = logits * scale

            # Edge bias from adjacency matrix
            adj_matrix_casted = tf.cast(adj_matrix, logits.dtype)
            eps = tf.constant(1e-9, dtype=logits.dtype)
            edge_bias = tf.math.log(adj_matrix_casted + eps)
            edge_bias = tf.expand_dims(edge_bias, axis=1)
            logits = logits + edge_bias

            attn = tf.nn.softmax(logits, axis=-1)
            attn = self.dropout(attn, training=training)
            attn_out = tf.matmul(attn, v)

            attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
            attn_out = tf.reshape(attn_out, (batch_size, -1, self.num_heads * self.head_dim))
            attn_out = self.out_dense(attn_out)

            x = self.layer_norm1(inputs + attn_out)
            ffn_out = self.ffn(x)
            out = self.layer_norm2(x + ffn_out)
            return out

        def get_config(self):
            config = super().get_config()
            config.update({
                'num_heads': self.num_heads, 'hidden_units': self.hidden_units,
                'dropout_rate': self.dropout_rate, 'activation': self.activation,
            })
            return config

    class EnhancedMSTGAT(Model):
        """增强型 MSTGAT 模型（完整版，与 ceo_qaadam_emstgat_trainer.py 一致）"""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config

            # 特征转换层
            self.feature_transform = Dense(
                config.hidden_units, activation=config.activation,
                kernel_initializer=config.kernel_initializer
            )

            # 序列化层
            self.sequence_layer = RepeatVector(config.max_sequence_length)

            # 位置编码
            self.positional_embeddings = self.add_weight(
                name="positional_embeddings",
                shape=(config.max_sequence_length, config.hidden_units),
                initializer="glorot_uniform", trainable=True,
            )

            # 因果卷积层
            self.causal_conv = DilatedCausalConv(
                filters=config.hidden_units, kernel_size=3, dilation_rate=config.dilation_rate,
                dropout_rate=config.dropout_rate, use_batch_norm=config.use_batch_norm
            )

            # 残差卷积层
            self.res_conv = DilatedCausalConv(
                filters=config.hidden_units, kernel_size=3, dilation_rate=config.dilation_rate,
                dropout_rate=config.dropout_rate, use_batch_norm=config.use_batch_norm
            )

            # 时序特征提取
            self.gru = layers.Bidirectional(layers.GRU(
                config.hidden_units // 2, return_sequences=True,
                dropout=config.dropout_rate, recurrent_dropout=0
            ))

            # 图构建器
            self.graph_builder = KNNGraphBuilder(top_k=config.knn_top_k)

            # 注意力机制
            self.attention = MultiHeadAttention(
                num_heads=config.attention_heads, hidden_units=config.hidden_units,
                dropout_rate=config.dropout_rate, activation=config.activation
            )
            self.edge_attention = EdgeAwareMultiHeadAttention(
                num_heads=config.attention_heads, hidden_units=config.hidden_units,
                dropout_rate=config.dropout_rate, activation=config.activation
            )

            # 额外注意力层
            self.attention_2 = MultiHeadAttention(
                num_heads=config.attention_heads, hidden_units=config.hidden_units,
                dropout_rate=config.dropout_rate, activation=config.activation
            )

            # 注意力池化
            self.attn_pool_score = tf.keras.Sequential([
                layers.Dense(config.hidden_units, activation=config.activation),
                layers.Dense(1)
            ])
            self.dropout = layers.Dropout(config.dropout_rate)

            # 分类头
            self.fc = layers.Dense(config.hidden_units, activation=config.activation,
                                   kernel_initializer=config.kernel_initializer)
            self.classifier = layers.Dense(config.num_classes, activation='softmax',
                                           kernel_initializer=config.kernel_initializer,
                                           dtype='float32')

        @tf.function
        def call(self, inputs, training=None):
            x = tf.cast(inputs, tf.float32)

            # 特征转换
            x = self.feature_transform(x)
            x = self.sequence_layer(x)

            # 位置编码
            pos = tf.expand_dims(self.positional_embeddings, axis=0)
            x = x + pos

            # 因果卷积
            x = self.causal_conv(x, training=training)

            # 残差卷积
            x = self.res_conv(x, training=training)

            # 时序特征提取
            x = self.gru(x)

            # 图结构
            adj_matrix = self.graph_builder(x, training=training)

            # 边感知注意力
            x = self.edge_attention(x, adj_matrix, training=training)

            # 第一层标准注意力
            x = self.attention(x, training=training)

            # 第二层标准注意力
            x = self.attention_2(x, training=training)

            # 注意力池化
            scores = self.attn_pool_score(x)
            weights = tf.nn.softmax(scores, axis=1)
            x = tf.reduce_sum(weights * x, axis=1)
            x = self.dropout(x, training=training)
            x = self.fc(x)

            # 分类
            outputs = self.classifier(x)
            return outputs

        def get_config(self):
            return {
                'input_shape': self.config.input_shape,
                'num_nodes': self.config.num_nodes,
                'num_classes': self.config.num_classes,
                'hidden_units': self.config.hidden_units,
                'attention_heads': self.config.attention_heads,
                'dropout_rate': self.config.dropout_rate,
                'max_sequence_length': self.config.max_sequence_length,
                'knn_top_k': self.config.knn_top_k,
            }

        @classmethod
        def from_config(cls, config):
            if 'input_shape' in config and isinstance(config['input_shape'], list):
                config['input_shape'] = tuple(config['input_shape'])
            # 兼容：如果包含不在 ModelConfig 里的键，可以过滤掉或者直接传入
            # 安全地实例化 ModelConfig
            valid_keys = ['input_shape', 'num_nodes', 'num_classes', 'hidden_units', 
                          'attention_heads', 'dropout_rate', 'embedding_dim', 
                          'max_sequence_length', 'use_batch_norm', 'activation', 
                          'kernel_initializer', 'graph_type', 'knn_top_k', 'dilation_rate']
            filtered_config = {k: v for k, v in config.items() if k in valid_keys}
            model_config = ModelConfig(**filtered_config)
            return cls(model_config)

    # 自定义对象注册表（用于 load_model）
    CUSTOM_OBJECTS = {
        'DilatedCausalConv': DilatedCausalConv,
        'KNNGraphBuilder': KNNGraphBuilder,
        'MultiHeadAttention': MultiHeadAttention,
        'EdgeAwareMultiHeadAttention': EdgeAwareMultiHeadAttention,
        'EnhancedMSTGAT': EnhancedMSTGAT,
    }


# ================== 在线诊断器 ==================
class OnlineDiagnosis:
    """
    在线故障诊断器（基于 EnhancedMSTGAT 深度学习模型）
    - 加载预训练的 .keras 模型进行推理
    - 自动特征标准化
    - 当模型不可用时回退到规则诊断
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "ceo_qaadam_emstgat.keras")
        self.data_path = os.path.join(model_dir, "feedback_data.json")

        # 确保目录存在
        os.makedirs(model_dir, exist_ok=True)

        # 初始化模型
        self.model = None
        self.is_trained = False
        self.sample_count = 0
        self.feedback_samples: List[Dict] = []
        self._feedback_lock = threading.Lock()
        self._predict_interval = 0.5  # seconds between predictions
        self._last_predict_time = 0.0

        # 标准化参数（基于训练集通过 StandardScaler 拟合的真实结果）
        # 特征顺序: ['U_totV', 'T_Stack_inlet', 'PW', 'RH_H2', 'T_Air_inlet', 'T_3', 'm_Air']
        self.scaler_mean = np.array([
            0.54341371, 34.80577512, 7.52547566, 94.05782605, 
            30.39686013, 28.85840481, 6.1809033
        ])
        self.scaler_scale = np.array([
            0.132746038, 9.58184703, 4.40340010, 4.09520911, 
            0.417523187, 6.35361892, 0.0242430148
        ])

        # 加载模型
        self._initialize()

    def _initialize(self):
        """初始化：加载预训练的 EnhancedMSTGAT 模型"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow 不可用，诊断仅使用规则模式")
            return

        if not os.path.exists(self.model_path):
            logger.warning(f"模型文件不存在: {self.model_path}，诊断仅使用规则模式")
            return

        try:
            logger.info(f"正在加载 EnhancedMSTGAT 模型: {self.model_path}")

            # 使用自定义对象注册表完整加载模型
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=CUSTOM_OBJECTS,
                compile=False  # 推理模式不需要编译
            )

            self.is_trained = True
            logger.info(f"✓ EnhancedMSTGAT 模型加载成功 (参数量: {self.model.count_params():,})")

            # 加载反馈数据计数
            if os.path.exists(self.data_path):
                try:
                    with open(self.data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.sample_count = data.get("sample_count", 0)
                        self.feedback_samples = data.get("samples", [])[-100:]
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"加载 EnhancedMSTGAT 模型失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.is_trained = False

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
                power.get("stackVoltage", 0.0),           # U_totV
                sensors.get("stackTemp", 25.0),           # T_Stack_inlet
                power.get("stackPower", 0.0),             # PW
                sensors.get("h2Concentration", 0.0),      # RH_H2
                sensors.get("airInletTemp", 30.4),        # T_Air_inlet (定值替代默认 30.4)
                sensors.get("ambientTemp", 25.0),         # T_3 (环境温度)
                sensors.get("airFlow", 6.18)              # m_Air (定值替代默认 6.18)
            ]

            return np.array(features, dtype=np.float32)

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

        now = datetime.now().timestamp()
        if now - self._last_predict_time < self._predict_interval:
            if hasattr(self, '_last_result'):
                return self._last_result
            return result
        self._last_predict_time = now

        if not TF_AVAILABLE or self.model is None:
            # 回退到规则诊断
            return self._rule_based_diagnosis(machine_state, result)

        # 提取特征
        features = self.extract_features(machine_state)
        if features is None:
            return result

        try:
            # 标准化
            X = features.reshape(1, -1)
            X_scaled = (X - self.scaler_mean) / self.scaler_scale

            # 模型推理（TensorFlow 模型预测）
            probas = self.model(X_scaled, training=False)
            probas = probas.numpy()[0]

            pred_class = int(np.argmax(probas))
            confidence = float(probas[pred_class]) * 100

            label = DIAGNOSIS_LABELS.get(pred_class, "normal")
            result["label"] = label
            result["label_cn"] = DIAGNOSIS_LABELS_CN.get(label, "未知")
            result["confidence"] = round(confidence, 1)

            # 所有类别概率
            for i, prob in enumerate(probas):
                lbl = DIAGNOSIS_LABELS.get(i, f"class_{i}")
                result["probabilities"][lbl] = round(float(prob) * 100, 1)

        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            result = self._rule_based_diagnosis(machine_state, result)

        self._last_result = result
        return result

    def _rule_based_diagnosis(self, machine_state: Dict, result: Dict) -> Dict:
        """
        基于规则的简单诊断（当模型不可用时使用）
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
        添加用户反馈标注（记录到文件，供未来重新训练使用）

        注意：EnhancedMSTGAT 不支持在线增量学习，
        反馈数据仅保存到文件，供离线重新训练时使用。

        Args:
            machine_state: 当时的机器状态
            label: 用户标注的标签

        Returns:
            是否成功添加
        """
        # 提取特征
        features = self.extract_features(machine_state)
        if features is None:
            return False

        # 标签验证
        valid_labels = set(DIAGNOSIS_LABELS.values())
        if label not in valid_labels:
            logger.error(f"未知标签: {label}")
            return False

        try:
            # 保存反馈样本记录
            sample_record = {
                "timestamp": datetime.now().isoformat(),
                "features": features.tolist(),
                "label": label,
                "machine_state_snapshot": {
                    "power": machine_state.get("power", {}),
                    "sensors": machine_state.get("sensors", {}),
                }
            }
            self.feedback_samples.append(sample_record)
            if len(self.feedback_samples) > 100:
                self.feedback_samples = self.feedback_samples[-100:]

            self.sample_count += 1

            # 每10个样本保存一次
            if self.sample_count % 10 == 0:
                self._save_feedback()

            logger.info(f"✓ 反馈已记录: 标签={label}, 累计样本={self.sample_count}")
            return True

        except Exception as e:
            logger.error(f"保存反馈失败: {e}")
            return False

    def _save_feedback(self):
        """保存反馈数据到文件（线程安全）"""
        with self._feedback_lock:
            try:
                with open(self.data_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "sample_count": self.sample_count,
                        "samples": self.feedback_samples,
                        "last_update": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"✓ 反馈数据已保存: {self.data_path}")
            except Exception as e:
                logger.error(f"保存反馈数据失败: {e}")

    def save_model(self):
        """保存反馈数据（兼容旧接口）"""
        self._save_feedback()

    def get_status(self) -> Dict:
        """获取诊断器状态"""
        return {
            "available": TF_AVAILABLE,
            "is_trained": self.is_trained,
            "sample_count": self.sample_count,
            "model_type": "EnhancedMSTGAT (CEO-QAAdamW)",
            "model_path": self.model_path if os.path.exists(self.model_path) else None
        }
