# CEO-QAAdamW-EnhancedMSTGAT 训练器
# 用于燃料电池状态分类

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================== GPU 配置（必须在导入 TensorFlow 前设置环境变量）==================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 减少 TF 日志输出

import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, RepeatVector

from core.preprocess_utils import (
    prepare_training_dataframe,
    save_preprocess_meta,
    train_preprocess_and_select,
    train_preprocess_and_select_from_df,
)
from scripts.paper_figure_artifacts import (
    build_training_figure_artifacts,
    save_training_figure_artifacts,
)

# ================== 日志配置 ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ================== GPU 配置模块 ==================
def setup_gpu(use_mixed_precision: bool = True, use_xla: bool = True) -> Dict[str, Any]:
    """配置 GPU 加速（针对 Kaggle 环境优化）

    Args:
        use_mixed_precision: 是否启用混合精度训练（推荐对于 P100/T4/V100）
        use_xla: 是否启用 XLA JIT 编译

    Returns:
        GPU 配置信息字典
    """
    gpu_info = {
        "gpus_available": [],
        "mixed_precision": False,
        "xla_enabled": False,
        "memory_growth": False,
    }

    # 检测可用 GPU
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        logger.info(f"✅ 检测到 {len(gpus)} 个 GPU:")
        for i, gpu in enumerate(gpus):
            logger.info(f"   GPU {i}: {gpu.name}")
            gpu_info["gpus_available"].append(gpu.name)

            # 启用内存增长（避免一次性占满显存）
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                gpu_info["memory_growth"] = True
            except RuntimeError as e:
                logger.warning(f"   无法设置内存增长: {e}")

        # 启用混合精度训练（float16）
        if use_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                gpu_info["mixed_precision"] = True
                logger.info(f"✅ 已启用混合精度训练 (mixed_float16)")
            except Exception as e:
                logger.warning(f"⚠️ 无法启用混合精度: {e}")

        # 启用 XLA JIT 编译
        if use_xla:
            try:
                tf.config.optimizer.set_jit(True)
                gpu_info["xla_enabled"] = True
                logger.info(f"✅ 已启用 XLA JIT 编译")
            except Exception as e:
                logger.warning(f"⚠️ 无法启用 XLA: {e}")
    else:
        logger.warning("⚠️ 未检测到 GPU，将使用 CPU 训练")

    return gpu_info


def get_gpu_memory_info() -> str:
    """获取 GPU 显存使用信息"""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # 尝试使用 nvidia-smi（Kaggle 环境可用）
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                used, total = result.stdout.strip().split(",")
                return f"{used.strip()}MB / {total.strip()}MB"
    except Exception:
        pass
    return "N/A"


# ================== 模型配置 ==================
@dataclass
class ModelConfig:
    """增强型MSTGAT模型配置（基于最优超参数）"""

    input_shape: Tuple[int, ...]
    num_nodes: int
    num_classes: int
    hidden_units: int = 192  # 最优: 192
    attention_heads: int = 16  # 最优: 16 (之前是8)
    dropout_rate: float = 0.206  # 最优: 0.20559
    embedding_dim: int = 160  # 最优: 160 (之前是16)
    max_sequence_length: int = 15  # 最优: 15 (之前是100)
    use_batch_norm: bool = False  # 最优: False (之前是True)
    activation: str = "gelu"
    kernel_initializer: str = "he_normal"
    graph_type: str = "knn"
    knn_top_k: int = 6  # 最优: 6 (之前是8)
    dilation_rate: int = 4  # 最优: 4 (之前是2)


@dataclass
class TrainingConfig:
    """训练配置"""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    patience: int = 15
    seed: int = 42


# ================== 自定义层 ==================
class DilatedCausalConv(layers.Layer):
    """扩张因果卷积层"""

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rate: int = 2,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.use_batch_norm_flag = use_batch_norm

        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            kernel_initializer="he_normal",
        )
        self.dropout = layers.Dropout(dropout_rate)
        if use_batch_norm:
            self.norm = layers.BatchNormalization()
        self.activation_layer = layers.Activation("gelu")
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
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm_flag,
            }
        )
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
        mask = tf.tensor_scatter_nd_update(
            neg_inf, tf.reshape(scatter_indices, [-1, 3]), tf.reshape(values, [-1])
        )

        adj = tf.nn.softmax(mask, axis=-1)
        return adj

    def get_config(self):
        config = super().get_config()
        config.update({"top_k": self.top_k})
        return config


class MultiHeadAttention(layers.Layer):
    """多头注意力机制"""

    def __init__(
        self,
        num_heads: int = 8,
        hidden_units: int = 64,
        dropout_rate: float = 0.2,
        activation: str = "gelu",
        **kwargs,
    ):
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

        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(hidden_units * 4, activation=activation),
                layers.Dropout(dropout_rate),
                layers.Dense(hidden_units),
            ]
        )

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
        # 使用固定的缩放因子，避免混合精度类型问题
        scale = tf.cast(1.0 / tf.math.sqrt(float(self.head_dim)), logits.dtype)
        logits = logits * scale

        if mask is not None:
            logits += mask * -1e9

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
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_units": self.hidden_units,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
            }
        )
        return config


class EdgeAwareMultiHeadAttention(layers.Layer):
    """边感知多头注意力"""

    def __init__(
        self,
        num_heads: int = 8,
        hidden_units: int = 64,
        dropout_rate: float = 0.2,
        activation: str = "gelu",
        **kwargs,
    ):
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

        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(hidden_units * 4, activation=activation),
                layers.Dropout(dropout_rate),
                layers.Dense(hidden_units),
            ]
        )

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
        # 使用固定的缩放因子，避免混合精度类型问题
        scale = tf.cast(1.0 / tf.math.sqrt(float(self.head_dim)), logits.dtype)
        logits = logits * scale

        # Edge bias from adjacency matrix (确保类型与logits一致)
        adj_matrix_casted = tf.cast(adj_matrix, logits.dtype)
        eps = tf.constant(1e-9, dtype=logits.dtype)
        edge_bias = tf.math.log(adj_matrix_casted + eps)
        edge_bias = tf.expand_dims(edge_bias, axis=1)
        logits = logits + edge_bias

        attn = tf.nn.softmax(logits, axis=-1)
        attn = self.dropout(attn, training=training)
        attn_out = tf.matmul(attn, v)

        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
        attn_out = tf.reshape(
            attn_out, (batch_size, -1, self.num_heads * self.head_dim)
        )
        attn_out = self.out_dense(attn_out)

        x = self.layer_norm1(inputs + attn_out)
        ffn_out = self.ffn(x)
        out = self.layer_norm2(x + ffn_out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_units": self.hidden_units,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
            }
        )
        return config


# ================== EnhancedMSTGAT 模型 ==================
class EnhancedMSTGAT(Model):
    """增强型 MSTGAT 模型（完整版，与 enhanced_mstgat_with_ceo.py 一致）"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 特征转换层
        self.feature_transform = Dense(
            config.hidden_units,
            activation=config.activation,
            kernel_initializer=config.kernel_initializer,
        )

        # 序列化层
        self.sequence_layer = RepeatVector(config.max_sequence_length)

        # 位置编码
        self.positional_embeddings = self.add_weight(
            name="positional_embeddings",
            shape=(config.max_sequence_length, config.hidden_units),
            initializer="glorot_uniform",
            trainable=True,
        )

        # 因果卷积层
        self.causal_conv = DilatedCausalConv(
            filters=config.hidden_units,
            kernel_size=3,
            dilation_rate=config.dilation_rate,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
        )

        # 残差卷积层（原版有此层）
        self.res_conv = DilatedCausalConv(
            filters=config.hidden_units,
            kernel_size=3,
            dilation_rate=config.dilation_rate,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
        )

        # 时序特征提取
        self.gru = layers.Bidirectional(
            layers.GRU(
                config.hidden_units // 2,
                return_sequences=True,
                dropout=config.dropout_rate,
                recurrent_dropout=0,
            )
        )

        # 图构建器
        self.graph_builder = KNNGraphBuilder(top_k=config.knn_top_k)

        # 注意力机制
        self.attention = MultiHeadAttention(
            num_heads=config.attention_heads,
            hidden_units=config.hidden_units,
            dropout_rate=config.dropout_rate,
            activation=config.activation,
        )
        self.edge_attention = EdgeAwareMultiHeadAttention(
            num_heads=config.attention_heads,
            hidden_units=config.hidden_units,
            dropout_rate=config.dropout_rate,
            activation=config.activation,
        )

        # 额外注意力层（原版有此层）
        self.attention_2 = MultiHeadAttention(
            num_heads=config.attention_heads,
            hidden_units=config.hidden_units,
            dropout_rate=config.dropout_rate,
            activation=config.activation,
        )

        # 注意力池化
        self.attn_pool_score = tf.keras.Sequential(
            [
                layers.Dense(config.hidden_units, activation=config.activation),
                layers.Dense(1),
            ]
        )
        self.dropout = layers.Dropout(config.dropout_rate)

        # 分类头（使用 float32 确保混合精度训练稳定性）
        self.fc = layers.Dense(
            config.hidden_units,
            activation=config.activation,
            kernel_initializer=config.kernel_initializer,
        )
        self.classifier = layers.Dense(
            config.num_classes,
            activation="softmax",
            kernel_initializer=config.kernel_initializer,
            dtype="float32",
        )

    @tf.function
    def call(self, inputs, training=None):
        # 确保输入是 float32
        x = tf.cast(inputs, tf.float32)

        # 特征转换
        x = self.feature_transform(x)
        x = self.sequence_layer(x)

        # 位置编码
        pos = tf.expand_dims(self.positional_embeddings, axis=0)
        x = x + pos

        # 因果卷积
        x = self.causal_conv(x, training=training)

        # 残差卷积（新增）
        x = self.res_conv(x, training=training)

        # 时序特征提取
        x = self.gru(x)

        # 图结构
        adj_matrix = self.graph_builder(x, training=training)

        # 边感知注意力
        x = self.edge_attention(x, adj_matrix, training=training)

        # 第一层标准注意力
        x = self.attention(x, training=training)

        # 第二层标准注意力（新增）
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
            "input_shape": self.config.input_shape,
            "num_nodes": self.config.num_nodes,
            "num_classes": self.config.num_classes,
            "hidden_units": self.config.hidden_units,
            "attention_heads": self.config.attention_heads,
            "dropout_rate": self.config.dropout_rate,
            "embedding_dim": self.config.embedding_dim,
            "max_sequence_length": self.config.max_sequence_length,
            "use_batch_norm": self.config.use_batch_norm,
            "activation": self.config.activation,
            "kernel_initializer": self.config.kernel_initializer,
            "graph_type": self.config.graph_type,
            "knn_top_k": self.config.knn_top_k,
            "dilation_rate": self.config.dilation_rate,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if "input_shape" in config and isinstance(config["input_shape"], list):
            config["input_shape"] = tuple(config["input_shape"])
        return cls(ModelConfig(**config))


# ================== 优化器：QAAdamW_Lite ==================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """带 Warmup 的余弦衰减学习率调度

    修复：添加边界保护，确保学习率不会异常衰减
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = float(max(total_steps, 1))
        self.warmup_steps = float(max(warmup_steps, 1))
        self.min_lr = self.base_lr * min_lr_ratio  # 最小学习率，防止衰减到0

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Warmup 阶段
        warmup_lr = self.base_lr * (step / self.warmup_steps)

        # Cosine 衰减阶段
        decay_steps = self.total_steps - self.warmup_steps
        decay_step = step - self.warmup_steps
        # 确保 progress 在 [0, 1] 范围内
        progress = tf.minimum(decay_step / tf.maximum(decay_steps, 1.0), 1.0)
        progress = tf.maximum(progress, 0.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * progress))
        decayed_lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        # 选择阶段
        lr = tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)
        # 确保最小学习率
        lr = tf.maximum(lr, self.min_lr)
        return lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "min_lr": self.min_lr,
        }


class QAAdamW_Lite(tf.keras.optimizers.Optimizer):
    """轻量级 QAAdamW 优化器（AdaBelief + 权重衰减 + 噪声注入）

    注意：为兼容 CEO 超参数搜索中不同 trial 的模型形状变化，
    本优化器使用字典而非列表存储累积变量。
    """

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        initial_noise_factor=0.02,
        use_adabelief=True,
        total_steps=10000,
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self._weight_decay = weight_decay
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._initial_noise_factor = initial_noise_factor
        self._use_adabelief = use_adabelief
        self._total_steps = total_steps
        # 使用字典存储累积变量，键为变量引用ID
        self._m_dict = {}
        self._v_dict = {}

    def build(self, var_list):
        super().build(var_list)
        # 每次 build 都根据当前 var_list 创建/更新累积变量
        for var in var_list:
            var_key = id(var)
            if var_key not in self._m_dict:
                self._m_dict[var_key] = self.add_variable_from_reference(var, "m")
                self._v_dict[var_key] = self.add_variable_from_reference(var, "v")

    def update_step(self, gradient, variable, learning_rate):
        # 处理稀疏梯度 (IndexedSlices)
        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.convert_to_tensor(gradient)

        var_key = id(variable)

        # 动态获取或创建累积变量（处理运行时新增的变量）
        if var_key not in self._m_dict:
            self._m_dict[var_key] = self.add_variable_from_reference(variable, "m")
            self._v_dict[var_key] = self.add_variable_from_reference(variable, "v")

        m = self._m_dict[var_key]
        v = self._v_dict[var_key]

        lr = tf.cast(learning_rate, variable.dtype)
        step = tf.cast(self.iterations + 1, variable.dtype)
        beta1 = tf.cast(self._beta_1, variable.dtype)
        beta2 = tf.cast(self._beta_2, variable.dtype)
        eps = tf.cast(self._epsilon, variable.dtype)

        # 添加噪声（逐渐衰减）
        noise_factor = self._initial_noise_factor * tf.maximum(
            1.0 - step / self._total_steps, 0.0
        )
        noise = tf.random.normal(
            tf.shape(gradient), stddev=noise_factor, dtype=gradient.dtype
        )
        gradient = gradient + noise

        # AdaBelief or Adam
        m.assign(beta1 * m + (1 - beta1) * gradient)
        if self._use_adabelief:
            v.assign(beta2 * v + (1 - beta2) * tf.square(gradient - m))
        else:
            v.assign(beta2 * v + (1 - beta2) * tf.square(gradient))

        m_hat = m / (1 - tf.pow(beta1, step))
        v_hat = v / (1 - tf.pow(beta2, step))

        # 更新变量
        update = lr * m_hat / (tf.sqrt(v_hat) + eps)

        # 权重衰减
        if self._weight_decay > 0:
            variable.assign_sub(lr * self._weight_decay * variable)

        variable.assign_sub(update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "weight_decay": self._weight_decay,
                "beta_1": self._beta_1,
                "beta_2": self._beta_2,
                "epsilon": self._epsilon,
                "initial_noise_factor": self._initial_noise_factor,
                "use_adabelief": self._use_adabelief,
                "total_steps": self._total_steps,
            }
        )
        return config


# 模型加载时所需的自定义对象注册表
CUSTOM_OBJECTS = {
    "DilatedCausalConv": DilatedCausalConv,
    "KNNGraphBuilder": KNNGraphBuilder,
    "MultiHeadAttention": MultiHeadAttention,
    "EdgeAwareMultiHeadAttention": EdgeAwareMultiHeadAttention,
    "EnhancedMSTGAT": EnhancedMSTGAT,
    "WarmupCosineDecay": WarmupCosineDecay,
    "QAAdamW_Lite": QAAdamW_Lite,
}


# ================== 数据预处理函数 ==================
def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """兼容旧逻辑：使用当前数据本身填充缺失值。"""
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"检测到 {missing_before} 个缺失值")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode = df[col].mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else ""
                    df[col] = df[col].fillna(fill_value)
        logger.info(f"缺失值处理完成，剩余 {df.isnull().sum().sum()} 个缺失值")
    return df


def _fit_missing_values(
    train_df: pd.DataFrame, label_col: Optional[str] = None
) -> Dict[str, Any]:
    """仅基于训练集拟合缺失值填充值。"""
    fill_values = {}
    for col in train_df.columns:
        if label_col and col == label_col:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            fill_values[col] = float(train_df[col].median())
        else:
            mode = train_df[col].mode(dropna=True)
            fill_values[col] = mode.iloc[0] if not mode.empty else ""
    return fill_values


def _apply_missing_values(
    df: pd.DataFrame, fill_values: Dict[str, Any]
) -> pd.DataFrame:
    """将训练集拟合出的缺失值规则应用到任意数据集。"""
    df = df.copy()
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"检测到 {missing_before} 个缺失值")
    for col, fill_value in fill_values.items():
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(fill_value)
    if missing_before > 0:
        logger.info(f"缺失值处理完成，剩余 {df.isnull().sum().sum()} 个缺失值")
    return df


def _remove_outliers(
    df: pd.DataFrame, label_col: Optional[str] = None, threshold: float = 3.0
) -> pd.DataFrame:
    """兼容旧逻辑：基于当前数据自身裁剪异常值。"""
    bounds = _fit_outlier_bounds(df, label_col=label_col, threshold=threshold)
    return _apply_outlier_bounds(df, bounds)


def _fit_outlier_bounds(
    train_df: pd.DataFrame, label_col: Optional[str] = None, threshold: float = 3.0
) -> Dict[str, Tuple[float, float]]:
    """仅基于训练集拟合异常值裁剪边界。"""
    bounds = {}
    numeric_cols = train_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if label_col and col == label_col:
            continue
        mean = float(train_df[col].mean())
        std = float(train_df[col].std())
        if np.isnan(std) or std < 1e-8:
            bounds[col] = (mean, mean)
        else:
            margin = threshold * std
            bounds[col] = (mean - margin, mean + margin)
    return bounds


def _apply_outlier_bounds(
    df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """将训练集拟合出的异常值边界应用到任意数据集。"""
    df = df.copy()
    outliers_count = 0
    for col, (lower, upper) in bounds.items():
        if col not in df.columns:
            continue
        clipped = df[col].clip(lower=lower, upper=upper)
        outliers_count += int((clipped != df[col]).sum())
        df[col] = clipped
    if outliers_count > 0:
        logger.info(f"处理异常值 {outliers_count} 个")
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """移除重复行。"""
    dup = int(df.duplicated().sum())
    if dup > 0:
        logger.info(f"检测到 {dup} 行重复数据，已移除")
        df = df.drop_duplicates(keep="first")
    return df


# ================== 数据加载（支持 CSV / Excel）==================
def _load_tabular_data(data_path: str) -> pd.DataFrame:
    """按扩展名读取 CSV 或 Excel 数据。"""
    file_ext = os.path.splitext(data_path)[1].lower()
    if file_ext == ".csv":
        return pd.read_csv(data_path)
    if file_ext in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)
    raise ValueError(f"不支持的数据文件格式: {file_ext}")


def _get_class_names(raw_labels: np.ndarray, encoded_labels: np.ndarray) -> List[str]:
    """返回用于展示的真实类别名。"""
    class_names = []
    for class_id in np.unique(encoded_labels):
        first_idx = int(np.where(encoded_labels == class_id)[0][0])
        class_names.append(str(raw_labels[first_idx]))
    return class_names


def _normalize_feature_name(name: str) -> str:
    """标准化特征名，便于做中英文关键词识别。"""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(name).strip().lower())


def _identify_power_and_current_columns(
    feature_names: List[str],
) -> Tuple[List[str], List[str]]:
    """识别功率列与电流列。"""
    power_columns: List[str] = []
    current_columns: List[str] = []

    power_keywords = ("功率", "power", "stackpower", "pw")
    current_keywords = ("电流", "current", "stackcurrent", "iwrite", "currenta")
    current_exclude_keywords = ("aircurrent", "currentlimit")

    for feature_name in feature_names:
        normalized = _normalize_feature_name(feature_name)
        if any(keyword in normalized for keyword in power_keywords):
            power_columns.append(feature_name)
            continue

        if any(keyword in normalized for keyword in current_keywords) and not any(
            keyword in normalized for keyword in current_exclude_keywords
        ):
            current_columns.append(feature_name)

    return power_columns, current_columns


def _assert_feature_policy(feature_names: List[str]):
    """校验训练输入特征中不含功率且至少保留一个电流列。"""
    power_columns, current_columns = _identify_power_and_current_columns(feature_names)
    if power_columns:
        raise ValueError(f"训练输入中不能包含功率列: {power_columns}")
    if not current_columns:
        raise ValueError("训练输入中必须至少保留一个电流列")


def _ensure_current_columns_selected(
    feature_names: List[str],
    selected_idx: np.ndarray,
    importances: np.ndarray,
    current_columns: List[str],
) -> np.ndarray:
    """若重要性筛选后没有电流列，则按重要性补回所有电流列。"""
    if not current_columns:
        return np.asarray(selected_idx, dtype=int)

    current_idx = [
        idx for idx, name in enumerate(feature_names) if name in current_columns
    ]
    selected_set = {int(idx) for idx in np.asarray(selected_idx, dtype=int).tolist()}

    if any(idx in selected_set for idx in current_idx):
        return np.asarray(
            sorted(selected_set, key=lambda idx: (-importances[idx], idx)), dtype=int
        )

    selected_set.update(current_idx)
    logger.warning(f"重要性筛选未保留电流列，已强制补回: {current_columns}")
    return np.asarray(
        sorted(selected_set, key=lambda idx: (-importances[idx], idx)), dtype=int
    )


def load_and_preprocess_data(
    data_path: str,
    test_size: float = 0.2,
    seed: int = 42,
    prepared_df: Optional["pd.DataFrame"] = None,
    label_col: Optional[str] = None,
):
    """加载并预处理数据（委托给 preprocess_utils 公共模块）。"""
    logger.info(f"加载数据: {data_path}")
    if prepared_df is None:
        data = train_preprocess_and_select(
            data_path=data_path,
            test_size=test_size,
            seed=seed,
            label_col=label_col,
        )
    else:
        data = train_preprocess_and_select_from_df(
            df=prepared_df,
            test_size=test_size,
            seed=seed,
            label_col=label_col,
        )

    logger.info(
        f"选择 {len(data['feature_names'])} 个特征，类别数: {data['num_classes']}"
    )
    logger.info(f"最终选中特征: {data['feature_names']}")

    removed_power = data.get("removed_power_columns", [])
    retained_current = data.get("retained_current_columns", [])
    if removed_power:
        logger.info(f"已移除功率列: {removed_power}")
    if retained_current:
        logger.info(f"保留电流列: {retained_current}")

    return data


def create_datasets(X_train, y_train, X_val, y_val, batch_size: int = 32):
    """创建 TensorFlow 数据集"""
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (tf.cast(X_train, tf.float32), tf.cast(y_train, tf.int32))
        )
        .cache()
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (tf.cast(X_val, tf.float32), tf.cast(y_val, tf.int32))
        )
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


# ================== 可视化模块 ==================
PLOT_FONT_FAMILY = "Times New Roman"
PLOT_CJK_FONT = "SimHei"  # 中文回退字体（黑体）


def _apply_plot_font():
    """统一设置图表字体（支持中英文混合显示）。"""
    plt.rcParams["font.sans-serif"] = [PLOT_CJK_FONT, PLOT_FONT_FAMILY, "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_history(history: Dict, save_path: str = None):
    """绘制训练历史曲线"""
    _apply_plot_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history["loss"], label="Train Loss", color="#1f77b4")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss", color="#ff7f0e")
    axes[0].set_xlabel("Epoch", fontname=PLOT_FONT_FAMILY)
    axes[0].set_ylabel("Loss", fontname=PLOT_FONT_FAMILY)
    axes[0].set_title("Training & Validation Loss", fontname=PLOT_FONT_FAMILY)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(history["accuracy"], label="Train Acc", color="#1f77b4")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="Val Acc", color="#ff7f0e")
    axes[1].set_xlabel("Epoch", fontname=PLOT_FONT_FAMILY)
    axes[1].set_ylabel("Accuracy", fontname=PLOT_FONT_FAMILY)
    axes[1].set_title("Training & Validation Accuracy", fontname=PLOT_FONT_FAMILY)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"训练曲线已保存: {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: List[str] = None,
    save_path: str = None,
    normalize: bool = False,
):
    """绘制混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_to_plot = cm.astype(np.float64) / row_sums
    else:
        cm_to_plot = cm

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    _apply_plot_font()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_to_plot, interpolation="nearest", cmap="Blues")
    ax.set_title(
        "Normalized Confusion Matrix" if normalize else "Confusion Matrix",
        fontsize=14,
        fontweight="bold",
        fontname=PLOT_FONT_FAMILY,
    )

    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontname=PLOT_FONT_FAMILY)
    ax.set_yticklabels(class_names, fontname=PLOT_FONT_FAMILY)
    ax.set_xlabel("Predicted Label", fontsize=12, fontname=PLOT_FONT_FAMILY)
    ax.set_ylabel("True Label", fontsize=12, fontname=PLOT_FONT_FAMILY)

    thresh = cm_to_plot.max() / 2.0 if cm_to_plot.size else 0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            label = f"{cm_to_plot[i, j]:.2f}" if normalize else format(cm[i, j], "d")
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if cm_to_plot[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"混淆矩阵已保存: {save_path}")
    plt.close()


def plot_per_class_metrics(
    metrics: Dict[str, np.ndarray], class_names: List[str], save_path: str = None
):
    """绘制各类别 Precision / Recall / F1 指标。"""
    _apply_plot_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, metrics["precision"], width=width, label="Precision")
    ax.bar(x, metrics["recall"], width=width, label="Recall")
    ax.bar(x + width, metrics["f1_score"], width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontname=PLOT_FONT_FAMILY)
    ax.set_xlabel("Class", fontsize=12, fontname=PLOT_FONT_FAMILY)
    ax.set_ylabel("Score", fontsize=12, fontname=PLOT_FONT_FAMILY)
    ax.set_title("Per-class Precision / Recall / F1", fontname=PLOT_FONT_FAMILY)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"各类别指标图已保存: {save_path}")
    plt.close()


def plot_split_metrics_comparison(
    split_results: List[Dict[str, Any]], save_path: str = None
):
    """绘制不同测试比例下的总体指标对比图。"""
    _apply_plot_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    split_labels = [f"{int(round(result['test_size'] * 100))}%" for result in split_results]
    x = np.arange(len(split_labels))
    ax.bar(x - 0.3, [result["accuracy"] for result in split_results], width=0.2, label="Accuracy")
    ax.bar(x - 0.1, [result["precision"] for result in split_results], width=0.2, label="Precision")
    ax.bar(x + 0.1, [result["recall"] for result in split_results], width=0.2, label="Recall")
    ax.bar(x + 0.3, [result["f1_score"] for result in split_results], width=0.2, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontname=PLOT_FONT_FAMILY)
    ax.set_xlabel("Test Split", fontsize=12, fontname=PLOT_FONT_FAMILY)
    ax.set_ylabel("Score", fontsize=12, fontname=PLOT_FONT_FAMILY)
    ax.set_title("Metrics Comparison Across Test Splits", fontname=PLOT_FONT_FAMILY)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"多比例指标对比图已保存: {save_path}")
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15,
    save_path: str = None,
):
    """绘制特征重要性条形图"""
    # 排序
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    _apply_plot_font()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_names)), sorted_importances, color="#2196F3")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12, fontname=PLOT_FONT_FAMILY)
    ax.set_title(
        f"Top {top_n} Feature Importance",
        fontsize=14,
        fontweight="bold",
        fontname=PLOT_FONT_FAMILY,
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"特征重要性图已保存: {save_path}")
    plt.close()


SHAP_BACKGROUND_SAMPLE_SIZE = 128
SHAP_EXPLANATION_SAMPLE_SIZE = 64
SHAP_KERNEL_NSAMPLES = 100


def aggregate_shap_global_importance(
    shap_values, feature_count: Optional[int] = None
) -> np.ndarray:
    """聚合 SHAP 值为单个全局特征重要性向量。"""
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    values = np.asarray(shap_values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    if values.ndim == 1:
        return np.abs(values)

    if feature_count is not None:
        candidate_axes = [
            axis for axis, size in enumerate(values.shape) if int(size) == int(feature_count)
        ]
        if len(candidate_axes) == 1:
            feature_axis = candidate_axes[0]
            reduction_axes = tuple(
                axis for axis in range(values.ndim) if axis != feature_axis
            )
            return np.mean(np.abs(values), axis=reduction_axes)

    reduction_axes = tuple(range(values.ndim - 1))
    return np.mean(np.abs(values), axis=reduction_axes)



def build_shap_importance_records(
    feature_names: List[str], importances: np.ndarray
) -> List[Dict[str, float]]:
    """构建按重要性降序排列的 SHAP JSON 记录。"""
    importance_values = np.asarray(importances, dtype=float).reshape(-1)
    if len(feature_names) != len(importance_values):
        raise ValueError("feature_names and importances must have the same length")

    sorted_idx = np.argsort(importance_values)[::-1]
    return [
        {
            "feature": str(feature_names[idx]),
            "importance": float(importance_values[idx]),
        }
        for idx in sorted_idx
    ]



def build_shap_metadata(
    status: str,
    background_samples: int,
    explanation_samples: int,
    feature_count: int,
    detail: str = "",
) -> Dict[str, Any]:
    """构建 SHAP 元数据。"""
    return {
        "status": status,
        "background_samples": int(background_samples),
        "explanation_samples": int(explanation_samples),
        "feature_count": int(feature_count),
        "detail": detail,
    }



def generate_shap_artifacts(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """生成 SHAP 图像、重要性 JSON 与元数据。"""
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "shap_summary.png")
    bar_path = os.path.join(output_dir, "shap_bar.png")
    importance_path = os.path.join(output_dir, "shap_importance.json")
    meta_path = os.path.join(output_dir, "shap_meta.json")
    feature_count = len(feature_names)

    def persist_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        result = dict(payload)
        result["meta_path"] = meta_path
        if payload.get("status") == "success":
            result["summary_path"] = summary_path
            result["bar_path"] = bar_path
            result["importance_path"] = importance_path
        return result

    try:
        shap = __import__("shap")
    except Exception as exc:
        logger.warning(f"SHAP 导入失败，跳过解释性分析: {exc}")
        return persist_metadata(
            build_shap_metadata(
                status="failed",
                background_samples=0,
                explanation_samples=0,
                feature_count=feature_count,
                detail=str(exc),
            )
        )

    try:
        X_train_array = np.asarray(X_train, dtype=float)
        X_test_array = np.asarray(X_test, dtype=float)

        if X_train_array.size == 0 or X_test_array.size == 0 or feature_count == 0:
            return persist_metadata(
                build_shap_metadata(
                    status="skipped",
                    background_samples=0,
                    explanation_samples=0,
                    feature_count=feature_count,
                    detail="insufficient data for SHAP analysis",
                )
            )

        background_limit = int(globals().get("SHAP_BACKGROUND_SAMPLE_SIZE", 128))
        explanation_limit = int(globals().get("SHAP_EXPLANATION_SAMPLE_SIZE", 64))
        kernel_nsamples = int(globals().get("SHAP_KERNEL_NSAMPLES", 100))

        rng = np.random.default_rng(seed)
        background_size = min(len(X_train_array), max(1, background_limit))
        explanation_size = min(len(X_test_array), max(1, explanation_limit))

        if len(X_train_array) > background_size:
            background_idx = rng.choice(
                len(X_train_array), size=background_size, replace=False
            )
            background = X_train_array[background_idx]
        else:
            background = X_train_array

        if len(X_test_array) > explanation_size:
            explanation_idx = rng.choice(
                len(X_test_array), size=explanation_size, replace=False
            )
            explanation = X_test_array[explanation_idx]
        else:
            explanation = X_test_array

        def predict_fn(values):
            return model.predict(np.asarray(values), verbose=0)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(explanation, nsamples=kernel_nsamples)
        global_importance = aggregate_shap_global_importance(
            shap_values,
            feature_count=feature_count,
        )

        if global_importance.shape[0] != feature_count:
            raise ValueError(
                f"SHAP importance shape mismatch: expected {feature_count}, got {global_importance.shape[0]}"
            )

        importance_records = build_shap_importance_records(
            feature_names, global_importance
        )
        with open(importance_path, "w", encoding="utf-8") as f:
            json.dump(importance_records, f, indent=2, ensure_ascii=False)

        apply_plot_font = globals().get("_apply_plot_font")
        if callable(apply_plot_font):
            apply_plot_font()
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            explanation,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()

        if callable(apply_plot_font):
            apply_plot_font()
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            explanation,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
        )
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()

        return persist_metadata(
            build_shap_metadata(
                status="success",
                background_samples=len(background),
                explanation_samples=len(explanation),
                feature_count=feature_count,
                detail="",
            )
        )
    except Exception as exc:
        logger.warning(f"SHAP 分析失败: {exc}")
        return persist_metadata(
            build_shap_metadata(
                status="failed",
                background_samples=0,
                explanation_samples=0,
                feature_count=feature_count,
                detail=str(exc),
            )
        )


# ================== CEO 优化算法（内联实现）==================
def _ceo_edm(x0, y0, itermax: int):
    """Exponential Discrete Memristor (E-DM) 混沌映射

    支持标量或向量初值：
    - 若 x0/y0 为标量，返回形状为 (itermax,) 的序列
    - 若 x0/y0 为向量，返回形状为 (itermax, Dim) 的序列
    """
    k = 2.66
    xo = np.asarray(x0, dtype=np.float64)
    yo = np.asarray(y0, dtype=np.float64)
    x_seq = []
    y_seq = []
    for _ in range(itermax):
        xn = k * (np.exp(-np.cos(np.pi * yo)) - 1.0) * xo
        yn = yo + xo
        x_seq.append(xn)
        y_seq.append(yn)
        xo, yo = xn, yn
    return np.array(x_seq), np.array(y_seq)


def _ceo_bound_constraint(v, lu):
    """处理边界约束违规"""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    low_state = np.tile(lu[0], (v.shape[0], 1))
    up_state = np.tile(lu[1], (v.shape[0], 1))
    # 处理下界违规
    vio_low = v < low_state
    v[vio_low] = np.minimum(up_state[vio_low], 2 * low_state[vio_low] - v[vio_low])
    # 处理上界违规
    vio_upper = v > up_state
    v[vio_upper] = np.maximum(
        low_state[vio_upper], 2 * up_state[vio_upper] - v[vio_upper]
    )
    return v


def _ceo_binomial_crossover(p, v, CR):
    """二项交叉操作"""
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    # 确保 p 和 v 形状一致
    if p.shape[0] == 1 and v.shape[0] > 1:
        p = np.tile(p, (v.shape[0], 1))
    N, dim = v.shape
    j_rand = np.floor(np.random.rand(N) * dim).astype(int)
    t = np.random.rand(N, dim) < CR
    t[np.arange(N), j_rand] = True
    return np.where(t, v, p)


def ceo_optimize(
    func, Np: int, Dim: int, Varmin, Varmax, N: int = 20, max_iter: int = 100
):
    """CEO (Chaotic Evolution Optimization) 优化算法

    Args:
        func: 目标函数（最小化）
        Np: 种群大小（必须为偶数）
        Dim: 问题维度
        Varmin: 下界数组
        Varmax: 上界数组
        N: 混沌样本数（默认20）
        max_iter: 最大迭代次数

    Returns:
        (最佳解, 最佳适应度, 历史记录)
    """
    # 确保 Np 为偶数
    if Np % 2 != 0:
        Np = max(2, Np + 1)

    # 处理搜索范围
    Varmin = np.asarray(Varmin, dtype=np.float64).flatten()
    Varmax = np.asarray(Varmax, dtype=np.float64).flatten()
    lu = np.vstack([Varmin, Varmax])
    if lu.shape[1] == 1:
        lu = np.tile(lu, (1, Dim))

    # 定义混沌搜索域
    low_chacos = np.array([-0.5, -0.25])
    up_chacos = np.array([0.5, 0.25])

    # 初始化种群
    Population = lu[0] + np.random.rand(Np, Dim) * (lu[1] - lu[0])
    fit = np.array([func(ind) for ind in Population])
    fBest = float(np.min(fit))
    Best = Population[np.argmin(fit)].copy()

    history = [fBest]
    counter = 0

    for t in range(1, max_iter + 1):
        oldfBest = fBest
        rand_num = np.random.permutation(Np)

        ub = np.max(Population, axis=0)
        lb = np.min(Population, axis=0)
        range_pop = ub - lb
        range_pop = np.where(range_pop < 1e-10, 1e-10, range_pop)

        for i in range(0, Np, 2):
            if i + 1 >= Np:
                break
            index = rand_num[i : i + 2]
            xy = Population[index]  # (2, Dim)

            # 区间映射：从问题空间 -> 混沌空间
            xy_dot = ((xy - lb) / range_pop) * (up_chacos - low_chacos)[
                :, None
            ] + low_chacos[:, None]

            # 生成混沌个体
            x_chaos, y_chaos = _ceo_edm(xy_dot[0], xy_dot[1], N)
            chaos_total = np.vstack([x_chaos, y_chaos])  # (2*N, Dim)

            for k in range(2):
                xy_chaos = chaos_total[k * N : (k + 1) * N]  # (N, Dim)

                # 映射回实际优化问题空间
                xy_chaos_dot = (
                    (xy_chaos - low_chacos[k]) / (up_chacos[k] - low_chacos[k])
                ) * range_pop + lb

                # 变异
                if np.random.rand() < 0.5:
                    xy_hat = xy[k] + np.random.rand(N, 1) * (xy_chaos_dot - xy[k])
                else:
                    xy_hat = Best + np.random.rand(N, 1) * (xy_chaos_dot - xy[k])

                # 交叉
                CR = np.random.rand()
                xy_trial = _ceo_binomial_crossover(xy[k], xy_hat, CR)

                # 边界约束
                xy_trial = _ceo_bound_constraint(xy_trial, lu)

                # 适应度评估
                fit_xy_trial = np.array([func(trial) for trial in xy_trial])
                best_trial_idx = np.argmin(fit_xy_trial)

                # 选择
                if fit_xy_trial[best_trial_idx] < fit[index[k]]:
                    Population[index[k]] = xy_trial[best_trial_idx]
                    fit[index[k]] = fit_xy_trial[best_trial_idx]

        # 更新最佳解
        fBest = float(np.min(fit))
        Best = Population[np.argmin(fit)].copy()

        # 终止条件
        if abs(oldfBest - fBest) < 1e-8:
            counter += 1
            if counter > 50:
                break
        else:
            counter = 0

        history.append(fBest)

    return Best, fBest, history


# ================== 最优参数保存和加载 ==================
def _get_best_params_path(output_dir: str = "results") -> str:
    """获取最优参数保存路径"""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "ceo_best_params.json")


def save_best_params(
    params: Dict[str, Any], val_accuracy: float, output_dir: str = "results"
):
    """保存最优超参数到JSON文件

    Args:
        params: 超参数字典
        val_accuracy: 验证准确率
        output_dir: 输出目录
    """
    filepath = _get_best_params_path(output_dir)
    data = {
        "val_accuracy": float(val_accuracy),
        "hyperparameters": params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 最优参数已保存: {filepath} (val_acc={val_accuracy:.4f})")
    except Exception as e:
        logger.warning(f"⚠️ 保存最优参数失败: {e}")


def load_best_params(output_dir: str = "results") -> Optional[Dict[str, Any]]:
    """加载历史最优超参数

    Returns:
        包含 'hyperparameters' 和 'val_accuracy' 的字典，如果文件不存在则返回 None
    """
    filepath = _get_best_params_path(output_dir)
    if not os.path.exists(filepath):
        logger.info(f"未找到历史最优参数文件: {filepath}")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            f"✅ 已加载历史最优参数 (val_acc={data.get('val_accuracy', 0):.4f})"
        )
        return data
    except Exception as e:
        logger.warning(f"⚠️ 加载最优参数失败: {e}")
        return None


# ================== CEO 超参数搜索 ==================
def define_search_space():
    """定义超参数搜索空间（基于最优参数配置）

    注意: 形状相关参数固定为最优值，避免与自定义优化器冲突。
    其他参数以最优值为中心进行小范围搜索。
    """
    return {
        # 可调参数（以最优值为中心的小范围搜索）
        "learning_rate": {
            "type": "continuous",
            "range": (5e-4, 1.5e-3),
        },  # 最优: 0.000813
        "weight_decay": {"type": "continuous", "range": (2e-4, 6e-4)},  # 最优: 0.000427
        "dropout_rate": {"type": "continuous", "range": (0.15, 0.25)},  # 最优: 0.206
        "initial_noise_factor": {
            "type": "continuous",
            "range": (0.02, 0.03),
        },  # 最优: 0.024
        # 固定参数（使用最优值）
        "hidden_units": {"type": "discrete", "choices": [192]},  # 最优: 192
        "attention_heads": {"type": "discrete", "choices": [16]},  # 最优: 16
        "max_sequence_length": {"type": "discrete", "choices": [15]},  # 最优: 15
        "knn_top_k": {"type": "discrete", "choices": [6]},  # 最优: 6
    }


def run_ceo_search(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes: int,
    budget: int = 10,
    epochs_per_trial: int = 25,
    output_dir: str = "results",
):
    """运行 CEO 超参数搜索（使用内联的CEO算法）

    功能：
    1. 加载历史最优参数（如果存在）
    2. 运行CEO搜索找到新的最优参数
    3. 比较新旧参数，保留更好的
    4. 保存最终最优参数

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        num_classes: 类别数
        budget: CEO搜索预算
        epochs_per_trial: 每次trial的训练轮数
        output_dir: 输出目录（用于保存最优参数）

    Returns:
        最优超参数字典
    """
    # ===== 1. 加载历史最优参数 =====
    historical_best = load_best_params(output_dir)
    historical_acc = 0.0
    if historical_best is not None:
        historical_acc = historical_best.get("val_accuracy", 0.0)
        logger.info(f"历史最优验证准确率: {historical_acc:.4f}")

    search_space = define_search_space()
    param_names = list(search_space.keys())
    dim = len(param_names)

    logger.info(f"开始 CEO 超参数搜索 (预算: {budget}, 维度: {dim})")

    def objective(x):
        """CEO目标函数 - 每次调用都创建全新的模型实例"""
        # ===== 1. 调用前先清理上一次的会话状态 =====
        tf.keras.backend.clear_session()
        gc.collect()

        # ===== 2. 解析超参数 =====
        params = {}
        for i, name in enumerate(param_names):
            spec = search_space[name]
            if spec["type"] == "continuous":
                lo, hi = spec["range"]
                params[name] = float(lo + x[i] * (hi - lo))
            else:
                choices = spec["choices"]
                idx = min(int(x[i] * len(choices)), len(choices) - 1)
                params[name] = choices[idx]

        # 确保 hidden_units 能被 attention_heads 整除
        params["hidden_units"] = (params["hidden_units"] // 8) * 8

        model = None
        val_acc = 0.0

        try:
            # ===== 3. 创建全新的模型配置（使用最优默认值）=====
            config = ModelConfig(
                input_shape=(X_train.shape[1],),
                num_nodes=X_train.shape[1],
                num_classes=num_classes,
                hidden_units=params["hidden_units"],
                attention_heads=params["attention_heads"],
                dropout_rate=params["dropout_rate"],
                max_sequence_length=params["max_sequence_length"],
                knn_top_k=params["knn_top_k"],
                # dilation_rate, use_batch_norm, embedding_dim 使用 ModelConfig 默认值
            )

            # ===== 4. 创建全新的模型实例 =====
            model = EnhancedMSTGAT(config)

            # ===== 5. 创建优化器（CEO搜索使用标准Adam快速评估，与原版一致）=====
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=params["learning_rate"],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
            )

            # ===== 6. 编译模型 =====
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"],
            )

            # ===== 7. 创建数据集 =====
            train_ds, val_ds = create_datasets(X_train, y_train, X_val, y_val)

            # ===== 8. 训练模型 =====
            history = model.fit(
                train_ds,
                epochs=epochs_per_trial,
                validation_data=val_ds,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy",
                        patience=5,  # 增加耐心
                        restore_best_weights=True,
                    )
                ],
            )

            val_acc = float(max(history.history.get("val_accuracy", [0.0])))
            logger.info(f"  Trial: val_acc={val_acc:.4f}, params={params}")

        except Exception as e:
            logger.warning(f"  Trial failed: {e}")
            import traceback

            traceback.print_exc()
            val_acc = 0.0

        finally:
            # ===== 9. 彻底清理模型和会话 =====
            if model is not None:
                try:
                    del model
                except:
                    pass
            gc.collect()
            tf.keras.backend.clear_session()

        return -val_acc

    # 运行 CEO（使用内联的ceo_optimize函数）
    Np = 4  # 种群大小（偶数）
    N = 20  # 混沌样本数
    max_iter = max(1, budget // 2)

    logger.info(f"CEO 参数: Np={Np}, N={N}, max_iter={max_iter}, dim={dim}")

    best_x, best_val, _ = ceo_optimize(
        func=objective,
        Np=Np,
        Dim=dim,
        Varmin=np.zeros(dim),
        Varmax=np.ones(dim),
        N=N,
        max_iter=max_iter,
    )

    # 解码最佳参数
    best_params = {}
    for i, name in enumerate(param_names):
        spec = search_space[name]
        if spec["type"] == "continuous":
            lo, hi = spec["range"]
            best_params[name] = float(lo + best_x[i] * (hi - lo))
        else:
            choices = spec["choices"]
            idx = min(int(best_x[i] * len(choices)), len(choices) - 1)
            best_params[name] = choices[idx]

    best_params["hidden_units"] = (best_params["hidden_units"] // 8) * 8

    # ===== 比较搜索结果与历史最优 =====
    search_val_acc = -best_val  # CEO返回的是负值（因为是最小化）
    logger.info(f"CEO 搜索完成！本次搜索最佳验证准确率: {search_val_acc:.4f}")

    # 比较并决定使用哪个参数
    if historical_best is not None and historical_acc > search_val_acc:
        # 历史参数更好，使用历史参数
        logger.info(
            f"⚠️ 本次搜索结果 ({search_val_acc:.4f}) 不如历史最优 ({historical_acc:.4f})"
        )
        logger.info(f"✅ 使用历史最优参数")
        final_params = historical_best["hyperparameters"]
        final_acc = historical_acc
    else:
        # 本次搜索更好或没有历史记录
        if historical_best is not None:
            logger.info(
                f"✅ 本次搜索结果 ({search_val_acc:.4f}) 优于历史最优 ({historical_acc:.4f})"
            )
        else:
            logger.info(f"✅ 首次搜索，记录最优参数")
        final_params = best_params
        final_acc = search_val_acc
        # 保存新的最优参数
        save_best_params(final_params, final_acc, output_dir)

    logger.info(f"最终使用参数: {final_params}")

    return final_params


# ================== 主训练流程 ==================
def run_training_pipeline(
    data_path: str,
    test_size: float = 0.2,
    epochs: int = 100,
    budget: int = 10,
    seed: int = 42,
    use_gpu: bool = True,
    skip_ceo: bool = False,
    output_dir: str = "results",
    prepared_df: Optional["pd.DataFrame"] = None,
    prepared_label_col: Optional[str] = None,
    gpu_info: Optional[Dict[str, Any]] = None,
):
    """完整训练流程

    Args:
        data_path: 数据集路径
        test_size: 测试集比例
        epochs: 训练轮数
        budget: CEO 搜索预算
        seed: 随机种子
        use_gpu: 是否使用 GPU 加速
        skip_ceo: 跳过 CEO 搜索，使用默认参数
        output_dir: 输出目录
    """
    start_time = time.time()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # GPU 配置
    if gpu_info is None:
        if use_gpu:
            gpu_info = setup_gpu()
        else:
            gpu_info = {"gpus_available": [], "mixed_precision": False}

    # 设置随机种子
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 加载数据
    if prepared_df is None and prepared_label_col is None:
        data = load_and_preprocess_data(
            data_path,
            test_size=test_size,
            seed=seed,
        )
    else:
        data = load_and_preprocess_data(
            data_path,
            test_size=test_size,
            seed=seed,
            prepared_df=prepared_df,
            label_col=prepared_label_col,
        )
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    num_classes = data["num_classes"]
    feature_names = data["feature_names"]
    class_names = data["class_names"]

    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
    )

    # CEO 超参数搜索
    if skip_ceo:
        # 使用最优超参数（来自 enhanced_mstgat_best_params_ts20.json）
        best_params = {
            "learning_rate": 0.0008127349453725082,
            "weight_decay": 0.00042655193657822774,
            "hidden_units": 192,
            "attention_heads": 16,  # 最优: 16
            "max_sequence_length": 15,  # 最优: 15
            "knn_top_k": 6,  # 最优: 6
            "dropout_rate": 0.20559374642540468,
            "initial_noise_factor": 0.024223792861225474,
            "dilation_rate": 4,  # 最优: 4
            "use_batch_norm": False,  # 最优: False
            "embedding_dim": 160,  # 最优: 160
        }
        logger.info("使用最优超参数（跳过 CEO 搜索）")
    else:
        best_params = run_ceo_search(
            X_train,
            y_train,
            X_val,
            y_val,
            num_classes,
            budget=budget,
            epochs_per_trial=10,
            output_dir=output_dir,
        )
        # CEO搜索完成后清理会话，确保最终模型在干净环境中创建
        tf.keras.backend.clear_session()
        gc.collect()

    # 创建模型
    logger.info("\n创建最终模型...")
    config = ModelConfig(
        input_shape=(X_train.shape[1],),
        num_nodes=X_train.shape[1],
        num_classes=num_classes,
        hidden_units=best_params["hidden_units"],
        attention_heads=best_params["attention_heads"],
        dropout_rate=best_params["dropout_rate"],
        max_sequence_length=best_params["max_sequence_length"],
        knn_top_k=best_params["knn_top_k"],
        dilation_rate=best_params.get("dilation_rate", 4),
        use_batch_norm=best_params.get("use_batch_norm", False),
        embedding_dim=best_params.get("embedding_dim", 160),
    )

    model = EnhancedMSTGAT(config)
    model.build(input_shape=(None, X_train.shape[1]))
    logger.info(f"模型参数量: {model.count_params():,}")

    # 创建优化器
    steps_per_epoch = int(np.ceil(X_train.shape[0] / 32))
    total_steps = epochs * steps_per_epoch

    lr_schedule = WarmupCosineDecay(
        base_lr=best_params["learning_rate"],
        total_steps=total_steps,
        warmup_steps=int(0.1 * total_steps),
    )

    # 使用改进的 QAAdamW_Lite 优化器
    optimizer = QAAdamW_Lite(
        learning_rate=lr_schedule,
        weight_decay=best_params["weight_decay"],
        initial_noise_factor=best_params["initial_noise_factor"],
        use_adabelief=True,
        total_steps=total_steps,
    )

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # 创建数据集
    train_ds, val_ds = create_datasets(X_train, y_train, X_val, y_val)

    # 回调函数
    # 注意：由于使用了自定义学习率调度器 WarmupCosineDecay，不再使用 ReduceLROnPlateau
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, mode="max", restore_best_weights=True
        ),
    ]

    # 训练
    logger.info("\n开始训练...")
    history = model.fit(
        train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=1
    )

    # 评估
    logger.info("\n评估模型...")
    pred_start = time.time()
    y_pred_prob = model.predict(X_test, verbose=0)
    pred_time = time.time() - pred_start
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_test, y_pred)
    per_class_metrics = {
        "precision": precision_score(y_test, y_pred, average=None, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=None, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=None, zero_division=0),
    }

    logger.info(f"\n{'=' * 60}")
    logger.info(f"测试结果:")
    logger.info(f"  准确率: {acc:.4f}")
    logger.info(f"  精确率: {prec:.4f}")
    logger.info(f"  召回率: {rec:.4f}")
    logger.info(f"  F1 分数: {f1:.4f}")
    logger.info(f"  预测时间: {pred_time:.4f}s")
    logger.info(f"{'=' * 60}")

    # 保存模型
    model_path = os.path.join(output_dir, "ceo_qaadam_emstgat.keras")
    model.save(model_path)
    logger.info(f"模型已保存: {model_path}")

    # 保存预处理元数据，供继续训练与离线评估/预测复用
    preprocess_meta_path = os.path.join(output_dir, "preprocess_meta.json")
    save_preprocess_meta(data["preprocess_meta"], preprocess_meta_path)
    logger.info(f"预处理元数据已保存: {preprocess_meta_path}")

    # 生成可视化
    logger.info("\n生成可视化...")
    plot_training_history(
        history.history, os.path.join(output_dir, "training_history.png")
    )

    display_class_names = [
        class_names[i] if i < len(class_names) else f"Class {i}"
        for i in range(num_classes)
    ]

    plot_confusion_matrix(
        y_test,
        y_pred,
        display_class_names,
        os.path.join(output_dir, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        y_test,
        y_pred,
        display_class_names,
        os.path.join(output_dir, "confusion_matrix_normalized.png"),
        normalize=True,
    )
    plot_per_class_metrics(
        per_class_metrics,
        display_class_names,
        save_path=os.path.join(output_dir, "per_class_metrics.png"),
    )

    plot_feature_importance(
        feature_names,
        data["feature_importances"],
        save_path=os.path.join(output_dir, "feature_importance.png"),
    )

    shap_results = generate_shap_artifacts(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        output_dir=output_dir,
        seed=seed,
    )

    results_split = int(round(test_size * 100))
    results_train = 100 - results_split

    # 保存结果
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "cohen_kappa": kappa,
        "prediction_time": pred_time,
        "best_params": best_params,
        "gpu_info": gpu_info,
        "training_time": time.time() - start_time,
        "test_size": test_size,
        "train_ratio": results_train,
        "test_ratio": results_split,
        "output_dir": output_dir,
        "shap": shap_results,
    }

    paper_artifacts = build_training_figure_artifacts(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_pred_prob,
        feature_names=feature_names,
        class_names=display_class_names,
        history=history.history,
        metrics=results,
        model_name="CEO-QAAdamW-EMSTGAT",
        random_state=seed,
    )
    figure_data_paths = save_training_figure_artifacts(paper_artifacts, output_dir)
    results["paper_figure_artifacts"] = figure_data_paths

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n✅ 训练完成！总耗时: {time.time() - start_time:.2f}s")
    logger.info(f"   结果保存于: {output_dir}")

    return results


def run_multi_split_experiments(
    data_path: str,
    test_sizes: List[float],
    epochs: int = 100,
    budget: int = 10,
    seed: int = 42,
    use_gpu: bool = True,
    skip_ceo: bool = True,
):
    """运行多组测试集比例实验。"""
    summaries = []
    shared_gpu_info: Optional[Dict[str, Any]] = None
    if use_gpu:
        shared_gpu_info = setup_gpu()

    prepared_df = None
    prepared_label_col = None
    prepare_df_fn = globals().get("prepare_training_dataframe")
    if callable(prepare_df_fn):
        prepared_df, prepared_label_col = prepare_df_fn(data_path)
        logger.info("已完成一次性通用清洗，后续各比例仅执行划分后拟合型预处理")

    for test_size in test_sizes:
        test_ratio = int(round(test_size * 100))
        train_ratio = 100 - test_ratio
        output_dir = f"results_testdata_{train_ratio}_{test_ratio}"
        logger.info(f"\n开始测试比例实验: {train_ratio}/{test_ratio}")
        result = run_training_pipeline(
            data_path=data_path,
            test_size=test_size,
            epochs=epochs,
            budget=budget,
            seed=seed,
            use_gpu=use_gpu,
            skip_ceo=skip_ceo,
            output_dir=output_dir,
            prepared_df=prepared_df,
            prepared_label_col=prepared_label_col,
            gpu_info=shared_gpu_info,
        )
        summaries.append(
            {
                "data_path": data_path,
                "test_size": float(test_size),
                "train_ratio": train_ratio,
                "test_ratio": test_ratio,
                "epochs": epochs,
                "budget": budget,
                "seed": seed,
                "use_gpu": use_gpu,
                "skip_ceo": skip_ceo,
                "output_dir": output_dir,
                "accuracy": float(result["accuracy"]),
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1_score": float(result["f1_score"]),
                "prediction_time": float(result["prediction_time"]),
            }
        )
    return summaries


# ================== 主入口 ==================
if __name__ == "__main__":
    data_path = os.path.join("数据文件", "测试数据.xlsx")
    test_sizes = [0.2, 0.3, 0.4]

    if not os.path.exists(data_path):
        logger.error(f"❌ 数据集文件不存在: {data_path}")
        logger.info("请确保数据集文件位于正确的位置")
        sys.exit(1)

    logger.info(f"数据集路径: {data_path}")
    logger.info(f"测试比例: {[int(size * 100) for size in test_sizes]}")

    split_summaries = run_multi_split_experiments(
        data_path=data_path,
        test_sizes=test_sizes,
        epochs=100,
        budget=10,
        seed=42,
        use_gpu=True,
        skip_ceo=True,
    )

    summary_output_dir = "results_testdata_summary"
    os.makedirs(summary_output_dir, exist_ok=True)
    summary_path = os.path.join(summary_output_dir, "split_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summaries, f, indent=2, ensure_ascii=False)
    logger.info(f"多比例汇总已保存: {summary_path}")

    plot_split_metrics_comparison(
        split_summaries,
        save_path=os.path.join(summary_output_dir, "split_metrics_comparison.png"),
    )

    print("\n" + "=" * 60)
    print("多比例训练完成！")
    for summary in split_summaries:
        print(
            f"  测试集 {summary['test_ratio']}% | 准确率: {summary['accuracy']:.4f} | "
            f"精确率: {summary['precision']:.4f} | 召回率: {summary['recall']:.4f} | "
            f"F1: {summary['f1_score']:.4f}"
        )
    print("=" * 60)
