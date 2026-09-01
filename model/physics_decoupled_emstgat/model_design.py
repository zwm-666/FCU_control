"""Physics-decoupled EMSTGAT model design."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
try:
    from tensorflow.keras import layers, Model
    from tensorflow.keras.layers import RepeatVector
    KERAS = tf.keras
except ModuleNotFoundError:
    import keras as KERAS
    from keras import layers, Model
    from keras.layers import RepeatVector


PHYSICAL_GROUP_ORDER = ("electric", "thermal", "gas", "humidity")
PHYSICAL_FEATURE_GROUPS: Dict[str, List[str]] = {
    "electric": [
        "voltage", "current", "power", "cell", "stack", "电压", "电流", "功率", "电堆",
    ],
    "thermal": [
        "temp", "temperature", "thermal", "coolant", "heat", "温度", "水温", "空温", "热",
    ],
    "gas": [
        "pressure", "air", "hydrogen", "h2", "o2", "gas", "压", "氢", "空压", "气",
    ],
    "humidity": [
        "humidity", "humid", "water", "moisture", "purge", "flood", "dry", "湿", "水", "膜干", "水淹",
    ],
}


def build_physical_feature_group_indices(
    feature_names: Optional[List[str]],
    input_dim: int,
) -> Dict[str, List[int]]:
    """Build readable electric/thermal/gas/humidity index groups."""
    input_dim = int(input_dim)
    groups: Dict[str, List[int]] = {name: [] for name in PHYSICAL_GROUP_ORDER}
    if feature_names:
        for index, feature in enumerate(feature_names[:input_dim]):
            text = str(feature).lower()
            matched = False
            for group_name in PHYSICAL_GROUP_ORDER:
                if any(keyword.lower() in text for keyword in PHYSICAL_FEATURE_GROUPS[group_name]):
                    groups[group_name].append(index)
                    matched = True
                    break
            if not matched:
                groups[PHYSICAL_GROUP_ORDER[index % len(PHYSICAL_GROUP_ORDER)]].append(index)
    else:
        splits = np.array_split(np.arange(input_dim), len(PHYSICAL_GROUP_ORDER))
        groups = {
            group_name: [int(index) for index in split.tolist()]
            for group_name, split in zip(PHYSICAL_GROUP_ORDER, splits)
        }

    assigned = {index for values in groups.values() for index in values}
    for index in range(input_dim):
        if index not in assigned:
            groups[PHYSICAL_GROUP_ORDER[index % len(PHYSICAL_GROUP_ORDER)]].append(index)

    return groups


def select_physics_aware_features(
    feature_names: Optional[List[str]],
    importances: np.ndarray,
    cumulative_threshold: float = 0.95,
    min_features_per_group: int = 1,
) -> List[int]:
    """Select important features while preserving each physical branch.

    The global cumulative-importance pass keeps the strongest features, then
    each non-empty physical group contributes its top features so downstream
    electric/thermal/gas/humidity branches are not accidentally emptied.
    """
    importances = np.asarray(importances, dtype=np.float64).reshape(-1)
    input_dim = int(importances.shape[0])
    if input_dim == 0:
        return []

    if feature_names is not None and len(feature_names) < input_dim:
        raise ValueError("feature_names length must cover all importances")

    cumulative_threshold = float(np.clip(cumulative_threshold, 0.0, 1.0))
    min_features_per_group = max(0, int(min_features_per_group))

    sorted_idx = np.argsort(-importances, kind="mergesort")
    total_importance = float(np.sum(importances))
    if total_importance > 0:
        cumsum_norm = np.cumsum(importances[sorted_idx]) / total_importance
        n_selected = int(np.searchsorted(cumsum_norm, cumulative_threshold) + 1)
        selected = set(int(index) for index in sorted_idx[:n_selected])
    else:
        selected = set()

    groups = build_physical_feature_group_indices(feature_names, input_dim)
    for group_indices in groups.values():
        if not group_indices or min_features_per_group <= 0:
            continue
        ranked_group = sorted(
            (int(index) for index in group_indices),
            key=lambda index: (-importances[index], index),
        )
        selected.update(ranked_group[:min_features_per_group])

    if not selected:
        selected.add(int(sorted_idx[0]))

    return sorted(selected, key=lambda index: (-importances[index], index))

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
    activation: str = 'gelu'
    kernel_initializer: str = 'he_normal'
    graph_type: str = 'knn'
    knn_top_k: int = 6  # 最优: 6 (之前是8)
    dilation_rate: int = 4  # 最优: 4 (之前是2)
    feature_group_indices: Optional[Dict[str, List[int]]] = None
    branch_units: Optional[int] = None



# ================== 自定义层 ==================
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
        mask = tf.tensor_scatter_nd_update(neg_inf, tf.reshape(scatter_indices, [-1, 3]), tf.reshape(values, [-1]))
        
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
        
        self.ffn = KERAS.Sequential([
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
        # 使用固定的缩放因子，避免混合精度类型问题
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
        
        self.ffn = KERAS.Sequential([
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


# ================== EnhancedMSTGAT 模型 ==================
class PhysicalBranchExtractor(layers.Layer):
    """Independent branch extractor for one physical feature group."""

    def __init__(self, hidden_units: int, dropout_rate: float, activation: str = "gelu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.net = KERAS.Sequential([
            layers.Dense(hidden_units, activation=activation, kernel_initializer="he_normal"),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_units, activation=activation, kernel_initializer="he_normal"),
        ])

    def call(self, inputs, training=None):
        return self.net(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        })
        return config


class EnhancedMSTGAT(Model):
    """Physics-decoupled MSTGAT with branch extraction and graph fusion."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        input_dim = int(config.input_shape[-1])
        self.feature_group_indices = config.feature_group_indices or build_physical_feature_group_indices(None, input_dim)
        missing_groups = set(PHYSICAL_GROUP_ORDER) - set(self.feature_group_indices)
        if missing_groups:
            raise ValueError(f"Missing physical feature groups: {sorted(missing_groups)}")
        for group_name, indices in self.feature_group_indices.items():
            if not indices:
                raise ValueError(f"Physical feature group {group_name!r} has no feature indices.")
            if min(indices) < 0 or max(indices) >= input_dim:
                raise ValueError(f"Physical feature group {group_name!r} contains out-of-range indices.")

        branch_units = int(config.branch_units or config.hidden_units)
        self.branch_extractors = {
            group_name: PhysicalBranchExtractor(
                hidden_units=branch_units,
                dropout_rate=config.dropout_rate,
                activation=config.activation,
                name=f"{group_name}_branch_extractor",
            )
            for group_name in PHYSICAL_GROUP_ORDER
        }

        # Four physical branches are treated as graph nodes before global fusion.
        self.branch_graph_builder = KNNGraphBuilder(top_k=min(config.knn_top_k, len(PHYSICAL_GROUP_ORDER)))
        self.branch_graph_attention = EdgeAwareMultiHeadAttention(
            num_heads=config.attention_heads,
            hidden_units=branch_units,
            dropout_rate=config.dropout_rate,
            activation=config.activation,
            name="physical_branch_graph_attention",
        )
        self.joint_fusion = KERAS.Sequential([
            layers.Flatten(),
            layers.Dense(config.hidden_units, activation=config.activation,
                         kernel_initializer=config.kernel_initializer),
            layers.Dropout(config.dropout_rate),
            layers.Dense(config.hidden_units, activation=config.activation,
                         kernel_initializer=config.kernel_initializer),
        ], name="multi_physics_joint_fusion")
        
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
        
        # 残差卷积层（原版有此层）
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
        
        # 额外注意力层（原版有此层）
        self.attention_2 = MultiHeadAttention(
            num_heads=config.attention_heads, hidden_units=config.hidden_units,
            dropout_rate=config.dropout_rate, activation=config.activation
        )
        
        # 注意力池化
        self.attn_pool_score = KERAS.Sequential([
            layers.Dense(config.hidden_units, activation=config.activation),
            layers.Dense(1)
        ])
        self.dropout = layers.Dropout(config.dropout_rate)
        
        # 分类头（使用 float32 确保混合精度训练稳定性）
        self.fc = layers.Dense(config.hidden_units, activation=config.activation,
                               kernel_initializer=config.kernel_initializer)
        self.classifier = layers.Dense(config.num_classes, activation='softmax',
                                       kernel_initializer=config.kernel_initializer,
                                       dtype='float32')

    @tf.function
    def call(self, inputs, training=None):
        # 确保输入是 float32
        x = tf.cast(inputs, tf.float32)

        branch_tokens = []
        for group_name in PHYSICAL_GROUP_ORDER:
            group_x = tf.gather(x, self.feature_group_indices[group_name], axis=1)
            branch_tokens.append(self.branch_extractors[group_name](group_x, training=training))
        branch_nodes = tf.stack(branch_tokens, axis=1)

        branch_adj = self.branch_graph_builder(branch_nodes, training=training)
        branch_nodes = self.branch_graph_attention(branch_nodes, branch_adj, training=training)
        x = self.joint_fusion(branch_nodes, training=training)
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
            'input_shape': self.config.input_shape,
            'num_nodes': self.config.num_nodes,
            'num_classes': self.config.num_classes,
            'hidden_units': self.config.hidden_units,
            'attention_heads': self.config.attention_heads,
            'dropout_rate': self.config.dropout_rate,
            'max_sequence_length': self.config.max_sequence_length,
            'knn_top_k': self.config.knn_top_k,
            'feature_group_indices': self.feature_group_indices,
            'branch_units': self.config.branch_units,
        }

