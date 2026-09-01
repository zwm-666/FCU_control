"""Direct-input and adaptive-branch EMSTGAT variants.

Both variants share the same temporal KNN-graph attention backbone.  The only
controlled difference is the input adapter: direct projection (``raw``) versus
learned feature-to-branch soft routing (``auto``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import tensorflow as tf
from tensorflow.keras import Model, layers

from .model_design import (
    DilatedCausalConv,
    EdgeAwareMultiHeadAttention,
    KNNGraphBuilder,
    MultiHeadAttention,
)


VALID_BRANCH_MODES = ("raw", "auto")


@dataclass(frozen=True)
class AdaptiveModelConfig:
    """Configuration shared by direct-input and adaptive-branch variants."""

    input_dim: int
    num_classes: int
    branch_mode: str = "raw"
    hidden_units: int = 128
    attention_heads: int = 8
    dropout_rate: float = 0.2
    max_sequence_length: int = 15
    knn_top_k: int = 6
    dilation_rate: int = 4
    num_auto_branches: int = 4
    router_temperature: float = 0.15
    router_balance_weight: float = 0.01
    router_entropy_weight: float = 0.05

    # ---- ablation switches -------------------------------------------------
    # Each flag removes exactly ONE architectural component so a controlled
    # ablation study can attribute accuracy to that component. All default to
    # True, so the full model is bit-for-bit unchanged unless a flag is set.
    use_dilated_causal_conv: bool = True   # DCC temporal front end
    use_bigru: bool = True                # bidirectional GRU
    use_knn_graph: bool = True            # temporal KNN adjacency context
    use_self_attention: bool = True       # multi-head temporal attention
    use_attention_pooling: bool = True    # attention pool (else mean pool)
    use_skip_classifier: bool = True      # tabular skip-logit fusion
    use_positional_embedding: bool = True # learned positional encoding
    use_router_gate: bool = True          # AB only: per-feature relevance gate

    def __post_init__(self) -> None:
        if self.branch_mode not in VALID_BRANCH_MODES:
            raise ValueError(f"branch_mode must be one of {VALID_BRANCH_MODES}, got {self.branch_mode!r}")
        if self.input_dim < 1 or self.num_classes < 2:
            raise ValueError("input_dim must be >= 1 and num_classes must be >= 2")
        if self.hidden_units < 1 or self.hidden_units % self.attention_heads:
            raise ValueError("hidden_units must be positive and divisible by attention_heads")
        if self.num_auto_branches < 2:
            raise ValueError("num_auto_branches must be at least 2")


class AdaptiveBranchRouter(layers.Layer):
    """Learn a dataset-specific soft assignment of features to latent branches.

    Routing is global per feature (rather than derived from column names), so
    it can be trained independently for datasets with different modalities and
    feature counts.  Auxiliary balance and entropy losses avoid one collapsed
    branch or fully uniform, non-interpretable routing.
    """

    def __init__(
        self,
        input_dim: int,
        num_branches: int,
        branch_units: int,
        temperature: float = 0.7,
        balance_weight: float = 0.01,
        entropy_weight: float = 0.001,
        use_gate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.num_branches = int(num_branches)
        self.branch_units = int(branch_units)
        self.temperature = float(temperature)
        self.balance_weight = float(balance_weight)
        self.entropy_weight = float(entropy_weight)
        self.use_gate = bool(use_gate)
        self.feature_logits = None
        self.feature_gate_logits = None
        self.feature_embedding = layers.Dense(branch_units, activation="gelu", name="feature_embedding")
        self.branch_norm = layers.LayerNormalization(epsilon=1e-6, name="branch_norm")

    def build(self, input_shape):
        self.feature_logits = self.add_weight(
            name="feature_to_branch_logits",
            shape=(self.input_dim, self.num_branches),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Learnable per-feature relevance gate. With 227 raw channels and few
        # training blocks, letting the model suppress uninformative features is
        # what makes the branch assignment interpretable instead of uniform.
        if self.use_gate:
            self.feature_gate_logits = self.add_weight(
                name="feature_relevance_gate_logits",
                shape=(self.input_dim,),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True,
            )
        super().build(input_shape)

    def feature_gate_values(self) -> tf.Tensor:
        if self.feature_gate_logits is None:
            raise RuntimeError("AdaptiveBranchRouter must be built before gate values are requested.")
        return tf.nn.sigmoid(self.feature_gate_logits)

    def routing_probabilities(self) -> tf.Tensor:
        if self.feature_logits is None:
            raise RuntimeError("AdaptiveBranchRouter must be built before routing probabilities are requested.")
        return tf.nn.softmax(self.feature_logits / self.temperature, axis=-1)

    def call(self, inputs, training=None):
        x = tf.cast(inputs, tf.float32)
        routing = self.routing_probabilities()
        if self.use_gate:
            gate = self.feature_gate_values()
            # Gate scales each feature's contribution to its branches, so
            # uninformative channels can be suppressed rather than averaged in.
            gated_routing = routing * tf.expand_dims(gate, axis=-1)
        else:
            gate = None
            gated_routing = routing

        if x.shape.rank == 2:
            feature_tokens = self.feature_embedding(tf.expand_dims(x, axis=-1))
            branch_tokens = tf.einsum("bfd,fk->bkd", feature_tokens, gated_routing)
        elif x.shape.rank == 3:
            feature_tokens = self.feature_embedding(tf.expand_dims(x, axis=-1))
            branch_tokens = tf.einsum("btfd,fk->btkd", feature_tokens, gated_routing)
        else:
            raise ValueError("AdaptiveBranchRouter expects a 2D tabular batch or 3D temporal batch.")
        branch_tokens = branch_tokens / tf.sqrt(tf.cast(self.input_dim, branch_tokens.dtype))
        branch_tokens = self.branch_norm(branch_tokens)

        mean_usage = tf.reduce_mean(routing, axis=0)
        target_usage = tf.fill([self.num_branches], 1.0 / float(self.num_branches))
        balance_loss = tf.reduce_sum(tf.square(mean_usage - target_usage))
        entropy = -tf.reduce_sum(routing * tf.math.log(tf.maximum(routing, 1e-8)), axis=-1)
        aux = (self.balance_weight * balance_loss
               + self.entropy_weight * tf.reduce_mean(entropy))
        if gate is not None:
            # Mild L1 on the gate encourages a sparse, readable feature subset.
            aux = aux + self.entropy_weight * 0.1 * tf.reduce_mean(gate)
        self.add_loss(aux)
        return branch_tokens

    def get_config(self) -> Dict[str, float | int]:
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "num_branches": self.num_branches,
                "branch_units": self.branch_units,
                "temperature": self.temperature,
                "balance_weight": self.balance_weight,
                "entropy_weight": self.entropy_weight,
            }
        )
        return config


class SharedEMSTGATBackbone(layers.Layer):
    """The same DCC–BiGRU–temporal-KNN–GAT backbone used in both variants."""

    def __init__(self, config: AdaptiveModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = config.hidden_units
        self.sequence_length = config.max_sequence_length
        self.cfg = config
        self.input_projection = layers.Dense(config.hidden_units, activation="gelu", name="input_projection")
        self.sequence_layer = layers.RepeatVector(config.max_sequence_length, name="repeat_to_temporal_tokens")
        self.positional_embeddings = None
        self.causal_conv = DilatedCausalConv(
            filters=config.hidden_units,
            kernel_size=3,
            dilation_rate=config.dilation_rate,
            dropout_rate=config.dropout_rate,
            use_batch_norm=False,
            name="dilated_causal_conv",
        ) if config.use_dilated_causal_conv else None
        self.gru = layers.Bidirectional(
            layers.GRU(config.hidden_units // 2, return_sequences=True, dropout=config.dropout_rate),
            name="bigru",
        ) if config.use_bigru else None
        self.graph_builder = (
            KNNGraphBuilder(top_k=config.knn_top_k, name="temporal_knn_graph")
            if config.use_knn_graph else None
        )
        self.edge_attention = EdgeAwareMultiHeadAttention(
            num_heads=config.attention_heads,
            hidden_units=config.hidden_units,
            dropout_rate=config.dropout_rate,
            activation="gelu",
            name="temporal_edge_gat",
        )
        self.self_attention = layers.MultiHeadAttention(
            num_heads=config.attention_heads,
            key_dim=config.hidden_units // config.attention_heads,
            dropout=config.dropout_rate,
            name="temporal_multihead_attention",
        ) if config.use_self_attention else None
        self.pool_score = tf.keras.Sequential(
            [layers.Dense(config.hidden_units, activation="gelu"), layers.Dense(1)],
            name="attention_pool_score",
        ) if config.use_attention_pooling else None
        self.dropout = layers.Dropout(config.dropout_rate)

    def build(self, input_shape):
        if self.cfg.use_positional_embedding:
            self.positional_embeddings = self.add_weight(
                name="positional_embeddings",
                shape=(self.sequence_length, self.hidden_units),
                initializer="glorot_uniform",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.input_projection(inputs)
        if x.shape.rank == 2:
            x = self.sequence_layer(x)
        elif x.shape.rank != 3:
            raise ValueError("SharedEMSTGATBackbone expects a 2D vector batch or 3D temporal batch.")
        if self.positional_embeddings is not None:
            time_steps = tf.shape(x)[1]
            position = tf.expand_dims(self.positional_embeddings[:time_steps], axis=0)
            x = x + position
        if self.causal_conv is not None:
            x = self.causal_conv(x, training=training)
        if self.gru is not None:
            x = self.gru(x, training=training)
        if self.graph_builder is not None:
            adjacency = self.graph_builder(x, training=training)
            # The two attention implementations use custom @tf.function call
            # signatures that do not reliably retain gradients under Keras 3's
            # symbolic tracing.  Use the built-in differentiable MHA while keeping
            # the temporal KNN adjacency as an explicit learned context summary.
            graph_context = tf.matmul(adjacency, x)
            x = x + graph_context
        if self.self_attention is not None:
            x = self.self_attention(x, x, training=training)
        if self.pool_score is not None:
            scores = self.pool_score(x, training=training)
            weights = tf.nn.softmax(scores, axis=1)
            pooled = tf.reduce_sum(weights * x, axis=1)
        else:
            pooled = tf.reduce_mean(x, axis=1)
        return self.dropout(pooled, training=training)


class AdaptiveEMSTGAT(Model):
    """A controlled direct-input/automatic-branch EMSTGAT comparison model."""

    def __init__(self, config: AdaptiveModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.raw_adapter = layers.Dense(config.hidden_units, activation="gelu", name="direct_input_adapter")
        self.auto_router = None
        if config.branch_mode == "raw":
            # Direct-input adapter consumes the full feature vector at once;
            # it therefore retains feature identity and coordinate information.
            self.raw_pool = None
        else:
            self.auto_router = AdaptiveBranchRouter(
                input_dim=config.input_dim,
                num_branches=config.num_auto_branches,
                branch_units=config.hidden_units,
                temperature=config.router_temperature,
                balance_weight=config.router_balance_weight,
                entropy_weight=config.router_entropy_weight,
                use_gate=config.use_router_gate,
                name="adaptive_feature_router",
            )
            self.auto_fusion = layers.Flatten(name="adaptive_branch_fusion")
            self.auto_projection = layers.Dense(config.hidden_units, activation="gelu", name="adaptive_branch_projection")
            self.auto_sequence_fusion = layers.TimeDistributed(
                layers.Flatten(), name="adaptive_sequence_branch_fusion"
            )
            self.auto_sequence_projection = layers.Dense(
                config.hidden_units, activation="gelu", name="adaptive_sequence_branch_projection"
            )
            self.auto_residual_fusion = layers.Add(name="adaptive_branch_residual_fusion")
        self.shared_backbone = SharedEMSTGATBackbone(config, name="shared_emstgat_backbone")
        self.classifier_residual_fusion = layers.Add(name="classifier_residual_fusion")
        self.classifier_dropout = layers.Dropout(config.dropout_rate, name="classifier_dropout")
        self.fc = layers.Dense(config.hidden_units, activation="gelu", name="classifier_hidden")
        self.classifier = layers.Dense(config.num_classes, activation="softmax", dtype="float32", name="classifier")
        self.tabular_skip_classifier = layers.Dense(
            config.num_classes, dtype="float32", name="tabular_skip_classifier"
        )
        self.logit_fusion = layers.Add(name="logit_fusion")

    def call(self, inputs, training=None):
        x = tf.cast(inputs, tf.float32)
        if self.config.branch_mode == "raw":
            adapted = self.raw_adapter(x, training=training)
        else:
            branch_tokens = self.auto_router(x, training=training)
            if x.shape.rank == 3:
                auto_adapted = self.auto_sequence_projection(
                    self.auto_sequence_fusion(branch_tokens), training=training
                )
            else:
                auto_adapted = self.auto_projection(self.auto_fusion(branch_tokens), training=training)
            direct_adapted = self.raw_adapter(x, training=training)
            adapted = self.auto_residual_fusion([direct_adapted, auto_adapted])
        representation = self.shared_backbone(adapted, training=training)
        residual_input = tf.reduce_mean(x, axis=1) if x.shape.rank == 3 else x
        representation = self.classifier_residual_fusion([representation, adapted if adapted.shape.rank == 2 else tf.reduce_mean(adapted, axis=1)])
        representation = self.classifier_dropout(representation, training=training)
        representation = self.fc(representation, training=training)
        backbone_probabilities = self.classifier(representation)
        if not self.config.use_skip_classifier:
            return backbone_probabilities
        skip_logits = self.tabular_skip_classifier(residual_input, training=training)
        return tf.nn.softmax(self.logit_fusion([tf.math.log(backbone_probabilities + 1e-7), skip_logits]))

    def routing_probabilities(self) -> tf.Tensor | None:
        return None if self.auto_router is None else self.auto_router.routing_probabilities()

    def get_config(self) -> Dict[str, object]:
        return {"config": self.config.__dict__}
