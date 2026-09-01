"""Public exports for the physics-decoupled EMSTGAT package."""

from .model_design import (
    PHYSICAL_FEATURE_GROUPS,
    PHYSICAL_GROUP_ORDER,
    DilatedCausalConv,
    EdgeAwareMultiHeadAttention,
    EnhancedMSTGAT,
    KNNGraphBuilder,
    ModelConfig,
    MultiHeadAttention,
    PhysicalBranchExtractor,
    build_physical_feature_group_indices,
    select_physics_aware_features,
)
from .plotting import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_training_history,
)

__all__ = [
    'PHYSICAL_FEATURE_GROUPS',
    'PHYSICAL_GROUP_ORDER',
    'DilatedCausalConv',
    'EdgeAwareMultiHeadAttention',
    'EnhancedMSTGAT',
    'KNNGraphBuilder',
    'ModelConfig',
    'MultiHeadAttention',
    'PhysicalBranchExtractor',
    'build_physical_feature_group_indices',
    'select_physics_aware_features',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_training_history',
]
