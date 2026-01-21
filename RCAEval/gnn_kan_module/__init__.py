"""
GNN-KAN Module: Simplified pure KAN implementation
Focus on core functionality, remove redundant components
"""

# Core configuration
from .config import (
    GNNKANConfig,
    SimplifiedGNNKANConfig,
    create_config
)

# Core models
from .models import (
    GNNKANModel,
    SimplifiedGNNKAN
)

# Training module
from .training import (
    train_gnn_kan_model,
)

# Feature processing
from .feature_processing import (
    ica_metric_processing,
    simplified_metric_processing,
    enhanced_ica_with_temporal_contrast
)

# KAN components
from .kan_components import (
    AdvancedKANLayer,
    SimplifiedKANLayer
)

# Input optimizer
from .optimized_input_processor import (
    GNNKANInputOptimizer,
    KANOptimizedData
)
