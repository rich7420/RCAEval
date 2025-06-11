"""
GNN-KAN Module: 完整的模組化GNN-KAN實現
整合所有核心功能到統一模組中
"""

# 核心配置
from .config import SimplifiedGNNKANConfig

# 特徵提取模組
from .feature_extractors import (
    MultiModalFeatureExtractor,
    enhanced_feature_fusion
)

# 模型核心組件
from .models import (
    GNNKANModel,
    SimplifiedGNNKAN,
    TemporalAttention,
    create_model_with_config,
    compute_loss_stable
)

# 訓練模組
from .training import (
    train_gnn_kan_model,
    ModelManager,
    AdvancedGNNKANTrainer
)

# 圖構建
from .graph_constructors import (
    SimplifiedGraphConstructor,
    IntelligentServiceGraphConstructor
)

# 高級處理器
from .advanced_processors import (
    DynamicModelAdjuster,
    create_advanced_processor
)

# 特徵處理函數
from .feature_processing import (
    simplified_metric_processing,
    enhanced_trace_processing,
    psm_metric_processing
)

# KAN 組件
from .kan_components import (
    OptimizedGNNKANEncoder,
    UltraFastKANLayer,
    GradientStabilizer,
    StabilizedKANLayer
)

# 工具函數
from .utils import (
    compute_service_criticality_weights,
    validate_model_setup
)

# 別名定義 - 確保向後兼容
AdvancedTrainingManager = AdvancedGNNKANTrainer

# 確保所有主要組件都可以被導入
__all__ = [
    'SimplifiedGNNKANConfig',
    'MultiModalFeatureExtractor', 
    'GNNKANModel',
    'SimplifiedGNNKAN',
    'train_gnn_kan_model',
    'create_model_with_config',
    'validate_model_setup',
    'AdvancedTrainingManager',
    'AdvancedGNNKANTrainer',
    'SimplifiedGraphConstructor',
    'IntelligentServiceGraphConstructor', 
    'DynamicModelAdjuster',
    'create_advanced_processor',
    'enhanced_feature_fusion',
    'simplified_metric_processing',
    'enhanced_trace_processing', 
    'psm_metric_processing',
    'compute_service_criticality_weights',
    'TemporalAttention',
    'ModelManager',
    'OptimizedGNNKANEncoder',
    'UltraFastKANLayer',
    'GradientStabilizer',
    'StabilizedKANLayer',
    'compute_loss_stable'
]

print("✅ GNN-KAN模組完全載入成功 - 所有功能已模組化")