"""
GNN-KAN Module: 簡化的純粹KAN實現
專注核心功能，移除冗餘組件
"""

# 核心配置
from .config import (
    GNNKANConfig,
    SimplifiedGNNKANConfig,
    create_config
)

# 核心模型
from .models import (
    GNNKANModel,
    SimplifiedGNNKAN
)

# 訓練模組
from .training import (
    train_gnn_kan_model,
)

# 特徵處理
from .feature_processing import (
    ica_metric_processing,
    simplified_metric_processing,
    enhanced_ica_with_temporal_contrast
)

# KAN 組件
from .kan_components import (
    AdvancedKANLayer,
    SimplifiedKANLayer
)

# 輸入優化器
from .optimized_input_processor import (
    GNNKANInputOptimizer,
    KANOptimizedData
)

print("✅ 簡化版 GNN-KAN 模組載入成功")