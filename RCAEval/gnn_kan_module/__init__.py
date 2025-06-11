"""
GNN-KAN Module: 純粹KAN模組化實現
專注於KAN取代MLP的核心價值，確保功能完整性
"""

# 核心配置 - 支持多種配置模式
from .config import (
    SimplifiedGNNKANConfig,
    HighCapacityGNNKANConfig,
    FastGNNKANConfig
)

# 特徵提取模組 - 支持ICA/kPCA新方法
from .feature_extractors import (
    MultiModalFeatureExtractor,
    simplified_feature_fusion
)

# 模型核心組件 - 純粹KAN實現
from .models import (
    GNNKANModel,
    SimplifiedGNNKAN,
    TemporalAttention,
    create_model_with_config,
    compute_loss_stable
)

# 訓練模組 - 針對KAN優化
from .training import (
    train_gnn_kan_model,
    ModelManager,
    AdvancedGNNKANTrainer
)

# 圖構建 - 支持可學習圖結構
from .graph_constructors import (
    SimplifiedGraphConstructor,
    IntelligentServiceGraphConstructor,
    LearnableGraphConstructor,
    DynamicModelAdjuster
)

# 高級處理器
from .advanced_processors import (
    create_advanced_processor
)

# 特徵處理函數 - 新增ICA/kPCA支持
from .feature_processing import (
    ica_metric_processing,
    kpca_metric_processing,
    simplified_metric_processing,
    enhanced_trace_processing,
    psm_metric_processing
)

# 純粹KAN組件 - 只保留有效的KAN層
from .kan_components import (
    AdvancedKANLayer,
    SimplifiedKANLayer,
    OptimizedGNNKANEncoder,
    GradientStabilizer
)

# 工具函數
from .utils import (
    compute_service_criticality_weights,
    validate_model_setup
)

# 別名定義 - 確保向後兼容
AdvancedTrainingManager = AdvancedGNNKANTrainer
enhanced_feature_fusion = simplified_feature_fusion  # 統一接口

# 確保所有主要組件都可以被導入
__all__ = [
    # 配置類
    'SimplifiedGNNKANConfig',
    'HighCapacityGNNKANConfig', 
    'FastGNNKANConfig',
    
    # 特徵處理
    'MultiModalFeatureExtractor',
    'ica_metric_processing',
    'kpca_metric_processing', 
    'simplified_metric_processing',
    'enhanced_trace_processing',
    'psm_metric_processing',
    'simplified_feature_fusion',
    'enhanced_feature_fusion',
    
    # 核心模型
    'GNNKANModel',
    'SimplifiedGNNKAN',
    'TemporalAttention',
    'create_model_with_config',
    'compute_loss_stable',
    
    # KAN組件
    'AdvancedKANLayer',
    'SimplifiedKANLayer', 
    'OptimizedGNNKANEncoder',
    'GradientStabilizer',
    
    # 訓練
    'train_gnn_kan_model',
    'ModelManager',
    'AdvancedGNNKANTrainer',
    'AdvancedTrainingManager',
    
    # 圖構建
    'SimplifiedGraphConstructor',
    'IntelligentServiceGraphConstructor',
    'LearnableGraphConstructor', 
    'DynamicModelAdjuster',
    
    # 高級處理
    'create_advanced_processor',
    
    # 工具函數
    'compute_service_criticality_weights',
    'validate_model_setup'
]

print("✅ 純粹KAN模組完全載入成功 - 專注於KAN取代MLP的核心價值")