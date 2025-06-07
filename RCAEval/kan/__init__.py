"""
KAN (Kolmogorov-Arnold Networks) module for RCAEval
包含 GPU 優化的 KAN 實現和特徵提取功能
"""

from .kan_layer import (
    KANLayer, OptimizedGNNKANEncoder as GNNKANEncoder,  # 統一使用優化版本
    SimplifiedKANLayer, UltraFastKANLayer, FastKANLayer, OptimizedGNNKANEncoder,
    AdvancedKANLayer
)
from .gradient_stabilizer import GradientStabilizer, StabilizedKANLayer
from .feature_extraction import (
    sliding_window_alignment,
    extract_log_features,
    stl_decomposition,
    kll_feature_processing,
    compute_topology_features,
    extract_error_features,
    feature_fusion,
    enhanced_feature_fusion,
    extract_trace_features,
    build_service_dependency_graph,
    extract_service_topology_features
)

__all__ = [
    # 統一的 KAN 編碼器
    'KANLayer',
    'GNNKANEncoder',  # 指向 OptimizedGNNKANEncoder
    'OptimizedGNNKANEncoder',
    
    # GPU 優化的 KAN 層
    'AdvancedKANLayer',
    'SimplifiedKANLayer',
    'UltraFastKANLayer', 
    'FastKANLayer',
    
    # 梯度穩定化組件
    'GradientStabilizer',
    'StabilizedKANLayer',
    
    # 特徵提取函數
    'sliding_window_alignment',
    'extract_log_features',
    'stl_decomposition',
    'kll_feature_processing',
    'compute_topology_features',
    'extract_error_features',
    'feature_fusion',
    'enhanced_feature_fusion',
    'extract_trace_features',
    'build_service_dependency_graph',
    'extract_service_topology_features'
]