"""
KAN (Kolmogorov-Arnold Networks) module for RCAEval
包含 GPU 優化的 KAN 實現和特徵提取功能
"""

from .kan_layer import (
    KANLayer, GNNKANEncoder,
    SimplifiedKANLayer, UltraFastKANLayer, FastKANLayer, OptimizedGNNKANEncoder
)
from .feature_extraction import (
    sliding_window_alignment,
    extract_log_features,
    stl_decomposition,
    kll_feature_processing,
    compute_topology_features,
    extract_error_features,
    feature_fusion,
    extract_trace_features,
    build_service_dependency_graph,
    extract_service_topology_features
)

__all__ = [
    # 原始 KAN 層
    'KANLayer',
    'GNNKANEncoder',
    
    # GPU 優化的 KAN 層
    'SimplifiedKANLayer',
    'UltraFastKANLayer', 
    'FastKANLayer',
    'OptimizedGNNKANEncoder',
    
    # 特徵提取功能
    'sliding_window_alignment',
    'extract_log_features',
    'stl_decomposition',
    'kll_feature_processing',
    'compute_topology_features',
    'extract_error_features',
    'feature_fusion',
    'extract_trace_features',
    'build_service_dependency_graph',
    'extract_service_topology_features'
]