"""
KAN Components Module
包含所有 KAN 相關的核心組件和功能
"""

# 核心 KAN 層
from .kan_layer import (
    KANLayer,
    SimplifiedKANLayer, 
    UltraFastKANLayer,
    FastKANLayer,
    AdvancedKANLayer,
    OptimizedGNNKANEncoder,
    OptimizedGNNKANEncoder as GNNKANEncoder  # 統一使用優化版本
)

# 梯度穩定器
from .gradient_stabilizer import GradientStabilizer, StabilizedKANLayer

# 高容量穩定 KAN
from .high_capacity_stable_kan import (
    HighCapacityStableKANLayer,
    create_high_capacity_kan_encoder
)

# 特徵提取功能
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

# 從 utils 導入統一的權重計算函數
def compute_service_criticality_weights(node_names):
    """
    基於服務名稱計算重要性權重 - 統一實現
    """
    # 導入時避免循環依賴
    try:
        from ..utils import compute_service_criticality_weights as _compute_weights
        return _compute_weights(node_names)
    except ImportError:
        # 回退實現
        critical_services = {
            'frontend': 3.0, 'front-end': 3.0,
            'checkout': 2.8, 'payment': 2.8,
            'cart': 2.5, 'catalog': 2.2,
            'currency': 2.0, 'redis': 2.3,
            'database': 2.5, 'db': 2.5,
            'email': 1.8, 'ad': 1.6,
            'recommendation': 1.7
        }
        
        weights = []
        for name in node_names:
            name_str = str(name).lower()
            weight = 1.0
            
            for service, service_weight in critical_services.items():
                if service in name_str:
                    weight = max(weight, service_weight)
            
            weights.append(min(weight, 3.0))
        
        import torch
        return torch.tensor(weights, dtype=torch.float)

# 導出所有重要功能 - 與原始 kan 模組保持一致
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
    
    # 高容量穩定 KAN
    'HighCapacityStableKANLayer',
    'create_high_capacity_kan_encoder',
    
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
    'extract_service_topology_features',
    'compute_service_criticality_weights'
]