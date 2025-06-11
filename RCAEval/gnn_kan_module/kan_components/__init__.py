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

# 實用函數
def compute_service_criticality_weights(service_graph=None, criticality_metrics=None):
    """計算服務關鍵度權重"""
    if isinstance(service_graph, list):
        # 如果傳入的是服務名稱列表
        return [1.0] * len(service_graph)
    
    if service_graph is None or (hasattr(service_graph, 'number_of_nodes') and service_graph.number_of_nodes() == 0):
        return {}
    
    try:
        import networkx as nx
        
        # 基於拓撲結構計算關鍵度
        pagerank = nx.pagerank(service_graph)
        betweenness = nx.betweenness_centrality(service_graph)
        degree_centrality = nx.degree_centrality(service_graph)
        
        # 組合權重
        weights = {}
        for node in service_graph.nodes():
            weight = (
                0.4 * pagerank.get(node, 0) +
                0.3 * betweenness.get(node, 0) +
                0.3 * degree_centrality.get(node, 0)
            )
            weights[node] = weight
        
        # 如果有業務關鍵度指標，結合使用
        if criticality_metrics:
            for node, business_weight in criticality_metrics.items():
                if node in weights:
                    weights[node] = 0.7 * weights[node] + 0.3 * business_weight
        
        return weights
        
    except Exception as e:
        print(f"Failed to compute service criticality weights: {e}")
        return {}

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