"""
Pure KAN Components Module
純粹KAN組件模組 - 專注於KAN取代MLP的核心功能
"""

# 純粹KAN層實現 - 核心組件
from .kan_layers import (
    AdvancedKANLayer,
    SimplifiedKANLayer,
    OptimizedGNNKANEncoder,
    KANLayer,  # 向後兼容
    GNNKANEncoder  # 向後兼容
)

# 梯度穩定器 - KAN特有的穩定性組件
from .gradient_stabilizer import (
    GradientStabilizer
)

# 高容量穩定KAN
from .high_capacity_stable_kan import (
    HighCapacityKANEncoder
)

# 工具函數
def compute_service_criticality_weights(adj_matrix, method='pagerank', alpha=0.85):
    """計算服務關鍵性權重 - 用於KAN模型的權重初始化"""
    import numpy as np
    
    try:
        n = adj_matrix.shape[0]
        if n == 0:
            return np.array([])
        
        if method == 'pagerank':
            # 簡化PageRank實現
            adj_norm = adj_matrix / (np.sum(adj_matrix, axis=1, keepdims=True) + 1e-8)
            pr = np.ones(n) / n
            
            for _ in range(100):
                pr_new = (1 - alpha) / n + alpha * np.dot(adj_norm.T, pr)
                if np.linalg.norm(pr_new - pr, 1) < 1e-6:
                    break
                pr = pr_new
            
            return pr
        elif method == 'degree':
            degrees = np.sum(adj_matrix, axis=1)
            return degrees / (np.sum(degrees) + 1e-8)
        else:
            return np.ones(n) / n
            
    except Exception:
        return np.ones(adj_matrix.shape[0]) / max(adj_matrix.shape[0], 1)

# 確保所有KAN組件都可以被導入
__all__ = [
    # 純粹KAN層
    'AdvancedKANLayer',
    'SimplifiedKANLayer', 
    'OptimizedGNNKANEncoder',
    'KANLayer',
    'GNNKANEncoder',
    
    # 穩定性組件
    'GradientStabilizer',
    'HighCapacityKANEncoder',
    
    # 工具函數
    'compute_service_criticality_weights'
]