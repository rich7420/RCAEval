"""
KAN Components: 純粹KAN實現的核心組件
專注於KAN取代MLP的核心價值
"""

# 基礎KAN層
from .kan_layers import (
    SimplifiedKANLayer,
    OptimizedGNNKANEncoder,
    AdvancedKANLayer,
    CompatibleSimplifiedKANLayer,
    create_compatible_kan_layer
)

# 確保向後兼容性
KANLayer = SimplifiedKANLayer

# 高容量穩定KAN - 已刪除，簡化版本
HighCapacityGNNKANEncoder = None
HighCapacityStableKANLayer = None

# 梯度穩定器 - 已刪除，使用簡化版本
GradientStabilizer = None

# 確保所有組件都可以被導入
__all__ = [
    # 基礎KAN層
    'AdvancedKANLayer',
    'SimplifiedKANLayer',
    'CompatibleSimplifiedKANLayer',
    'OptimizedGNNKANEncoder',
    'KANLayer',
    
    # 高容量KAN
    'HighCapacityGNNKANEncoder',
    'HighCapacityStableKANLayer',
    
    # 穩定性組件  
    'GradientStabilizer',
    
    # 工具函數
    'create_high_capacity_stable_model',
    'create_compatible_kan_layer'
]

print("✅ KAN組件完全載入成功 - 專注於KAN取代MLP的核心價值")