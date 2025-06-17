"""
KAN Components: 純粹KAN實現的核心組件
專注於KAN取代MLP的核心價值
"""

# 基礎KAN層
from .kan_layers import (
    SimplifiedKANLayer,
    OptimizedGNNKANEncoder,
    AdvancedKANLayer  # alias to SimplifiedKANLayer for compatibility
)

# 高容量穩定KAN (可選)
try:
    from .high_capacity_stable_kan import (
        HighCapacityGNNKANEncoder,
        HighCapacityStableKANLayer
    )
except ImportError:
    HighCapacityGNNKANEncoder = None
    HighCapacityStableKANLayer = None

# 梯度穩定器
from .gradient_stabilizer import GradientStabilizer

# 確保所有組件都可以被導入
__all__ = [
    # 基礎KAN層
    'AdvancedKANLayer',
    'SimplifiedKANLayer',
    'OptimizedGNNKANEncoder',
    
    # 高容量KAN
    'HighCapacityGNNKANEncoder',
    'HighCapacityStableKANLayer',
    
    # 穩定性組件
    'GradientStabilizer',
    
    # 工具函數
    'create_high_capacity_stable_model'
]

print("✅ KAN組件完全載入成功 - 專注於KAN取代MLP的核心價值")