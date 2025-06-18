"""
GNN-KAN Core Module
核心模組 - 提供統一的數據接口和基礎類
"""

from .data_interface import (
    UnifiedDataInterface,
    StandardizedData,
    DataType
)
from .base_classes import (
    BaseFeatureProcessor,
    BaseGraphConstructor,
    BaseKANProcessor
)

__all__ = [
    'UnifiedDataInterface',
    'StandardizedData', 
    'DataType',
    'BaseFeatureProcessor',
    'BaseGraphConstructor',
    'BaseKANProcessor'
] 