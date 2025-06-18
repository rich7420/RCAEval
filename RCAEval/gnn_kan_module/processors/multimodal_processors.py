"""
統一多模態處理器
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

from ..core.base_classes import BaseFeatureProcessor
from ..core.data_interface import StandardizedData
from .log_processors import UnifiedLogProcessor
from .metric_processors import UnifiedMetricProcessor
from .trace_processors import UnifiedTraceProcessor


class UnifiedMultiModalProcessor(BaseFeatureProcessor):
    """統一多模態處理器"""
    
    def __init__(self, target_dim: int = 64, method: str = 'auto', **kwargs):
        super().__init__(target_dim, method, **kwargs)
        self.log_processor = UnifiedLogProcessor(target_dim//3, method)
        self.metric_processor = UnifiedMetricProcessor(target_dim//3, method)
        self.trace_processor = UnifiedTraceProcessor(target_dim//3, method)
    
    def fit(self, data: Union[StandardizedData, Any], **kwargs) -> 'UnifiedMultiModalProcessor':
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[StandardizedData, Any]) -> Tuple[np.ndarray, List[str]]:
        """轉換多模態數據為特徵"""
        if not isinstance(data, StandardizedData):
            from ..core.data_interface import UnifiedDataInterface
            data = UnifiedDataInterface.standardize_input(data, data_type='multimodal')
        
        # 多模態數據已經在標準化時處理了
        return data.data, data.feature_names 