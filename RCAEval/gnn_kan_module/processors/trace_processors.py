"""
統一鏈路追蹤處理器
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union

from ..core.base_classes import BaseFeatureProcessor
from ..core.data_interface import StandardizedData


class UnifiedTraceProcessor(BaseFeatureProcessor):
    """統一鏈路追蹤處理器"""
    
    def fit(self, data: Union[StandardizedData, Any], **kwargs) -> 'UnifiedTraceProcessor':
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[StandardizedData, Any]) -> Tuple[np.ndarray, List[str]]:
        """轉換鏈路追蹤數據為特徵"""
        try:
            from ..feature_processing import enhanced_trace_processing
            features, feature_names = enhanced_trace_processing(data, kwargs.get('inject_time'))
            return features, feature_names
        except Exception as e:
            print(f"⚠️ 鏈路追蹤處理失敗: {e}")
            features = np.zeros((1, self.target_dim))
            feature_names = [f'trace_feature_{i}' for i in range(self.target_dim)]
            return features, feature_names


def extract_trace_features(trace_data, inject_time=None, target_dim=64, **kwargs):
    """向後兼容的鏈路追蹤特徵提取函數"""
    processor = UnifiedTraceProcessor(target_dim=target_dim)
    return processor.process(trace_data, inject_time=inject_time) 