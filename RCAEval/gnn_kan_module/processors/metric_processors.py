"""
統一指標處理器 - 合併重複的指標特徵提取函數
整合 ICA, kPCA, PCA 等方法
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import warnings

from ..core.base_classes import BaseFeatureProcessor
from ..core.data_interface import StandardizedData

warnings.filterwarnings("ignore")


class UnifiedMetricProcessor(BaseFeatureProcessor):
    """統一指標處理器 - 整合所有指標處理方法"""
    
    def __init__(self, target_dim: int = 64, method: str = 'ica', **kwargs):
        super().__init__(target_dim, method, **kwargs)
        self.scaler = StandardScaler()
        self.processor = None
        
    def fit(self, data: Union[StandardizedData, Any], **kwargs) -> 'UnifiedMetricProcessor':
        """訓練指標處理器"""
        if not isinstance(data, StandardizedData):
            from ..core.data_interface import UnifiedDataInterface
            data = UnifiedDataInterface.standardize_input(data, data_type='metrics')
        
        # 準備數據
        metric_data = self._prepare_metric_data(data)
        
        # 根據方法選擇處理器
        if self.method == 'ica':
            self._fit_ica(metric_data)
        elif self.method == 'kpca':
            self._fit_kpca(metric_data)
        elif self.method == 'pca':
            self._fit_pca(metric_data)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[StandardizedData, Any]) -> Tuple[np.ndarray, List[str]]:
        """轉換指標數據為特徵"""
        if not isinstance(data, StandardizedData):
            from ..core.data_interface import UnifiedDataInterface
            data = UnifiedDataInterface.standardize_input(data, data_type='metrics')
        
        metric_data = self._prepare_metric_data(data)
        
        if self.method == 'ica':
            return self._transform_ica(metric_data)
        elif self.method == 'kpca':
            return self._transform_kpca(metric_data)
        elif self.method == 'pca':
            return self._transform_pca(metric_data)
        else:
            return self._transform_simplified(metric_data)
    
    def _prepare_metric_data(self, data: StandardizedData) -> np.ndarray:
        """準備指標數據"""
        if isinstance(data.data, pd.DataFrame):
            return data.data.select_dtypes(include=[np.number]).fillna(0).values
        else:
            return np.array(data.data)
    
    def _fit_ica(self, data: np.ndarray):
        """訓練ICA"""
        try:
            from sklearn.decomposition import FastICA
            n_components = min(data.shape[1], self.target_dim // 8, data.shape[0])
            self.processor = FastICA(n_components=n_components, random_state=42)
            self.scaler.fit(data)
        except Exception as e:
            print(f"⚠️ ICA初始化失敗: {e}")
            self.processor = None
    
    def _transform_ica(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """ICA變換"""
        try:
            # 使用原有的ICA處理邏輯
            from ..feature_processing import ica_metric_processing
            features, feature_names, _ = ica_metric_processing(data, target_dim=self.target_dim)
            return features, feature_names
        except Exception as e:
            print(f"⚠️ ICA變換失敗: {e}")
            return self._transform_simplified(data)
    
    def _fit_pca(self, data: np.ndarray):
        """訓練PCA"""
        try:
            n_components = min(data.shape[1], self.target_dim, data.shape[0])
            self.processor = PCA(n_components=n_components, random_state=42)
            self.scaler.fit(data)
        except Exception as e:
            print(f"⚠️ PCA初始化失敗: {e}")
            self.processor = None
    
    def _transform_pca(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """PCA變換"""
        try:
            scaled_data = self.scaler.transform(data)
            features = self.processor.fit_transform(scaled_data)
            feature_names = [f'pca_component_{i}' for i in range(features.shape[1])]
            return features, feature_names
        except Exception as e:
            print(f"⚠️ PCA變換失敗: {e}")
            return self._transform_simplified(data)
    
    def _fit_kpca(self, data: np.ndarray):
        """訓練kPCA"""
        try:
            from sklearn.decomposition import KernelPCA
            n_components = min(data.shape[1], self.target_dim // 8, data.shape[0])
            self.processor = KernelPCA(n_components=n_components, kernel='rbf', random_state=42)
            self.scaler.fit(data)
        except Exception as e:
            print(f"⚠️ kPCA初始化失敗: {e}")
            self.processor = None
    
    def _transform_kpca(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """kPCA變換"""
        try:
            from ..feature_processing import kpca_metric_processing
            features, feature_names = kpca_metric_processing(data, target_dim=self.target_dim)
            return features, feature_names
        except Exception as e:
            print(f"⚠️ kPCA變換失敗: {e}")
            return self._transform_simplified(data)
    
    def _transform_simplified(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """簡化變換"""
        try:
            from ..feature_processing import simplified_metric_processing
            features, feature_names = simplified_metric_processing(data, target_dim=self.target_dim)
            return features, feature_names
        except Exception as e:
            print(f"⚠️ 簡化處理失敗: {e}")
            # 最基本的回退
            features = np.zeros((1, self.target_dim))
            feature_names = [f'default_metric_{i}' for i in range(self.target_dim)]
            return features, feature_names


# 向後兼容函數
def extract_metric_features(metrics_data, method='ica', target_dim=64, **kwargs):
    """向後兼容的指標特徵提取函數"""
    processor = UnifiedMetricProcessor(target_dim=target_dim, method=method, **kwargs)
    return processor.process(metrics_data) 