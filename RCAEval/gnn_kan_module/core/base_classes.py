"""
基礎處理器類 - 統一特徵處理接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from .data_interface import StandardizedData, UnifiedDataInterface


class BaseFeatureProcessor(ABC):
    """基礎特徵處理器 - 統一所有特徵處理的接口"""
    
    def __init__(self, target_dim: int = 64, method: str = 'auto', **kwargs):
        """
        初始化特徵處理器
        
        Args:
            target_dim: 目標特徵維度
            method: 處理方法 ('ica', 'kpca', 'pca', 'simplified', 'auto')
            **kwargs: 其他參數
        """
        self.target_dim = target_dim
        self.method = method
        self.config = kwargs
        self.is_fitted = False
        self.feature_names_ = []
        self.scaler_ = None
    
    @abstractmethod
    def fit(self, data: Union[StandardizedData, Any], **kwargs) -> 'BaseFeatureProcessor':
        """訓練特徵處理器"""
        pass
    
    @abstractmethod
    def transform(self, data: Union[StandardizedData, Any]) -> Tuple[np.ndarray, List[str]]:
        """轉換數據為特徵"""
        pass
    
    def fit_transform(self, data: Union[StandardizedData, Any], **kwargs) -> Tuple[np.ndarray, List[str]]:
        """訓練並轉換數據"""
        return self.fit(data, **kwargs).transform(data)
    
    def process(self, data: Any, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        統一的處理接口
        
        Args:
            data: 任意格式的輸入數據
            **kwargs: 其他參數
            
        Returns:
            features: 處理後的特徵矩陣
            feature_names: 特徵名稱列表
        """
        # 1. 數據標準化
        if not isinstance(data, StandardizedData):
            data = UnifiedDataInterface.standardize_input(data, **kwargs)
        
        # 2. 特徵提取和轉換
        features, feature_names = self.fit_transform(data, **kwargs)
        
        # 3. 維度適配
        features = self._adapt_dimensions(features)
        
        # 4. 數值穩定化
        features = self._stabilize_features(features)
        
        # 更新特徵名稱
        if len(feature_names) != features.shape[1]:
            feature_names = [f'{self.method}_feature_{i}' for i in range(features.shape[1])]
        
        return features, feature_names
    
    def _adapt_dimensions(self, features: np.ndarray) -> np.ndarray:
        """適配維度到目標維度"""
        if features.size == 0:
            return np.zeros((1, self.target_dim))
        
        # 確保是二維數組
        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim == 0:
            features = features.reshape(1, 1)
        
        current_dim = features.shape[1]
        
        if current_dim == self.target_dim:
            return features
        elif current_dim > self.target_dim:
            # PCA 降維
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.target_dim, random_state=42)
                features = pca.fit_transform(features)
            except:
                # 簡單截斷
                features = features[:, :self.target_dim]
        else:
            # 零填充或特徵複製
            if current_dim > 0:
                # 重複特徵填充
                repeat_times = (self.target_dim + current_dim - 1) // current_dim
                repeated_features = np.tile(features, (1, repeat_times))
                features = repeated_features[:, :self.target_dim]
            else:
                # 零填充
                features = np.zeros((features.shape[0], self.target_dim))
        
        return features
    
    def _stabilize_features(self, features: np.ndarray) -> np.ndarray:
        """數值穩定化處理"""
        # 處理無效值
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 避免過大或過小的值
        features = np.clip(features, -1e6, 1e6)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """獲取特徵名稱"""
        return self.feature_names_


class BaseGraphConstructor(ABC):
    """基礎圖構建器"""
    
    def __init__(self, config: Any = None):
        """
        初始化圖構建器
        
        Args:
            config: 配置對象
        """
        self.config = config
        self.is_learnable = getattr(config, 'learnable_graph', False) if config else False
    
    @abstractmethod
    def build_graph(self, features: np.ndarray, node_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        構建圖結構
        
        Args:
            features: 節點特徵矩陣
            node_names: 節點名稱列表
            
        Returns:
            edge_index: 邊索引 [2, num_edges]
            edge_weights: 邊權重 [num_edges]
        """
        pass
    
    def _create_fully_connected_graph(self, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """創建全連接圖"""
        edges = []
        weights = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # 排除自環
                    edges.append([i, j])
                    weights.append(1.0)
        
        if not edges:
            # 單節點情況
            edges = [[0, 0]]
            weights = [1.0]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(weights, dtype=torch.float32)
        
        return edge_index, edge_weights
    
    def _create_similarity_graph(self, features: np.ndarray, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """基於相似度創建圖"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            # 計算餘弦相似度
            similarity_matrix = cosine_similarity(features)
            
            # 應用閾值
            edges = []
            weights = []
            
            num_nodes = similarity_matrix.shape[0]
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and similarity_matrix[i, j] > threshold:
                        edges.append([i, j])
                        weights.append(similarity_matrix[i, j])
            
            if not edges:
                # 如果沒有邊，回退到全連接
                return self._create_fully_connected_graph(num_nodes)
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(weights, dtype=torch.float32)
            
            return edge_index, edge_weights
            
        except Exception as e:
            print(f"⚠️ 相似度圖構建失敗: {e}，使用全連接圖")
            return self._create_fully_connected_graph(features.shape[0])


class BaseKANProcessor(ABC):
    """基礎 KAN 處理器"""
    
    def __init__(self, config: Any = None):
        """
        初始化 KAN 處理器
        
        Args:
            config: 配置對象
        """
        self.config = config
        self.kan_config = self._extract_kan_config(config)
    
    def _extract_kan_config(self, config: Any) -> Dict[str, Any]:
        """提取 KAN 相關配置"""
        kan_config = {
            'num_basis': getattr(config, 'kan_grid_size', 8),
            'grid_size': getattr(config, 'kan_grid_size', 8),
            'spline_order': getattr(config, 'kan_spline_order', 3),
            'adaptive_spline_order': getattr(config, 'adaptive_spline_order', True),
            'learnable_activation': getattr(config, 'learnable_activation', True)
        }
        return kan_config
    
    @abstractmethod
    def create_kan_layer(self, input_dim: int, output_dim: int) -> nn.Module:
        """創建 KAN 層"""
        pass
    
    @abstractmethod  
    def process_with_kan(self, data: torch.Tensor) -> torch.Tensor:
        """使用 KAN 處理數據"""
        pass
    
    def get_kan_parameters(self) -> Dict[str, Any]:
        """獲取 KAN 參數配置"""
        return self.kan_config.copy()


# 工具函數
def create_processor_factory(processor_type: str) -> BaseFeatureProcessor:
    """
    處理器工廠函數
    
    Args:
        processor_type: 處理器類型 ('metrics', 'logs', 'traces', 'multimodal')
        
    Returns:
        對應的處理器實例
    """
    if processor_type == 'metrics':
        from ..feature_processing import MetricProcessor
        return MetricProcessor()
    elif processor_type == 'logs':
        from ..feature_processing import LogProcessor  
        return LogProcessor()
    elif processor_type == 'traces':
        from ..feature_processing import TraceProcessor
        return TraceProcessor()
    elif processor_type == 'multimodal':
        from ..feature_processing import MultiModalProcessor
        return MultiModalProcessor()
    else:
        raise ValueError(f"未知的處理器類型: {processor_type}")


def validate_processor_config(config: Dict[str, Any]) -> List[str]:
    """
    驗證處理器配置
    
    Args:
        config: 配置字典
        
    Returns:
        問題列表（空列表表示無問題）
    """
    issues = []
    
    # 檢查必需的配置項
    required_keys = ['target_dim', 'method']
    for key in required_keys:
        if key not in config:
            issues.append(f"缺少必需配置項: {key}")
    
    # 檢查數值範圍
    if 'target_dim' in config:
        target_dim = config['target_dim']
        if not isinstance(target_dim, int) or target_dim <= 0:
            issues.append(f"target_dim 必須是正整數，當前值: {target_dim}")
    
    # 檢查方法名稱
    if 'method' in config:
        method = config['method']
        valid_methods = ['ica', 'kpca', 'pca', 'simplified', 'auto']
        if method not in valid_methods:
            issues.append(f"無效的方法名稱: {method}，支持的方法: {valid_methods}")
    
    return issues 