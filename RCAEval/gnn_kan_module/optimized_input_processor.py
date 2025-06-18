"""
GNN+KAN 優化輸入處理器
專為KAN特性設計的高效輸入預處理系統
目標：最大化KAN性能，最小化輸入處理開銷
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import time
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import FastICA, PCA
import warnings

warnings.filterwarnings("ignore")


@dataclass
class KANOptimizedData:
    """專為KAN優化的數據格式"""
    node_features: torch.Tensor      # (num_nodes, feature_dim)
    edge_index: torch.Tensor         # (2, num_edges)  
    edge_weights: torch.Tensor       # (num_edges,)
    node_names: List[str]            # 節點名稱
    feature_names: List[str]         # 特徵名稱
    metadata: Dict[str, Any]         # 元數據
    
    def to_device(self, device: str):
        """移動到指定設備"""
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_weights = self.edge_weights.to(device)
        return self


class FastServiceExtractor:
    """快速微服務提取器"""
    
    def __init__(self):
        self.service_patterns = [
            'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
            'emailservice', 'paymentservice', 'productcatalogservice', 
            'recommendationservice', 'shippingservice', 'frontend'
        ]
        
    def extract_services_batch(self, columns: List[str]) -> Dict[str, List[str]]:
        """批量提取微服務對應的列"""
        service_columns = {}
        columns_lower = [col.lower() for col in columns]
        
        for service in self.service_patterns:
            matching_cols = [
                columns[i] for i, col_lower in enumerate(columns_lower)
                if service in col_lower
            ]
            if matching_cols:
                service_columns[service] = matching_cols
        
        return service_columns


class KANFeatureProcessor:
    """專為KAN優化的特徵處理器"""
    
    def __init__(self, method='ica', target_dim=64):
        self.method = method
        self.target_dim = target_dim
        self.service_extractor = FastServiceExtractor()
        
    def process_features_optimized(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """優化的特徵處理"""
        service_columns = self.service_extractor.extract_services_batch(data.columns.tolist())
        
        if not service_columns:
            return self._global_feature_processing(data)
        
        service_features = []
        service_names = []
        
        for service_name, cols in service_columns.items():
            service_data = data[cols].fillna(0).replace([np.inf, -np.inf], 0)
            
            if service_data.shape[1] == 0:
                continue
                
            if self.method == 'ica' and service_data.shape[1] >= 2:
                features = self._fast_ica_processing(service_data)
            else:
                features = self._fast_statistical_processing(service_data)
            
            service_features.append(features)
            service_names.append(service_name)
        
        aligned_features = self._fast_feature_alignment(service_features)
        return aligned_features, service_names
    
    def _fast_ica_processing(self, service_data: pd.DataFrame) -> np.ndarray:
        """快速ICA處理"""
        try:
            max_components = min(service_data.shape[1], service_data.shape[0] - 1, self.target_dim // 4)
            if max_components < 1:
                return self._fast_statistical_processing(service_data)
            
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(service_data)
            
            ica = FastICA(
                n_components=max_components,
                random_state=42,
                max_iter=200,
                tol=1e-3,
                algorithm='parallel'
            )
            
            components = ica.fit_transform(scaled_data)
            
            features = []
            for i in range(components.shape[1]):
                comp = components[:, i]
                features.extend([
                    np.mean(comp), np.std(comp), np.min(comp), np.max(comp)
                ])
            
            if len(features) >= self.target_dim:
                return np.array(features[:self.target_dim])
            else:
                padding = np.zeros(self.target_dim - len(features))
                return np.concatenate([features, padding])
                
        except Exception:
            return self._fast_statistical_processing(service_data)
    
    def _fast_statistical_processing(self, service_data: pd.DataFrame) -> np.ndarray:
        """快速統計處理"""
        features = [
            service_data.values.mean(),
            service_data.values.std(),
            service_data.values.min(),
            service_data.values.max(),
            np.median(service_data.values),
        ]
        
        if service_data.shape[1] <= 10:
            col_means = service_data.mean().values
            col_stds = service_data.std().values
            features.extend(col_means.tolist()[:5])
            features.extend(col_stds.tolist()[:5])
        
        current_len = len(features)
        if current_len >= self.target_dim:
            return np.array(features[:self.target_dim])
        else:
            padding = np.zeros(self.target_dim - current_len)
            return np.concatenate([features, padding])
    
    def _global_feature_processing(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """全局特徵處理"""
        features = self._fast_statistical_processing(data)
        return np.array([features]), ['global_service']
    
    def _fast_feature_alignment(self, service_features: List[np.ndarray]) -> np.ndarray:
        """快速特徵對齊"""
        if not service_features:
            return np.array([[0.0] * self.target_dim])
        
        aligned = []
        for features in service_features:
            if len(features) >= self.target_dim:
                aligned.append(features[:self.target_dim])
            else:
                padding = np.zeros(self.target_dim - len(features))
                aligned.append(np.concatenate([features, padding]))
        
        return np.array(aligned)


class OptimizedGraphBuilder:
    """優化的圖構建器"""
    
    def __init__(self, similarity_threshold=0.3, max_edges_per_node=5):
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
    
    def build_graph_fast(self, node_features: np.ndarray, node_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """快速圖構建"""
        num_nodes = len(node_names)
        
        if num_nodes <= 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long) if num_nodes == 1 else torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.tensor([1.0]) if num_nodes == 1 else torch.empty(0)
            return edge_index, edge_weights
        
        # 快速相似性計算
        node_features_norm = node_features / (np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(node_features_norm, node_features_norm.T)
        
        edges = []
        weights = []
        
        for i in range(num_nodes):
            similarities = similarity_matrix[i]
            similarities[i] = -1  # 排除自環
            
            if np.max(similarities) > self.similarity_threshold:
                top_indices = np.argsort(similarities)[-self.max_edges_per_node:]
                for j in top_indices:
                    if similarities[j] > self.similarity_threshold:
                        edges.append([i, j])
                        weights.append(similarities[j])
            
            # 添加自環
            edges.append([i, i])
            weights.append(1.0)
        
        if not edges:
            edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
            weights = [0.5] * num_nodes
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_weights = torch.tensor(weights, dtype=torch.float)
        
        return edge_index, edge_weights


class GNNKANInputOptimizer:
    """GNN+KAN 輸入優化器"""
    
    def __init__(self, 
                 feature_method='ica',
                 target_dim=64,
                 similarity_threshold=0.3,
                 max_edges_per_node=5):
        
        self.feature_processor = KANFeatureProcessor(feature_method, target_dim)
        self.graph_builder = OptimizedGraphBuilder(similarity_threshold, max_edges_per_node)
    
    def optimize_input(self, data: Any, inject_time: Optional[float] = None) -> KANOptimizedData:
        """優化輸入處理"""
        start_time = time.time()
        
        df = self._fast_data_standardization(data)
        node_features, node_names = self.feature_processor.process_features_optimized(df)
        edge_index, edge_weights = self.graph_builder.build_graph_fast(node_features, node_names)
        
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        processing_time = time.time() - start_time
        
        return KANOptimizedData(
            node_features=node_features_tensor,
            edge_index=edge_index,
            edge_weights=edge_weights,
            node_names=node_names,
            feature_names=[f'feature_{i}' for i in range(node_features.shape[1])],
            metadata={
                'processing_time': processing_time,
                'num_nodes': len(node_names),
                'num_edges': edge_index.size(1),
                'feature_method': self.feature_processor.method,
                'inject_time': inject_time
            }
        )
    
    def _fast_data_standardization(self, data: Any) -> pd.DataFrame:
        """快速數據標準化"""
        if isinstance(data, pd.DataFrame):
            df = data.select_dtypes(include=[np.number])
        elif isinstance(data, dict):
            if 'metrics' in data:
                df = pd.DataFrame(data['metrics'])
            else:
                combined_data = {}
                for key, value in data.items():
                    if isinstance(value, pd.DataFrame):
                        for col in value.select_dtypes(include=[np.number]).columns:
                            combined_data[f'{key}_{col}'] = value[col]
                df = pd.DataFrame(combined_data)
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({'metric': [float(data)] if np.isscalar(data) else [0.0]})
        
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        time_cols = [col for col in df.columns if 'time' in str(col).lower()]
        if time_cols:
            df = df.drop(columns=time_cols)
        
        return df


def optimize_gnn_kan_input(data: Any, 
                          feature_method: str = 'ica',
                          target_dim: int = 64,
                          inject_time: Optional[float] = None,
                          **kwargs) -> KANOptimizedData:
    """一鍵優化GNN+KAN輸入"""
    optimizer = GNNKANInputOptimizer(
        feature_method=feature_method,
        target_dim=target_dim,
        **kwargs
    )
    
    return optimizer.optimize_input(data, inject_time) 