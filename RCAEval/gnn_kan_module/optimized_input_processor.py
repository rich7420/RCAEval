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
import os
from RCAEval.io.time_series import (
    drop_constant,
    drop_near_constant,
    convert_mem_mb,
    drop_extra,
    drop_time,
    select_useful_cols,
)

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
    """快速微服務提取器 - 學習其他e2e方法的通用性"""
    
    def __init__(self):
        # 🎯 分層的微服務模式匹配策略
        
        # 第一層：已知微服務名稱模式（來自數據集）
        self.known_services = {
            # Online Boutique
            'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
            'emailservice', 'paymentservice', 'productcatalogservice', 
            'recommendationservice', 'shippingservice', 'frontend',
            # Sock Shop
            'front-end', 'user', 'carts', 'orders', 'shipping', 'payment',
            'catalogue', 'user-db', 'carts-db', 'orders-db', 'catalogue-db',
            # Train Ticket  
            'ts-ui-dashboard', 'ts-auth-service', 'ts-user-service', 'ts-verification-code-service',
            'ts-account-service', 'ts-route-service', 'ts-train-service', 'ts-travel-service',
            'ts-preserve-service', 'ts-security-service', 'ts-inside-payment-service',
            'ts-execute-service', 'ts-contacts-service', 'ts-order-service', 'ts-order-other-service',
            'ts-config-service', 'ts-station-service', 'ts-travel2-service', 'ts-preserve-other-service',
            'ts-basic-service', 'ts-ticketinfo-service', 'ts-price-service', 'ts-notification-service',
            'ts-seat-service', 'ts-travel-plan-service', 'ts-route-plan-service', 'ts-food-service',
            'ts-consign-service', 'ts-consign-price-service', 'ts-admin-order-service',
            'ts-admin-basic-info-service', 'ts-admin-route-service', 'ts-admin-travel-service',
            'ts-admin-user-service', 'ts-cancel-service', 'ts-rebook-service', 'ts-assurance-service',
            'ts-food-map-service', 'ts-gateway-service'
        }
        
        # 第二層：通用組件模式
        self.component_patterns = [
            'service', 'api', 'backend', 'frontend', 'database', 'db', 'web', 'app',
            'server', 'client', 'worker', 'processor', 'handler', 'gateway', 'proxy',
            'auth', 'user', 'admin', 'ui', 'dashboard', 'config', 'monitor'
        ]
        
        # 第三層：指標類型模式（如果無法按服務分組）
        self.metric_patterns = {
            'cpu': ['cpu', 'processor'],
            'memory': ['mem', 'memory', 'ram'],
            'network': ['net', 'network', 'io', 'rx', 'tx', 'bytes'],
            'disk': ['disk', 'storage', 'volume'],
            'latency': ['lat', 'latency', 'response', 'duration'],
            'throughput': ['rps', 'qps', 'throughput', 'rate'],
            'error': ['error', 'err', 'exception', 'fail'],
            'availability': ['up', 'down', 'available', 'health']
        }
        
    def extract_services_batch(self, columns: List[str]) -> Dict[str, List[str]]:
        """批量提取微服務對應的列 - 多層策略"""
        service_columns = {}
        
        # 🎯 策略1：精確匹配已知微服務
        service_columns = self._extract_known_services(columns)
        
        # 🎯 策略2：如果精確匹配結果不足，使用模式匹配
        if len(service_columns) <= 1:
            service_columns.update(self._extract_pattern_services(columns))
        
        # 🎯 策略3：如果仍然不足，使用前綴分組
        if len(service_columns) <= 1:
            service_columns.update(self._extract_prefix_services(columns))
        
        # 🎯 策略4：如果還是不足，按指標類型分組
        if len(service_columns) <= 1:
            service_columns = self._extract_metric_services(columns)
        
        # 🚀 策略5：最後手段，智能分割確保多節點
        if len(service_columns) <= 1 and len(columns) > 1:
            service_columns = self._create_multiple_services(columns)
        
        # 📊 優化：清理空分組並限制分組數量
        service_columns = self._optimize_service_groups(service_columns, columns)
        
        return service_columns
    
    def _extract_known_services(self, columns: List[str]) -> Dict[str, List[str]]:
        """精確匹配已知微服務"""
        service_columns = {}
        
        for col in columns:
            col_lower = col.lower().replace('-', '_').replace('.', '_')
            
            # 檢查是否包含已知微服務名稱
            for service in self.known_services:
                service_normalized = service.replace('-', '_')
                if service_normalized in col_lower:
                    if service not in service_columns:
                        service_columns[service] = []
                    service_columns[service].append(col)
                    break
        
        return service_columns
    
    def _extract_pattern_services(self, columns: List[str]) -> Dict[str, List[str]]:
        """基於通用組件模式匹配"""
        service_columns = {}
        
        for col in columns:
            col_lower = col.lower()
            
            for pattern in self.component_patterns:
                if pattern in col_lower:
                    # 嘗試提取更具體的服務名
                    service_name = self._extract_specific_service_name(col, pattern)
                    if service_name not in service_columns:
                        service_columns[service_name] = []
                    service_columns[service_name].append(col)
                    break
        
        return service_columns
    
    def _extract_specific_service_name(self, col: str, pattern: str) -> str:
        """從列名中提取具體的服務名稱"""
        col_lower = col.lower()
        
        # 嘗試提取前綴
        if '_' in col_lower:
            parts = col_lower.split('_')
            for part in parts:
                if pattern in part:
                    # 找到包含模式的部分，返回前面的部分作為服務名
                    idx = parts.index(part)
                    if idx > 0:
                        return '_'.join(parts[:idx+1])
                    else:
                        return part
        
        # 如果沒有下劃線，嘗試提取包含模式的部分
        if pattern in col_lower:
            return f"{pattern}_service"
        
        return "unknown_service"
    
    def _extract_prefix_services(self, columns: List[str]) -> Dict[str, List[str]]:
        """基於前綴分組（學習BARO等方法的通用性）"""
        service_columns = {}
        
        # 收集所有前綴
        prefixes = {}
        for col in columns:
            # 嘗試多種分隔符
            for sep in ['_', '-', '.', ':']:
                if sep in col:
                    prefix = col.split(sep)[0].lower()
                    if len(prefix) >= 2:  # 避免太短的前綴
                        if prefix not in prefixes:
                            prefixes[prefix] = []
                        prefixes[prefix].append(col)
                        break
        
        # 只保留有多個列的前綴，或者所有前綴都只有一個列時保留所有
        if prefixes:
            multi_col_prefixes = {k: v for k, v in prefixes.items() if len(v) > 1}
            if multi_col_prefixes:
                service_columns.update({f"{k}_service": v for k, v in multi_col_prefixes.items()})
            else:
                # 如果所有前綴都只有一個列，也包含它們
                service_columns.update({f"{k}_service": v for k, v in prefixes.items()})
        
        return service_columns
    
    def _extract_metric_services(self, columns: List[str]) -> Dict[str, List[str]]:
        """按指標類型分組（回退策略）"""
        service_columns = {}
        
        for metric_type, patterns in self.metric_patterns.items():
            matching_cols = []
            for col in columns:
                col_lower = col.lower()
                for pattern in patterns:
                    if pattern in col_lower:
                        matching_cols.append(col)
                        break
            
            if matching_cols:
                service_columns[f"{metric_type}_metrics"] = matching_cols
        
        # 處理未分類的列
        classified_cols = set()
        for cols in service_columns.values():
            classified_cols.update(cols)
        
        unclassified = [col for col in columns if col not in classified_cols]
        if unclassified:
            service_columns['other_metrics'] = unclassified
        
        return service_columns
    
    def _create_multiple_services(self, columns: List[str]) -> Dict[str, List[str]]:
        """強制創建多個服務節點（最後手段）"""
        services = {}
        
        # 根據列數量動態決定服務數
        num_cols = len(columns)
        if num_cols <= 4:
            target_services = 2
        elif num_cols <= 12:
            target_services = min(4, num_cols // 2)
        else:
            target_services = min(8, num_cols // 3)
        
        cols_per_service = max(1, num_cols // target_services)
        
        for i in range(target_services):
            start_idx = i * cols_per_service
            end_idx = start_idx + cols_per_service if i < target_services - 1 else num_cols
            
            if start_idx < num_cols:
                service_cols = columns[start_idx:end_idx]
                services[f"auto_service_{i+1}"] = service_cols
        
        return services
    
    def _optimize_service_groups(self, service_columns: Dict[str, List[str]], 
                                columns: List[str]) -> Dict[str, List[str]]:
        """優化服務分組"""
        # 移除空分組
        service_columns = {k: v for k, v in service_columns.items() if v}
        
        # 限制分組數量（避免過度分割）
        max_services = min(15, max(2, len(columns) // 2))
        if len(service_columns) > max_services:
            # 保留最大的分組
            sorted_services = sorted(service_columns.items(), 
                                   key=lambda x: len(x[1]), reverse=True)
            service_columns = dict(sorted_services[:max_services])
        
        # 確保至少有2個分組
        if len(service_columns) < 2 and len(columns) > 1:
            service_columns = self._create_multiple_services(columns)
        
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
        # 檢查空數據
        if service_data.empty or service_data.shape[0] == 0 or service_data.shape[1] == 0:
            return np.zeros(self.target_dim)
        
        try:
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
            
            # 處理NaN值
            features = [f if not np.isnan(f) else 0.0 for f in features]
            
            current_len = len(features)
            if current_len >= self.target_dim:
                return np.array(features[:self.target_dim])
            else:
                padding = np.zeros(self.target_dim - current_len)
                return np.concatenate([features, padding])
        except Exception as e:
            # 如果統計處理失敗，返回零向量
            if os.environ.get('GNN_KAN_DEBUG') == '1':
                print(f"[WARN] 統計處理失敗: {e}，返回零向量")
            return np.zeros(self.target_dim)
    
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
        """快速數據標準化 + BARO 時間序列清理邏輯"""
        # === 1) 先將輸入統一轉成 DataFrame ===
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

        # === 2) 基本缺失值 / 無限值處理 ===
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 保存原始備份以防清理過度
        original_df = df.copy()
        original_col_count = len(df.columns)
        
        if os.environ.get('GNN_KAN_DEBUG') == '1':
            print(f"[DEBUG] 開始時間序列預處理: {original_col_count} 列")

        # === 3) 套用 BARO 的時間序列預處理 ===
        try:
            # 移除時間欄位（如果仍存在）
            df_before = len(df.columns)
            df = drop_time(df)
            if os.environ.get('GNN_KAN_DEBUG') == '1' and len(df.columns) != df_before:
                print(f"[DEBUG] drop_time: {df_before} → {len(df.columns)} 列")
            
            # 檢查是否還有列
            if len(df.columns) == 0:
                if os.environ.get('GNN_KAN_DEBUG') == '1':
                    print(f"[WARN] drop_time移除了所有列，恢復原始數據")
                df = original_df.copy()
                # 手動移除時間列
                time_cols = [col for col in df.columns if 'time' in str(col).lower()]
                if time_cols:
                    df = df.drop(columns=time_cols)
            
            # 移除常數列
            if len(df.columns) > 0:
                df_before = len(df.columns)
                df = drop_constant(df)
                if os.environ.get('GNN_KAN_DEBUG') == '1' and len(df.columns) != df_before:
                    print(f"[DEBUG] drop_constant: {df_before} → {len(df.columns)} 列")
            
            # 移除近似常數列 (變異度過低) - 更寬鬆的閾值
            if len(df.columns) > 0:
                df_before = len(df.columns)
                df = drop_near_constant(df, threshold=0.01)  # 更寬鬆閾值
                if os.environ.get('GNN_KAN_DEBUG') == '1' and len(df.columns) != df_before:
                    print(f"[DEBUG] drop_near_constant: {df_before} → {len(df.columns)} 列")
            
            # 轉換記憶體單位為 MB，保持量級一致
            if len(df.columns) > 0:
                df = convert_mem_mb(df)
                if os.environ.get('GNN_KAN_DEBUG') == '1':
                    print(f"[DEBUG] convert_mem_mb: 記憶體單位轉換完成")
            
            # 移除一些無用或雜訊列（如 queue / redis / istio 等）
            if len(df.columns) > 0:
                df_before = len(df.columns)
                df = drop_extra(df)
                if os.environ.get('GNN_KAN_DEBUG') == '1' and len(df.columns) != df_before:
                    print(f"[DEBUG] drop_extra: {df_before} → {len(df.columns)} 列")
            
            # 可選：挑選訊息量較高的列以降低維度 - 只在列數很多時使用
            if len(df.columns) > 20:  # 只在列數超過20時才挑選
                useful_cols = select_useful_cols(df)
                if len(useful_cols) > 0 and len(useful_cols) < len(df.columns):
                    if os.environ.get('GNN_KAN_DEBUG') == '1':
                        print(f"[DEBUG] select_useful_cols: {len(df.columns)} → {len(useful_cols)} 列")
                    df = df[useful_cols]
            
        except Exception as e:
            # 若任何步驟失敗，保留原 df，並在 debug 模式下輸出警告
            if os.environ.get('GNN_KAN_DEBUG') == '1':
                print(f"[WARN] 時間序列預處理失敗: {e}")
            df = original_df.copy()

        # === 4) 最終檢查與保護 ===
        # 如果清理後沒有列了，恢復到基本清理版本
        if len(df.columns) == 0:
            if os.environ.get('GNN_KAN_DEBUG') == '1':
                print(f"[WARN] 清理後無列，恢復到基本版本")
            df = original_df.copy()
            # 只移除明顯的時間列
            time_cols = [col for col in df.columns if col.lower() in ['time', 'timestamp', 'datetime']]
            if time_cols:
                df = df.drop(columns=time_cols)
        
        # 最終保險：再次移除時間欄位
        time_cols = [col for col in df.columns if 'time' in str(col).lower()]
        if time_cols and len(df.columns) > len(time_cols):  # 確保不會清空
            df = df.drop(columns=time_cols)
        
        if os.environ.get('GNN_KAN_DEBUG') == '1':
            print(f"[DEBUG] 時間序列預處理完成: {original_col_count} → {len(df.columns)} 列")
            if len(df.columns) > 0:
                print(f"[DEBUG] 最終列名: {df.columns.tolist()}")

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