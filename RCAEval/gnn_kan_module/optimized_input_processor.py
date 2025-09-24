"""
GNN+KAN 優化輸入處理器
專為KAN特性設計的高效輸入預處理系統
目標：最大化KAN性能，最小化輸入處理開銷
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
        
    def extract_services_batch(self, columns: List[str], force_expansion=False) -> Dict[str, List[str]]:
        """批量提取微服務對應的列 - 多層策略 + 強制擴展"""
        service_columns = {}
        
        # 🔥 如果啟用強制擴展，使用更激進的分組策略
        if force_expansion:
            print("✓ 啟用強制節點擴展模式")
            service_columns = self._create_individual_metric_nodes(columns)
        else:
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
        
        # 🔥 關鍵修復：強制確保至少2個節點
        if len(service_columns) <= 1:
            print(f"⚠️ 節點數不足({len(service_columns)})，強制創建多節點")
            service_columns = self._force_create_multiple_nodes(columns)
        
        # 📊 優化：清理空分組並限制分組數量
        service_columns = self._optimize_service_groups(service_columns, columns)
        
        return service_columns
    
    def _create_individual_metric_nodes(self, columns: List[str]) -> Dict[str, List[str]]:
        """創建單指標節點 - 最大化節點數量以提高準確度"""
        services = {}
        
        for i, col in enumerate(columns):
            # 基於列名的智能命名
            col_lower = col.lower()
            
            # 嘗試提取服務名和指標類型
            if any(svc in col_lower for svc in self.known_services):
                service_name = next(svc for svc in self.known_services if svc in col_lower)
                metric_type = self._extract_metric_type(col)
                node_name = f"{service_name}_{metric_type}" if metric_type != "metric" else service_name
            else:
                # 使用列名的前綴作為節點名
                if '_' in col:
                    parts = col.split('_')
                    if len(parts) >= 2:
                        prefix = parts[0]
                        suffix = parts[1] if len(parts[1]) <= 8 else parts[1][:8]
                        node_name = f"{prefix}_{suffix}"
                    else:
                        node_name = f"metric_{col[:12]}"
                elif '-' in col:
                    parts = col.split('-')
                    if len(parts) >= 2:
                        prefix = parts[0]
                        suffix = parts[1] if len(parts[1]) <= 8 else parts[1][:8]
                        node_name = f"{prefix}_{suffix}"
                    else:
                        node_name = f"metric_{col[:12]}"
                else:
                    node_name = f"metric_{col[:12]}" if len(col) <= 12 else f"node_{i+1}"
            
            # 確保節點名唯一
            if node_name in services:
                node_name = f"{node_name}_{i}"
                
            services[node_name] = [col]
        
        print(f"✓ 強制擴展完成：{len(columns)} 列 -> {len(services)} 節點")
        
        # 最小節點保證：確保至少3個節點以支持圖神經網絡
        min_nodes = max(3, min(5, len(columns)))
        if len(services) < min_nodes and len(columns) >= 1:
            print(f"⚠️ 節點數不足({len(services)})，擴展到最少{min_nodes}個節點")
            
            # 通過拆分現有節點來增加節點數
            expanded_services = {}
            original_keys = list(services.keys())
            
            for i, (service_name, cols) in enumerate(list(services.items())):
                if len(cols) > 1 and len(services) < min_nodes:
                    # 拆分多列節點
                    for j, col in enumerate(cols):
                        sub_node_name = f"{service_name}_part{j+1}"
                        expanded_services[sub_node_name] = [col]
                        if len(expanded_services) >= min_nodes:
                            break
                    # 移除原節點
                    if service_name in services:
                        del services[service_name]
                else:
                    expanded_services[service_name] = cols
                
                if len(expanded_services) >= min_nodes:
                    break
            
            # 如果還是不夠，創建合成節點
            if len(expanded_services) < min_nodes:
                for i in range(len(expanded_services), min_nodes):
                    synthetic_name = f"synthetic_metric_{i+1}"
                    # 複製第一個節點的列
                    if expanded_services:
                        first_cols = list(expanded_services.values())[0]
                        expanded_services[synthetic_name] = first_cols.copy()
                    else:
                        expanded_services[synthetic_name] = columns[:1] if columns else ["placeholder"]
            
            services = expanded_services
            print(f"✓ 節點擴展完成：{len(columns)} 列 -> {len(services)} 節點（保證最少{min_nodes}個）")
        return services
    
    def _force_create_multiple_nodes(self, columns: List[str]) -> Dict[str, List[str]]:
        """強制創建多個節點 - 解決單節點問題"""
        services = {}
        
        if len(columns) == 0:
            # 如果沒有列，創建默認節點
            services['default_node'] = ['placeholder_metric']
            return services
        
        if len(columns) == 1:
            # 單列情況：創建多個虛擬節點
            col = columns[0]
            services[f'{col}_cpu'] = [col]
            services[f'{col}_memory'] = [col]
            services[f'{col}_network'] = [col]
            print(f"✓ 單列強制擴展：{col} -> 3個節點")
        else:
            # 多列情況：按列分割
            for i, col in enumerate(columns):
                if i < 3:  # 最多創建3個節點
                    services[f'node_{i+1}'] = [col]
                else:
                    # 將多餘的列分配到現有節點
                    target_node = f'node_{(i % 3) + 1}'
                    services[target_node].append(col)
        
        print(f"✓ 強制多節點創建完成：{len(columns)} 列 -> {len(services)} 節點")
        return services
    
    def _extract_metric_type(self, col: str) -> str:
        """提取指標類型"""
        col_lower = col.lower()
        
        for metric_type, patterns in self.metric_patterns.items():
            for pattern in patterns:
                if pattern in col_lower:
                    return metric_type
        
        # 如果沒有匹配到已知模式，嘗試從列名推斷
        if any(word in col_lower for word in ['cpu', 'memory', 'mem']):
            return 'resource'
        elif any(word in col_lower for word in ['latency', 'response', 'time']):
            return 'performance'
        elif any(word in col_lower for word in ['error', 'fail']):
            return 'error'
        
        return "metric"
    
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
        """強制創建多個服務節點（最後手段）- 增強細粒度"""
        services = {}
        
        # 🔥 調整：增加更細粒度的服務劃分
        num_cols = len(columns)
        if num_cols <= 4:
            target_services = min(num_cols, 3)  # 從2增加到3
        elif num_cols <= 12:
            target_services = min(8, int(num_cols // 1.5))  # 從4增加到8，分母從2減少到1.5，確保整數
        else:
            target_services = min(15, num_cols // 2)  # 從8增加到15，分母從3增加到2
        
        # 🔥 支持單列服務：避免過度合並
        cols_per_service = max(1, num_cols // target_services)
        
        for i in range(target_services):
            start_idx = i * cols_per_service
            end_idx = start_idx + cols_per_service if i < target_services - 1 else num_cols
            
            if start_idx < num_cols:
                service_cols = columns[start_idx:end_idx]
                
                # 🔥 更智能的命名：基於列名內容
                if len(service_cols) == 1:
                    col_name = service_cols[0]
                    if any(svc in col_name.lower() for svc in ['frontend', 'backend', 'database', 'cache', 'queue']):
                        service_name = next((svc for svc in ['frontend', 'backend', 'database', 'cache', 'queue'] 
                                          if svc in col_name.lower()), f"metric_{i+1}")
                    else:
                        service_name = f"metric_{col_name.split('_')[0] if '_' in col_name else f'node_{i+1}'}"
                else:
                    service_name = f"group_{i+1}"
                    
                services[service_name] = service_cols
        
        return services
    
    def _optimize_service_groups(self, service_columns: Dict[str, List[str]], 
                                columns: List[str]) -> Dict[str, List[str]]:
        """優化服務分組 - 增加節點數量以提高準確度"""
        # 移除空分組
        service_columns = {k: v for k, v in service_columns.items() if v}
        
        # 🔥 調整：大幅增加分組數量以提高根因定位精度
        max_services = min(25, max(8, int(len(columns) // 1.5)))  # 從15增加到25，從2增加到8，確保整數
        if len(service_columns) > max_services:
            # 保留最大的分組
            sorted_services = sorted(service_columns.items(), 
                                   key=lambda x: len(x[1]), reverse=True)
            service_columns = dict(sorted_services[:max_services])
        
        # 🔥 確保更多分組：提高最低分組數量
        min_services = min(8, max(3, len(columns) // 3))  # 從2增加到8，新增3的最小值
        if len(service_columns) < min_services and len(columns) > 1:
            service_columns = self._create_multiple_services(columns)
        
        return service_columns


class KANFeatureProcessor:
    """專為KAN優化的特徵處理器"""
    
    def __init__(self, method='enhanced_ica', target_dim=64):
        self.method = method
        self.target_dim = target_dim
        self.service_extractor = FastServiceExtractor()
        
    def process_features_optimized(self, data: pd.DataFrame, force_expansion=False, inject_time=None) -> Tuple[np.ndarray, List[str]]:
        """簡化的特徵處理 - 使用增強版 ICA"""
        
        # 🚀 直接使用增強版特徵處理
        if self.method == 'enhanced_ica':
            from .feature_processing import enhanced_ica_with_temporal_contrast
            features, node_names = enhanced_ica_with_temporal_contrast(
                data, inject_time=inject_time, target_dim=self.target_dim
            )
        elif self.method == 'ica':
            from .feature_processing import ica_metric_processing
            features, node_names = ica_metric_processing(
                data, inject_time=inject_time, target_dim=self.target_dim
            )
        else:
            from .feature_processing import simplified_metric_processing
            features, node_names = simplified_metric_processing(data, target_dim=self.target_dim)
        
        # 🎯 方向1: 增強輸入特徵豐富度 - 解決圖稀疏根源
        features = self.enhanced_feature_enrichment(features, node_names)
        
        return features, node_names
    
    def enhanced_feature_enrichment(self, features, node_names):
        """RICH輸入特徵 - 時序異常檢測 + 融合, 完全數據驅動"""
        import torch
        
        # 轉換為tensor進行處理
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # 1. 時序異常檢測: 計算 z-score 異常分數 (無需預設類型)
        mean = features.mean(dim=1, keepdim=True)  # 時序平均
        std = features.std(dim=1, keepdim=True) + 1e-8
        z_scores = (features - mean) / std
        anomaly_scores = z_scores.abs().mean(dim=0)  # 節點平均異常度
        
        # 2. 變化率特徵: 計算相鄰時間點差異
        if features.shape[1] > 1:
            diff = features[:, 1:] - features[:, :-1]
            change_rate = diff.abs().sum(dim=1) / (features.shape[1] - 1 + 1e-8)  # 平均變化率
        else:
            change_rate = torch.zeros(features.shape[0], device=features.device)
        
        # 3. 融合: 使用簡單可學習權重 (數據驅動)
        anomaly_weight = 0.3
        change_weight = 0.3
        enriched = features * 0.4 + anomaly_scores.unsqueeze(0) * anomaly_weight + change_rate.unsqueeze(1) * change_weight
        
        # 4. 防止過度稀疏: 添加小量高斯噪聲 (0.01 std)
        noise = torch.randn(enriched.shape, device=enriched.device) * 0.01 * enriched.std()
        enriched = enriched + noise
        
        print(f"🔧 特徵增強: 原始{features.shape} -> 增強{enriched.shape}, 異常分數範圍[{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
        
        return enriched.numpy()
    
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
    
    def _extract_baro_style_features(self, service_data: pd.DataFrame, inject_time: float) -> np.ndarray:
        """提取 BARO 風格的異常檢測特徵"""
        from sklearn.preprocessing import RobustScaler
        
        try:
            # 檢查是否有時間列
            if 'time' not in service_data.columns:
                # 如果沒有時間列，使用行索引作為時間代理
                service_data = service_data.copy()
                service_data['time'] = range(len(service_data))
                inject_time = int(inject_time) if inject_time < len(service_data) else len(service_data) // 2
            
            # 分離故障前後數據（BARO 核心思想）
            normal_df = service_data[service_data['time'] < inject_time]
            anomal_df = service_data[service_data['time'] >= inject_time]
            
            # 如果數據量太少，使用簡單分割
            if len(normal_df) == 0 or len(anomal_df) == 0:
                mid_point = len(service_data) // 2
                normal_df = service_data.iloc[:mid_point]
                anomal_df = service_data.iloc[mid_point:]
            
            baro_scores = []
            
            # 對每列進行 BARO 風格異常檢測
            for col in service_data.columns:
                if col == 'time':
                    continue
                
                try:
                    # 正常時期數據
                    normal_values = normal_df[col].dropna().values
                    # 異常時期數據
                    anomal_values = anomal_df[col].dropna().values
                    
                    if len(normal_values) == 0 or len(anomal_values) == 0:
                        baro_scores.append(0.0)
                        continue
                    
                    # 使用 RobustScaler（BARO 的核心）
                    scaler = RobustScaler()
                    scaler.fit(normal_values.reshape(-1, 1))
                    
                    # 計算異常分數
                    z_scores = scaler.transform(anomal_values.reshape(-1, 1))[:, 0]
                    max_anomaly_score = np.max(np.abs(z_scores))
                    
                    baro_scores.append(max_anomaly_score)
                    
                except Exception as e:
                    baro_scores.append(0.0)
            
            # 擴展到目標維度
            current_len = len(baro_scores)
            if current_len >= self.target_dim:
                return np.array(baro_scores[:self.target_dim])
            else:
                # 添加統計特徵
                additional_features = []
                if current_len > 0:
                    additional_features.extend([
                        np.mean(baro_scores),
                        np.std(baro_scores),
                        np.max(baro_scores),
                        np.percentile(baro_scores, 90) if len(baro_scores) > 1 else baro_scores[0]
                    ])
                
                # 填充到目標維度
                total_features = baro_scores + additional_features
                if len(total_features) >= self.target_dim:
                    return np.array(total_features[:self.target_dim])
                else:
                    padding = np.zeros(self.target_dim - len(total_features))
                    return np.concatenate([total_features, padding])
                    
        except Exception as e:
            print(f"⚠️ BARO 特徵提取失敗: {e}，返回零向量")
            return np.zeros(self.target_dim)
    
    def _combine_features(self, baro_features: np.ndarray, original_features: np.ndarray) -> np.ndarray:
        """融合 BARO 特徵和原始特徵"""
        try:
            # 確保兩個特徵向量維度一致
            if len(baro_features) != len(original_features):
                min_len = min(len(baro_features), len(original_features))
                baro_features = baro_features[:min_len]
                original_features = original_features[:min_len]
            
            # 加權融合
            combined = self.baro_weight * baro_features + (1 - self.baro_weight) * original_features
            
            # 確保最終維度正確
            if len(combined) >= self.target_dim:
                return combined[:self.target_dim]
            else:
                padding = np.zeros(self.target_dim - len(combined))
                return np.concatenate([combined, padding])
                
        except Exception as e:
            print(f"⚠️ 特徵融合失敗: {e}，使用原始特徵")
            return original_features
    
    def _global_feature_processing(self, data: pd.DataFrame, inject_time=None) -> Tuple[np.ndarray, List[str]]:
        """全局特徵處理 + BARO 風格增強"""
        if self.use_baro_features and inject_time is not None:
            baro_features = self._extract_baro_style_features(data, inject_time)
            stat_features = self._fast_statistical_processing(data)
            combined_features = self._combine_features(baro_features, stat_features)
            return np.array([combined_features]), ['global_service']
        else:
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
    """優化的圖構建器 - 修復空圖問題"""
    
    def __init__(self, similarity_threshold=0.5, max_edges_per_node=4):  # 提高閾值降低密度 / Increase threshold to reduce density
        # 🔧 降低相似性閾值，避免空圖
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
        print(f"🔧 圖構建器初始化: 閾值={similarity_threshold}, 最大邊數={max_edges_per_node}")
    
    def build_graph_fast(self, node_features: np.ndarray, node_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """修復版圖構建 - 確保圖連通性 (改進方案3.3.1)"""
        num_nodes = len(node_names)
        
        if num_nodes <= 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long) if num_nodes == 1 else torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.tensor([1.0]) if num_nodes == 1 else torch.empty(0)
            return edge_index, edge_weights
        
        print(f"🔧 修復版圖構建: {num_nodes} 個節點, 閾值={self.similarity_threshold}")
        
        # 🎯 改進1: 增強特徵處理
        if node_features.shape[1] < 3:
            # 特徵維度太低，添加統計特徵
            mean_feat = np.mean(node_features, axis=1, keepdims=True)
            std_feat = np.std(node_features, axis=1, keepdims=True) + 1e-8
            node_features = np.hstack([node_features, mean_feat, std_feat])
        
        # 標準化特徵
        node_features_norm = node_features / (np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(node_features_norm, node_features_norm.T)
        
        # 🎯 改進方案3.3.1: 自適應閾值計算
        threshold = self._adaptive_threshold_calculation(similarity_matrix)
        
        edges = []
        weights = []
        
        # 🎯 改進3: 嚴格閾值執行與小圖規則
        # 更嚴格的度數限制：大幅減少最大連接數
        max_edges_per_node = 1 if num_nodes <= 4 else min(2, num_nodes // 3)
        
        for i in range(num_nodes):
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # 排除自環
            
            # 獲取最相似的節點
            top_indices = np.argsort(similarities)[-max_edges_per_node:]
            connected = False
            
            for j in top_indices:
                if similarities[j] > threshold:  # 🎯 嚴格: 移除 or not connected 繞過
                    # Sigmoid 平滑權重，避免硬閾值造成的突變
                    sim_val = float(similarities[j])
                    smooth_weight = 1.0 / (1.0 + np.exp(-(sim_val - threshold) * 10.0))
                    weight = max(smooth_weight, 0.2)
                    # 添加雙向邊
                    edges.extend([[i, j], [j, i]])
                    weights.extend([weight, weight])
                    connected = True
            
            # 僅在完全孤立時才強制連接最相似的節點
            if not connected:
                j = int(np.argmax(similarities))
                edges.extend([[i, j], [j, i]])
                weights.extend([0.2, 0.2])  # 使用最小權重
        
        # 🎯 改進方案3.3.2: 結構驗證與調整
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for edge, weight in zip(edges, weights):
            adj_matrix[edge[0], edge[1]] = weight
        
        # 檢查圖密度 - 基於節點數的自適應目標密度
        def compute_density(e):
            # 🎯 修正: 對於無向圖，最大邊數應該是 n*(n-1)/2
            max_possible = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
            return len(e) / 2 / max_possible if max_possible > 0 else 0  # 除以2因為雙向邊

        def adaptive_target_density(n_nodes):
            """基於節點數的自適應目標密度 - 細微調整大圖"""
            if n_nodes <= 5:
                return 0.25, 0.5  # 小圖：適中密度
            elif n_nodes <= 15:
                return 0.2, 0.4  # 中圖：適中密度
            else:
                return 0.18, 0.4  # 大圖：微調提升下限

        density = compute_density(edges)
        target_min, target_max = adaptive_target_density(num_nodes)
        
        # 優化調整邏輯 - 更精準, 減少迭代
        step_low = 0.95  # 溫和降低 (從0.9調整)
        step_high = 1.05  # 溫和提升 (從1.1調整)
        hysteresis = 0.05  # 遲滯避免震蕩
        iter_limit = 4  # 減少迭代次數
        it = 0
        
        print(f"🔧 圖密度調整: 當前={density:.3f}, 目標=[{target_min:.3f}, {target_max:.3f}]")
        
        while it < iter_limit:
            if density < target_min - hysteresis:
                print("⚠️ 密度過低, 降低閾值")
                threshold *= step_low
            elif density > target_max + hysteresis:
                print("⚠️ 密度過高, 提升閾值")
                threshold *= step_high
            else:
                break
            
            edges, weights = self._rebuild_edges(similarity_matrix, threshold, num_nodes)
            density = compute_density(edges)
            it += 1
            print(f"  迭代{it}: 密度={density:.3f}, 閾值={threshold:.3f}")
        
        # 檢查連通性
        if not self._is_connected(edges, num_nodes):
            print("⚠️ 圖不連通，添加最小連接")
            edges, weights = self._add_minimal_connections(edges, weights, num_nodes)
        
        # 最終密度檢查
        final_density = compute_density(edges)
        if final_density < 0.1:
            print(f"⚠️ 最終密度過低({final_density:.3f})，強制提升")
            # 降低閾值並重建
            threshold *= 0.8
            edges, weights = self._rebuild_edges(similarity_matrix, threshold, num_nodes)
            final_density = compute_density(edges)
            print(f"✓ 密度調整後: {final_density:.3f}")
        
        # 去重
        edge_set = set()
        final_edges = []
        final_weights = []
        
        for edge, weight in zip(edges, weights):
            edge_tuple = tuple(edge)
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                final_edges.append(edge)
                final_weights.append(weight)
        
        # 最終檢查
        if not final_edges:
            print("❌ 圖構建失敗，創建星形圖")
            for i in range(1, num_nodes):
                final_edges.extend([[0, i], [i, 0]])
                final_weights.extend([0.8, 0.8])
        
        # 最終密度報告
        final_density = compute_density(final_edges)
        print(f"✅ 圖構建完成: {len(final_edges)} 條邊, 密度={final_density:.3f}")
        
        edge_index = torch.tensor(final_edges, dtype=torch.long).T
        edge_weights = torch.tensor(final_weights, dtype=torch.float)
        
        final_density = len(final_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        print(f"✅ 圖構建完成: {len(final_edges)} 條邊, 密度={final_density:.3f}")
        
        return edge_index, edge_weights
    
    def _adaptive_threshold_calculation(self, similarity_matrix):
        """自適應閾值計算（更激進的分位數 + 方差調整）"""
        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        if len(upper_tri) == 0:
            return 0.3
        # 更激進的分位數：大幅提高閾值降低密度
        num_nodes = similarity_matrix.shape[0]
        if num_nodes <= 4:
            percentile = 95  # 從90提升到95
        elif num_nodes <= 8:
            percentile = 92  # 從85提升到92
        else:
            percentile = 88  # 從80提升到88
        threshold = np.percentile(upper_tri, percentile)
        # 依據分佈方差微調
        std = np.std(upper_tri)
        if std < 0.1:
            threshold *= 0.8
        elif std > 0.3:
            threshold *= 1.2
        # 邊界保護
        return max(0.1, min(0.9, threshold))
    
    def _rebuild_edges(self, similarity_matrix, threshold, num_nodes):
        """重新構建邊 - 嚴格閾值執行 + 度數限制"""
        edges = []
        weights = []
        # 更嚴格的度數限制：大幅減少最大連接數
        max_edges_per_node = 1 if num_nodes <= 4 else min(2, num_nodes // 3)
        
        for i in range(num_nodes):
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1
            # 獲取候選節點（過採樣後修剪）
            candidate_indices = np.argsort(similarities)[-max_edges_per_node * 2:]
            connected = False
            selected = []
            
            for j in candidate_indices:
                if similarities[j] > threshold and len(selected) < max_edges_per_node:
                    # 修剪過於相似的節點（避免平行邊）
                    if all(np.abs(similarities[j] - similarities[k]) > 0.02 for k in selected):
                        # Sigmoid 平滑權重，避免硬閾值造成的突變
                        sim_val = float(similarities[j])
                        smooth_weight = 1.0 / (1.0 + np.exp(-(sim_val - threshold) * 10.0))
                        weight = max(smooth_weight, 0.2)
                        # 添加雙向邊
                        edges.extend([[i, j], [j, i]])
                        weights.extend([weight, weight])
                        selected.append(j)
                        connected = True
            
            # 僅在完全孤立時才強制連接最相似的節點
            if not connected:
                j = int(np.argmax(similarities[:num_nodes]))
                edges.extend([[i, j], [j, i]])
                weights.extend([0.2, 0.2])  # 使用最小權重
        return edges, weights
    
    def _is_connected(self, edges, num_nodes):
        """檢查圖是否連通"""
        if not edges:
            return num_nodes <= 1
        
        # 使用並查集檢查連通性
        parent = list(range(num_nodes))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for edge in edges:
            union(edge[0], edge[1])
        
        return len(set(find(i) for i in range(num_nodes))) == 1
    
    def _add_minimal_connections(self, edges, weights, num_nodes):
        """添加最小連接確保連通性"""
        if num_nodes <= 1:
            return edges, weights
        
        # 添加環形連接
        for i in range(num_nodes):
            j = (i + 1) % num_nodes
            edges.extend([[i, j], [j, i]])
            weights.extend([0.5, 0.5])
        
        return edges, weights


class EnhancedFusion(nn.Module):
    """增強融合模組 - 改進方案3.2.3"""
    
    def __init__(self, input_dim, num_heads=4):
        super(EnhancedFusion, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, features_list):
        """融合多個特徵列表"""
        if len(features_list) == 1:
            return features_list[0]
        
        # 拼接特徵
        fused = torch.cat(features_list, dim=1)
        
        # 注意力融合
        attn_output, _ = self.attention(fused, fused, fused)
        
        # 殘差連接和歸一化
        output = self.norm(attn_output + fused)
        
        return output


class GNNKANInputOptimizer:
    """GNN+KAN 輸入優化器 - 支援 BARO 風格特徵"""
    
    def __init__(self, 
                 feature_method='enhanced_ica',
                 target_dim=64,
                 similarity_threshold=0.5,  # 提高相似性閾值以降低圖密度 / Increase similarity threshold to reduce graph density
                 max_edges_per_node=4,      # 減少每節點最大邊數 / Reduce max edges per node
                 force_node_expansion=False):
        
        self.feature_processor = KANFeatureProcessor(
            feature_method, target_dim
        )
        self.graph_builder = OptimizedGraphBuilder(similarity_threshold, max_edges_per_node)
        self.force_node_expansion = force_node_expansion
    
    def optimize_input(self, data: Any, inject_time: Optional[float] = None) -> KANOptimizedData:
        """優化輸入處理 + 強制節點擴展支持 + BARO 風格特徵"""
        start_time = time.time()
        
        df = self._fast_data_standardization(data)
        node_features, node_names = self.feature_processor.process_features_optimized(
            df, self.force_node_expansion, inject_time
        )

        # 防呆：行數對不上就以 features 為準重建名稱
        if isinstance(node_features, np.ndarray) and node_features.shape[0] != len(node_names):
            print(f"⚠️ 節點數不匹配: features={node_features.shape[0]} vs names={len(node_names)}，以 features 為準重建名稱")
            node_names = [f'node_{i}' for i in range(node_features.shape[0])]

        edge_index, edge_weights = self.graph_builder.build_graph_fast(node_features, node_names)

        # 保守的因果先後矩陣（lead-lag prior）：偏好「先異常→後影響」的方向
        lead_lag_prior = self._compute_lead_lag_prior(df, node_names)
        
        # 二次保險：強制一致
        if node_features.shape[0] != len(node_names):
            print(f"⚠️ build_graph_fast: features={node_features.shape[0]} vs num_nodes={len(node_names)}，截斷到一致")
            num_nodes = min(len(node_names), node_features.shape[0])
            node_features = node_features[:num_nodes]
            node_names = node_names[:num_nodes]

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
                'inject_time': inject_time,
                'lead_lag_prior': lead_lag_prior.tolist() if lead_lag_prior is not None else None
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

    def _compute_lead_lag_prior(self, df: pd.DataFrame, node_names: List[str]) -> Optional[torch.Tensor]:
        """保守計算服務間先後關係的先驗矩陣。
        方法：
        - 對每個節點，嘗試從欄位名稱包含節點名的數值列聚合成單一時間序列（均值）
        - 對聚合序列做z-score，取首次超過閾值的時間作為『異常起始時間』
        - 構建先驗 L[i,j] ~ sigmoid((t_j - t_i)/tau)，i 早於 j 時 > 0.5，否則 < 0.5
        退化處理：找不到對應列或時間不足時，返回全1矩陣（不影響原流程）
        """
        try:
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                return None
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                return None
            series_per_node = []
            for name in node_names:
                # 寬鬆匹配：名稱出現在欄位中或欄位在名稱中
                matched = [c for c in numeric_cols if (str(name).lower() in str(c).lower()) or (str(c).lower() in str(name).lower())]
                if not matched:
                    # 回退：選取與該節點索引相同模數的列，避免空
                    idx = len(series_per_node) % max(1, len(numeric_cols))
                    matched = [numeric_cols[idx]]
                values = df[matched].mean(axis=1).astype(float).values
                if len(values) < 5:
                    series_per_node.append(None)
                    continue
                mu = np.mean(values)
                sigma = np.std(values) + 1e-8
                z = (values - mu) / sigma
                # 首次顯著異常閾值（保守）：|z|>2.0
                thresh = 2.0
                indices = np.where(np.abs(z) > thresh)[0]
                onset = int(indices[0]) if len(indices) > 0 else len(values)  # 未觸發則視為很晚
                series_per_node.append(onset)
            if any(s is None for s in series_per_node):
                # 時間點不足，返回均勻先驗
                n = len(node_names)
                return torch.ones((n, n), dtype=torch.float32)
            onsets = np.array(series_per_node, dtype=float)
            n = len(onsets)
            if n == 0:
                return None
            # 時間尺度（tau）取序列長度的5%，至少為1
            T = max(1.0, 0.05 * max(onsets.max(), 1.0))
            # L[i,j] = sigmoid((t_j - t_i)/tau)
            diff = onsets.reshape(1, n) - onsets.reshape(n, 1)
            L = 1.0 / (1.0 + np.exp(-diff / T))
            # 歸一化並移除自環偏置
            np.fill_diagonal(L, 1.0)
            L = np.clip(L, 0.2, 0.8)
            return torch.tensor(L, dtype=torch.float32)
        except Exception:
            return None
    
    def compute_multi_modal_similarity(self, fused_embeds: torch.Tensor) -> torch.Tensor:
        """
        跨模態相似度計算
        基於融合特徵計算節點間相似度，避免單一模態偏誤
        
        Args:
            fused_embeds: 融合後的特徵嵌入 [nodes, embed_dim]
            
        Returns:
            similarity_matrix: 相似度矩陣 [nodes, nodes]
        """
        # 餘弦相似度計算
        norm_emb = fused_embeds / (fused_embeds.norm(dim=1, keepdim=True) + 1e-8)
        S = norm_emb @ norm_emb.T  # [nodes, nodes]
        
        # 正則化：懲罰過高相似度（避免密集聚類）
        S = torch.clamp(S, max=0.9)  # 避免 >0.9 以防止密集聚類
        S.fill_diagonal_(0)  # 無自環
        
        # 可選：模態特定增強（基於數據方差推斷）
        if torch.std(S) < 0.1:  # 低方差 → 增強對比度
            S = (S - S.mean()) / (S.std() + 1e-8) * 1.5
            S = torch.clamp(S, min=-1.0, max=1.0)
            
        return S
    
    def build_balanced_graph(self, fused_embeds: torch.Tensor, 
                           target_density: float = 0.35, 
                           k_out: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        建構平衡圖結構 - 修復過度凝合問題 
        Build Balanced Graph Structure - Fix Over-Dense Graph Issue
        使用增強特徵區分性 + 嚴格閾值控制 + 結構化稀疏性
        Using Enhanced Feature Discrimination + Strict Threshold Control + Structural Sparsity
        
        Args:
            fused_embeds: 融合後的特徵嵌入 [nodes, embed_dim] / Fused feature embeddings
            target_density: 目標圖密度 (降低到0.35) / Target graph density (reduced to 0.35)
            k_out: 每個節點的最大出度 (增加到3) / Max out-degree per node (increased to 3)
            
        Returns:
            edge_index: 邊索引 [2, num_edges] / Edge indices
            edge_weights: 邊權重 [num_edges] / Edge weights
        """
        S = self.compute_enhanced_similarity(fused_embeds)
        n = S.shape[0]
        
        print(f"🔧 圖構建開始 / Graph Construction Start: {n}個節點 / nodes，目標密度 / target density={target_density:.2f}")
        
        # 🔧 步驟2: 動態k值和嚴格互惠過濾 / Step 2: Dynamic k-value and Strict Reciprocal Filtering
        adaptive_k = max(1, min(k_out, n // 4))  # 動態調整k值 / Dynamically adjust k-value
        topk_values, topk_indices = torch.topk(S, adaptive_k + 1, dim=1)  # 獲取top-k和分數 / Get top-k and scores
        topk_indices = topk_indices[:, 1:]  # 排除自身 / Exclude self
        topk_values = topk_values[:, 1:]   # 排除自身分數 / Exclude self scores
        
        # 🔧 步驟3: 強化互惠過濾 + 分數門檻 / Step 3: Enhanced Reciprocal Filtering + Score Threshold
        adj = torch.zeros_like(S)
        connection_count = 0
        
        for i in range(n):
            for idx, j in enumerate(topk_indices[i]):
                j = j.item()
                # 嚴格互惠檢查 + 分數門檻 / Strict reciprocal check + score threshold
                if (i in topk_indices[j] and 
                    topk_values[i, idx] > 0.7 and  # 提高分數門檻 / Raise score threshold
                    S[i, j] > 0.6):  # 額外相似性檢查 / Additional similarity check
                    weight = float(S[i, j])
                    adj[i, j] = weight
                    adj[j, i] = weight  # 確保對稱性 / Ensure symmetry
                    connection_count += 1
        
        print(f"   互惠過濾後 / After Reciprocal Filtering: {connection_count}個連接 / connections")
        
        # 🔧 步驟4: 自適應閾值和結構化剪枝 / Step 4: Adaptive Threshold and Structural Pruning
        current_density = (adj > 0).float().mean().item()
        print(f"   當前密度 / Current Density: {current_density:.3f}")
        
        # 如果密度仍然過高，進行結構化剪枝 / If density is still too high, perform structural pruning
        if current_density > target_density:
            # 基於邊權重進行全局Top-k選擇 / Global Top-k selection based on edge weights
            edge_weights_flat = adj[adj > 0]
            if len(edge_weights_flat) > 0:
                num_target_edges = int(target_density * n * (n-1))
                threshold_idx = max(0, len(edge_weights_flat) - num_target_edges)
                weight_threshold = torch.sort(edge_weights_flat, descending=True)[0][threshold_idx]
                adj[adj < weight_threshold] = 0
                print(f"   權重剪枝閾值 / Weight Pruning Threshold: {weight_threshold:.3f}")
        
        # 🔧 步驟5: 連通性保證（最小生成樹）/ Step 5: Connectivity Guarantee (MST)
        final_density = (adj > 0).float().mean().item()
        if final_density < 0.1:  # 如果過於稀疏，確保連通性 / If too sparse, ensure connectivity
            print("   密度過低，添加最小連通結構 / Density too low, adding minimal connectivity structure")
            # 添加環形連接確保連通性 / Add ring connections to ensure connectivity
            for i in range(n):
                j = (i + 1) % n
                if adj[i, j] == 0:
                    adj[i, j] = 0.3
                    adj[j, i] = 0.3
        
        final_density = (adj > 0).float().mean().item()
        print(f"✅ 圖構建完成 / Graph Construction Complete: 最終密度 / final density={final_density:.3f}")
        
        # 轉換為稀疏格式 / Convert to sparse format
        edge_index, edge_weights = self._dense_to_sparse(adj)
        
        return edge_index, edge_weights
    
    def compute_enhanced_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:

        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        cos_sim = torch.mm(embeddings_norm, embeddings_norm.t())

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        max_dist = dist_matrix.max()
        euclidean_sim = 1.0 - (dist_matrix / (max_dist + 1e-8))
        # 皮爾森相關係數
        embeddings_centered = embeddings - embeddings.mean(dim=1, keepdim=True)
        std = embeddings_centered.std(dim=1, keepdim=True)
        embeddings_standardized = embeddings_centered / (std + 1e-8)
        pearson_sim = torch.mm(embeddings_standardized, embeddings_standardized.t()) / embeddings.shape[1]
        combined_sim = (0.5 * cos_sim + 0.3 * euclidean_sim + 0.2 * pearson_sim)
        
        combined_sim = torch.sigmoid(5 * (combined_sim - 0.5))
        
        return combined_sim
    
    def _dense_to_sparse(self, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将稠密邻接矩阵转换为稀疏格式
        """
        nonzero_indices = torch.nonzero(adj_matrix, as_tuple=False)
        if len(nonzero_indices) == 0:
            # 空图，返回自环
            n = adj_matrix.shape[0]
            edge_index = torch.stack([torch.arange(n), torch.arange(n)])
            edge_weights = torch.ones(n) * 0.1
            return edge_index, edge_weights
        
        edge_index = nonzero_indices.t()
        edge_weights = adj_matrix[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        
        return edge_index, edge_weights


class MultiModalGraphBuilder:
    
    def __init__(self, target_density=0.35):
        self.target_density = target_density
    
    def build_multimodal_graph(self, data_dict, inject_time=None):
        
        pass


def _check_function_placeholder():
    pass
    
    def _adaptive_threshold_calculation(self, similarity_matrix: torch.Tensor) -> float:
        """
        自適應閾值計算（更激進的分位數 + 方差調整）
        """
        upper_tri = similarity_matrix[torch.triu_indices(similarity_matrix.shape[0], similarity_matrix.shape[1], offset=1)]
        if len(upper_tri) == 0:
            return 0.3
            
        num_nodes = similarity_matrix.shape[0]
        
        # 更激進的分位數：大幅提高閾值降低密度
        if num_nodes <= 4:
            percentile = 98  # 從95提升到98
        elif num_nodes <= 8:
            percentile = 96  # 從92提升到96
        else:
            percentile = 94  # 從88提升到94
            
        threshold = np.percentile(upper_tri.detach().cpu().numpy(), percentile)
        
        # 依據分佈方差微調
        std = torch.std(upper_tri).item()
        if std < 0.1:
            threshold *= 0.8
        elif std > 0.3:
            threshold *= 1.2
            
        return max(0.2, min(0.8, threshold))
    
    def optimize_input_multimodal(self, data_dict: Dict[str, Any], 
                                inject_time: Optional[float] = None) -> KANOptimizedData:
        """
        多模態輸入優化處理
        
        Args:
            data_dict: 包含 'metrics', 'logs', 'traces' 的字典
            inject_time: 故障注入時間
            
        Returns:
            KANOptimizedData: 優化後的數據
        """
        from .feature_processing import MultiModalFeatureExtractor
        
        start_time = time.time()
        
        # 初始化多模態特徵提取器
        multimodal_extractor = MultiModalFeatureExtractor(embed_dim=64, num_heads=4)
        
        # 提取融合特徵 - 確保在推理模式下
        multimodal_extractor.eval()
        with torch.no_grad():
            fused_embeds, attention_weights = multimodal_extractor(data_dict)
        
        # 建構平衡圖
        edge_index, edge_weights = self.build_balanced_graph(fused_embeds, target_density=0.4)
        
        # 生成節點名稱
        num_nodes = fused_embeds.shape[0]
        node_names = [f'node_{i}' for i in range(num_nodes)]
        
        processing_time = time.time() - start_time
        
        return KANOptimizedData(
            node_features=fused_embeds,
            edge_index=edge_index,
            edge_weights=edge_weights,
            node_names=node_names,
            feature_names=[f'feature_{i}' for i in range(fused_embeds.shape[1])],
            metadata={
                'processing_time': processing_time,
                'num_nodes': num_nodes,
                'num_edges': edge_index.size(1),
                'feature_method': 'multimodal_fusion',
                'inject_time': inject_time,
                'attention_weights': attention_weights.detach().cpu().numpy() if attention_weights is not None else None
            }
        )


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