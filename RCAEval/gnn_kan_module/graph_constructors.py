"""
GNN-KAN Graph Construction Module
包含所有圖構建相關的類和函數
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SimplifiedGraphConstructor:
    """簡化的圖構建器 - 移除硬編碼服務依賴"""
    
    def __init__(self, config):
        self.config = config
        
    def build_graph(self, features, node_names):
        """
        構建圖結構 - 使用通用的特徵相似性方法
        """
        if features.size == 0 or len(node_names) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
        
        print(f"🔗 Building simplified graph for {len(node_names)} nodes...")
        
        # 使用特徵相似性構建圖
        edge_index, edge_weights = self._build_similarity_graph(features, node_names)
        
        print(f"✓ Built graph: {len(node_names)} nodes, {edge_index.size(1)} edges")
        
        return edge_index, edge_weights
    
    def _build_similarity_graph(self, features, node_names):
        """基於特徵相似性構建圖 - 替代硬編碼依賴"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 計算節點特徵
        node_features = self._compute_node_features(features, node_names)
        
        # 計算相似性矩陣
        similarity_matrix = cosine_similarity(node_features)
        
        # 🎯 動態閾值 - 基於服務重要性調整
        base_threshold = getattr(self.config, 'similarity_threshold', 0.3)
        adj_matrix = self._apply_dynamic_threshold(similarity_matrix, node_names, base_threshold)
        
        # 稀疏化 - 保持圖的連通性
        adj_matrix = self._apply_sparsification(adj_matrix, node_names)
        
        # 轉換為邊列表
        return self._adjacency_to_edges(adj_matrix)
    
    def _compute_node_features(self, features, node_names):
        """計算節點特徵 - 簡化版本"""
        num_nodes = len(node_names)
        
        if features.ndim == 2 and features.shape[0] > 1:
            # 每個節點對應特徵矩陣的統計量
            node_features = np.zeros((num_nodes, 8))  # 增加到8維特徵
            
            for i in range(num_nodes):
                col_idx = i % features.shape[1]
                feature_col = features[:, col_idx]
                
                # 🎯 增強的統計特徵 - 考慮服務語義
                service_weight = self._compute_service_importance_weight(node_names[i])
                
                node_features[i] = [
                    np.mean(feature_col) * service_weight,
                    np.std(feature_col) + 1e-8,
                    np.max(feature_col),
                    np.min(feature_col),
                    np.median(feature_col),
                    np.var(feature_col) + 1e-8,
                    service_weight,  # 服務重要性權重
                    len([name for name in node_names if any(kw in str(name).lower() for kw in ['error', 'latency', 'cpu'])])  # 關鍵指標計數
                ]
        else:
            # 回退到隨機特徵 - 但加入語義信息
            node_features = np.zeros((num_nodes, 8))
            for i in range(num_nodes):
                service_weight = self._compute_service_importance_weight(node_names[i])
                base_features = np.random.randn(6) * 0.1
                node_features[i] = np.concatenate([base_features, [service_weight, 1.0]])
        
        return node_features
    
    def _compute_service_importance_weight(self, node_name):
        """計算服務重要性權重"""
        node_str = str(node_name).lower()
        
        # 🎯 基於關鍵詞的動態權重計算
        critical_keywords = {
            'error': 3.0, 'exception': 2.8, 'fail': 2.5,
            'latency': 2.2, 'delay': 2.0, 'timeout': 2.3,
            'cpu': 1.8, 'memory': 1.6, 'mem': 1.6,
            'frontend': 2.0, 'checkout': 1.9, 'payment': 2.0,
            'cart': 1.6, 'catalog': 1.4, 'redis': 1.7
        }
        
        weight = 1.0
        for keyword, kw_weight in critical_keywords.items():
            if keyword in node_str:
                weight = max(weight, kw_weight)
        
        # 🎯 指標類型權重
        if any(metric in node_str for metric in ['max', 'std', 'trend']):
            weight *= 1.2
        
        return min(weight, 3.0)  # 限制最大權重
    
    def _apply_dynamic_threshold(self, similarity_matrix, node_names, base_threshold):
        """應用動態閾值 - 基於節點重要性調整"""
        num_nodes = similarity_matrix.shape[0]
        adj_matrix = np.zeros_like(similarity_matrix)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # 🎯 動態閾值調整
                    weight_i = self._compute_service_importance_weight(node_names[i])
                    weight_j = self._compute_service_importance_weight(node_names[j])
                    
                    # 重要節點之間使用更低的閾值（更容易連接）
                    dynamic_threshold = base_threshold * (2.0 / (weight_i + weight_j + 0.1))
                    dynamic_threshold = max(0.1, min(0.8, dynamic_threshold))
                    
                    if similarity_matrix[i, j] > dynamic_threshold:
                        adj_matrix[i, j] = similarity_matrix[i, j]
        
        return adj_matrix
    
    def _apply_sparsification(self, adj_matrix, node_names):
        """應用稀疏化 - 保持重要連接"""
        num_nodes = adj_matrix.shape[0]
        sparsified_adj = np.zeros_like(adj_matrix)
        max_edges = getattr(self.config, 'max_edges_per_node', 5)
        
        for i in range(num_nodes):
            weights = adj_matrix[i].copy()
            weights[i] = -1  # 排除自環
            
            # 🎯 優先保留與重要節點的連接
            node_importance = [self._compute_service_importance_weight(node_names[j]) for j in range(num_nodes)]
            
            # 選擇top-k連接，優先考慮重要性和相似性
            if np.max(weights) > 0:
                # 計算綜合分數：相似性 × 目標節點重要性
                combined_scores = weights * np.array(node_importance)
                top_indices = np.argsort(combined_scores)[-max_edges:]
                
                for j in top_indices:
                    if weights[j] > 0:
                        sparsified_adj[i, j] = weights[j]
        
        # 添加自環
        if getattr(self.config, 'use_self_loops', True):
            for i in range(num_nodes):
                sparsified_adj[i, i] = 1.0
        
        return sparsified_adj
    
    def _adjacency_to_edges(self, adj_matrix):
        """將鄰接矩陣轉換為邊列表 - 修復邊索引問題"""
        # 🔧 安全檢查：確保鄰接矩陣有效
        if adj_matrix.size == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
        
        # 獲取非零元素的索引
        edge_indices = np.transpose(np.nonzero(adj_matrix))
        
        if edge_indices.size == 0:
            # 如果沒有邊，創建最小連接（避免完全斷開的圖）
            num_nodes = adj_matrix.shape[0]
            if num_nodes > 1:
                # 創建環形連接
                edge_list = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
                edge_indices = np.array(edge_list)
                edge_weights = np.ones(len(edge_list)) * 0.1
            else:
                # 單節點情況：自環
                edge_indices = np.array([[0, 0]])
                edge_weights = np.array([1.0])
        else:
            edge_weights = adj_matrix[edge_indices[:, 0], edge_indices[:, 1]]
        
        # 🔧 確保邊索引在有效範圍內
        num_nodes = adj_matrix.shape[0]
        valid_mask = (edge_indices[:, 0] < num_nodes) & (edge_indices[:, 1] < num_nodes)
        edge_indices = edge_indices[valid_mask]
        edge_weights = edge_weights[valid_mask]
        
        if len(edge_indices) == 0:
            print(f"⚠️ 所有邊都無效，創建最小圖結構")
            if num_nodes > 1:
                edge_indices = np.array([[0, 1], [1, 0]])
                edge_weights = np.array([0.1, 0.1])
            else:
                edge_indices = np.array([[0, 0]])
                edge_weights = np.array([1.0])
        
        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), torch.tensor(edge_weights, dtype=torch.float)


class IntelligentServiceGraphConstructor:
    """智能服務圖構建器 - 基於真實服務依賴和動態特徵"""
    
    def __init__(self, config):
        self.config = config
        self.service_dependency_cache = {}
        self.critical_path_cache = {}
        
    def build_intelligent_graph(self, features, node_names, service_graph=None):
        """
        構建智能服務依賴圖
        
        Args:
            features: 特徵矩陣
            node_names: 節點名稱
            service_graph: 服務依賴圖（來自trace分析）
            
        Returns:
            edge_index: 邊索引
            edge_weights: 邊權重
        """
        print("🔗 Building intelligent service dependency graph...")
        
        if features.size == 0 or len(node_names) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
        
        # 1. 基於服務依賴構建基礎圖結構
        if service_graph is not None:
            base_adj_matrix = self._service_graph_to_adjacency(service_graph, node_names)
        else:
            base_adj_matrix = self._build_heuristic_service_graph(node_names)
        
        # 2. 基於特徵相似性增強圖結構
        feature_adj_matrix = self._build_feature_similarity_graph(features, node_names)
        
        # 3. 智能融合：服務依賴 + 特徵相似性
        combined_adj_matrix = self._intelligent_graph_fusion(
            base_adj_matrix, feature_adj_matrix, node_names
        )
        
        # 4. 關鍵路徑增強
        enhanced_adj_matrix = self._enhance_critical_paths(combined_adj_matrix, node_names)
        
        # 5. 動態稀疏化 - 保持重要連接
        final_adj_matrix = self._adaptive_sparsification(enhanced_adj_matrix, node_names)
        
        # 轉換為PyTorch格式
        edge_index, edge_weights = self._adjacency_to_edges(final_adj_matrix)
        
        print(f"✓ Built intelligent graph: {len(node_names)} nodes, {edge_index.size(1)} edges")
        return edge_index, edge_weights
    
    def _service_graph_to_adjacency(self, service_graph, node_names):
        """將服務依賴圖轉換為鄰接矩陣"""
        num_nodes = len(node_names)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        # 建立服務名稱到索引的映射
        service_to_idx = {}
        for i, node_name in enumerate(node_names):
            # 從節點名稱中提取服務名稱
            service_name = self._extract_service_name(node_name)
            if service_name not in service_to_idx:
                service_to_idx[service_name] = []
            service_to_idx[service_name].append(i)
        
        # 根據服務依賴關係填充鄰接矩陣
        try:
            import networkx as nx
            if isinstance(service_graph, nx.Graph):
                for source, target, edge_data in service_graph.edges(data=True):
                    weight = edge_data.get('weight', 1.0)
                    
                    # 正規化權重
                    normalized_weight = min(1.0, weight / 10.0)
                    
                    # 將服務級別的連接映射到節點級別
                    if source in service_to_idx and target in service_to_idx:
                        for src_idx in service_to_idx[source]:
                            for tgt_idx in service_to_idx[target]:
                                adj_matrix[src_idx, tgt_idx] = normalized_weight
        except Exception as e:
            print(f"⚠️ Service graph conversion failed: {e}")
        
        return adj_matrix
    
    def _extract_service_name(self, node_name):
        """從節點名稱中提取服務名稱"""
        node_str = str(node_name).lower()
        
        # 常見的微服務名稱模式
        service_patterns = [
            'frontend', 'front-end', 'cartservice', 'cart-service', 
            'checkoutservice', 'checkout-service', 'paymentservice', 'payment-service',
            'currencyservice', 'currency-service', 'emailservice', 'email-service',
            'adservice', 'ad-service', 'recommendationservice', 'recommendation-service',
            'productcatalogservice', 'catalog-service', 'redis', 'database', 'db'
        ]
        
        for pattern in service_patterns:
            if pattern in node_str:
                return pattern
        
        # 如果沒有匹配，嘗試提取前綴
        parts = node_str.split('_')
        if len(parts) > 1:
            return parts[0]
        
        return 'unknown_service'
    
    def _build_heuristic_service_graph(self, node_names):
        """基於啟發式規則構建服務依賴圖"""
        num_nodes = len(node_names)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        # 定義微服務依賴的啟發式規則
        dependency_rules = {
            'frontend': ['cart', 'checkout', 'currency', 'ad', 'recommendation', 'catalog'],
            'checkout': ['payment', 'cart', 'email', 'currency'],
            'cart': ['catalog', 'redis'],
            'recommendation': ['catalog'],
            'ad': ['catalog'],
            'payment': ['currency'],
            'email': [],  # 葉子節點
            'currency': [],  # 葉子節點
            'catalog': ['database', 'redis'],
            'redis': [],  # 葉子節點
            'database': []  # 葉子節點
        }
        
        for i, source_node in enumerate(node_names):
            source_service = self._extract_service_name(source_node)
            
            if source_service in dependency_rules:
                for target_service in dependency_rules[source_service]:
                    for j, target_node in enumerate(node_names):
                        if target_service in str(target_node).lower():
                            # 設置依賴權重
                            adj_matrix[i, j] = 0.8
        
        return adj_matrix
    
    def _build_feature_similarity_graph(self, features, node_names):
        """基於特徵相似性構建圖"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 計算節點特徵
        node_features = self._compute_enhanced_node_features(features, node_names)
        
        # 計算相似性矩陣
        similarity_matrix = cosine_similarity(node_features)
        
        # 應用動態閾值
        threshold = getattr(self.config, 'similarity_threshold', 0.3)
        adj_matrix = (similarity_matrix > threshold).astype(float) * similarity_matrix
        
        return adj_matrix
    
    def _compute_enhanced_node_features(self, features, node_names):
        """計算增強的節點特徵 - 結合統計特徵和語義特徵"""
        num_nodes = len(node_names)
        
        if features.ndim == 2 and features.shape[0] > 1:
            # 基於特徵矩陣計算節點統計量
            node_features = np.zeros((num_nodes, 12))  # 擴展到12維特徵
            
            for i in range(num_nodes):
                col_idx = i % features.shape[1]
                feature_col = features[:, col_idx]
                
                # 基本統計特徵
                basic_stats = [
                    np.mean(feature_col),
                    np.std(feature_col) + 1e-8,
                    np.max(feature_col),
                    np.min(feature_col),
                    np.median(feature_col),
                    np.var(feature_col) + 1e-8
                ]
                
                # 語義特徵
                semantic_features = self._compute_semantic_features(node_names[i])
                
                # 時序特徵（如果數據足夠）
                if len(feature_col) > 5:
                    temporal_features = self._compute_temporal_features(feature_col)
                else:
                    temporal_features = [0.0, 0.0, 0.0]
                
                node_features[i] = basic_stats + semantic_features + temporal_features
        else:
            # 回退到純語義特徵
            node_features = np.zeros((num_nodes, 12))
            for i in range(num_nodes):
                semantic_features = self._compute_semantic_features(node_names[i])
                base_features = np.random.randn(6) * 0.05  # 減少隨機性
                temporal_features = [0.0, 0.0, 0.0]
                node_features[i] = base_features + semantic_features + temporal_features
        
        return node_features
    
    def _compute_semantic_features(self, node_name):
        """計算節點的語義特徵"""
        node_str = str(node_name).lower()
        
        # 服務重要性
        service_importance = self._compute_service_criticality(node_str)
        
        # 指標類型重要性
        metric_importance = self._compute_metric_criticality(node_str)
        
        # 異常相關性
        anomaly_relevance = self._compute_anomaly_relevance(node_str)
        
        return [service_importance, metric_importance, anomaly_relevance]
    
    def _compute_service_criticality(self, node_str):
        """計算服務關鍵性"""
        critical_services = {
            'frontend': 3.0, 'front-end': 3.0,
            'checkout': 2.8, 'payment': 2.8,
            'cart': 2.5, 'catalog': 2.2,
            'currency': 2.0, 'redis': 2.3,
            'database': 2.5, 'db': 2.5
        }
        
        for service, weight in critical_services.items():
            if service in node_str:
                return weight
        
        return 1.0
    
    def _compute_metric_criticality(self, node_str):
        """計算指標關鍵性"""
        critical_metrics = {
            'error': 3.0, 'exception': 2.8, 'fail': 2.5,
            'latency': 2.5, 'delay': 2.2, 'timeout': 2.3,
            'cpu': 2.0, 'memory': 1.8, 'mem': 1.8,
            'disk': 1.6, 'network': 1.7, 'connection': 1.5
        }
        
        for metric, weight in critical_metrics.items():
            if metric in node_str:
                return weight
        
        return 1.0
    
    def _compute_anomaly_relevance(self, node_str):
        """計算異常相關性"""
        anomaly_indicators = {
            'max': 2.0, 'peak': 2.0, 'high': 1.8,
            'std': 1.5, 'variance': 1.5, 'deviation': 1.5,
            'trend': 1.3, 'change': 1.2, 'diff': 1.2
        }
        
        for indicator, weight in anomaly_indicators.items():
            if indicator in node_str:
                return weight
        
        return 1.0
    
    def _compute_temporal_features(self, feature_col):
        """計算時序特徵"""
        # 趨勢
        x = np.arange(len(feature_col))
        trend_coef = np.polyfit(x, feature_col, 1)[0]
        
        # 穩定性
        diff = np.diff(feature_col)
        stability = 1.0 / (1.0 + np.std(diff))
        
        # 週期性（簡化檢測）
        if len(feature_col) > 10:
            fft = np.fft.fft(feature_col)
            power = np.abs(fft) ** 2
            dominant_freq = np.argmax(power[1:len(power)//2]) + 1
            periodicity = 1.0 / max(dominant_freq, 1)
        else:
            periodicity = 0.0
        
        return [trend_coef, stability, periodicity]
    
    def _intelligent_graph_fusion(self, service_adj, feature_adj, node_names):
        """智能融合服務依賴圖和特徵相似性圖"""
        num_nodes = len(node_names)
        fused_adj = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # 服務依賴權重
                    service_weight = service_adj[i, j]
                    
                    # 特徵相似性權重
                    feature_weight = feature_adj[i, j]
                    
                    # 智能融合策略
                    if service_weight > 0.5:
                        # 強服務依賴：主要使用服務依賴，特徵相似性作為增強
                        fused_weight = 0.7 * service_weight + 0.3 * feature_weight
                    elif feature_weight > 0.6:
                        # 強特徵相似性：主要使用特徵相似性
                        fused_weight = 0.3 * service_weight + 0.7 * feature_weight
                    else:
                        # 均衡融合
                        fused_weight = 0.5 * service_weight + 0.5 * feature_weight
                    
                    fused_adj[i, j] = fused_weight
        
        return fused_adj
    
    def _enhance_critical_paths(self, adj_matrix, node_names):
        """增強關鍵路徑連接"""
        enhanced_adj = adj_matrix.copy()
        
        # 識別關鍵節點
        critical_nodes = []
        for i, node_name in enumerate(node_names):
            criticality = self._compute_service_criticality(str(node_name).lower())
            if criticality > 2.0:
                critical_nodes.append(i)
        
        # 增強關鍵節點之間的連接
        for i in critical_nodes:
            for j in critical_nodes:
                if i != j and enhanced_adj[i, j] > 0:
                    enhanced_adj[i, j] = min(1.0, enhanced_adj[i, j] * 1.5)
        
        return enhanced_adj
    
    def _adaptive_sparsification(self, adj_matrix, node_names):
        """自適應稀疏化 - 基於節點重要性"""
        num_nodes = adj_matrix.shape[0]
        sparsified_adj = np.zeros_like(adj_matrix)
        
        # 計算每個節點的重要性
        node_importance = [
            self._compute_service_criticality(str(name).lower()) for name in node_names
        ]
        
        for i in range(num_nodes):
            weights = adj_matrix[i].copy()
            weights[i] = -1  # 排除自環
            
            # 根據重要性調整最大邊數
            base_max_edges = getattr(self.config, 'max_edges_per_node', 5)
            max_edges = int(base_max_edges * min(2.0, node_importance[i]))
            
            if np.max(weights) > 0:
                # 優先保留與重要節點的連接
                combined_scores = weights * np.array(node_importance)
                top_indices = np.argsort(combined_scores)[-max_edges:]
                
                for j in top_indices:
                    if weights[j] > 0:
                        sparsified_adj[i, j] = weights[j]
        
        # 添加自環
        if getattr(self.config, 'use_self_loops', True):
            for i in range(num_nodes):
                sparsified_adj[i, i] = 1.0
        
        return sparsified_adj
    
    def _adjacency_to_edges(self, adj_matrix):
        """將鄰接矩陣轉換為邊列表 - 修復邊索引問題"""
        # 🔧 安全檢查：確保鄰接矩陣有效
        if adj_matrix.size == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
        
        # 獲取非零元素的索引
        edge_indices = np.transpose(np.nonzero(adj_matrix))
        
        if edge_indices.size == 0:
            # 如果沒有邊，創建最小連接（避免完全斷開的圖）
            num_nodes = adj_matrix.shape[0]
            if num_nodes > 1:
                # 創建環形連接
                edge_list = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
                edge_indices = np.array(edge_list)
                edge_weights = np.ones(len(edge_list)) * 0.1
            else:
                # 單節點情況：自環
                edge_indices = np.array([[0, 0]])
                edge_weights = np.array([1.0])
        else:
            edge_weights = adj_matrix[edge_indices[:, 0], edge_indices[:, 1]]
        
        # 🔧 確保邊索引在有效範圍內
        num_nodes = adj_matrix.shape[0]
        valid_mask = (edge_indices[:, 0] < num_nodes) & (edge_indices[:, 1] < num_nodes)
        edge_indices = edge_indices[valid_mask]
        edge_weights = edge_weights[valid_mask]
        
        if len(edge_indices) == 0:
            print(f"⚠️ 所有邊都無效，創建最小圖結構")
            if num_nodes > 1:
                edge_indices = np.array([[0, 1], [1, 0]])
                edge_weights = np.array([0.1, 0.1])
            else:
                edge_indices = np.array([[0, 0]])
                edge_weights = np.array([1.0])
        
        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), torch.tensor(edge_weights, dtype=torch.float)


class DynamicModelAdjuster:
    """動態模型調整器 - 根據數據特性自動調整模型參數"""
    
    def __init__(self, config):
        self.config = config
        self.data_characteristics = {}
        self.adjustment_history = []
        
    def analyze_data_characteristics(self, features, node_names, inject_time=None):
        """
        分析數據特性，為動態調整提供依據
        
        Args:
            features: 特徵矩陣
            node_names: 節點名稱
            inject_time: 故障注入時間
            
        Returns:
            characteristics: 數據特性分析結果
        """
        print("🔍 Analyzing data characteristics for dynamic adjustment...")
        
        characteristics = {
            'data_complexity': self._assess_data_complexity(features),
            'network_density': self._estimate_network_density(node_names),
            'temporal_patterns': self._analyze_temporal_patterns(features, inject_time),
            'anomaly_severity': self._assess_anomaly_severity(features, inject_time),
            'service_diversity': self._analyze_service_diversity(node_names),
            'feature_stability': self._assess_feature_stability(features)
        }
        
        self.data_characteristics = characteristics
        print(f"✓ Data analysis completed: complexity={characteristics['data_complexity']:.2f}, "
              f"density={characteristics['network_density']:.2f}")
        
        return characteristics
    
    def adjust_model_parameters(self, model, characteristics):
        """
        基於數據特性動態調整模型參數
        
        Args:
            model: GNN-KAN模型
            characteristics: 數據特性分析結果
            
        Returns:
            adjusted_config: 調整後的配置
        """
        print("⚙️ Dynamically adjusting model parameters...")
        
        adjusted_config = self.config.__dict__.copy()
        
        # 1. 基於數據複雜度調整KAN參數
        complexity = characteristics['data_complexity']
        if complexity > 0.8:
            # 高複雜度：增加KAN表達能力
            adjusted_config['kan_grid_size'] = min(8, self.config.kan_grid_size + 2)
            adjusted_config['kan_spline_order'] = min(5, self.config.kan_spline_order + 1)
            adjusted_config['hidden_dims'] = [dim * 2 for dim in self.config.hidden_dims]
            print("📈 High complexity detected: Enhanced KAN parameters")
        elif complexity < 0.3:
            # 低複雜度：簡化模型防止過擬合
            adjusted_config['kan_grid_size'] = max(3, self.config.kan_grid_size - 1)
            adjusted_config['dropout'] = min(0.3, self.config.dropout + 0.1)
            print("📉 Low complexity detected: Simplified KAN parameters")
        
        # 2. 基於網絡密度調整圖結構參數
        density = characteristics['network_density']
        if density > 0.7:
            # 高密度：增加稀疏化
            adjusted_config['max_edges_per_node'] = max(3, self.config.max_edges_per_node - 2)
            adjusted_config['similarity_threshold'] = self.config.similarity_threshold + 0.1
            print("🕸️ High density detected: Increased sparsification")
        elif density < 0.2:
            # 低密度：保持更多連接
            adjusted_config['max_edges_per_node'] = self.config.max_edges_per_node + 2
            adjusted_config['similarity_threshold'] = max(0.1, self.config.similarity_threshold - 0.1)
            print("🔗 Low density detected: Preserved more connections")
        
        # 3. 基於異常嚴重程度調整訓練參數
        anomaly_severity = characteristics['anomaly_severity']
        if anomaly_severity > 0.8:
            # 高異常程度：加強正則化和穩定性
            adjusted_config['base_l1_lambda'] = self.config.base_l1_lambda * 2
            adjusted_config['gradient_clip_norm'] = self.config.gradient_clip_norm * 0.8
            adjusted_config['epochs'] = min(50, self.config.epochs + 10)
            print("🚨 High anomaly severity: Enhanced regularization")
        
        # 4. 基於時序模式調整注意力機制
        if characteristics['temporal_patterns'] > 0.6:
            # 強時序模式：增強時序注意力
            adjusted_config['use_temporal_attention'] = True
            print("⏰ Strong temporal patterns: Enhanced attention mechanism")
        
        # 5. 基於特徵穩定性調整學習率
        feature_stability = characteristics['feature_stability']
        if feature_stability < 0.3:
            # 特徵不穩定：降低學習率
            adjusted_config['base_learning_rate'] = self.config.base_learning_rate * 0.5
            adjusted_config['warmup_epochs'] = self.config.warmup_epochs + 3
            print("📊 Low feature stability: Reduced learning rate")
        
        # 記錄調整歷史
        self.adjustment_history.append({
            'characteristics': characteristics,
            'adjustments': adjusted_config
        })
        
        return type(self.config)(**adjusted_config)
    
    def _assess_data_complexity(self, features):
        """評估數據複雜度"""
        if features.size == 0:
            return 0.5
        
        try:
            # 計算特徵間相關性
            if features.ndim == 2 and features.shape[1] > 1:
                corr_matrix = np.corrcoef(features.T)
                # 去除對角線元素
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                correlations = corr_matrix[mask]
                
                # 複雜度指標
                mean_corr = np.abs(correlations).mean()
                std_corr = np.std(correlations)
                
                # 特徵分佈複雜度
                feature_entropies = []
                for col in range(features.shape[1]):
                    hist, _ = np.histogram(features[:, col], bins=min(20, len(features)//5))
                    hist = hist / np.sum(hist)
                    entropy = -np.sum(hist * np.log(hist + 1e-10))
                    feature_entropies.append(entropy)
                
                mean_entropy = np.mean(feature_entropies)
                
                # 組合複雜度指標 (0-1範圍)
                complexity = (mean_corr * 0.4 + std_corr * 0.3 + min(mean_entropy/3, 1) * 0.3)
                return min(1.0, complexity)
            else:
                return 0.3
        except Exception as e:
            print(f"⚠️ Complexity assessment failed: {e}")
            return 0.5
    
    def _estimate_network_density(self, node_names):
        """估計網絡密度"""
        if len(node_names) == 0:
            return 0.5
        
        # 基於節點名稱估計潛在連接密度
        service_types = set()
        metric_types = set()
        
        for name in node_names:
            name_str = str(name).lower()
            
            # 識別服務類型
            for service in ['frontend', 'cart', 'checkout', 'payment', 'catalog', 'redis']:
                if service in name_str:
                    service_types.add(service)
                    break
            
            # 識別指標類型
            for metric in ['cpu', 'memory', 'latency', 'error', 'disk', 'network']:
                if metric in name_str:
                    metric_types.add(metric)
                    break
        
        # 密度估計：服務數量 × 指標數量 / 總節點數
        service_count = len(service_types)
        metric_count = len(metric_types)
        
        if service_count > 0 and metric_count > 0:
            density = (service_count * metric_count) / len(node_names)
            return min(1.0, density)
        else:
            return 0.3
    
    def _analyze_temporal_patterns(self, features, inject_time):
        """分析時序模式強度"""
        if features.size == 0 or inject_time is None:
            return 0.3
        
        try:
            # 分析特徵的時序相關性
            if features.ndim == 2 and features.shape[0] > 5:
                temporal_scores = []
                
                for col in range(min(features.shape[1], 10)):  # 限制計算量
                    feature_col = features[:, col]
                    
                    # 計算自相關性
                    if len(feature_col) > 10:
                        autocorr = np.correlate(feature_col, feature_col, mode='full')
                        mid = len(autocorr) // 2
                        autocorr = autocorr[mid:]
                        
                        # 尋找週期性
                        if len(autocorr) > 3:
                            peak_score = np.max(autocorr[1:min(len(autocorr)//2, 10)])
                            temporal_scores.append(peak_score / np.max(autocorr))
                
                if temporal_scores:
                    return min(1.0, np.mean(temporal_scores))
                else:
                    return 0.3
            else:
                return 0.3
        except Exception as e:
            print(f"⚠️ Temporal analysis failed: {e}")
            return 0.3
    
    def _assess_anomaly_severity(self, features, inject_time):
        """評估異常嚴重程度"""
        if features.size == 0:
            return 0.5
        
        try:
            # 計算特徵分佈的異常程度
            anomaly_scores = []
            
            for col in range(min(features.shape[1], 10)):
                feature_col = features[:, col]
                
                # 基於IQR的異常檢測
                Q1, Q3 = np.percentile(feature_col, [25, 75])
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((feature_col < lower_bound) | (feature_col > upper_bound)).sum()
                    anomaly_ratio = outliers / len(feature_col)
                    anomaly_scores.append(anomaly_ratio)
                
                # 基於Z-score的異常檢測
                if np.std(feature_col) > 0:
                    z_scores = np.abs((feature_col - np.mean(feature_col)) / np.std(feature_col))
                    extreme_count = (z_scores > 2.5).sum()
                    extreme_ratio = extreme_count / len(feature_col)
                    anomaly_scores.append(extreme_ratio)
            
            if anomaly_scores:
                return min(1.0, np.mean(anomaly_scores) * 3)  # 放大異常信號
            else:
                return 0.5
                
        except Exception as e:
            print(f"⚠️ Anomaly assessment failed: {e}")
            return 0.5
    
    def _analyze_service_diversity(self, node_names):
        """分析服務多樣性"""
        if len(node_names) == 0:
            return 0.5
        
        # 識别不同類型的服務和指標
        categories = {
            'services': set(),
            'metrics': set(),
            'operations': set()
        }
        
        for name in node_names:
            name_str = str(name).lower()
            
            # 服務類型
            services = ['frontend', 'cart', 'checkout', 'payment', 'catalog', 'redis', 'currency', 'email']
            for service in services:
                if service in name_str:
                    categories['services'].add(service)
            
            # 指標類型
            metrics = ['cpu', 'memory', 'latency', 'error', 'disk', 'network', 'connection', 'queue']
            for metric in metrics:
                if metric in name_str:
                    categories['metrics'].add(metric)
            
            # 操作類型
            operations = ['get', 'post', 'put', 'delete', 'create', 'update', 'read', 'write']
            for op in operations:
                if op in name_str:
                    categories['operations'].add(op)
        
        # 計算多樣性分數
        total_diversity = len(categories['services']) + len(categories['metrics']) + len(categories['operations'])
        diversity_score = min(1.0, total_diversity / 15)  # 正規化到0-1
        
        return diversity_score
    
    def _assess_feature_stability(self, features):
        """評估特徵穩定性"""
        if features.size == 0:
            return 0.5
        
        try:
            if features.ndim == 2 and features.shape[0] > 3:
                stability_scores = []
                
                for col in range(min(features.shape[1], 10)):
                    feature_col = features[:, col]
                    
                    # 計算變異係數
                    if np.mean(feature_col) != 0:
                        cv = np.std(feature_col) / np.abs(np.mean(feature_col))
                        stability = 1.0 / (1.0 + cv)  # 變異係數越小，穩定性越高
                        stability_scores.append(stability)
                    
                    # 計算平穩性
                    if len(feature_col) > 5:
                        diff = np.diff(feature_col)
                        stationarity = 1.0 / (1.0 + np.std(diff))
                        stability_scores.append(stability)
                
                if stability_scores:
                    return np.mean(stability_scores)
                else:
                    return 0.5
            else:
                return 0.5
                
        except Exception as e:
            print(f"⚠️ Stability assessment failed: {e}")
            return 0.5


class LearnableGraphConstructor(nn.Module):
    """可學習的圖構建器 - 支持動態邊權重更新"""
    
    def __init__(self, config, num_nodes):
        super(LearnableGraphConstructor, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # 可學習的邊權重參數
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(num_nodes * 2, 64),  # 節點對特徵
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 圖結構更新頻率控制
        self.update_counter = 0
        self.update_frequency = getattr(config, 'graph_update_frequency', 10)
        
        # 保存初始圖結構
        self.register_buffer('base_edge_index', torch.empty((2, 0), dtype=torch.long))
        self.register_buffer('base_edge_weights', torch.empty(0))
        
    def initialize_base_graph(self, features, node_names):
        """初始化基礎圖結構"""
        constructor = SimplifiedGraphConstructor(self.config)
        edge_index, edge_weights = constructor.build_graph(features, node_names)
        
        self.base_edge_index = edge_index
        self.base_edge_weights = edge_weights
        
        return edge_index, edge_weights
    
    def update_edge_weights(self, node_features):
        """動態更新邊權重"""
        if self.base_edge_index.size(1) == 0:
            return self.base_edge_index, self.base_edge_weights
        
        # 只在指定頻率更新
        self.update_counter += 1
        if self.update_counter % self.update_frequency != 0:
            return self.base_edge_index, self.base_edge_weights
        
        # 計算邊特徵
        edge_features = self._compute_edge_features(node_features, self.base_edge_index)
        
        # 通過MLP更新邊權重
        with torch.no_grad():  # 避免影響主要訓練
            new_weights = self.edge_weight_mlp(edge_features).squeeze(-1)
            
            # 與原始權重結合
            combined_weights = 0.7 * self.base_edge_weights + 0.3 * new_weights
            
            # 過濾弱連接
            threshold = torch.quantile(combined_weights, 0.3)  # 保留前70%的邊
            mask = combined_weights > threshold
            
            filtered_edges = self.base_edge_index[:, mask]
            filtered_weights = combined_weights[mask]
            
            return filtered_edges, filtered_weights
    
    def _compute_edge_features(self, node_features, edge_index):
        """計算邊特徵"""
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        src_features = node_features[src_nodes]
        dst_features = node_features[dst_nodes]
        
        # 拼接源節點和目標節點特徵
        edge_features = torch.cat([src_features, dst_features], dim=1)
        
        return edge_features
    
    def forward(self, node_features=None):
        """前向傳播 - 返回當前圖結構"""
        if node_features is not None and self.config.learnable_edges:
            return self.update_edge_weights(node_features)
        else:
            return self.base_edge_index, self.base_edge_weights