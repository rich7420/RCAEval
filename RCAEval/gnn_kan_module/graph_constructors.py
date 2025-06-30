"""
圖構建器模組 - 主要的圖構建類
包含SimplifiedGraphConstructor和IntelligentServiceGraphConstructor
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# 移除循環導入 - 高級類可以在需要時延遲導入
# DynamicModelAdjuster 和 LearnableGraphConstructor 在 advanced_graph_constructors.py 中

# 基本圖構建器實現

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
        
        # 🔧 確保邊索引在有效範圍內 - 強化檢查
        num_nodes = adj_matrix.shape[0]
        
        # 檢查索引範圍
        valid_row_mask = (edge_indices[:, 0] >= 0) & (edge_indices[:, 0] < num_nodes)
        valid_col_mask = (edge_indices[:, 1] >= 0) & (edge_indices[:, 1] < num_nodes)
        valid_mask = valid_row_mask & valid_col_mask
        
        if valid_mask.sum() > 0:
            edge_indices = edge_indices[valid_mask]
            edge_weights = edge_weights[valid_mask]
        else:
            edge_indices = np.array([])
            edge_weights = np.array([])
        
        if len(edge_indices) == 0:
            print(f"⚠️ 所有邊都無效，為{num_nodes}個節點創建安全圖結構")
            if num_nodes > 1:
                # 創建星形連接（更安全）
                edges = []
                weights = []
                center_node = 0
                for i in range(1, num_nodes):
                    edges.extend([[center_node, i], [i, center_node]])
                    weights.extend([0.1, 0.1])
                edge_indices = np.array(edges)
                edge_weights = np.array(weights)
            else:
                # 單節點自環
                edge_indices = np.array([[0, 0]])
                edge_weights = np.array([1.0])
        
        # 🔧 最終安全檢查
        max_index = edge_indices.max() if len(edge_indices) > 0 else 0
        if max_index >= num_nodes:
            print(f"⚠️ 修復後仍有超出範圍的索引: max={max_index}, nodes={num_nodes}")
            # 截斷超出範圍的索引
            edge_indices = np.clip(edge_indices, 0, num_nodes - 1)
        
        print(f"✓ 圖構建完成: {num_nodes}節點, {len(edge_indices)}條邊, 索引範圍[0,{num_nodes-1}]")
        
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