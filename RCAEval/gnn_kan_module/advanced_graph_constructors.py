"""
高級圖構建器模組 - 包含動態調整和可學習圖結構
從 graph_constructors.py 拆分出來的高級功能
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# 使用延遲導入避免循環依賴 - SimplifiedGraphConstructor 在需要時導入


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
            adjusted_config['max_edges_per_node'] = max(3, getattr(self.config, 'max_edges_per_node', 5) - 2)
            adjusted_config['similarity_threshold'] = getattr(self.config, 'similarity_threshold', 0.3) + 0.1
            print("🕸️ High density detected: Increased sparsification")
        elif density < 0.2:
            # 低密度：保持更多連接
            adjusted_config['max_edges_per_node'] = getattr(self.config, 'max_edges_per_node', 5) + 2
            adjusted_config['similarity_threshold'] = max(0.1, getattr(self.config, 'similarity_threshold', 0.3) - 0.1)
            print("🔗 Low density detected: Preserved more connections")
        
        # 3. 基於異常嚴重程度調整訓練參數
        anomaly_severity = characteristics['anomaly_severity']
        if anomaly_severity > 0.8:
            # 高異常程度：加強正則化和穩定性
            adjusted_config['base_l1_lambda'] = getattr(self.config, 'base_l1_lambda', 1e-3) * 2
            adjusted_config['gradient_clip_norm'] = self.config.gradient_clip_norm * 0.8
            adjusted_config['epochs'] = min(150, getattr(self.config, 'epochs', 100) + 10)
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
            adjusted_config['base_learning_rate'] = getattr(self.config, 'base_learning_rate', 0.001) * 0.5
            adjusted_config['warmup_epochs'] = getattr(self.config, 'warmup_epochs', 10) + 3
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
        # 延遲導入避免循環依賴
        from .graph_constructors import SimplifiedGraphConstructor
        
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
        if node_features is not None and getattr(self.config, 'learnable_edges', False):
            return self.update_edge_weights(node_features)
        else:
            return self.base_edge_index, self.base_edge_weights 