"""
高級特徵處理模組 - 整合所有特徵處理功能
從大檔案中提取並優化的關鍵函數
新增：ICA特徵提取，替代複雜的STL分解
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")


@dataclass
class RCAFeatureConfig:
    """RCA特徵提取配置"""

    # SLI識別配置
    sli_patterns: List[str] = field(default_factory=lambda: [
        'latency', 'response_time', 'response', 'duration', 'delay',
        'error_rate', 'error', 'error_total', 'errors', 'failures',
        'throughput', 'requests_per_second', 'qps', 'rps',
        'availability', 'uptime', 'health'
    ])
    sli_fallback_threshold: float = 0.3

    # 時序分析配置
    pre_inject_window: int = 60      # 注入前窗口大小（分鐘）
    post_inject_window: int = 60     # 注入後窗口大小（分鐘）
    temporal_scales: List[int] = field(default_factory=lambda: [5, 10, 20, 60])

    # 特徵選擇配置
    correlation_threshold: float = 0.2
    mutual_info_threshold: float = 0.1
    max_features_per_service: int = 16
    feature_selection_method: str = 'hybrid'  # 'correlation', 'mutual_info', 'hybrid'

    # 異常檢測配置
    anomaly_z_threshold: float = 2.0
    anomaly_persistence_threshold: float = 3
    propagation_delay_threshold: float = 10.0

    # 服務分組配置
    min_service_columns: int = 1
    max_service_columns: int = 50

    # 特徵降維配置
    target_dim: int = 48
    enable_feature_normalization: bool = True
    enable_pca_reduction: bool = True

    # 故障類型特化配置
    fault_type: str = 'auto'  # 'loss', 'delay', 'auto'
    loss_patterns: List[str] = field(default_factory=lambda: ['error', 'timeout', 'retry', '5xx', '4xx', 'exception'])
    delay_patterns: List[str] = field(default_factory=lambda: ['latency', 'response', 'queue', 'duration'])

    # 多模態配置
    enable_multimodal_fusion: bool = True
    multimodal_embed_dim: int = 48
    multimodal_num_heads: int = 4

    # 性能優化配置
    enable_caching: bool = True
    cache_size: int = 100
    parallel_processing: bool = False
    max_workers: int = 4

    def __post_init__(self):
        """配置驗證和調整"""
        # 驗證數值範圍
        assert 0 < self.correlation_threshold <= 1, "相關性閾值必須在0-1之間"
        assert 0 < self.mutual_info_threshold <= 1, "互信息閾值必須在0-1之間"
        assert self.target_dim > 0, "目標維度必須大於0"
        assert self.pre_inject_window > 0, "注入前窗口必須大於0"
        assert self.post_inject_window > 0, "注入後窗口必須大於0"

        # 調整相關性閾值（根據故障類型）
        if self.fault_type == 'loss':
            self.correlation_threshold = max(0.1, self.correlation_threshold * 0.8)
        elif self.fault_type == 'delay':
            self.correlation_threshold = min(0.4, self.correlation_threshold * 1.2)


class RCAConfigManager:
    """RCA配置管理器"""

    # 預定義配置模板
    CONFIG_TEMPLATES = {
        'train_ticket_loss': RCAFeatureConfig(
            fault_type='loss',
            correlation_threshold=0.15,
            target_dim=32,
            pre_inject_window=30,
            post_inject_window=30,
            sli_patterns=['error', 'timeout', '5xx', '4xx', 'retry']
        ),
        'train_ticket_delay': RCAFeatureConfig(
            fault_type='delay',
            correlation_threshold=0.25,
            target_dim=40,
            pre_inject_window=45,
            post_inject_window=45,
            sli_patterns=['latency', 'response_time', 'duration']
        ),
        're1_loss': RCAFeatureConfig(
            fault_type='loss',
            correlation_threshold=0.18,
            target_dim=36,
            pre_inject_window=40,
            post_inject_window=40,
            sli_patterns=['error', 'timeout', 'failures']
        ),
        're1_delay': RCAFeatureConfig(
            fault_type='delay',
            correlation_threshold=0.22,
            target_dim=44,
            pre_inject_window=50,
            post_inject_window=50,
            sli_patterns=['latency', 'response', 'duration']
        ),
        'default': RCAFeatureConfig(),
        'fast': RCAFeatureConfig(
            target_dim=24,
            enable_pca_reduction=False,
            parallel_processing=True,
            max_workers=8
        ),
        'comprehensive': RCAFeatureConfig(
            target_dim=64,
            pre_inject_window=120,
            post_inject_window=120,
            temporal_scales=[3, 5, 10, 20, 60, 120],
            max_features_per_service=24,
            feature_selection_method='hybrid'
        )
    }

    @classmethod
    def get_config(cls, template_name: str = 'default',
                   dataset_name: str = None,
                   fault_type: str = None,
                   **overrides) -> RCAFeatureConfig:
        """
        獲取配置實例

        Args:
            template_name: 配置模板名稱
            dataset_name: 數據集名稱（用於自動配置調整）
            fault_type: 故障類型
            **overrides: 其他配置覆蓋

        Returns:
            配置實例
        """
        # 獲取基礎配置
        if template_name in cls.CONFIG_TEMPLATES:
            config = cls.CONFIG_TEMPLATES[template_name]
        else:
            config = cls.CONFIG_TEMPLATES['default']

        # 創建副本以避免修改原配置
        config = RCAFeatureConfig(**config.__dict__)

        # 根據數據集調整配置
        if dataset_name:
            config = cls._adjust_for_dataset(config, dataset_name)

        # 根據故障類型調整配置
        if fault_type and fault_type != 'auto':
            config.fault_type = fault_type
            config = cls._adjust_for_fault_type(config, fault_type)

        # 應用覆蓋配置
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # 後處理調整
        config.__post_init__()

        return config

    @classmethod
    def _adjust_for_dataset(cls, config: RCAFeatureConfig, dataset_name: str) -> RCAFeatureConfig:
        """根據數據集調整配置"""
        dataset_lower = dataset_name.lower()

        if 'train-ticket' in dataset_lower:
            # train-ticket特定調整
            config.sli_patterns.extend([
                'ts-ui-dashboard_istio-latency', 'ts-ui-dashboard_latency', 'istio-latency',
                'ui_latency', 'frontend_latency', 'dashboard_latency'
            ])
            config.min_service_columns = 1
            config.max_service_columns = 30

        elif 're1' in dataset_lower:
            # RE1特定調整
            config.sli_patterns.extend([
                're1_latency', 're1_error', 're1_throughput'
            ])
            config.correlation_threshold *= 0.9  # 降低閾值以捕捉更多信號

        elif 'sock-shop' in dataset_lower:
            # Sock Shop特定調整
            config.sli_patterns.extend([
                'istio-request-duration', 'istio-requests-total', 'istio-requests-error'
            ])

        return config

    @classmethod
    def _adjust_for_fault_type(cls, config: RCAFeatureConfig, fault_type: str) -> RCAFeatureConfig:
        """根據故障類型調整配置"""
        if fault_type == 'loss':
            config.correlation_threshold = max(0.1, config.correlation_threshold * 0.8)
            config.sli_patterns = config.loss_patterns + config.sli_patterns
            config.anomaly_z_threshold = 2.5  # 提高異常檢測閾值

        elif fault_type == 'delay':
            config.correlation_threshold = min(0.4, config.correlation_threshold * 1.2)
            config.sli_patterns = config.delay_patterns + config.sli_patterns
            config.propagation_delay_threshold = 5.0  # 降低延遲閾值

        return config

    @classmethod
    def print_config_info(cls, config: RCAFeatureConfig):
        """打印配置信息"""
        print("🔧 RCA特徵提取配置:")
        print(f"   故障類型: {config.fault_type}")
        print(f"   SLI模式: {config.sli_patterns[:3]}{'...' if len(config.sli_patterns) > 3 else ''}")
        print(f"   相關性閾值: {config.correlation_threshold}")
        print(f"   目標維度: {config.target_dim}")
        print(f"   時窗口: 前{config.pre_inject_window}min, 後{config.post_inject_window}min")
        print(f"   特徵選擇方法: {config.feature_selection_method}")
        print(f"   異常檢測閾值: {config.anomaly_z_threshold}σ")


class RCACache:
    """RCA特徵緩衝器"""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, key: str) -> Any:
        """獲取緩衝項目"""
        if key in self.cache:
            # 更新訪問順序
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        """添加緩衝項目"""
        if key in self.cache:
            # 更新現有項目
            self.cache[key] = value
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # 添加新項目
            if len(self.cache) >= self.max_size:
                # 移除最舊的項目
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]

            self.cache[key] = value
            self.access_order.append(key)

    def clear(self):
        """清空緩衝"""
        self.cache.clear()
        self.access_order.clear()

    def stats(self) -> Dict[str, int]:
        """獲取緩衝統計"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_ratio': len([k for k in self.access_order if k in self.cache]) / max(1, len(self.access_order))
        }


class RCAPerformanceOptimizer:
    """RCA性能優化器"""

    def __init__(self, config: RCAFeatureConfig):
        self.config = config
        self.cache = RCACache(config.cache_size) if config.enable_caching else None
        self.processing_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }

    def optimize_features(self, features: np.ndarray, operation: str = 'normalize') -> np.ndarray:
        """優化特徵處理"""
        if operation == 'normalize':
            return self._normalize_features(features)
        elif operation == 'reduce':
            return self._reduce_dimensions(features)
        elif operation == 'select':
            return self._select_features(features)
        else:
            return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """標準化特徵"""
        if not self.config.enable_feature_normalization:
            return features

        if features.shape[0] <= 1:
            return features

        # 每列標準化
        normalized = features.copy()
        for col in range(features.shape[1]):
            col_data = features[:, col]
            if np.std(col_data) > 0:
                normalized[:, col] = (col_data - np.mean(col_data)) / np.std(col_data)

        return normalized

    def _reduce_dimensions(self, features: np.ndarray) -> np.ndarray:
        """降維處理"""
        if not self.config.enable_pca_reduction:
            return features

        if features.shape[1] <= self.config.target_dim:
            return features

        try:
            # 使用PCA降維
            pca = PCA(n_components=self.config.target_dim, random_state=42)
            reduced = pca.fit_transform(features)
            return reduced
        except Exception:
            # 如果PCA失敗，返回原始特徵
            return features

    def _select_features(self, features: np.ndarray) -> np.ndarray:
        """特徵選擇"""
        # 簡單的特徵選擇：移除低方差特徵
        if features.shape[1] <= self.config.max_features_per_service:
            return features

        # 計算每列的方差
        variances = np.var(features, axis=0)
        # 保留方差最大的特徵
        top_indices = np.argsort(variances)[-self.config.max_features_per_service:]
        return features[:, top_indices]

    def get_cache_key(self, data_hash: str, operation: str, **params) -> str:
        """生成緩衝鍵"""
        param_str = '_'.join([f"{k}:{v}" for k, v in sorted(params.items())])
        return f"{data_hash}_{operation}_{param_str}"

    def update_stats(self, cache_hit: bool, processing_time: float):
        """更新統計信息"""
        self.processing_stats['total_calls'] += 1
        if cache_hit:
            self.processing_stats['cache_hits'] += 1
        else:
            self.processing_stats['cache_misses'] += 1

        self.processing_stats['total_time'] += processing_time
        self.processing_stats['avg_time'] = (
            self.processing_stats['total_time'] / self.processing_stats['total_calls']
        )

    def print_performance_stats(self):
        """打印性能統計"""
        stats = self.processing_stats
        cache_hit_ratio = stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses'])

        print("⚡ RCA性能統計:")
        print(f"   總調用次數: {stats['total_calls']}")
        print(f"   緩衝命中率: {cache_hit_ratio:.3f}")
        print(f"   平均處理時間: {stats['avg_time']:.4f}s")

        if self.cache:
            cache_stats = self.cache.stats()
            print(f"   緩衝大小: {cache_stats['size']}/{cache_stats['max_size']}")
            print(f"   緩衝命中率: {cache_stats['hit_ratio']:.3f}")


class MultiScaleTemporalProcessor:
    """
    多尺度時序特徵處理器
    提取不同時間窗口的時序特徵，增強對延遲和磁盤故障的檢測能力
    """
    
    def __init__(self, window_sizes=[3, 5, 10, 20, 50]):
        self.window_sizes = window_sizes
    
    def extract_features(self, data, inject_time):
        """
        提取多尺度時序特徵
        
        Args:
            data: 指標數據
            inject_time: 故障注入時間
            
        Returns:
            features: 多尺度時序特徵矩陣
        """
        features = []
        
        for window in self.window_sizes:
            # 短期特徵
            short_term = self._extract_short_term(data, window, inject_time)
            # 長期特徵  
            long_term = self._extract_long_term(data, window, inject_time)
            # 趨勢特徵
            trend = self._extract_trend(data, window, inject_time)
            
            features.extend([short_term, long_term, trend])
        
        return np.array(features)
    
    def _extract_short_term(self, data, window, inject_time):
        """提取短期時序特徵"""
        try:
            post_data = data[data['timestamp'] > inject_time].tail(window)
            if len(post_data) < 2:
                return np.zeros(3)  # 均值、標準差、變化率
            
            values = post_data.select_dtypes(include=[np.number]).values
            if values.size == 0:
                return np.zeros(3)
            
            # 計算統計特徵
            mean_val = np.mean(values)
            std_val = np.std(values)
            change_rate = (values[-1] - values[0]) / (values[0] + 1e-8) if len(values) > 1 else 0
            
            return np.array([mean_val, std_val, change_rate])
        except:
            return np.zeros(3)
    
    def _extract_long_term(self, data, window, inject_time):
        """提取長期時序特徵"""
        try:
            pre_data = data[data['timestamp'] <= inject_time].tail(window)
            if len(pre_data) < 2:
                return np.zeros(3)
            
            values = pre_data.select_dtypes(include=[np.number]).values
            if values.size == 0:
                return np.zeros(3)
            
            # 計算長期統計特徵
            mean_val = np.mean(values)
            std_val = np.std(values)
            trend_slope = self._calculate_trend_slope(values)
            
            return np.array([mean_val, std_val, trend_slope])
        except:
            return np.zeros(3)
    
    def _extract_trend(self, data, window, inject_time):
        """提取趨勢特徵"""
        try:
            post_data = data[data['timestamp'] > inject_time].tail(window)
            if len(post_data) < 2:
                return np.zeros(2)
            
            values = post_data.select_dtypes(include=[np.number]).values
            if values.size == 0:
                return np.zeros(2)
            
            # 多項式趨勢擬合
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values.flatten(), min(1, len(values)-1))
            
            return coeffs[:2]  # 斜率和截距
        except:
            return np.zeros(2)
    
    def _calculate_trend_slope(self, values):
        """計算趨勢斜率"""
        try:
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values.flatten(), 1)
            return slope
        except:
            return 0


def ica_metric_processing(metrics_data, n_components=None, target_dim=64, inject_time=None):
    """
    改進的 ICA 特徵提取 - 保留關鍵資訊，支援故障前後對比
    
    改進點：
    1. 使用故障前後對比增強異常檢測
    2. 自適應維度分配避免過度壓縮
    3. 保留原始統計特徵補充 ICA 資訊
    
    Args:
        metrics_data: 指標數據
        n_components: ICA成分數量 (None = 自動確定)
        target_dim: 目標維度
        inject_time: 故障注入時間，用於前後對比
    
    Returns:
        enhanced_features: 增強特徵 (num_services, target_dim)
        feature_names: 微服務節點名稱列表
    """
    print("🚀 改進版 ICA 特徵處理 - 保留關鍵資訊，支援時序對比...")
    
    try:
        from sklearn.decomposition import FastICA
        
        # 數據預處理 - 同 simplified_metric_processing
        if isinstance(metrics_data, pd.DataFrame):
            data = metrics_data.select_dtypes(include=[np.number])
        else:
            data = pd.DataFrame(metrics_data) if isinstance(metrics_data, np.ndarray) else pd.DataFrame({'metric': [0]})
        
        data = data.fillna(0).replace([np.inf, -np.inf], 0)
        
        if data.shape[0] < 2 or data.shape[1] == 0:
            return simplified_metric_processing(metrics_data, target_dim)
        
        # 🎯 正確提取微服務名稱
        services = extract_service_names_from_columns(data.columns)
        
        if len(services) == 0:
            print("⚠️ 無法提取微服務名稱，回退到簡化處理")
            return simplified_metric_processing(metrics_data, target_dim)
        
        print(f"✓ 檢測到 {len(services)} 個微服務節點: {services}")
        
        # 🎯 為每個微服務計算ICA特徵
        service_features = []
        
        for service in services:
            # 找到屬於該服務的所有列
            service_cols = [col for col in data.columns if service.lower() in col.lower()]
            
            if not service_cols:
                service_feature = np.zeros(target_dim)
                service_features.append(service_feature)
                continue
            
            # 獲取該服務的數據
            service_data = data[service_cols]
            
            if service_data.shape[1] < 2:
                # 不足以進行ICA，使用基本統計
                series = service_data.iloc[:, 0].dropna()
                if len(series) > 0:
                    basic_stats = [series.mean(), series.std(), series.min(), series.max()]
                    service_feature = np.array(basic_stats + [0] * (target_dim - 4))[:target_dim]
                else:
                    service_feature = np.zeros(target_dim)
                service_features.append(service_feature)
                continue
            
            # 標準化該服務的數據
            scaler = StandardScaler()
            scaled_service_data = scaler.fit_transform(service_data)
            
            # 確定ICA成分數量
            max_components = min(service_data.shape[1], service_data.shape[0] - 1, target_dim // 4)
            if n_components is None:
                ica_components = max_components
            else:
                ica_components = min(n_components, max_components)
            
            if ica_components < 1:
                service_feature = np.zeros(target_dim)
                service_features.append(service_feature)
                continue
            
            # 應用ICA
            ica = FastICA(n_components=ica_components, random_state=42, max_iter=1000, tol=1e-4)
            
            try:
                ica_components_data = ica.fit_transform(scaled_service_data)
                
                # 為該服務提取ICA特徵
                service_ica_features = []
                
                for i in range(ica_components_data.shape[1]):
                    component = ica_components_data[:, i]
                    
                    # 基本統計特徵
                    stats = [
                        np.mean(component),
                        np.std(component),
                        np.median(component),
                        np.min(component),
                        np.max(component),
                        _compute_neg_entropy_approx(component),  # ICA特有：負熵
                        _compute_mutual_info_reduction(component)  # ICA特有：互信息減少
                    ]
                    service_ica_features.extend(stats)
                
                # 調整到目標維度
                if len(service_ica_features) >= target_dim:
                    service_feature = np.array(service_ica_features[:target_dim])
                else:
                    padding = np.zeros(target_dim - len(service_ica_features))
                    service_feature = np.concatenate([service_ica_features, padding])
                
            except Exception as e:
                print(f"⚠️ ICA失敗 for {service}: {e}")
                service_feature = np.zeros(target_dim)
            
            service_features.append(service_feature)
        
        # 轉換為矩陣格式 (num_services, target_dim)
        feature_matrix = np.array(service_features)
        
        # 確保數值穩定性
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"✓ ICA處理: {feature_matrix.shape[0]} 個微服務節點，每個節點 {feature_matrix.shape[1]} 維特徵")
        return feature_matrix, services
        
    except Exception as e:
        print(f"⚠️ ICA processing failed: {e}, falling back to simplified processing")
        return simplified_metric_processing(metrics_data, target_dim)


def _compute_neg_entropy_approx(x):
    """計算負熵近似 - 衡量非高斯性的關鍵指標"""
    try:
        # 標準化
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
        
        # G函數近似：G(u) = 1/α1 * log(cosh(α1*u))
        alpha1 = 1.0
        g_gauss = np.mean(np.log(np.cosh(alpha1 * np.random.randn(len(x_norm)))))
        g_x = np.mean(np.log(np.cosh(alpha1 * x_norm)))
        
        # 負熵近似
        neg_entropy = (g_x - g_gauss)**2
        return neg_entropy if not np.isnan(neg_entropy) else 0.0
    except:
        return 0.0


def _compute_mutual_info_reduction(x):
    """計算互信息減少 - ICA的優化目標"""
    try:
        # 簡化的互信息估計
        # 基於直方圖的方法
        hist, _ = np.histogram(x, bins=min(10, len(x)//2), density=True)
        hist = hist + 1e-8  # 避免log(0)
        
        # 計算熵
        entropy = -np.sum(hist * np.log(hist))
        
        # 與高斯分布的熵差異
        gauss_entropy = 0.5 * np.log(2 * np.pi * np.e * np.var(x))
        mutual_info_reduction = abs(entropy - gauss_entropy)
        
        return mutual_info_reduction if not np.isnan(mutual_info_reduction) else 0.0
    except:
        return 0.0


def kpca_metric_processing(metrics_data, kernel='rbf', gamma=None, target_dim=64):
    """
    基於核PCA (kPCA) 的特徵提取 - 修正微服務節點提取
    處理非線性關係，作為ICA的補充
    
    Args:
        metrics_data: 指標數據
        kernel: 核函數類型 ('rbf', 'poly', 'sigmoid')
        gamma: RBF核參數
        target_dim: 目標維度
    
    Returns:
        kpca_features: kPCA特徵 (num_services, target_dim)
        feature_names: 微服務節點名稱列表
    """
    print("🔧 Using kernel PCA feature processing - 正確提取微服務節點...")
    
    try:
        from sklearn.decomposition import KernelPCA
        
        # 數據預處理
        if isinstance(metrics_data, pd.DataFrame):
            data = metrics_data.select_dtypes(include=[np.number])
        else:
            data = pd.DataFrame(metrics_data) if isinstance(metrics_data, np.ndarray) else pd.DataFrame({'metric': [0]})
        
        data = data.fillna(0).replace([np.inf, -np.inf], 0)
        
        if data.shape[0] < 2 or data.shape[1] == 0:
            return simplified_metric_processing(metrics_data, target_dim)
        
        # 🎯 正確提取微服務名稱
        services = extract_service_names_from_columns(data.columns)
        
        if len(services) == 0:
            print("⚠️ 無法提取微服務名稱，回退到簡化處理")
            return simplified_metric_processing(metrics_data, target_dim)
        
        print(f"✓ 檢測到 {len(services)} 個微服務節點: {services}")
        
        # 🎯 為每個微服務計算kPCA特徵
        service_features = []
        
        for service in services:
            # 找到屬於該服務的所有列
            service_cols = [col for col in data.columns if service.lower() in col.lower()]
            
            if not service_cols:
                service_feature = np.zeros(target_dim)
                service_features.append(service_feature)
                continue
            
            # 獲取該服務的數據
            service_data = data[service_cols]
            
            if service_data.shape[1] < 2:
                # 不足以進行kPCA，使用基本統計
                series = service_data.iloc[:, 0].dropna()
                if len(series) > 0:
                    basic_stats = [series.mean(), series.std(), series.min(), series.max()]
                    service_feature = np.array(basic_stats + [0] * (target_dim - 4))[:target_dim]
                else:
                    service_feature = np.zeros(target_dim)
                service_features.append(service_feature)
                continue
            
            # 標準化該服務的數據
            scaler = StandardScaler()
            scaled_service_data = scaler.fit_transform(service_data)
            
            # 確定kPCA成分數量
            n_components = min(target_dim // 5, service_data.shape[0] - 1, service_data.shape[1])
            
            if n_components < 1:
                service_feature = np.zeros(target_dim)
                service_features.append(service_feature)
                continue
            
            # 應用kPCA
            if gamma is None:
                gamma = 1.0 / service_data.shape[1]
            
            kpca = KernelPCA(
                n_components=n_components,
                kernel=kernel,
                gamma=gamma,
                random_state=42,
                eigen_solver='auto'
            )
            
            try:
                kpca_components = kpca.fit_transform(scaled_service_data)
                
                # 為該服務提取kPCA特徵
                service_kpca_features = []
                
                for i in range(kpca_components.shape[1]):
                    component = kpca_components[:, i]
                    
                    # 核空間特徵統計
                    stats = [
                        np.mean(component),
                        np.std(component),
                        np.median(component),
                        np.min(component),
                        np.max(component)
                    ]
                    service_kpca_features.extend(stats)
                
                # 調整到目標維度
                if len(service_kpca_features) >= target_dim:
                    service_feature = np.array(service_kpca_features[:target_dim])
                else:
                    padding = np.zeros(target_dim - len(service_kpca_features))
                    service_feature = np.concatenate([service_kpca_features, padding])
                
            except Exception as e:
                print(f"⚠️ kPCA失敗 for {service}: {e}")
                service_feature = np.zeros(target_dim)
            
            service_features.append(service_feature)
        
        # 轉換為矩陣格式 (num_services, target_dim)
        feature_matrix = np.array(service_features)
        
        # 確保數值穩定性
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"✓ kPCA處理: {feature_matrix.shape[0]} 個微服務節點，每個節點 {feature_matrix.shape[1]} 維特徵")
        return feature_matrix, services
        
    except Exception as e:
        print(f"⚠️ kPCA processing failed: {e}, falling back to simplified processing")
        return simplified_metric_processing(metrics_data, target_dim)


def simplified_metric_processing(metrics_data, target_dim=64):
    """
    簡化的指標處理 - 🔧 修正微服務節點提取邏輯
    每個微服務應該是一個獨立的節點，而不是將所有特徵壓縮為單一節點
    
    Args:
        metrics_data: 指標數據
        target_dim: 目標維度
    
    Returns:
        processed_features: 處理後的特徵 (num_services, target_dim)
        feature_names: 微服務節點名稱列表
    """
    print("🔧 Using simplified metric processing - 正確提取微服務節點...")
    
    if isinstance(metrics_data, pd.DataFrame):
        data = metrics_data.select_dtypes(include=[np.number])
    elif isinstance(metrics_data, np.ndarray):
        data = pd.DataFrame(metrics_data) if metrics_data.ndim == 2 else pd.DataFrame({'metric': metrics_data})
    else:
        try:
            data = pd.DataFrame(metrics_data)
        except:
            return np.array([[0]]), ['default_feature']

    # 🎯 關鍵修正：正確提取微服務名稱
    services = extract_service_names_from_columns(data.columns)
    
    if len(services) == 0:
        print("⚠️ 無法提取微服務名稱，使用列名作為節點")
        # 回退：每列作為一個節點
        services = list(data.columns)
    
    print(f"✓ 檢測到 {len(services)} 個微服務節點: {services}")
    
    # 🎯 為每個微服務計算節點特徵
    service_features = []
    
    for service in services:
        # 找到屬於該服務的所有列
        service_cols = [col for col in data.columns if service.lower() in col.lower()]
        
        if not service_cols:
            # 如果沒有找到匹配列，用服務名直接匹配
            service_cols = [col for col in data.columns if col == service]
        
        if not service_cols:
            # 最後回退：為該服務創建零特徵
            service_feature = np.zeros(target_dim)
        else:
            # 計算該服務的綜合特徵
            service_data = data[service_cols]
            all_features = []
            
            for col in service_cols:
                series = service_data[col].dropna()
                
                if len(series) < 3:
                    # 數據太少，使用基本統計
                    basic_stats = [series.mean() if len(series) > 0 else 0, 
                                 series.std() if len(series) > 0 else 0, 
                                 series.min() if len(series) > 0 else 0, 
                                 series.max() if len(series) > 0 else 0]
                    all_features.extend(basic_stats)
                    continue
                
                # 核心統計特徵
                core_features = [
                    series.mean(),                    # 中心趨勢
                    series.std(),                     # 離散程度
                    series.min(),                     # 最小值
                    series.max(),                     # 最大值
                    series.median(),                  # 中位數
                    np.percentile(series, 25),        # 第一四分位數
                    np.percentile(series, 75),        # 第三四分位數
                    series.skew() if len(series) > 3 else 0,  # 偏度
                ]
                
                # 簡化的趨勢特徵
                if len(series) >= 5:
                    # 線性趨勢
                    x = np.arange(len(series))
                    trend_coef = np.polyfit(x, series.values, 1)[0]
                    
                    # 變化率
                    diff = np.diff(series.values)
                    change_rate = np.mean(np.abs(diff))
                    
                    # 穩定性
                    stability = 1.0 / (1.0 + np.std(diff))
                    
                    trend_features = [trend_coef, change_rate, stability]
                else:
                    trend_features = [0.0, 0.0, 1.0]
                
                # 異常檢測特徵
                Q1, Q3 = np.percentile(series, [25, 75])
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
                    outlier_ratio = outliers / len(series)
                else:
                    outlier_ratio = 0.0
                
                anomaly_features = [outlier_ratio]
                
                # 組合該列的所有特徵
                col_features = core_features + trend_features + anomaly_features
                all_features.extend(col_features)
            
            # 將該服務的所有特徵調整到目標維度
            if all_features:
                service_feature_vector = np.array(all_features)
                
                # 安全的維度調整
                if len(service_feature_vector) >= target_dim:
                    service_feature = service_feature_vector[:target_dim]
                else:
                    # 填充到目標維度
                    padding = np.zeros(target_dim - len(service_feature_vector))
                    service_feature = np.concatenate([service_feature_vector, padding])
            else:
                service_feature = np.zeros(target_dim)
        
        service_features.append(service_feature)
    
    # 轉換為矩陣格式 (num_services, target_dim)
    if service_features:
        feature_matrix = np.array(service_features)
        
        # 確保數值穩定性
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        # 沒有服務，創建單一默認節點
        feature_matrix = np.zeros((1, target_dim))
        services = ['default_service']
    
    print(f"✓ 正確處理: {feature_matrix.shape[0]} 個微服務節點，每個節點 {feature_matrix.shape[1]} 維特徵")
    return feature_matrix, services


def extract_service_names_from_columns(columns: list) -> list:
    """從列名中提取服務名稱 - 更智能的版本 + IP 地址過濾"""
    services = set()
    
    # 常見的微服務模式
    service_patterns = [
        'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
        'emailservice', 'paymentservice', 'productcatalogservice', 
        'recommendationservice', 'shippingservice', 'frontend'
    ]
    
    for col in columns:
        col_lower = str(col).lower()
        
        # 檢查是否為 IP 地址格式 (如 192-168-xx-xx-xxxx)
        is_ip_format = (
            (col_lower.startswith("192-168-") and col_lower.count("-") >= 4) or
            # 檢查是否符合 IP 地址的一般模式 (數字-數字-數字-數字-端口)
            (col_lower.count("-") >= 4 and all(part.isdigit() for part in col_lower.split("-")))
        )
        
        if is_ip_format:
            # 跳過 IP 地址格式的列名
            continue
        
        # 檢查是否包含已知的微服務名稱
        for pattern in service_patterns:
            if pattern in col_lower:
                services.add(pattern)
                break
        else:
            # 嘗試從列名中提取前綴
            if '_' in col_lower:
                prefix = col_lower.split('_')[0]
                if len(prefix) > 2:  # 避免太短的前綴
                    services.add(prefix)
            elif '-' in col_lower:
                prefix = col_lower.split('-')[0]
                if len(prefix) > 2 and not prefix.isdigit():  # 避免數字前綴
                    services.add(prefix)
    
    return sorted(list(services))


def psm_metric_processing(metrics_data, target_dim=64):
    """重定向到 advanced_processors 中的統一實現 - 避免重複定義"""
    from .advanced_processors import psm_metric_processing as _psm_impl
    return _psm_impl(metrics_data, target_dim)


def _phase_space_embedding(series, embedding_dim=3, delay=1):
    """重定向到 advanced_processors 中的統一實現 - 避免重複定義"""
    from .advanced_processors import _phase_space_embedding as _phase_impl
    return _phase_impl(series, embedding_dim, delay)


def _compute_dynamics_invariants(series):
    """重定向到 advanced_processors 中的統一實現 - 避免重複定義"""
    from .advanced_processors import _compute_dynamics_invariants as _dynamics_impl
    return _dynamics_impl(series)


def _compute_stability_metrics(series):
    """重定向到 advanced_processors 中的統一實現 - 避免重複定義"""
    from .advanced_processors import _compute_stability_metrics as _stability_impl
    return _stability_impl(series)


def _compute_frequency_features(series):
    """重定向到 advanced_processors 中的統一實現 - 避免重複定義"""
    from .advanced_processors import _compute_frequency_features as _freq_impl
    return _freq_impl(series)


def _compute_sample_entropy(series, m=2, r=None):
    """重定向到 advanced_processors 中的統一實現 - 避免重複定義"""
    from .advanced_processors import _compute_sample_entropy as _entropy_impl
    return _entropy_impl(series, m, r)


# =============================================
# RCA導向特徵提取方法
# =============================================

def _identify_sli_column(df: pd.DataFrame, dataset_name: str = "") -> str:
    """
    識別SLI (Service Level Indicator) 列

    Args:
        df: 輸入數據框
        dataset_name: 數據集名稱，用於特定數據集的SLI識別

    Returns:
        SLI列名，如果找不到返回None
    """
    if df.empty:
        return None

    # 定義SLI候選模式，按優先級排序
    sli_candidates = [
        # 通用SLI指標
        'latency', 'response_time', 'response', 'duration', 'delay',
        'error_rate', 'error', 'error_total', 'errors', 'failures',
        'throughput', 'requests_per_second', 'qps', 'rps',
        'availability', 'uptime', 'health',
        # train-ticket 特定模式
        'ts-ui-dashboard_istio-latency', 'ts-ui-dashboard_latency', 'istio-latency',
        'ui_latency', 'frontend_latency', 'dashboard_latency',
        # RE1 特定模式
        're1_latency', 're1_error', 're1_throughput',
        # 常見錯誤指標
        '5xx', '4xx', 'timeout', 'retry', 'exception', 'failure'
    ]

    # 第一輪：精確匹配
    for candidate in sli_candidates:
        for col in df.columns:
            if candidate.lower() in col.lower():
                return col

    # 第二輪：模糊匹配
    for col in df.columns:
        col_lower = col.lower()
        # 延遲相關
        if any(keyword in col_lower for keyword in ['lat', 'delay', 'response', 'duration']):
            return col
        # 錯誤相關
        elif any(keyword in col_lower for keyword in ['error', 'fail', 'exception', 'timeout']):
            return col
        # 吞吐量相關
        elif any(keyword in col_lower for keyword in ['throughput', 'qps', 'rps', 'rate']):
            return col

    # 第三輪：如果找不到明確SLI，選擇變異度最大的數值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # 計算每個數值列的標準差，選擇變異度最大的
        max_std_col = None
        max_std = 0
        for col in numeric_cols:
            if col in df.columns:
                std_val = df[col].std()
                if std_val > max_std:
                    max_std = std_val
                    max_std_col = col
        if max_std_col:
            return max_std_col

    return None


def _extract_service_groups_rca(df: pd.DataFrame, dataset_name: str = "") -> Dict[str, List[str]]:
    """
    RCA導向的服務分組策略

    Args:
        df: 輸入數據框
        dataset_name: 數據集名稱

    Returns:
        服務名到列名列表的映射
    """
    if df.empty:
        return {}

    columns = df.columns.tolist()
    service_groups = {}

    # 定義已知服務模式
    known_services = {
        # Online Boutique
        'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
        'emailservice', 'paymentservice', 'productcatalogservice',
        'recommendationservice', 'shippingservice', 'frontend',
        # Sock Shop
        'front-end', 'user', 'carts', 'orders', 'shipping', 'payment',
        'catalogue', 'user-db', 'carts-db', 'orders-db', 'catalogue-db',
        # Train Ticket (ts- 前綴)
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

    # 策略1：精確匹配已知服務
    for col in columns:
        col_lower = col.lower()

        # 檢查是否包含已知服務名稱
        for service in known_services:
            if service in col_lower:
                if service not in service_groups:
                    service_groups[service] = []
                service_groups[service].append(col)
                break

    # 如果沒有找到服務，使用列名前綴分組
    if not service_groups:
        prefix_groups = {}
        for col in columns:
            # 處理 ts- 前綴的特殊情況
            if col.startswith('ts-'):
                parts = col.split('-')
                if len(parts) >= 3:
                    service_name = f"ts-{parts[1]}-{parts[2]}"
                    if service_name not in prefix_groups:
                        prefix_groups[service_name] = []
                    prefix_groups[service_name].append(col)
            else:
                # 常規前綴分組
                if '_' in col:
                    prefix = col.split('_')[0]
                elif '-' in col:
                    prefix = col.split('-')[0]
                else:
                    prefix = col

                prefix_lower = prefix.lower()
                if prefix_lower not in prefix_groups:
                    prefix_groups[prefix_lower] = []
                prefix_groups[prefix_lower].append(col)

        service_groups = prefix_groups

    # 清理小分組（少於2列的分組可能不是真正的服務）
    cleaned_groups = {}
    for service, cols in service_groups.items():
        if len(cols) >= 1:  # 放寬條件，每個服務至少1列
            cleaned_groups[service] = cols

    return cleaned_groups


def _compute_cross_correlation_with_lag(series1: np.ndarray, series2: np.ndarray,
                                      max_lag: int = 10) -> Tuple[float, int]:
    """
    計算兩個時間序列的互相關和最佳滯後

    Args:
        series1: 第一個時間序列
        series2: 第二個時間序列
        max_lag: 最大滯後步數

    Returns:
        最大相關係數, 對應的滯後
    """
    if len(series1) < 2 or len(series2) < 2:
        return 0.0, 0

    # 標準化序列
    s1 = (series1 - np.nanmean(series1)) / (np.nanstd(series1) + 1e-8)
    s2 = (series2 - np.nanmean(series2)) / (np.nanstd(series2) + 1e-8)

    best_corr = 0.0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # series1 滯後
            x = s1[-lag:] if -lag <= len(s1) else s1
            y = s2[:len(x)] if len(x) <= len(s2) else s2[:len(x)]
        elif lag > 0:
            # series2 滯後
            x = s1[:-lag] if lag <= len(s1) else s1
            y = s2[lag:] if lag <= len(s2) else s2
        else:
            # 同步
            x = s1
            y = s2

        if len(x) < 3 or len(y) < 3:
            continue

        # 計算相關係數
        corr = np.corrcoef(x, y)[0, 1]
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return best_corr, best_lag


def _extract_temporal_features(data: np.ndarray, inject_time: int = None) -> Dict[str, float]:
    """
    提取時序特徵

    Args:
        data: 時間序列數據
        inject_time: 注入時間點

    Returns:
        時序特徵字典
    """
    if len(data) < 2:
        return {'mean': 0.0, 'std': 0.0, 'trend': 0.0, 'anomaly_score': 0.0}

    # 基本統計特徵
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)

    # 趨勢特徵
    if len(data) >= 3:
        x = np.arange(len(data))
        try:
            trend_coef = np.polyfit(x, data, 1)[0]
        except:
            trend_coef = 0.0
    else:
        trend_coef = 0.0

    # 異常分數
    if std_val > 0:
        z_scores = np.abs((data - mean_val) / std_val)
        anomaly_score = np.percentile(z_scores, 95)  # 95分位數異常分數
    else:
        anomaly_score = 0.0

    return {
        'mean': float(mean_val),
        'std': float(std_val),
        'trend': float(trend_coef),
        'anomaly_score': float(anomaly_score)
    }


def rca_aware_metric_processing(metrics_df: pd.DataFrame,
                              inject_time: Optional[float] = None,
                              dataset_name: str = "",
                              target_dim: int = 48) -> Tuple[np.ndarray, List[str]]:
    """
    RCA導向特徵提取方法

    Args:
        metrics_df: 指標數據
        inject_time: 故障注入時間
        dataset_name: 數據集名稱
        target_dim: 目標特徵維度

    Returns:
        特徵矩陣和服務名稱列表
    """
    print("🔍 RCA導向特徵提取 - 專注SLI對齊和時序傳播分析...")

    if metrics_df.empty:
        return np.zeros((1, target_dim)), ['default_service']

    # 1. 數據預處理
    df = metrics_df.copy()
    if 'time' in df.columns:
        df = df.drop(columns=['time'])

    # 填充缺失值
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0)

    # 2. 識別SLI
    sli_col = _identify_sli_column(df, dataset_name)
    if sli_col:
        sli_data = df[sli_col].values
        print(f"✓ 識別SLI: {sli_col}")
    else:
        # 如果找不到SLI，使用所有數據的均值作為代理
        sli_data = df.mean(axis=1).values
        sli_col = 'mean_proxy'
        print(f"⚠️ 未找到明確SLI，使用均值代理")

    # 3. 服務分組
    service_groups = _extract_service_groups_rca(df, dataset_name)

    if not service_groups:
        # 如果無法分組，創建默認分組
        service_groups = {'default_service': df.columns.tolist()}
        print("⚠️ 無法分組，使用默認服務")

    print(f"✓ 識別 {len(service_groups)} 個服務: {list(service_groups.keys())}")

    # 4. 特徵提取
    all_features = []
    all_service_names = []

    for service_name, columns in service_groups.items():
        service_data = df[columns].values

        if service_data.shape[0] < 2 or service_data.shape[1] == 0:
            # 數據不足，創建零特徵
            service_feature = np.zeros(target_dim)
            all_features.append(service_feature)
            all_service_names.append(service_name)
            continue

        # 計算該服務的聚合時間序列（均值）
        service_series = np.nanmean(service_data, axis=1)

        # 時序分割（前後注入點）
        if inject_time is not None and len(df) > 0:
            inject_idx = np.searchsorted(np.arange(len(df)), inject_time, side='left')
            pre_end = max(0, inject_idx)
            post_start = min(len(df), inject_idx)
        else:
            # 如果沒有注入時間，使用中點分割
            mid_point = len(df) // 2
            pre_end = mid_point
            post_start = mid_point

        pre_data = service_series[:pre_end]
        post_data = service_series[post_start:]

        # 5. 計算各類特徵
        features = []

        # 5.1 SLI相關性特徵
        if len(service_series) >= 3 and len(sli_data) >= 3:
            # 同步相關性
            sync_corr = np.corrcoef(service_series, sli_data)[0, 1]
            # 互相關和滯後
            xcorr, lag = _compute_cross_correlation_with_lag(service_series, sli_data)

            features.extend([
                float(sync_corr),
                float(xcorr),
                float(lag)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 5.2 時序差異特徵
        pre_features = _extract_temporal_features(pre_data)
        post_features = _extract_temporal_features(post_data)

        # 變化量（後 - 前）
        change_features = {
            'mean_change': post_features['mean'] - pre_features['mean'],
            'std_change': post_features['std'] - pre_features['std'],
            'trend_change': post_features['trend'] - pre_features['trend'],
            'anomaly_change': post_features['anomaly_score'] - pre_features['anomaly_score']
        }

        features.extend([
            change_features['mean_change'],
            change_features['std_change'],
            change_features['trend_change'],
            change_features['anomaly_change']
        ])

        # 5.3 異常檢測特徵
        if len(post_data) > 0 and pre_features['std'] > 0:
            # 相對於基準的異常分數
            baseline_mean = pre_features['mean']
            baseline_std = pre_features['std'] + 1e-8
            z_scores = np.abs((post_data - baseline_mean) / baseline_std)
            max_z_score = np.max(z_scores)
            mean_z_score = np.mean(z_scores)

            features.extend([float(max_z_score), float(mean_z_score)])
        else:
            features.extend([0.0, 0.0])

        # 5.4 故障類型特化特徵
        # 識別關鍵指標類型
        latency_cols = [col for col in columns if any(k in col.lower() for k in ['lat', 'delay', 'response', 'duration'])]
        error_cols = [col for col in columns if any(k in col.lower() for k in ['error', 'fail', 'exception', 'timeout'])]

        if latency_cols:
            latency_data = df[latency_cols].values
            latency_series = np.nanmean(latency_data, axis=1)
            # 延遲變化特徵
            latency_change = _extract_temporal_features(latency_series[post_start:])['mean'] - _extract_temporal_features(latency_series[:pre_end])['mean']
            features.append(float(latency_change))
        else:
            features.append(0.0)

        if error_cols:
            error_data = df[error_cols].values
            error_series = np.nanmean(error_data, axis=1)
            # 錯誤率變化特徵
            error_change = _extract_temporal_features(error_series[post_start:])['mean'] - _extract_temporal_features(error_series[:pre_end])['mean']
            features.append(float(error_change))
        else:
            features.append(0.0)

        # 5.5 多尺度特徵（短/中/長窗口）
        try:
            total_len = len(service_series)
            # 定義多尺度窗口大小（以樣本數為單位）
            short_win = max(8, total_len // 12)   # 約 ~1/12 長度
            medium_win = max(16, total_len // 4) # 約 ~1/4 長度
            long_win = total_len                 # 全序列

            scales = [
                ("short", short_win),
                ("medium", medium_win),
                ("long", long_win),
            ]

            for _, win in scales:
                if win < 3:
                    # 窗口過短，填充佔位
                    features.extend([0.0, 0.0, 0.0])
                    continue

                # 針對當前窗口重建 pre/post 區段
                pre_slice_start = max(0, pre_end - win)
                pre_slice_end = pre_end
                post_slice_start = post_start
                post_slice_end = min(total_len, post_start + win)

                pre_slice = service_series[pre_slice_start:pre_slice_end]
                post_slice = service_series[post_slice_start:post_slice_end]

                if len(pre_slice) < 2 or len(post_slice) < 2:
                    features.extend([0.0, 0.0, 0.0])
                    continue

                # 多尺度均值變化
                ms_mean_change = float(np.nanmean(post_slice) - np.nanmean(pre_slice))

                # 多尺度與 SLI 的相關（同步）
                if len(sli_data) >= max(post_slice_end, pre_slice_end):
                    try:
                        sli_pre = sli_data[pre_slice_start:pre_slice_end]
                        sli_post = sli_data[post_slice_start:post_slice_end]
                        # 使用 post 段與 SLI 的相關性作為代表
                        if len(sli_post) >= 2 and len(post_slice) >= 2:
                            ms_sync_corr = float(np.corrcoef(post_slice, sli_post)[0, 1])
                        else:
                            ms_sync_corr = 0.0
                    except Exception:
                        ms_sync_corr = 0.0
                else:
                    ms_sync_corr = 0.0

                # 多尺度 lag（互相關最大值位置，限制在窗口內）
                try:
                    xcorr_val, lag_val = _compute_cross_correlation_with_lag(
                        post_slice, sli_data[post_slice_start:post_slice_end] if len(sli_data) >= post_slice_end else post_slice
                    )
                    # 只取 lag，xcorr 已在上方提供同步度量
                    ms_lag = float(lag_val)
                except Exception:
                    ms_lag = 0.0

                features.extend([ms_mean_change, ms_sync_corr, ms_lag])
        except Exception:
            # 忽略多尺度提取失敗，保持向後相容
            pass

        # 6. 填充到目標維度
        if len(features) >= target_dim:
            service_feature = np.array(features[:target_dim])
        else:
            # 填充零
            padding = np.zeros(target_dim - len(features))
            service_feature = np.concatenate([features, padding])

        all_features.append(service_feature)
        all_service_names.append(service_name)

    # 7. 特徵後處理
    feature_matrix = np.array(all_features)

    # 標準化特徵（每列）+ 穩健回退
    if feature_matrix.shape[0] > 1:
        for col in range(feature_matrix.shape[1]):
            col_data = feature_matrix[:, col]
            std_val = float(np.std(col_data))
            if std_val > 0:
                feature_matrix[:, col] = (col_data - np.mean(col_data)) / std_val
            else:
                # Robust scaling 回退：使用 IQR
                q1 = np.percentile(col_data, 25)
                q3 = np.percentile(col_data, 75)
                iqr = float(q3 - q1)
                if iqr > 0:
                    median = np.median(col_data)
                    feature_matrix[:, col] = (col_data - median) / iqr
                # 否則保持原值

    # 處理NaN和無限值
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    print(f"✅ RCA特徵提取完成: {feature_matrix.shape[0]} 服務 × {feature_matrix.shape[1]} 維特徵")
    print(f"   SLI: {sli_col}, 服務: {all_service_names}")

    return feature_matrix, all_service_names


def compute_anomaly_propagation_features(data: pd.DataFrame,
                                       service_groups: Dict[str, List[str]],
                                       inject_time: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    計算異常傳播特徵

    Args:
        data: 輸入數據
        service_groups: 服務分組
        inject_time: 注入時間

    Returns:
        異常傳播特徵字典
    """
    if data.empty or not service_groups or inject_time is None:
        return {}

    # 時序分割
    time_col = None
    if 'time' in data.columns:
        time_col = data['time']
        data_numeric = data.drop(columns=['time'])
    else:
        time_col = np.arange(len(data))
        data_numeric = data

    # 找到注入時間索引
    inject_idx = np.searchsorted(time_col, inject_time, side='left')
    pre_end = max(0, inject_idx)
    post_start = min(len(data), inject_idx)

    if pre_end == 0 or post_start >= len(data):
        return {}

    propagation_features = {}

    for service_name, columns in service_groups.items():
        service_data = data_numeric[columns].values

        if service_data.shape[0] < 2 or service_data.shape[1] == 0:
            continue

        # 計算服務聚合序列
        service_series = np.nanmean(service_data, axis=1)

        # 分割前後數據
        pre_data = service_series[:pre_end]
        post_data = service_series[post_start:]

        if len(pre_data) < 3 or len(post_data) < 3:
            continue

        # 1. 異常積累特徵
        anomaly_features = _compute_anomaly_accumulation(pre_data, post_data)

        # 2. 傳播延遲特徵
        propagation_delay = _compute_propagation_delay(service_series, inject_idx, time_col)

        # 3. 級聯異常特徵
        cascade_features = _compute_cascade_anomalies(service_series, pre_data, post_data, inject_idx)

        # 組合特徵
        service_features = np.concatenate([
            anomaly_features,
            [propagation_delay],
            cascade_features
        ])

        propagation_features[service_name] = service_features

    return propagation_features


def _compute_anomaly_accumulation(pre_data: np.ndarray, post_data: np.ndarray) -> np.ndarray:
    """計算異常積累特徵"""
    # 基準統計
    baseline_mean = np.mean(pre_data)
    baseline_std = np.std(pre_data) + 1e-8

    # 後窗口異常分數
    z_scores = np.abs((post_data - baseline_mean) / baseline_std)

    # 異常積累特徵
    anomaly_features = [
        np.mean(z_scores),                    # 平均異常分數
        np.max(z_scores),                     # 最大異常分數
        np.sum(z_scores > 2) / len(z_scores), # 高異常比例 (>2σ)
        np.sum(z_scores > 3) / len(z_scores), # 極端異常比例 (>3σ)
        _compute_anomaly_momentum(z_scores),  # 異常動量
    ]

    return np.array(anomaly_features)


def _compute_propagation_delay(service_series: np.ndarray, inject_idx: int, time_col) -> float:
    """計算傳播延遲特徵"""
    # 異常檢測閾值
    baseline_std = np.std(service_series[:inject_idx])
    threshold = 2.0 * baseline_std

    # 找到首次超過閾值的時間點
    baseline_mean = np.mean(service_series[:inject_idx])
    z_scores = np.abs((service_series - baseline_mean) / (baseline_std + 1e-8))

    anomaly_indices = np.where(z_scores > threshold)[0]
    if len(anomaly_indices) == 0:
        return 0.0  # 沒有異常

    first_anomaly_idx = anomaly_indices[0]
    propagation_delay = time_col[first_anomaly_idx] - time_col[inject_idx]

    return float(propagation_delay)


def _compute_cascade_anomalies(service_series: np.ndarray, pre_data: np.ndarray,
                             post_data: np.ndarray, inject_idx: int) -> np.ndarray:
    """計算級聯異常特徵"""
    # 計算前後窗口的異常模式
    baseline_mean = np.mean(pre_data)
    baseline_std = np.std(pre_data) + 1e-8

    pre_z_scores = np.abs((pre_data - baseline_mean) / baseline_std)
    post_z_scores = np.abs((post_data - baseline_mean) / baseline_std)

    # 級聯特徵
    cascade_features = [
        np.mean(post_z_scores) / (np.mean(pre_z_scores) + 1e-8),  # 異常放大倍數
        np.max(post_z_scores) / (np.max(pre_z_scores) + 1e-8),    # 峰值放大倍數
        _compute_anomaly_persistence(post_z_scores),               # 異常持續性
        _compute_anomaly_spread(post_z_scores),                    # 異常擴散
    ]

    return np.array(cascade_features)


def _compute_anomaly_momentum(z_scores: np.ndarray) -> float:
    """計算異常動量（異常變化趨勢）"""
    if len(z_scores) < 3:
        return 0.0

    # 計算異常分數的趨勢
    x = np.arange(len(z_scores))
    try:
        slope, _ = np.polyfit(x, z_scores, 1)
        return float(abs(slope))
    except:
        return 0.0


def _compute_anomaly_persistence(z_scores: np.ndarray) -> float:
    """計算異常持續性"""
    if len(z_scores) < 2:
        return 0.0

    # 計算連續異常段的持續時間
    anomaly_mask = z_scores > 2.0
    if not np.any(anomaly_mask):
        return 0.0

    # 找到最長連續異常段
    max_persistence = 0
    current_persistence = 0

    for is_anomaly in anomaly_mask:
        if is_anomaly:
            current_persistence += 1
            max_persistence = max(max_persistence, current_persistence)
        else:
            current_persistence = 0

    return float(max_persistence) / len(z_scores)


def _compute_anomaly_spread(z_scores: np.ndarray) -> float:
    """計算異常擴散程度"""
    if len(z_scores) < 2:
        return 0.0

    # 計算異常分數的方差變化
    anomaly_values = z_scores[z_scores > 2.0]
    if len(anomaly_values) < 2:
        return 0.0

    return float(np.std(anomaly_values) / (np.mean(anomaly_values) + 1e-8))


def compute_mutual_information_matrix(data: pd.DataFrame,
                                    sli_col: str = None,
                                    max_features: int = 32) -> Tuple[np.ndarray, List[str]]:
    """
    計算互信息矩陣並選擇最重要的特徵

    Args:
        data: 輸入數據
        sli_col: SLI列名，如果為None則自動識別
        max_features: 最大特徵數量

    Returns:
        選擇的特徵矩陣和特徵名稱列表
    """
    if data.empty:
        return np.array([]), []

    # 識別SLI列
    if sli_col is None:
        sli_col = _identify_sli_column(data, "unknown")

    # 準備數據
    numeric_data = data.select_dtypes(include=[np.number])
    if sli_col and sli_col in numeric_data.columns:
        sli_data = numeric_data[sli_col].values
        feature_cols = [col for col in numeric_data.columns if col != sli_col]
    else:
        # 如果沒有SLI，使用第一列作為代理
        sli_data = numeric_data.iloc[:, 0].values
        feature_cols = numeric_data.columns.tolist()[1:]

    if len(feature_cols) == 0:
        return np.array([]), []

    # 計算互信息
    mutual_info_scores = []

    for col in feature_cols:
        try:
            feature_data = numeric_data[col].values

            # 計算互信息（簡化的版本）
            mi_score = _compute_mutual_information(sli_data, feature_data)
            mutual_info_scores.append((col, mi_score))
        except:
            mutual_info_scores.append((col, 0.0))

    # 按互信息分數排序
    mutual_info_scores.sort(key=lambda x: x[1], reverse=True)

    # 選擇最重要的特徵
    selected_features = mutual_info_scores[:max_features]
    selected_cols = [col for col, score in selected_features]

    if len(selected_cols) == 0:
        return np.array([]), []

    # 返回選擇的特徵矩陣
    feature_matrix = numeric_data[selected_cols].values

    return feature_matrix, selected_cols


def _compute_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """計算兩個序列的互信息（簡化版本）"""
    if len(x) < 10 or len(y) < 10:
        return 0.0

    try:
        # 離散化數據
        x_binned = _discretize_data(x, n_bins=10)
        y_binned = _discretize_data(y, n_bins=10)

        # 計算聯合分佈和邊緣分佈
        joint_hist, _, _ = np.histogram2d(x_binned, y_binned, bins=10)
        x_hist, _ = np.histogram(x_binned, bins=10)
        y_hist, _ = np.histogram(y_binned, bins=10)

        # 避免零概率
        joint_hist = joint_hist + 1e-8
        x_hist = x_hist + 1e-8
        y_hist = y_hist + 1e-8

        # 計算互信息
        joint_prob = joint_hist / joint_hist.sum()
        x_prob = x_hist / x_hist.sum()
        y_prob = y_hist / y_hist.sum()

        mutual_info = 0.0
        for i in range(10):
            for j in range(10):
                if joint_prob[i, j] > 0:
                    mutual_info += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))

        return max(0.0, mutual_info)

    except Exception:
        # 如果計算失敗，返回相關係數作為替代
        try:
            return abs(np.corrcoef(x, y)[0, 1])
        except:
            return 0.0


def _discretize_data(data: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """將連續數據離散化"""
    if len(data) < 2:
        return np.zeros_like(data)

    # 移除NaN值
    data_clean = data[~np.isnan(data)]
    if len(data_clean) < 2:
        return np.zeros_like(data)

    # 計算分位數邊界
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(data_clean, percentiles[1:-1])  # 排除0和100

    # 離散化
    return np.digitize(data, bin_edges, right=False)


def select_features_by_correlation(data: pd.DataFrame,
                                 sli_col: str = None,
                                 threshold: float = 0.2,
                                 max_features: int = 32) -> Tuple[np.ndarray, List[str]]:
    """
    基於相關性選擇特徵

    Args:
        data: 輸入數據
        sli_col: SLI列名
        threshold: 相關性閾值
        max_features: 最大特徵數量

    Returns:
        選擇的特徵矩陣和特徵名稱列表
    """
    if data.empty:
        return np.array([]), []

    # 識別SLI列
    if sli_col is None:
        sli_col = _identify_sli_column(data, "unknown")

    # 準備數據
    numeric_data = data.select_dtypes(include=[np.number])
    if sli_col and sli_col in numeric_data.columns:
        sli_data = numeric_data[sli_col].values
        feature_cols = [col for col in numeric_data.columns if col != sli_col]
    else:
        sli_data = numeric_data.iloc[:, 0].values
        feature_cols = numeric_data.columns.tolist()[1:]

    if len(feature_cols) == 0:
        return np.array([]), []

    # 計算相關性
    correlations = []
    valid_cols = []

    for col in feature_cols:
        try:
            feature_data = numeric_data[col].values

            # 計算相關係數
            corr = np.corrcoef(sli_data, feature_data)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, abs(corr)))
                valid_cols.append(col)
        except:
            continue

    # 按相關性排序
    correlations.sort(key=lambda x: x[1], reverse=True)

    # 選擇相關性高於閾值的特徵
    selected_features = [col for col, corr in correlations if corr >= threshold]
    selected_features = selected_features[:max_features]  # 限制數量

    if len(selected_features) == 0:
        # 如果沒有特徵超過閾值，選擇最相關的特徵
        selected_features = [col for col, _ in correlations[:max_features]]

    if len(selected_features) == 0:
        return np.array([]), []

    # 返回選擇的特徵矩陣
    feature_matrix = numeric_data[selected_features].values

    return feature_matrix, selected_features


def rca_feature_selection_and_reduction(data: pd.DataFrame,
                                      inject_time: Optional[float] = None,
                                      dataset_name: str = "",
                                      sli_col: str = None,
                                      method: str = 'hybrid',
                                      correlation_threshold: float = 0.2,
                                      max_features: int = 32) -> Tuple[np.ndarray, List[str]]:
    """
    RCA導向特徵選擇和降維

    Args:
        data: 輸入數據
        inject_time: 注入時間（用於異常檢測）
        dataset_name: 數據集名稱
        sli_col: SLI列名
        method: 選擇方法 ('correlation', 'mutual_info', 'hybrid')
        correlation_threshold: 相關性閾值
        max_features: 最大特徵數量

    Returns:
        特徵矩陣和特徵名稱列表
    """
    print(f"🔧 RCA特徵選擇和降維 - 方法: {method}, 閾值: {correlation_threshold}")

    if data.empty:
        return np.array([]), []

    # 1. 識別SLI列
    if sli_col is None:
        sli_col = _identify_sli_column(data, dataset_name)
        if sli_col:
            print(f"✓ 識別SLI: {sli_col}")

    # 2. 選擇特徵
    if method == 'correlation':
        feature_matrix, selected_cols = select_features_by_correlation(
            data, sli_col, correlation_threshold, max_features
        )
    elif method == 'mutual_info':
        feature_matrix, selected_cols = compute_mutual_information_matrix(
            data, sli_col, max_features
        )
    elif method == 'hybrid':
        # 先用相關性選擇，然後用互信息精煉
        corr_matrix, corr_cols = select_features_by_correlation(
            data, sli_col, correlation_threshold * 0.8, max_features * 2
        )
        if len(corr_cols) > 0:
            hybrid_data = pd.DataFrame(corr_matrix, columns=corr_cols)
            hybrid_data[sli_col] = data[sli_col].values if sli_col in data.columns else hybrid_data.iloc[:, 0]
            feature_matrix, selected_cols = compute_mutual_information_matrix(
                hybrid_data, sli_col, max_features
            )
        else:
            feature_matrix, selected_cols = corr_matrix, corr_cols
    else:
        # 默認使用相關性
        feature_matrix, selected_cols = select_features_by_correlation(
            data, sli_col, correlation_threshold, max_features
        )

    if len(selected_cols) == 0:
        print("⚠️ 沒有選擇到特徵，使用原始數據")
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            feature_matrix = numeric_data.values
            selected_cols = numeric_data.columns.tolist()
        else:
            return np.array([]), []

    # 3. 特徵降維（如果特徵數量太多）
    if feature_matrix.shape[1] > max_features:
        print(f"🔧 特徵降維: {feature_matrix.shape[1]} -> {max_features}")

        # 使用PCA降維
        from sklearn.decomposition import PCA
        pca = PCA(n_components=max_features, random_state=42)
        feature_matrix = pca.fit_transform(feature_matrix)

        # 創建新的特徵名稱
        selected_cols = [f"rca_feature_{i}" for i in range(max_features)]

    # 4. 特徵標準化
    if feature_matrix.shape[0] > 1 and feature_matrix.shape[1] > 0:
        for col in range(feature_matrix.shape[1]):
            col_data = feature_matrix[:, col]
            if np.std(col_data) > 0:
                feature_matrix[:, col] = (col_data - np.mean(col_data)) / np.std(col_data)

    # 5. 處理NaN和無限值
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    print(f"✅ RCA特徵選擇完成: {feature_matrix.shape[1]} 特徵, 選擇方法: {selected_cols[:5]}{'...' if len(selected_cols) > 5 else ''}")

    return feature_matrix, selected_cols


def rca_multimodal_feature_processing(data_dict: Dict[str, Any],
                                    inject_time: Optional[float] = None,
                                    dataset_name: str = "",
                                    target_dim: int = 48) -> Tuple[np.ndarray, List[str]]:
    """
    RCA導向多模態特徵提取方法

    Args:
        data_dict: 包含 'metrics', 'logs', 'traces' 的字典
        inject_time: 故障注入時間
        dataset_name: 數據集名稱
        target_dim: 目標特徵維度

    Returns:
        特徵矩陣和節點名稱列表
    """
    print("🔍 RCA導向多模態特徵提取 - 整合metrics, logs, traces...")

    # 初始化RCA多模態特徵提取器
    multimodal_extractor = RCAMultiModalFeatureExtractor(
        embed_dim=target_dim,
        dataset_name=dataset_name
    )

    try:
        # 提取特徵
        fused_embeddings, attention_weights = multimodal_extractor(data_dict)

        # 轉換為numpy數組
        features = fused_embeddings.detach().cpu().numpy()

        # 創建節點名稱（基於可用的模態）
        node_names = []

        if 'metrics' in data_dict and data_dict['metrics'] is not None:
            # 從metrics數據提取服務名稱
            if isinstance(data_dict['metrics'], pd.DataFrame):
                service_groups = _extract_service_groups_rca(data_dict['metrics'], dataset_name)
                node_names.extend(list(service_groups.keys()))

        if not node_names:
            # 如果無法提取服務名稱，創建默認名稱
            node_names = [f'modal_node_{i}' for i in range(features.shape[0])]

        # 確保特徵數量與節點數量匹配
        if features.shape[0] != len(node_names):
            min_nodes = min(features.shape[0], len(node_names))
            features = features[:min_nodes]
            node_names = node_names[:min_nodes]

        # 特徵後處理
        if features.shape[0] > 1:
            # 標準化特徵
            for col in range(features.shape[1]):
                col_data = features[:, col]
                if np.std(col_data) > 0:
                    features[:, col] = (col_data - np.mean(col_data)) / np.std(col_data)

        # 處理NaN和無限值
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        print(f"✅ RCA多模態特徵提取完成: {features.shape[0]} 節點 × {features.shape[1]} 維特徵")
        print(f"   可用模態: {list(data_dict.keys())}")
        print(f"   節點名稱: {node_names}")

        return features, node_names

    except Exception as e:
        print(f"❌ RCA多模態特徵提取失敗: {e}")
        # 回退到單一模態處理
        if 'metrics' in data_dict and data_dict['metrics'] is not None:
            print("⚠️ 回退到metrics單模態處理")
            return rca_aware_metric_processing(
                data_dict['metrics'],
                inject_time=inject_time,
                dataset_name=dataset_name,
                target_dim=target_dim
            )
        else:
            return np.zeros((1, target_dim)), ['default_multimodal_node']

    
def _maxdist(xi, xj, m):
    """重定向到 advanced_processors 中的統一實現 - 避免重複定義"""
    from .advanced_processors import _maxdist as _maxdist_impl
    return _maxdist_impl(xi, xj, m)


def _phi(m):
    """重定向到 advanced_processors 中的統一實現 - 避免重複定義"""
    from .advanced_processors import _phi as _phi_impl
    return _phi_impl(m)


# 注意：gnn_kan_rca 主實現已移至 e2e/gnnkan.py
# 此檔案專注於特徵處理功能，不包含主要的 RCA 函數


class RCAMultiModalFeatureExtractor(nn.Module):
    """
    RCA導向多模態特徵提取器
    整合 metrics, logs, traces 三種模態的數據，專注於RCA任務
    特點：SLI對齊、異常傳播分析、跨模態相關性
    """

    def __init__(self, embed_dim=48, num_heads=4, dataset_name="unknown"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dataset_name = dataset_name

        # 多模態注意力機制
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # 模態特定的特徵提取器
        self.metrics_extractor = RCAMetricsFeatureExtractor(embed_dim)
        self.logs_extractor = RCALogsFeatureExtractor(embed_dim)
        self.traces_extractor = RCATracesFeatureExtractor(embed_dim)

        # 模態特定的投影層
        self.metrics_proj = nn.Linear(embed_dim, embed_dim)
        self.logs_proj = nn.Linear(embed_dim, embed_dim)
        self.traces_proj = nn.Linear(embed_dim, embed_dim)

        # RCA專用的融合層
        self.rca_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # SLI增強模組
        self.sli_enhancer = nn.Sequential(
            nn.Linear(embed_dim + 16, embed_dim),  # 16是SLI特徵維度
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, data_dict):
        """
        RCA導向多模態特徵融合

        Args:
            data_dict: 包含 'metrics', 'logs', 'traces' 的字典

        Returns:
            fused_embeddings: 融合後的特徵嵌入 [nodes, embed_dim]
            attention_weights: 注意力權重 [nodes, num_modals, num_modals]
        """
        available_modals = []
        modal_embeddings = []

        # 1. 提取各模態特徵
        if 'metrics' in data_dict and data_dict['metrics'] is not None:
            metrics_features = self.metrics_extractor(data_dict['metrics'])
            metrics_proj = self.metrics_proj(metrics_features)
            available_modals.append('metrics')
            modal_embeddings.append(metrics_proj)

        if 'logs' in data_dict and data_dict['logs'] is not None:
            logs_features = self.logs_extractor(data_dict['logs'])
            logs_proj = self.logs_proj(logs_features)
            available_modals.append('logs')
            modal_embeddings.append(logs_proj)

        if 'traces' in data_dict and data_dict['traces'] is not None:
            traces_features = self.traces_extractor(data_dict['traces'])
            traces_proj = self.traces_proj(traces_features)
            available_modals.append('traces')
            modal_embeddings.append(traces_proj)

        if not modal_embeddings:
            # 沒有可用模態，返回零特徵
            return torch.zeros(1, self.embed_dim), torch.ones(1, 1, 1)

        # 2. 對齊模態維度（確保所有模態有相同的節點數）
        max_nodes = max(emb.shape[0] for emb in modal_embeddings)
        aligned_embeddings = []

        for emb in modal_embeddings:
            if emb.shape[0] < max_nodes:
                # 填充到最大節點數
                padding = torch.zeros(max_nodes - emb.shape[0], emb.shape[1], device=emb.device)
                aligned_emb = torch.cat([emb, padding], dim=0)
            else:
                aligned_emb = emb
            aligned_embeddings.append(aligned_emb)

        # 3. 注意力融合
        if len(aligned_embeddings) == 1:
            fused = aligned_embeddings[0]
            attention_weights = torch.ones(1, 1, 1, device=fused.device)
        else:
            # 堆疊模態嵌入 [nodes, num_modals, embed_dim]
            stacked_embeddings = torch.stack(aligned_embeddings, dim=1)

            # 注意力機制
            fused, attention_weights = self.attention(stacked_embeddings, stacked_embeddings, stacked_embeddings)
            fused = self.norm(fused.mean(dim=1))  # 平均融合

        # 4. RCA專用融合
        if len(aligned_embeddings) > 1:
            # 串聯所有模態特徵
            concatenated = torch.cat(aligned_embeddings, dim=1)  # [nodes, num_modals * embed_dim]
            fused = self.rca_fusion(concatenated)  # [nodes, embed_dim]

        # 5. SLI增強（如果有metrics數據）
        if 'metrics' in data_dict and data_dict['metrics'] is not None:
            sli_features = self._extract_sli_features(data_dict['metrics'])
            if sli_features is not None:
                # 擴展SLI特徵以匹配節點數
                sli_features_expanded = sli_features.repeat(max_nodes, 1) if sli_features.shape[0] == 1 else sli_features[:max_nodes]

                # 與主特徵融合
                enhanced_input = torch.cat([fused, sli_features_expanded], dim=1)
                fused = self.sli_enhancer(enhanced_input)

        return fused, attention_weights

    def _extract_sli_features(self, metrics_df):
        """提取SLI特徵進行增強"""
        try:
            if isinstance(metrics_df, pd.DataFrame):
                # 識別SLI列
                sli_col = _identify_sli_column(metrics_df, self.dataset_name)
                if sli_col:
                    sli_data = metrics_df[sli_col].values

                    # 計算SLI特徵
                    sli_stats = [
                        np.mean(sli_data),
                        np.std(sli_data),
                        np.max(sli_data) - np.min(sli_data),  # 範圍
                        np.percentile(sli_data, 95) - np.percentile(sli_data, 5),  # 95-5分位數範圍
                    ]

                    # 異常檢測
                    if np.std(sli_data) > 0:
                        z_scores = np.abs((sli_data - np.mean(sli_data)) / np.std(sli_data))
                        sli_stats.extend([
                            np.max(z_scores),  # 最大異常分數
                            np.mean(z_scores[z_scores > 2]),  # 高異常平均
                        ])
                    else:
                        sli_stats.extend([0.0, 0.0])

                    return torch.tensor(sli_stats[:16], dtype=torch.float32).unsqueeze(0)

        except Exception as e:
            print(f"⚠️ SLI特徵提取失敗: {e}")

        return None


class RCAMetricsFeatureExtractor:
    """RCA導向的指標特徵提取器"""

    def __init__(self, output_dim=48):
        self.output_dim = output_dim
        self.pca = PCA(n_components=min(32, output_dim))

    def __call__(self, metrics_df):
        return self.extract_features(metrics_df)

    def extract_features(self, metrics_df):
        """提取RCA導向的指標特徵"""
        if isinstance(metrics_df, pd.DataFrame):
            data = metrics_df.values
        else:
            data = metrics_df

        if data.shape[0] < 2 or data.shape[1] < 2:
            return torch.zeros(1, self.output_dim)

        # 1. 基本統計特徵
        stats_features = []
        for col in range(min(data.shape[1], 16)):  # 限制列數避免過多計算
            col_data = data[:, col]
            if np.std(col_data) > 0:
                # 標準化統計
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                z_scores = (col_data - mean_val) / std_val

                stats = [
                    mean_val,
                    std_val,
                    np.max(z_scores),  # 最大異常分數
                    np.percentile(z_scores, 95),  # 95分位異常分數
                    np.mean(z_scores[z_scores > 1.5]) if np.any(z_scores > 1.5) else 0,  # 高異常平均
                ]
                stats_features.extend(stats)

        # 2. 時序特徵（如果數據足夠）
        if data.shape[0] >= 10:
            # 趨勢分析
            x = np.arange(data.shape[0])
            for col in range(min(data.shape[1], 8)):
                col_data = data[:, col]
                try:
                    slope, intercept = np.polyfit(x, col_data, 1)
                    # 計算趨勢強度
                    trend_strength = abs(slope) * data.shape[0] / (np.std(col_data) + 1e-8)
                    stats_features.append(float(trend_strength))
                except:
                    stats_features.append(0.0)

        # 3. 相關性特徵（如果有多列）
        if data.shape[1] >= 2:
            # 計算列間相關性矩陣的最大值（排除自相關）
            corr_matrix = np.corrcoef(data.T)
            np.fill_diagonal(corr_matrix, 0)
            max_corr = np.max(np.abs(corr_matrix))
            mean_corr = np.mean(np.abs(corr_matrix))
            stats_features.extend([float(max_corr), float(mean_corr)])

        # 4. 降維處理
        if len(stats_features) >= self.output_dim:
            features = np.array(stats_features[:self.output_dim])
        else:
            # 填充到目標維度
            features = np.zeros(self.output_dim)
            features[:len(stats_features)] = stats_features

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


class RCALogsFeatureExtractor:
    """RCA導向的日誌特徵提取器"""

    def __init__(self, output_dim=48):
        self.output_dim = output_dim
        self.tfidf = TfidfVectorizer(max_features=32, stop_words='english')

    def __call__(self, logs_data):
        return self.extract_features(logs_data)

    def extract_features(self, logs_data):
        """提取RCA導向的日誌特徵"""
        if not logs_data:
            return torch.zeros(1, self.output_dim)

        # 確保logs_data是列表
        if isinstance(logs_data, str):
            logs_list = [logs_data]
        elif not isinstance(logs_data, list):
            logs_list = [str(logs_data)]
        else:
            logs_list = logs_data

        # 清理日誌
        cleaned_logs = []
        for log in logs_list:
            if isinstance(log, str) and len(log.strip()) > 0:
                cleaned_logs.append(log.strip().lower())

        if not cleaned_logs:
            return torch.zeros(1, self.output_dim)

        try:
            # TF-IDF特徵
            tfidf_matrix = self.tfidf.fit_transform(cleaned_logs).toarray()

            # 統計特徵
            log_features = []

            # 詞頻統計
            total_words = sum(len(log.split()) for log in cleaned_logs)
            unique_words = len(self.tfidf.get_feature_names_out())
            log_features.extend([float(total_words), float(unique_words)])

            # 錯誤關鍵字統計
            error_keywords = ['error', 'exception', 'fail', 'timeout', 'crash', 'warning']
            error_count = sum(sum(log.count(keyword) for keyword in error_keywords) for log in cleaned_logs)
            log_features.append(float(error_count))

            # 延遲關鍵字統計
            latency_keywords = ['slow', 'delay', 'timeout', 'latency', 'response']
            latency_count = sum(sum(log.count(keyword) for keyword in latency_keywords) for log in cleaned_logs)
            log_features.append(float(latency_count))

            # TF-IDF統計
            if tfidf_matrix.shape[1] > 0:
                tfidf_means = np.mean(tfidf_matrix, axis=0)
                tfidf_stds = np.std(tfidf_matrix, axis=0) + 1e-8
                tfidf_max = np.max(tfidf_matrix, axis=0)

                # 取最重要的TF-IDF特徵
                importance_scores = tfidf_means * np.log(1 + tfidf_max / (tfidf_means + 1e-8))
                top_indices = np.argsort(importance_scores)[-8:]  # 取前8個

                for idx in top_indices:
                    log_features.extend([
                        float(tfidf_means[idx]),
                        float(tfidf_stds[idx]),
                        float(tfidf_max[idx])
                    ])

            # 填充到目標維度
            if len(log_features) >= self.output_dim:
                features = np.array(log_features[:self.output_dim])
            else:
                features = np.zeros(self.output_dim)
                features[:len(log_features)] = log_features

            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        except Exception as e:
            print(f"⚠️ 日誌特徵提取失敗: {e}")
            return torch.zeros(1, self.output_dim)


class RCATracesFeatureExtractor:
    """RCA導向的調用鏈特徵提取器"""

    def __init__(self, output_dim=48):
        self.output_dim = output_dim

    def __call__(self, traces_data):
        return self.extract_features(traces_data)

    def extract_features(self, traces_data):
        """提取RCA導向的調用鏈特徵"""
        if traces_data is None:
            return torch.zeros(1, self.output_dim)

        try:
            # 假設traces_data是NetworkX圖或類似結構
            if hasattr(traces_data, 'nodes') and hasattr(traces_data, 'edges'):
                # NetworkX圖
                return self._extract_from_networkx(traces_data)
            elif isinstance(traces_data, dict):
                # 字典格式
                return self._extract_from_dict(traces_data)
            else:
                # 其他格式，創建簡單特徵
                return self._create_default_features()

        except Exception as e:
            print(f"⚠️ 調用鏈特徵提取失敗: {e}")
            return torch.zeros(1, self.output_dim)

    def _extract_from_networkx(self, graph):
        """從NetworkX圖提取特徵"""
        features = []

        # 基本圖統計
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0

        features.extend([float(num_nodes), float(num_edges), float(density)])

        # 度數統計
        degrees = [d for n, d in graph.degree()]
        if degrees:
            features.extend([
                float(np.mean(degrees)),
                float(np.std(degrees)),
                float(np.max(degrees)),
                float(np.median(degrees))
            ])

        # 聚類係數（如果可用）
        try:
            clustering_coeffs = list(nx.clustering(graph).values())
            if clustering_coeffs:
                features.extend([
                    float(np.mean(clustering_coeffs)),
                    float(np.std(clustering_coeffs))
                ])
        except:
            features.extend([0.0, 0.0])

        # 填充到目標維度
        if len(features) >= self.output_dim:
            features = features[:self.output_dim]
        else:
            features.extend([0.0] * (self.output_dim - len(features)))

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _extract_from_dict(self, traces_dict):
        """從字典格式提取特徵"""
        features = []

        # 提取基本統計
        if 'spans' in traces_dict:
            spans = traces_dict['spans']
            num_spans = len(spans)

            # 持續時間統計
            durations = [span.get('duration', 0) for span in spans]
            if durations:
                features.extend([
                    float(num_spans),
                    float(np.mean(durations)),
                    float(np.std(durations)),
                    float(np.max(durations))
                ])

        # 填充到目標維度
        if len(features) >= self.output_dim:
            features = features[:self.output_dim]
        else:
            features.extend([0.0] * (self.output_dim - len(features)))

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _create_default_features(self):
        """創建默認特徵"""
        # 創建一些合理的默認值
        features = [
            1.0,    # 至少一個節點
            0.0,    # 沒有邊
            0.0,    # 密度0
            0.0,    # 平均度數0
            0.0,    # 度數標準差0
            0.0,    # 最大度數0
            0.0     # 中位度數0
        ]

        # 填充到目標維度
        features.extend([0.0] * (self.output_dim - len(features)))
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


class MultiModalFeatureExtractor(nn.Module):
    """
    多模態特徵提取器
    整合 metrics, logs, traces 三種模態的數據，避免單一模態偏誤
    目標：建構平衡的圖結構，密度控制在 0.3-0.5
    """
    
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.tfidf = TfidfVectorizer(max_features=32, stop_words='english')
        self.pca = PCA(n_components=32)
        
        # 模態特定的投影層
        self.metrics_proj = nn.Linear(32, embed_dim)
        self.logs_proj = nn.Linear(32, embed_dim)
        self.traces_proj = nn.Linear(32, embed_dim)
        
        # 融合層 - 修正維度問題
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  # 修正：輸入維度應該是 embed_dim 而不是 embed_dim * 3
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def extract_metrics(self, metrics_df):
        """
        提取時序指標特徵：統計 + FFT
        """
        if isinstance(metrics_df, pd.DataFrame):
            data = metrics_df.values
        else:
            data = metrics_df
            
        # 統計特徵：均值、標準差、偏度
        stats = np.array([
            np.mean(data, axis=0),
            np.std(data, axis=0),
            self._safe_skew(data)
        ]).T  # [features, 3]
        
        # FFT 特徵：前16個頻率分量
        fft_data = np.abs(np.fft.fft(data, axis=0))[:16]  # [16, features]
        
        # 組合特徵
        combined_features = np.hstack([stats, fft_data.T])  # [features, 19]
        
        # PCA 降維到 32
        if combined_features.shape[1] > 32:
            features_32d = self.pca.fit_transform(combined_features)
        else:
            # 如果特徵數不足32，用零填充
            features_32d = np.zeros((combined_features.shape[0], 32))
            features_32d[:, :combined_features.shape[1]] = combined_features
            
        return features_32d  # [nodes, 32]
    
    def extract_logs(self, logs_list):
        """
        提取日誌特徵：TF-IDF 嵌入
        """
        if not logs_list or len(logs_list) == 0:
            # 如果沒有日誌，返回零特徵
            return np.zeros((1, 32))
            
        # 確保 logs_list 是字符串列表
        if isinstance(logs_list, str):
            logs_list = [logs_list]
        elif not isinstance(logs_list, list):
            logs_list = [str(logs_list)]
            
        # 清理和預處理日誌
        cleaned_logs = []
        for log in logs_list:
            if isinstance(log, str) and len(log.strip()) > 0:
                cleaned_logs.append(log.strip())
        
        if len(cleaned_logs) == 0:
            return np.zeros((1, 32))
        
        try:
            # TF-IDF 特徵提取
            tfidf_matrix = self.tfidf.fit_transform(cleaned_logs).toarray()
            
            # PCA 降維
            if tfidf_matrix.shape[1] > 32:
                features_32d = self.pca.fit_transform(tfidf_matrix)
            else:
                features_32d = np.zeros((tfidf_matrix.shape[0], 32))
                features_32d[:, :tfidf_matrix.shape[1]] = tfidf_matrix
                
            return features_32d  # [nodes, 32]
        except Exception as e:
            print(f"⚠️ 日誌特徵提取失敗: {e}")
            return np.zeros((len(cleaned_logs), 32))
    
    def extract_traces(self, trace_graph):
        """
        提取調用鏈特徵：路徑長度 + 度數
        """
        if trace_graph is None or len(trace_graph.nodes()) == 0:
            return np.zeros((1, 32))
            
        try:
            # 計算所有節點對的最短路徑
            if hasattr(nx, 'all_pairs_shortest_path_length'):
                paths = dict(nx.all_pairs_shortest_path_length(trace_graph))
            else:
                # 如果 networkx 版本不支持，使用簡化方法
                paths = {}
                for node in trace_graph.nodes():
                    paths[node] = {node: 0}
                    for neighbor in trace_graph.neighbors(node):
                        paths[node][neighbor] = 1
            
            features = []
            for node in trace_graph.nodes():
                # 度數
                degree = trace_graph.degree(node)
                
                # 平均路徑長度
                path_lengths = [paths[node].get(other, 0) for other in trace_graph.nodes() if other != node]
                avg_path_length = np.mean(path_lengths) if path_lengths else 0
                
                # 聚類係數
                clustering = nx.clustering(trace_graph, node) if hasattr(nx, 'clustering') else 0
                
                features.append([degree, avg_path_length, clustering])
            
            features_array = np.array(features)  # [nodes, 3]
            
            # 擴展到 32 維
            if features_array.shape[1] < 32:
                # 使用線性變換擴展維度
                expansion = np.random.randn(features_array.shape[1], 32 - features_array.shape[1]) * 0.1
                features_32d = np.hstack([features_array, features_array @ expansion])
            else:
                features_32d = features_array[:, :32]
                
            return features_32d  # [nodes, 32]
        except Exception as e:
            print(f"⚠️ 調用鏈特徵提取失敗: {e}")
            return np.zeros((1, 32))
    
    def _safe_skew(self, data):
        """安全的偏度計算"""
        try:
            from scipy.stats import skew
            return skew(data, axis=0)
        except:
            # 如果 scipy 不可用，使用簡化計算
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            return np.mean(((data - mean) / (std + 1e-8)) ** 3, axis=0)
    
    def forward(self, data_dict):
        """
        多模態特徵融合前向傳播 - 動態權重調整版本
        
        Args:
            data_dict: 包含 'metrics', 'logs', 'traces' 的字典
            
        Returns:
            fused_embeddings: 融合後的特徵嵌入 [nodes, embed_dim]
            attention_weights: 注意力權重 [nodes, num_modals, num_modals]
        """
        # 1. 檢查可用模態
        has_metrics = data_dict.get('metrics') is not None
        has_logs = data_dict.get('logs') is not None and len(data_dict.get('logs', [])) > 0
        has_traces = data_dict.get('traces') is not None
        
        # 2. 動態權重調整
        if has_metrics and not has_logs and not has_traces:
            # 只有metrics：權重 = 1
            metrics_emb = self.extract_metrics(data_dict.get('metrics'))
            metrics_tensor = torch.tensor(metrics_emb, dtype=torch.float32)
            fused = self.metrics_proj(metrics_tensor)
            attention_weights = torch.ones(1, 1, 1)  # 單一模態
            print("✓ 使用單一模態: metrics (權重=1.0)")
            
        elif has_logs and not has_metrics and not has_traces:
            # 只有logs：權重 = 1
            logs_emb = self.extract_logs(data_dict.get('logs', []))
            logs_tensor = torch.tensor(logs_emb, dtype=torch.float32)
            fused = self.logs_proj(logs_tensor)
            attention_weights = torch.ones(1, 1, 1)
            print("✓ 使用單一模態: logs (權重=1.0)")
            
        elif has_traces and not has_metrics and not has_logs:
            # 只有traces：權重 = 1
            traces_emb = self.extract_traces(data_dict.get('traces'))
            traces_tensor = torch.tensor(traces_emb, dtype=torch.float32)
            fused = self.traces_proj(traces_tensor)
            attention_weights = torch.ones(1, 1, 1)
            print("✓ 使用單一模態: traces (權重=1.0)")
            
        else:
            # 多模態：動態權重分配
            available_modals = []
            modal_names = []
            
            if has_metrics:
                metrics_emb = self.extract_metrics(data_dict.get('metrics'))
                metrics_tensor = torch.tensor(metrics_emb, dtype=torch.float32)
                available_modals.append(self.metrics_proj(metrics_tensor))
                modal_names.append('metrics')
                
            if has_logs:
                logs_emb = self.extract_logs(data_dict.get('logs', []))
                logs_tensor = torch.tensor(logs_emb, dtype=torch.float32)
                available_modals.append(self.logs_proj(logs_tensor))
                modal_names.append('logs')
                
            if has_traces:
                traces_emb = self.extract_traces(data_dict.get('traces'))
                traces_tensor = torch.tensor(traces_emb, dtype=torch.float32)
                available_modals.append(self.traces_proj(traces_tensor))
                modal_names.append('traces')
            
            # 檢查是否有可用模態
            if len(available_modals) == 0:
                # 沒有可用模態，返回零特徵
                print("⚠️ 沒有可用模態，返回零特徵")
                fused = torch.zeros(1, self.embed_dim)
                attention_weights = torch.ones(1, 1, 1)
            else:
                # 確保所有模態的節點數一致（取最大值）
                max_nodes = max(modal.shape[0] for modal in available_modals)
                
                # 填充到相同節點數
                padded_modals = []
                for modal in available_modals:
                    if modal.shape[0] < max_nodes:
                        padding = torch.zeros(max_nodes - modal.shape[0], self.embed_dim)
                        padded_modal = torch.cat([modal, padding], dim=0)
                    else:
                        padded_modal = modal
                    padded_modals.append(padded_modal)
                
                # 注意力融合，權重自動分配
                modals = torch.stack(padded_modals, dim=1)  # [nodes, num_modals, embed_dim]
                fused, attention_weights = self.attention(modals, modals, modals)
                fused = self.norm(fused.mean(dim=1))  # 根據實際模態數量平均
                print(f"✓ 使用 {len(available_modals)} 個模態: {modal_names} (權重=1/{len(available_modals)})")
        
        # 額外的融合層
        fused = self.fusion_layer(fused)  # [nodes, embed_dim]
        
        return fused, attention_weights


def enhanced_ica_with_temporal_contrast(metrics_data, inject_time=None, target_dim=64):
    """
    🚀 改進的 ICA 特徵處理 + 時序對比增強
    
    核心改進：
    1. 故障前後直接對比
    2. 自適應維度分配 
    3. 保留關鍵統計特徵
    4. 避免過度壓縮
    """
    print("🔥 增強版 ICA + 時序對比特徵處理")
    
    try:
        # 🎯 改進方案3.2.1: 自適應維度分配
        complexity = _calculate_data_complexity(metrics_data)
        if complexity > 0.8:
            ica_dim = int(target_dim * 0.6)  # 高複雜度：ICA佔60%
        else:
            ica_dim = int(target_dim * 0.4)  # 低複雜度：ICA佔40%
        
        contrast_dim = target_dim - ica_dim
        
        # 使用原有 ICA 處理作為基礎
        base_features, service_names = ica_metric_processing(
            metrics_data, target_dim=ica_dim, inject_time=inject_time
        )
        
        # 添加時序對比特徵
        if inject_time is not None:
            contrast_features = _extract_enhanced_temporal_contrast(
                metrics_data, inject_time, contrast_dim, service_names
            )
            # 組合基礎特徵和對比特徵
            if contrast_features is not None and contrast_features.shape[0] == base_features.shape[0]:
                enhanced_features = np.hstack([base_features, contrast_features])
            else:
                # 如果對比特徵失敗，填充零
                padding = np.zeros((base_features.shape[0], contrast_dim))
                enhanced_features = np.hstack([base_features, padding])
        else:
            # 沒有注入時間，填充統計特徵
            stats_features = _extract_statistical_features(metrics_data, service_names, contrast_dim)
            enhanced_features = np.hstack([base_features, stats_features])
        
        print(f"✅ 增強特徵: {enhanced_features.shape[0]} 服務 × {enhanced_features.shape[1]} 維 (ICA:{ica_dim}, 對比:{contrast_dim})")
        return enhanced_features, service_names
        
    except Exception as e:
        print(f"❌ 增強處理失敗: {e}")
        return ica_metric_processing(metrics_data, target_dim=target_dim)


def _calculate_data_complexity(metrics_data):
    """計算數據複雜度 - 用於自適應維度分配"""
    try:
        if isinstance(metrics_data, pd.DataFrame):
            data = metrics_data.select_dtypes(include=[np.number])
        else:
            data = pd.DataFrame(metrics_data)
        
        if data.empty:
            return 0.5  # 默認中等複雜度
        
        # 計算熵作為複雜度指標
        entropy_sum = 0
        valid_cols = 0
        
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) > 1:
                # 計算直方圖熵
                hist, _ = np.histogram(col_data, bins=min(10, len(col_data)//2), density=True)
                hist = hist + 1e-8  # 避免log(0)
                entropy = -np.sum(hist * np.log(hist))
                entropy_sum += entropy
                valid_cols += 1
        
        if valid_cols == 0:
            return 0.5
        
        avg_entropy = entropy_sum / valid_cols
        # 歸一化到[0,1]範圍
        complexity = min(1.0, max(0.0, avg_entropy / 2.0))
        
        return complexity
        
    except Exception as e:
        print(f"⚠️ 複雜度計算失敗: {e}")
        return 0.5  # 默認中等複雜度


def _extract_enhanced_temporal_contrast(metrics_data, inject_time, target_dim, service_names):
    """提取增強的時序對比特徵"""
    try:
        if isinstance(metrics_data, pd.DataFrame):
            data = metrics_data.select_dtypes(include=[np.number])
        else:
            data = pd.DataFrame(metrics_data)
        
        # 假設時間戳在索引或第一列
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            data = data.drop('timestamp', axis=1)
        else:
            # 使用行索引作為時間代理
            timestamps = pd.to_datetime(data.index, unit='s', errors='coerce')
        
        inject_timestamp = pd.to_datetime(inject_time, unit='s')
        
        # 分離故障前後數據
        pre_fault_mask = timestamps < inject_timestamp
        post_fault_mask = timestamps >= inject_timestamp
        
        pre_fault_data = data[pre_fault_mask]
        post_fault_data = data[post_fault_mask]
        
        if pre_fault_data.empty or post_fault_data.empty:
            return None
        
        # 計算對比特徵
        contrast_features = []
        
        for service in service_names:
            service_cols = [col for col in data.columns if service.lower() in col.lower()]
            if not service_cols:
                contrast_features.append(np.zeros(target_dim))
                continue
            
            pre_service = pre_fault_data[service_cols]
            post_service = post_fault_data[service_cols]
            
            # 計算變化特徵
            service_contrast = []
            
            for col in service_cols[:min(len(service_cols), target_dim//4)]:
                pre_vals = pre_service[col].dropna()
                post_vals = post_service[col].dropna()
                
                if len(pre_vals) > 0 and len(post_vals) > 0:
                    # 🎯 關鍵變化特徵
                    mean_change = (post_vals.mean() - pre_vals.mean()) / (pre_vals.std() + 1e-8)  # 標準化變化
                    std_change = post_vals.std() / (pre_vals.std() + 1e-8)  # 方差比率
                    trend_change = np.polyfit(range(len(post_vals)), post_vals, 1)[0] if len(post_vals) > 1 else 0
                    # 🔥 增強異常檢測 - 使用更敏感的異常分數計算 (來自改進方案)
                    baseline_mean = pre_vals.mean()
                    baseline_std = pre_vals.std() + 1e-8
                    
                    # Z-score 異常檢測
                    z_scores = np.abs((post_vals - baseline_mean) / baseline_std)
                    anomaly_score = np.percentile(z_scores, 95)  # 使用95分位數突出極值
                    
                    # 變化幅度檢測
                    magnitude_change = np.abs(post_vals.mean() - baseline_mean) / (baseline_mean + 1e-8)
                    anomaly_score = max(anomaly_score, magnitude_change * 3)  # 放大變化幅度
                    
                    # 🎯 改進方案3.2.2: 增強時序對比 - 多尺度分析
                    # 短期特徵 (5個時間點窗口)
                    short_term_features = _extract_short_term_features(post_vals, window=5)
                    # 長期特徵 (20個時間點窗口)
                    long_term_features = _extract_long_term_features(post_vals, window=20)
                    
                    # 組合多尺度特徵
                    multi_scale_features = np.concatenate([
                        short_term_features, long_term_features
                    ])
                    
                    # 加入多尺度特徵與核心統計變化
                    service_contrast.extend([
                        mean_change, std_change, trend_change, anomaly_score
                    ])
                    service_contrast.extend(multi_scale_features.tolist())
                else:
                    service_contrast.extend([0, 0, 0, 0])
            
            # 調整到目標維度
            if len(service_contrast) >= target_dim:
                service_features = np.array(service_contrast[:target_dim])
            else:
                padding = np.zeros(target_dim - len(service_contrast))
                service_features = np.concatenate([service_contrast, padding])
            
            contrast_features.append(service_features)
        
        contrast_matrix = np.array(contrast_features)
        contrast_matrix = np.nan_to_num(contrast_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # 特徵標準化與去相關（PCA whiten）以增強判別力
        try:
            from sklearn.decomposition import PCA
            # 標準化到零均值
            contrast_matrix = contrast_matrix - contrast_matrix.mean(axis=0, keepdims=True)
            contrast_matrix_std = contrast_matrix.std(axis=0, keepdims=True) + 1e-8
            contrast_matrix = contrast_matrix / contrast_matrix_std
            # PCA 白化，保持 target_dim 維度
            n_components = min(target_dim, contrast_matrix.shape[1])
            pca = PCA(n_components=n_components, whiten=True, random_state=0)
            contrast_matrix = pca.fit_transform(contrast_matrix)
            # 如有需要，填充至 target_dim
            if contrast_matrix.shape[1] < target_dim:
                pad = np.zeros((contrast_matrix.shape[0], target_dim - contrast_matrix.shape[1]))
                contrast_matrix = np.hstack([contrast_matrix, pad])
        except Exception as _:
            pass
        
        print(f"✅ 時序對比特徵: {contrast_matrix.shape}")
        return contrast_matrix
        
    except Exception as e:
        print(f"❌ 時序對比提取失敗: {e}")
        return None


def _extract_short_term_features(series, window=5):
    """提取短期特徵 - 改進方案3.2.2"""
    try:
        if len(series) < window:
            return np.zeros(3)  # 返回零特徵
        
        # 滑動窗口統計
        rolling_mean = np.mean(series[-window:])
        rolling_std = np.std(series[-window:])
        rolling_trend = np.polyfit(range(window), series[-window:], 1)[0] if len(series[-window:]) > 1 else 0
        
        return np.array([rolling_mean, rolling_std, rolling_trend])
    except:
        return np.zeros(3)


def _extract_long_term_features(series, window=20):
    """提取長期特徵 - 改進方案3.2.2"""
    try:
        if len(series) < window:
            return np.zeros(3)  # 返回零特徵
        
        # 長期趨勢分析
        long_mean = np.mean(series[-window:])
        long_std = np.std(series[-window:])
        long_trend = np.polyfit(range(window), series[-window:], 1)[0] if len(series[-window:]) > 1 else 0
        
        return np.array([long_mean, long_std, long_trend])
    except:
        return np.zeros(3)


def _extract_statistical_features(metrics_data, service_names, target_dim):
    """提取統計特徵作為補充"""
    try:
        if isinstance(metrics_data, pd.DataFrame):
            data = metrics_data.select_dtypes(include=[np.number])
        else:
            data = pd.DataFrame(metrics_data)
        
        stats_features = []
        
        for service in service_names:
            service_cols = [col for col in data.columns if service.lower() in col.lower()]
            if not service_cols:
                stats_features.append(np.zeros(target_dim))
                continue
            
            service_data = data[service_cols]
            service_stats = []
            
            for col in service_cols[:min(len(service_cols), target_dim//6)]:
                vals = service_data[col].dropna()
                if len(vals) > 0:
                    stats = [
                        vals.mean(),
                        vals.std(),
                        vals.median(),
                        vals.min(),
                        vals.max(),
                        vals.skew() if len(vals) > 2 else 0  # 偏度
                    ]
                    service_stats.extend(stats)
                else:
                    service_stats.extend([0, 0, 0, 0, 0, 0])
            
            # 調整到目標維度
            if len(service_stats) >= target_dim:
                service_features = np.array(service_stats[:target_dim])
            else:
                padding = np.zeros(target_dim - len(service_stats))
                service_features = np.concatenate([service_stats, padding])
            
            stats_features.append(service_features)
        
        stats_matrix = np.array(stats_features)
        stats_matrix = np.nan_to_num(stats_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return stats_matrix
        
    except Exception as e:
        print(f"❌ 統計特徵提取失敗: {e}")
        return np.zeros((len(service_names), target_dim))


__all__ = [
    'ica_metric_processing',
    'kpca_metric_processing',
    'simplified_metric_processing',
    'psm_metric_processing',
    'enhanced_trace_processing',
    'enhanced_ica_with_temporal_contrast',
    'rca_aware_metric_processing',
    'rca_multimodal_feature_processing',
    'rca_feature_selection_and_reduction',
    'compute_anomaly_propagation_features',
    'compute_mutual_information_matrix',
    'select_features_by_correlation',
    'RCAFeatureConfig',
    'RCAConfigManager',
    'RCACache',
    'RCAPerformanceOptimizer',
    'RCAMultiModalFeatureExtractor',
    'RCAMetricsFeatureExtractor',
    'RCALogsFeatureExtractor',
    'RCATracesFeatureExtractor'
]
