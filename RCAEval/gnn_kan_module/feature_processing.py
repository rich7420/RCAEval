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

warnings.filterwarnings("ignore")


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
    'enhanced_ica_with_temporal_contrast'
]
