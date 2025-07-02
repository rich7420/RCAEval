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
import warnings

warnings.filterwarnings("ignore")


def ica_metric_processing(metrics_data, n_components=None, target_dim=64):
    """
    基於ICA (Independent Component Analysis) 的特徵提取 - 修正微服務節點提取
    專門用於分離混合信號中的獨立成分，適合微服務指標分析
    
    Args:
        metrics_data: 指標數據
        n_components: ICA成分數量 (None = 自動確定)
        target_dim: 目標維度
    
    Returns:
        ica_features: ICA特徵 (num_services, target_dim)
        feature_names: 微服務節點名稱列表
    """
    print("🔧 Using ICA feature processing - 正確提取微服務節點...")
    
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
    """從列名中提取服務名稱 - 更智能的版本"""
    services = set()
    
    # 常見的微服務模式
    service_patterns = [
        'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
        'emailservice', 'paymentservice', 'productcatalogservice', 
        'recommendationservice', 'shippingservice', 'frontend'
    ]
    
    for col in columns:
        col_lower = str(col).lower()
        
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
                if len(prefix) > 2:
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

__all__ = [
    'ica_metric_processing',
    'kpca_metric_processing', 
    'simplified_metric_processing',
    'psm_metric_processing',
    'enhanced_trace_processing'
]
