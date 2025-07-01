"""
Utility Functions for GNN-KAN Module
工具函數模組 - 提供GNN-KAN模組的輔助功能
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def safe_pca_transform(features, target_dim, random_state=42):
    """
    安全的PCA降維函數 - 自動處理維度限制
    
    Args:
        features: 輸入特徵矩陣 (n_samples, n_features)
        target_dim: 目標維度
        random_state: 隨機種子
    
    Returns:
        transformed_features: 降維後的特徵矩陣
    """
    try:
        from sklearn.decomposition import PCA
        
        # 檢查輸入
        if features is None or features.size == 0:
            return np.zeros((1, target_dim))
        
        # 確保是2D數組
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        n_samples, n_features = features.shape
        
        # 如果特徵數已經等於或小於目標維度
        if n_features <= target_dim:
            if n_features < target_dim:
                # 填充零到目標維度
                padding = np.zeros((n_samples, target_dim - n_features))
                return np.hstack([features, padding])
            else:
                return features
        
        # 計算最大可能的PCA成分數
        max_components = min(n_samples, n_features)
        actual_components = min(target_dim, max_components)
        
        if actual_components <= 0:
            # 無法進行PCA，創建默認特徵
            return np.zeros((n_samples, target_dim))
        
        # 執行安全的PCA
        pca = PCA(n_components=actual_components, random_state=random_state)
        transformed = pca.fit_transform(features)
        
        # 如果降維後維度仍不足target_dim，用零填充
        if transformed.shape[1] < target_dim:
            padding = np.zeros((transformed.shape[0], target_dim - transformed.shape[1]))
            transformed = np.hstack([transformed, padding])
        
        return transformed
        
    except Exception as e:
        print(f"⚠️ PCA變換失敗: {e}, 返回零填充特徵")
        return np.zeros((features.shape[0] if features.ndim > 1 else 1, target_dim))


def safe_feature_alignment(features_list, target_dim=None, method='pad'):
    """
    安全的特徵對齊函數
    
    Args:
        features_list: 特徵列表
        target_dim: 目標維度
        method: 對齊方法 ('pad', 'truncate', 'pca')
    
    Returns:
        aligned_features: 對齊後的特徵列表
    """
    if not features_list:
        return []
    
    # 過濾空特徵
    valid_features = [f for f in features_list if f is not None and f.size > 0]
    if not valid_features:
        return []
    
    # 對齊節點數（行數）
    max_nodes = max(f.shape[0] for f in valid_features)
    aligned_features = []
    
    for features in valid_features:
        if features.shape[0] < max_nodes:
            # 重複最後一行以對齊節點數
            padding = np.repeat(features[-1:], max_nodes - features.shape[0], axis=0)
            features = np.vstack([features, padding])
        aligned_features.append(features)
    
    # 對齊特徵維度（列數）
    if target_dim is not None:
        final_features = []
        for features in aligned_features:
            if method == 'pca':
                features = safe_pca_transform(features, target_dim)
            elif method == 'truncate':
                if features.shape[1] > target_dim:
                    features = features[:, :target_dim]
                elif features.shape[1] < target_dim:
                    padding = np.zeros((features.shape[0], target_dim - features.shape[1]))
                    features = np.hstack([features, padding])
            else:  # 'pad'
                if features.shape[1] < target_dim:
                    padding = np.zeros((features.shape[0], target_dim - features.shape[1]))
                    features = np.hstack([features, padding])
                elif features.shape[1] > target_dim:
                    # 使用安全PCA
                    features = safe_pca_transform(features, target_dim)
            
            final_features.append(features)
        
        return final_features
    
    return aligned_features


def safe_tensor_operation(func, *args, **kwargs):
    """
    安全的張量操作包裝器
    
    Args:
        func: 要執行的函數
        *args: 函數參數
        **kwargs: 函數關鍵字參數
        
    Returns:
        結果或None（如果出錯）
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Tensor operation error: {e}")
        return None


def validate_inputs(node_features, edge_index=None):
    """
    驗證輸入數據的有效性
    
    Args:
        node_features: 節點特徵張量
        edge_index: 邊索引張量（可選）
        
    Returns:
        bool: 是否有效
        str: 錯誤訊息（如果無效）
    """
    # 檢查節點特徵
    if node_features is None:
        return False, "Node features cannot be None"
    
    if not isinstance(node_features, torch.Tensor):
        return False, "Node features must be a torch.Tensor"
    
    if node_features.dim() != 2:
        return False, f"Node features must be 2D, got {node_features.dim()}D"
    
    if node_features.size(0) == 0:
        return False, "Node features cannot be empty"
    
    # 檢查是否包含NaN或Inf
    if torch.isnan(node_features).any():
        return False, "Node features contain NaN values"
    
    if torch.isinf(node_features).any():
        return False, "Node features contain Inf values"
    
    # 檢查邊索引
    if edge_index is not None:
        if not isinstance(edge_index, torch.Tensor):
            return False, "Edge index must be a torch.Tensor"
        
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return False, "Edge index must be 2D with shape [2, num_edges]"
        
        # 檢查邊索引範圍
        num_nodes = node_features.size(0)
        if edge_index.size(1) > 0:
            if edge_index.max() >= num_nodes or edge_index.min() < 0:
                return False, f"Edge index out of range [0, {num_nodes-1}]"
    
    return True, "Valid inputs"


def normalize_features(features, method='standard'):
    """
    特徵標準化
    
    Args:
        features: 輸入特徵張量
        method: 標準化方法 ('standard', 'minmax', 'robust')
        
    Returns:
        normalized_features: 標準化後的特徵
        stats: 標準化統計信息
    """
    if method == 'standard':
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)  # 避免除零
        normalized = (features - mean) / std
        stats = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = features.min(dim=0, keepdim=True)[0]
        max_val = features.max(dim=0, keepdim=True)[0]
        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        normalized = (features - min_val) / range_val
        stats = {'min': min_val, 'max': max_val}
        
    elif method == 'robust':
        median = features.median(dim=0, keepdim=True)[0]
        mad = torch.median(torch.abs(features - median), dim=0, keepdim=True)[0]
        mad = torch.where(mad == 0, torch.ones_like(mad), mad)
        normalized = (features - median) / mad
        stats = {'median': median, 'mad': mad}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats


def create_adjacency_matrix(edge_index, num_nodes, edge_weights=None):
    """
    從邊索引創建鄰接矩陣
    
    Args:
        edge_index: 邊索引張量 [2, num_edges]
        num_nodes: 節點數量
        edge_weights: 邊權重（可選）
        
    Returns:
        adj_matrix: 鄰接矩陣
    """
    device = edge_index.device
    adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
    
    if edge_index.size(1) > 0:
        if edge_weights is None:
            adj_matrix[edge_index[0], edge_index[1]] = 1.0
        else:
            adj_matrix[edge_index[0], edge_index[1]] = edge_weights
    
    return adj_matrix


def adjacency_to_edge_index(adj_matrix, threshold=0.5):
    """
    將鄰接矩陣轉換為邊索引
    
    Args:
        adj_matrix: 鄰接矩陣
        threshold: 閾值
        
    Returns:
        edge_index: 邊索引
        edge_weights: 邊權重
    """
    # 找到大於閾值的邊
    mask = adj_matrix > threshold
    edge_index = mask.nonzero().t()
    edge_weights = adj_matrix[mask]
    
    return edge_index, edge_weights


def compute_graph_metrics(pred_adj, true_adj, threshold=0.5):
    """
    計算圖重構的評估指標
    
    Args:
        pred_adj: 預測的鄰接矩陣
        true_adj: 真實的鄰接矩陣
        threshold: 二值化閾值
        
    Returns:
        metrics: 評估指標字典
    """
    # 轉換為numpy
    if isinstance(pred_adj, torch.Tensor):
        pred_adj = pred_adj.detach().cpu().numpy()
    if isinstance(true_adj, torch.Tensor):
        true_adj = true_adj.detach().cpu().numpy()
    
    # 二值化預測
    pred_binary = (pred_adj > threshold).astype(int)
    true_binary = true_adj.astype(int)
    
    # 展平為1D數組（排除對角線）
    mask = ~np.eye(pred_adj.shape[0], dtype=bool)
    pred_flat = pred_binary[mask]
    true_flat = true_binary[mask]
    pred_prob_flat = pred_adj[mask]
    
    # 計算指標
    metrics = {}
    
    try:
        metrics['accuracy'] = accuracy_score(true_flat, pred_flat)
        metrics['precision'] = precision_score(true_flat, pred_flat, zero_division=0)
        metrics['recall'] = recall_score(true_flat, pred_flat, zero_division=0)
        metrics['f1'] = f1_score(true_flat, pred_flat, zero_division=0)
        
        # AUC（如果有正負樣本）
        if len(np.unique(true_flat)) > 1:
            metrics['auc'] = roc_auc_score(true_flat, pred_prob_flat)
        else:
            metrics['auc'] = 0.0
            
        # 圖結構指標
        metrics['edge_density_pred'] = np.mean(pred_binary)
        metrics['edge_density_true'] = np.mean(true_binary)
        metrics['density_diff'] = abs(metrics['edge_density_pred'] - metrics['edge_density_true'])
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        metrics = {key: 0.0 for key in ['accuracy', 'precision', 'recall', 'f1', 'auc', 
                                       'edge_density_pred', 'edge_density_true', 'density_diff']}
    
    return metrics


def memory_usage_gpu():
    """獲取GPU記憶體使用情況"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {'allocated': 0, 'cached': 0, 'max_allocated': 0}


def save_model_checkpoint(model, optimizer, epoch, loss, metrics, save_path, config=None):
    """
    💾 保存模型檢查點 (統一版本)
    
    Args:
        model: 要保存的模型
        optimizer: 優化器
        epoch: 當前epoch
        loss: 當前損失
        metrics: 評估指標
        save_path: 保存路徑
        config: 配置對象
    """
    print(f"💾 Saving checkpoint to {save_path}")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics,
        'timestamp': time.time()
    }
    
    if config:
        # To avoid circular dependencies or saving large objects,
        # we can serialize the config to a dict.
        if hasattr(config, 'to_dict'):
             checkpoint['config'] = config.to_dict()
        else:
             checkpoint['config'] = vars(config)
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved successfully")
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")


def load_model_checkpoint(model, optimizer, filepath):
    """
    載入模型檢查點
    
    Args:
        model: 模型
        optimizer: 優化器
        filepath: 檢查點路徑
        
    Returns:
        epoch: 載入的epoch
        loss: 載入的損失
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint['epoch'], checkpoint['loss']


def export_results(results, filepath, format='json'):
    """
    導出結果到檔案
    
    Args:
        results: 結果字典
        filepath: 檔案路徑
        format: 檔案格式 ('json', 'pickle')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'json':
        # 轉換torch tensors為lists
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.detach().cpu().tolist()
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
            
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    print(f"Results exported to {filepath}")


def create_synthetic_graph_data(num_nodes, num_features, density=0.1, seed=42):
    """
    創建合成圖數據用於測試
    
    Args:
        num_nodes: 節點數量
        num_features: 特徵維度
        density: 邊密度
        seed: 隨機種子
        
    Returns:
        node_features: 節點特徵
        edge_index: 邊索引
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 創建節點特徵
    node_features = torch.randn(num_nodes, num_features)
    
    # 創建邊
    num_edges = int(num_nodes * (num_nodes - 1) * density / 2)
    
    # 隨機選擇邊
    all_edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            all_edges.append([i, j])
    
    if len(all_edges) > 0:
        selected_indices = np.random.choice(
            len(all_edges), 
            size=min(num_edges, len(all_edges)), 
            replace=False
        )
        selected_edges = [all_edges[i] for i in selected_indices]
        
        # 創建雙向邊
        edge_list = []
        for edge in selected_edges:
            edge_list.extend([[edge[0], edge[1]], [edge[1], edge[0]]])
        
        if edge_list:
            edge_index = torch.tensor(edge_list).t()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    
    return node_features, edge_index


class ProgressTracker:
    """進度追蹤器"""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.description = description
        
    def update(self, step=1):
        self.current_step += step
        self._print_progress()
    
    def _print_progress(self):
        if self.total_steps > 0:
            progress = self.current_step / self.total_steps
            elapsed = time.time() - self.start_time
            eta = elapsed / progress - elapsed if progress > 0 else 0
            
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f'\r{self.description}: |{bar}| {progress:.1%} '
                  f'({self.current_step}/{self.total_steps}) '
                  f'ETA: {eta:.0f}s', end='', flush=True)
            
            if self.current_step >= self.total_steps:
                print()  # 換行


def benchmark_function(func, *args, num_runs=5, **kwargs):
    """
    對函數進行基準測試
    
    Args:
        func: 要測試的函數
        *args: 函數參數
        num_runs: 運行次數
        **kwargs: 函數關鍵字參數
        
    Returns:
        dict: 基準測試結果
    """
    times = []
    results = []
    
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        times.append(end_time - start_time)
        results.append(result)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'results': results
    }


def compute_service_criticality_weights(node_names):
    """
    基於服務名稱計算重要性權重
    
    Args:
        node_names: 節點名稱列表
        
    Returns:
        weights: 重要性權重張量
    """
    critical_services = {
        'frontend': 3.0, 'front-end': 3.0,
        'checkout': 2.8, 'payment': 2.8,
        'cart': 2.5, 'catalog': 2.2,
        'currency': 2.0, 'redis': 2.3,
        'database': 2.5, 'db': 2.5,
        'email': 1.8, 'ad': 1.6,
        'recommendation': 1.7
    }
    
    critical_metrics = {
        'error': 3.0, 'exception': 2.8, 'fail': 2.5,
        'latency': 2.5, 'delay': 2.2, 'timeout': 2.3,
        'cpu': 2.0, 'memory': 1.8, 'mem': 1.8,
        'disk': 1.6, 'network': 1.7, 'connection': 1.5
    }
    
    weights = []
    for name in node_names:
        name_str = str(name).lower()
        weight = 1.0
        
        # 檢查關鍵服務
        for service, service_weight in critical_services.items():
            if service in name_str:
                weight = max(weight, service_weight)
        
        # 檢查關鍵指標
        for metric, metric_weight in critical_metrics.items():
            if metric in name_str:
                weight = max(weight, metric_weight)
        
        # 指標統計類型加權
        if any(stat in name_str for stat in ['max', 'std', 'trend', 'peak']):
            weight *= 1.2
        
        weights.append(min(weight, 3.0))  # 限制最大權重
    
    return torch.tensor(weights, dtype=torch.float)


def validate_model_setup(model, node_features, edge_index):
    """
    驗證模型設置的完整性
    
    Args:
        model: GNN-KAN模型
        node_features: 節點特徵張量
        edge_index: 邊索引張量
        
    Returns:
        bool: 驗證是否通過
    """
    try:
        # 檢查設備一致性
        model_device = next(model.parameters()).device
        if node_features.device != model_device:
            print(f"⚠️ Device mismatch: model on {model_device}, features on {node_features.device}")
            return False
            
        if edge_index.device != model_device:
            print(f"⚠️ Device mismatch: model on {model_device}, edge_index on {edge_index.device}")
            return False
        
        # 測試前向傳播
        model.eval()
        with torch.no_grad():
            embeddings, adj = model(node_features, edge_index)
            
            # 檢查輸出形狀
            if embeddings.size(0) != node_features.size(0):
                print(f"⚠️ Embedding shape mismatch: {embeddings.shape} vs {node_features.shape}")
                return False
                
            if adj.size() != (node_features.size(0), node_features.size(0)):
                print(f"⚠️ Adjacency shape mismatch: {adj.shape}")
                return False
            
            # 檢查數值穩定性
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print("⚠️ NaN/Inf in embeddings")
                return False
                
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                print("⚠️ NaN/Inf in adjacency matrix")
                return False
        
        print("✓ Model validation passed")
        return True
        
    except Exception as e:
        print(f"⚠️ Model validation failed: {e}")
        return False