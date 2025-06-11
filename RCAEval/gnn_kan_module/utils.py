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


def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    保存模型檢查點
    
    Args:
        model: 模型
        optimizer: 優化器
        epoch: 當前epoch
        loss: 當前損失
        filepath: 保存路徑
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


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