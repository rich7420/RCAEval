"""
Pure GNN RCA - Using standard GNN with MLP layers (without KAN)
Pure GNN method using standard MLP layers (without KAN)
Uses the same feature extraction as GNN-KAN, but uses traditional MLP for intermediate layers
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

def get_best_device():
    """
    Get best available device
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# Import required components from GNN-KAN module (reuse feature extraction)
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import configuration and input processor (reuse GNN-KAN feature extraction)
from RCAEval.gnn_kan_module import GNNKANConfig, SimplifiedGNNKANConfig, create_config
from RCAEval.gnn_kan_module.optimized_input_processor import GNNKANInputOptimizer
from RCAEval.graph_heads.page_rank import page_rank


def gnn_rca(data, inject_time=None, dataset=None, with_bg=False, 
            config_type='simplified', feature_method='enhanced_ica', 
            use_optimized_input=True, **kwargs):
    """
    Pure GNN RCA analysis (without KAN)
    Reuses GNN-KAN feature extraction but uses standard MLP architecture
    
    Args:
        data: 輸入數據
        inject_time: 故障注入時間
        dataset: 數據集名稱
        with_bg: 是否使用背景數據
        config_type: 配置類型
        feature_method: 特徵提取方法
        use_optimized_input: 是否使用優化輸入處理
        **kwargs: 其他參數
        
    Returns:
        RCA 分析結果
    """
    start_time = time.time()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Set parameters
    kwargs.setdefault('learning_rate', 1e-3)
    kwargs.setdefault('num_epochs', 100)
    kwargs.setdefault('use_cuda', True)
    kwargs.setdefault('cpu_fallback', True)
    kwargs.setdefault('target_feature_dim', 64)
    kwargs.setdefault('hidden_dim', 64)
    kwargs.setdefault('similarity_threshold', 0.3)
    kwargs.setdefault('max_edges_per_node', 5)
    
    # Device detection
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        device = get_best_device()
        
        force_gpu = kwargs.get('use_cuda', device != 'cpu')
        if force_gpu and device != 'cpu':
            try:
                test_tensor = torch.randn(10, 10).to(device)
                _ = test_tensor @ test_tensor
                use_gpu = True
            except Exception as test_error:
                device = 'cpu'
                use_gpu = False
        else:
            use_gpu = False
            device = 'cpu'
    except Exception as device_error:
        use_gpu = False
        device = 'cpu'
    
    # 1. Create configuration (reuse GNN-KAN configuration)
    config = create_config(**kwargs)
    config.feature_method = feature_method
    config.use_cuda = use_gpu
    
    if use_gpu:
        config.use_cuda = True
        config.device = device
        config.batch_size = min(config.batch_size * 2, 128)
    else:
        config.use_cuda = False
        config.device = 'cpu'
    
    # 2. Feature processing (completely reuse GNN-KAN feature extraction)
    
    similarity_threshold = kwargs.get('similarity_threshold', 0.3)
    max_edges_per_node = kwargs.get('max_edges_per_node', 5)
    force_node_expansion = kwargs.get('force_node_expansion', False)
    
    processor = GNNKANInputOptimizer(
        feature_method=feature_method,
        target_dim=config.target_feature_dim,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges_per_node,
        force_node_expansion=force_node_expansion
    )
    
    start_proc = time.time()
    optimized_data = processor.optimize_input(data, inject_time)
    
    node_features = optimized_data.node_features
    edge_index = optimized_data.edge_index
    edge_weights = optimized_data.edge_weights
    node_names = optimized_data.node_names
    
    proc_time = time.time() - start_proc
    
    # 3. Initialize pure GNN model (use standard MLP)
    from RCAEval.gnn_kan_module.gnn_model import PureGNNModel
    
    model = PureGNNModel(config, len(node_names))
    
    # Move to device
    try:
        if device != 'cpu':
            model = model.to(device)
            node_features = node_features.to(device)
            edge_index = edge_index.to(device) 
            edge_weights = edge_weights.to(device)
        else:
            model = model.cpu()
            node_features = node_features.cpu()
            edge_index = edge_index.cpu()
            edge_weights = edge_weights.cpu()
    except Exception as device_error:
        device = 'cpu'
        model = model.cpu()
        node_features = node_features.cpu()
        edge_index = edge_index.cpu()
        edge_weights = edge_weights.cpu()
    
    # 4. Train pure GNN model
    from RCAEval.gnn_kan_module.gnn_training import train_gnn_model
    
    model, training_history = train_gnn_model(
        model, 
        node_features, 
        edge_index, 
        config
    )
    
    # 5. Inference and ranking
    model.eval()
    
    with torch.no_grad():
        try:
            # GNN inference
            embeddings, adj_scores = model(node_features, edge_index)
            
            if adj_scores is not None:
                adj_matrix = torch.sigmoid(adj_scores)
            else:
                # Use degree centrality fallback
                from sklearn.metrics.pairwise import cosine_similarity
                embeddings_np = embeddings.cpu().numpy()
                similarity_matrix = cosine_similarity(embeddings_np)
                adj_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
            
            # Move to CPU
            if device != 'cpu':
                adj_matrix = adj_matrix.cpu()
                embeddings = embeddings.cpu()
                
        except Exception as inference_error:
            num_nodes = len(node_names)
            adj_matrix = torch.eye(num_nodes)
            embeddings = node_features.cpu()
    
    # 6. PageRank ranking
    numpy_adj = adj_matrix.detach().numpy()
    
    try:
        ranks = page_rank(numpy_adj)
        pagerank_ranks = [node_names[i] for i in np.argsort(ranks)[::-1]]
        pagerank_scores = {node_names[i]: ranks[i] for i in range(len(node_names))}
    except Exception as pagerank_error:
        degrees = numpy_adj.sum(axis=1)
        sorted_indices = np.argsort(degrees)[::-1]
        pagerank_ranks = [node_names[i] for i in sorted_indices]
        pagerank_scores = {node_names[i]: degrees[i] for i in range(len(node_names))}
    
    total_time = time.time() - start_time
    
    # Clean memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'ranks': pagerank_ranks,
        'adj': numpy_adj,
        'node_names': node_names,
        'embeddings': embeddings.detach().numpy(),
        'processing_time': total_time,
        'device_used': device,
        'gpu_accelerated': use_gpu,
        'pagerank_scores': pagerank_scores,
        'config_info': {
            'config_type': config_type,
            'feature_method': feature_method,
            'use_optimized_input': use_optimized_input,
            'extra_kwargs': kwargs
        }
    }



