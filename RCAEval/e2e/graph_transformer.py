"""
Graph Transformer RCA - Modern GNN Baseline

Ensures fair comparison:
1. Same input features: RCA-aware features
2. Same number of layers: 4 layers
3. Same hidden dimensions: [128, 96, 64]
4. Same output dimension: 96
5. Same training strategy: same learning rate, epochs, early stopping
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
    Get the best available device
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


def graph_transformer_rca(data, inject_time=None, dataset=None, with_bg=False, 
                          config_type='simplified', feature_method='enhanced_ica', 
                          use_optimized_input=True, **kwargs):
    """
    Graph Transformer RCA Analysis - Modern GNN Baseline Method
    
    Uses exactly the same settings as GNN-KAN to ensure fair comparison:
    - Same input features (RCA-aware features)
    - Same number of layers and dimensions
    - Same training strategy
    - Uses self-attention mechanism (modern transformer architecture)
    
    Args:
        data: Input data
        inject_time: Fault injection time
        dataset: Dataset name
        with_bg: Whether to use background data
        config_type: Configuration type
        feature_method: Feature extraction method (must be same as GNN-KAN)
        use_optimized_input: Whether to use optimized input processing
        **kwargs: Other parameters
        
    Returns:
        RCA analysis results
    """
    print("Using Graph Transformer architecture for RCA analysis")
    print("Goal: Modern GNN baseline comparison for GNN-KAN")
    print(f"Feature method: {feature_method} (same as GNN-KAN)")
    print("Fair comparison settings: same layers, dimensions, training strategy")
    
    start_time = time.time()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Set parameters (exactly same as GNN-KAN)
    kwargs.setdefault('learning_rate', 2e-4)  # Same as GNN-KAN
    kwargs.setdefault('num_epochs', 400)      # Same as GNN-KAN
    kwargs.setdefault('use_cuda', True)
    kwargs.setdefault('cpu_fallback', True)
    kwargs.setdefault('target_feature_dim', 128)  # Same as GNN-KAN
    kwargs.setdefault('hidden_dim', 128)         # Same as GNN-KAN
    kwargs.setdefault('similarity_threshold', 0.2)  # Same as GNN-KAN
    kwargs.setdefault('max_edges_per_node', 15)    # Same as GNN-KAN
    
    # Ensure same configuration as GNN-KAN
    kwargs.setdefault('num_gnn_layers', 4)         # Same as GNN-KAN
    kwargs.setdefault('hidden_dims', [128, 96, 64])  # Same as GNN-KAN
    kwargs.setdefault('output_dim', 96)            # Same as GNN-KAN
    kwargs.setdefault('input_dim', 128)            # Same as GNN-KAN
    kwargs.setdefault('dropout', 0.1)              # Same as GNN-KAN
    kwargs.setdefault('weight_decay', 5e-6)        # Same as GNN-KAN
    kwargs.setdefault('gradient_clip_norm', 0.5)   # Same as GNN-KAN
    kwargs.setdefault('patience', 60)              # Same as GNN-KAN
    
    print(f"  Core parameters: LR={kwargs['learning_rate']}, Epochs={kwargs['num_epochs']}")
    print(f"  Architecture parameters: layers={kwargs['num_gnn_layers']}, hidden_dims={kwargs['hidden_dims']}, output_dim={kwargs['output_dim']}")
    
    # Device detection
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        device = get_best_device()
        
        print(f"Device detection:")
        print(f"  - CUDA: {'Available' if cuda_available else 'Not available'}")
        print(f"  - MPS (Apple GPU): {'Available' if mps_available else 'Not available'}")
        print(f"  - Selected device: {device.upper()}")
        
        force_gpu = kwargs.get('use_cuda', device != 'cpu')
        if force_gpu and device != 'cpu':
            try:
                test_tensor = torch.randn(10, 10).to(device)
                _ = test_tensor @ test_tensor
                print(f"{device.upper()} test successful, will use {device.upper()} acceleration")
                use_gpu = True
            except Exception as test_error:
                print(f"Warning: {device.upper()} test failed: {test_error}, switching to CPU")
                device = 'cpu'
                use_gpu = False
        else:
            print("Will use CPU mode")
            use_gpu = False
            device = 'cpu'
    except Exception as device_error:
        print(f"Warning: Device detection failed: {device_error}, switching to CPU")
        use_gpu = False
        device = 'cpu'
    
    # 1. Create configuration (same as GNN-KAN)
    config = create_config(**kwargs)
    config.feature_method = feature_method
    config.use_cuda = use_gpu
    
    if use_gpu:
        config.use_cuda = True
        config.device = device
        config.batch_size = min(config.batch_size * 2, 128)
        print(f"{device.upper()} acceleration config: batch_size={config.batch_size}, epochs={config.num_epochs}")
    else:
        config.use_cuda = False
        config.device = 'cpu'
        print("CPU config: using standard parameters")
    
    # 2. Feature processing (completely reuse GNN-KAN feature extraction for fairness)
    print(f"Using optimized input processor (feature method: {feature_method})...")
    print("   Using exactly the same feature extraction as GNN-KAN to ensure fair comparison")
    
    similarity_threshold = kwargs.get('similarity_threshold', 0.2)
    max_edges_per_node = kwargs.get('max_edges_per_node', 15)
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
    print(f"Optimization processing completed: {proc_time:.3f}s")
    print(f"Processing results: {len(node_names)} nodes, {edge_index.shape[1]} edges")
    
    # 3. Initialize Graph Transformer model (same architecture as GNN-KAN)
    print("Initializing Graph Transformer model (same layers and dimensions as GNN-KAN)...")
    from RCAEval.gnn_kan_module.graph_transformer_model import GraphTransformerModel
    
    model = GraphTransformerModel(config, len(node_names))
    
    # Move to device
    print(f"Using device: {device}")
    try:
        if device != 'cpu':
            model = model.to(device)
            node_features = node_features.to(device)
            edge_index = edge_index.to(device) 
            edge_weights = edge_weights.to(device)
            print(f"Model and data successfully moved to {device.upper()}")
        else:
            model = model.cpu()
            node_features = node_features.cpu()
            edge_index = edge_index.cpu()
            edge_weights = edge_weights.cpu()
            print("Model and data running on CPU")
    except Exception as device_error:
        print(f"Warning: Device movement failed: {device_error}, forcing CPU")
        device = 'cpu'
        model = model.cpu()
        node_features = node_features.cpu()
        edge_index = edge_index.cpu()
        edge_weights = edge_weights.cpu()
    
    # 4. Train Graph Transformer model (same training strategy as GNN-KAN)
    print("Starting Graph Transformer model training (same training strategy as GNN-KAN)...")
    from RCAEval.gnn_kan_module.graph_transformer_training import train_graph_transformer_model
    
    model, training_history = train_graph_transformer_model(
        model, 
        node_features, 
        edge_index, 
        config
    )
    
    # 5. Inference and ranking
    print("Step 5: Inference and root cause ranking...")
    model.eval()
    
    with torch.no_grad():
        try:
            # Graph Transformer inference
            embeddings, adj_scores = model(node_features, edge_index)
            print(f"Graph Transformer inference successful: input {node_features.shape} -> embeddings {embeddings.shape}")
            
            if adj_scores is not None:
                adj_matrix = torch.sigmoid(adj_scores)
                print(f"Graph Transformer graph structure learning successful: {adj_matrix.shape}, density: {adj_matrix.mean().item():.3f}")
            else:
                # Use degree centrality fallback
                print("Warning: Graph structure learning failed, using simple fallback")
                from sklearn.metrics.pairwise import cosine_similarity
                embeddings_np = embeddings.cpu().numpy()
                similarity_matrix = cosine_similarity(embeddings_np)
                adj_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
            
            # Move to CPU
            if device != 'cpu':
                adj_matrix = adj_matrix.cpu()
                embeddings = embeddings.cpu()
                
        except Exception as inference_error:
            print(f"Warning: Graph Transformer inference failed: {inference_error}")
            num_nodes = len(node_names)
            adj_matrix = torch.eye(num_nodes)
            embeddings = node_features.cpu()
    
    # 6. PageRank ranking (same as GNN-KAN)
    print("Computing PageRank importance ranking...")
    numpy_adj = adj_matrix.detach().numpy()
    print(f"Built adjacency matrix: {numpy_adj.shape}, density: {numpy_adj.mean():.3f}")
    
    try:
        ranks = page_rank(numpy_adj)
        pagerank_ranks = [node_names[i] for i in np.argsort(ranks)[::-1]]
        pagerank_scores = {node_names[i]: ranks[i] for i in range(len(node_names))}
    except Exception as pagerank_error:
        print(f"Warning: PageRank computation failed: {pagerank_error}, using degree centrality")
        degrees = numpy_adj.sum(axis=1)
        sorted_indices = np.argsort(degrees)[::-1]
        pagerank_ranks = [node_names[i] for i in sorted_indices]
        pagerank_scores = {node_names[i]: degrees[i] for i in range(len(node_names))}
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.3f}s")
    print(f"PageRank completed, top-3: {pagerank_ranks[:3]}")
    
    # Clean up memory
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
            'extra_kwargs': kwargs,
            'fair_comparison': True,  # Mark as fair comparison
            'architecture': {
                'num_layers': config.num_gnn_layers,
                'hidden_dims': config.hidden_dims,
                'output_dim': config.output_dim
            }
        }
    }

