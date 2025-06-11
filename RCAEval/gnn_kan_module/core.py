"""
GNN-KAN Core Implementation
核心GNN-KAN功能實現
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import networkx as nx
from sknetwork.ranking import PageRank
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import traceback

# Import from our kan_components instead of RCAEval.kan
from .kan_components import (
    OptimizedGNNKANEncoder,
    GradientStabilizer,
    extract_trace_features,
    build_service_dependency_graph,
    extract_service_topology_features,
    stl_decomposition,
    compute_service_criticality_weights,
    sliding_window_alignment,
    extract_log_features,
    feature_fusion
)
from RCAEval.io.time_series import preprocess, drop_constant

# Import from other modules
from .config import GNNKANConfig
from .feature_extraction import MultiModalFeatureExtractor
from .graph_construction import SimplifiedGraphConstructor
from .training import train_gnn_kan_model, TemporalAttention

warnings.filterwarnings("ignore")

# 向後兼容
SimplifiedGNNKANConfig = GNNKANConfig


class GNNKANModel(nn.Module):
    """GNN-KAN 模型 - 結合 GNN 和 KAN 的優勢"""
    
    def __init__(self, config, num_nodes):
        super(GNNKANModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # 特徵投影層
        self.feature_projection = nn.Linear(config.target_feature_dim, config.input_dim)
        
        # GNN-KAN編碼器
        self.gnn_encoder = OptimizedGNNKANEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            kan_grid_size=config.kan_grid_size,
            kan_spline_order=config.kan_spline_order,
            dropout=config.dropout
        )
        
        # 時序注意力機制
        self.temporal_attention = TemporalAttention(config.output_dim)
        
        # KAN 解碼器 - 用於計算鄰接矩陣
        self.graph_decoder = nn.Sequential(
            nn.Linear(config.output_dim * 2, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, node_features, edge_index):
        """
        前向傳播
        
        Args:
            node_features: 節點特徵 [num_nodes, feature_dim]
            edge_index: 邊索引 [2, num_edges]
            
        Returns:
            node_embeddings: 節點嵌入
            adj_scores: 鄰接矩陣分數
        """
        # 特徵投影
        projected_features = self.feature_projection(node_features)
        
        # GNN-KAN 編碼
        node_embeddings = self.gnn_encoder(projected_features, edge_index)
        
        # 時序注意力
        if node_embeddings.dim() == 2:
            # 增加時間維度用於注意力計算
            node_embeddings_expanded = node_embeddings.unsqueeze(1)  # [num_nodes, 1, feature_dim]
            attended_embeddings = self.temporal_attention(node_embeddings_expanded)
            node_embeddings = attended_embeddings.squeeze(1)  # [num_nodes, feature_dim]
        else:
            node_embeddings = self.temporal_attention(node_embeddings)
        
        # 計算鄰接矩陣分數 (批量化處理)
        adj_scores = self._compute_adjacency_scores_batch(node_embeddings)
        
        return node_embeddings, adj_scores
    
    def _compute_adjacency_scores_batch(self, embeddings):
        """批量化計算鄰接矩陣分數"""
        num_nodes = embeddings.size(0)
        
        # 創建所有可能的邊對
        i_indices = torch.arange(num_nodes, device=embeddings.device).repeat_interleave(num_nodes)
        j_indices = torch.arange(num_nodes, device=embeddings.device).repeat(num_nodes)
        
        # 批量計算邊特徵
        edge_features = torch.cat([
            embeddings[i_indices], 
            embeddings[j_indices]
        ], dim=1)
        
        # 批量通過 KAN 解碼器
        scores = torch.sigmoid(self.graph_decoder(edge_features))
        
        # 重塑為鄰接矩陣
        adj_scores = scores.view(num_nodes, num_nodes)
        
        return adj_scores


def gnn_kan_rca(data, inject_time=None, dataset=None, with_bg=False, **kwargs):
    """
    主要的 GNN-KAN RCA 方法
    
    Args:
        data: 輸入數據 (multimodal 或 單一模態)
        inject_time: 注入時間點
        dataset: 數據集名稱
        with_bg: 是否包含背景數據
        **kwargs: 其他參數
    
    Returns:
        dict: 包含 adj, node_names, ranks 的結果
    """
    print("Starting GNN-KAN RCA analysis...")
    start_time = time.time()
    
    # 初始化配置
    config = GNNKANConfig()
    
    # 更新配置參數
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    try:
        # 1. 特徵提取
        print("Extracting features...")
        feature_extractor = MultiModalFeatureExtractor(config)
        features, node_names = feature_extractor.extract_features(data, inject_time)
        
        if features.size == 0 or len(node_names) == 0:
            print("No features extracted, returning empty result")
            return {"adj": np.array([]), "node_names": [], "ranks": []}
        
        print(f"Extracted features shape: {features.shape}, Nodes: {len(node_names)}")
        
        # 2. 圖構建
        print("Building graph...")
        graph_constructor = SimplifiedGraphConstructor(config)
        edge_index, edge_weights = graph_constructor.build_graph(features, node_names)
        
        print(f"Built graph with {len(node_names)} nodes and {edge_index.size(1)} edges")
        
        # 3. 準備節點特徵
        if features.ndim == 2 and features.shape[1] >= config.target_feature_dim:
            node_features = features[:len(node_names), :config.target_feature_dim]
        else:
            # 使用特徵統計作為節點特徵
            if features.ndim == 2 and features.shape[0] > 0:
                node_stats = np.array([
                    [
                        np.mean(features[:, i % features.shape[1]]),
                        np.std(features[:, i % features.shape[1]]),
                        np.max(features[:, i % features.shape[1]]),
                        np.min(features[:, i % features.shape[1]])
                    ]
                    for i in range(len(node_names))
                ])
                
                # 擴展到目標維度
                if node_stats.shape[1] < config.target_feature_dim:
                    padding = np.zeros((len(node_names), 
                                     config.target_feature_dim - node_stats.shape[1]))
                    node_features = np.hstack([node_stats, padding])
                else:
                    node_features = node_stats[:, :config.target_feature_dim]
            else:
                node_features = np.random.randn(len(node_names), config.target_feature_dim)
        
        # 4. 訓練 GNN-KAN 模型
        print("Training GNN-KAN model...")
        model = GNNKANModel(config, len(node_names))
        
        # 設備管理優化
        device = 'cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            try:
                model = model.cuda()
                edge_index = edge_index.cuda()
                node_features = torch.tensor(node_features, dtype=torch.float).cuda()
            except RuntimeError as e:
                print(f"CUDA initialization failed: {e}, falling back to CPU")
                device = 'cpu'
                model = model.cpu()
                edge_index = edge_index.cpu()
                node_features = torch.tensor(node_features, dtype=torch.float).cpu()
        else:
            model = model.cpu()
            node_features = torch.tensor(node_features, dtype=torch.float).cpu()
            edge_index = edge_index.cpu()
        
        # 訓練模型
        model, final_adj = train_gnn_kan_model(
            model, node_features, edge_index, config
        )
        
        # 確保最終評估時的設備一致性
        print("Getting final adjacency matrix...")
        model.eval()
        
        # 確保所有張量在相同設備上
        model_device = next(model.parameters()).device
        node_features = node_features.to(model_device)
        edge_index = edge_index.to(model_device)
        
        try:
            with torch.no_grad():
                _, final_adj = model(node_features, edge_index)
                
                # 確認結果有效性
                if torch.isnan(final_adj).any() or torch.isinf(final_adj).any():
                    print("NaN/Inf in final adjacency, using fallback...")
                    num_nodes = node_features.size(0)
                    final_adj = torch.eye(num_nodes, device=model_device) * 0.8
                    final_adj += torch.rand(num_nodes, num_nodes, device=model_device) * 0.2
                    
        except RuntimeError as e:
            if "CUDA" in str(e) or "device" in str(e).lower():
                print(f"Device error in final evaluation: {e}")
                # 強制切換到CPU並重新計算
                model = model.cpu()
                node_features = node_features.cpu()
                edge_index = edge_index.cpu()
                
                with torch.no_grad():
                    try:
                        _, final_adj = model(node_features, edge_index)
                    except:
                        # 最終回退
                        num_nodes = node_features.size(0)
                        final_adj = torch.eye(num_nodes, device='cpu')
            else:
                print(f"Error getting final adjacency: {e}")
                num_nodes = node_features.size(0)
                final_adj = torch.eye(num_nodes, device=model_device)
        
        # 5. 計算 PageRank 排名
        print("Computing PageRank rankings...")
        adj_numpy = final_adj.detach().cpu().numpy()
        
        try:
            # 使用 PageRank 算法
            pagerank = PageRank()
            scores = pagerank.fit_transform(adj_numpy)
            
            # 獲取排名
            ranked_indices = np.argsort(scores)[::-1]
            top_k_indices = ranked_indices[:config.top_k_results]
            
            # 確保返回字符串列表，處理可能的嵌套列表
            ranks = []
            for i in top_k_indices:
                if i < len(node_names):
                    node_name = node_names[i]
                    # 如果 node_name 是列表，取第一个元素；如果是字符串，直接使用
                    if isinstance(node_name, (list, tuple)):
                        if len(node_name) > 0:
                            ranks.append(str(node_name[0]))
                        else:
                            ranks.append(f"node_{i}")
                    else:
                        ranks.append(str(node_name))
                else:
                    ranks.append(f"node_{i}")
            
        except Exception as e:
            print(f"PageRank computation failed: {e}, using degree centrality")
            # 回退到度中心性
            degrees = np.sum(adj_numpy, axis=1)
            ranked_indices = np.argsort(degrees)[::-1]
            top_k_indices = ranked_indices[:config.top_k_results]
            
            # 確保返回字符串列表，處理可能的嵌套列表
            ranks = []
            for i in top_k_indices:
                if i < len(node_names):
                    node_name = node_names[i]
                    # 如果 node_name 是列表，取第一个元素；如果是字符串，直接使用
                    if isinstance(node_name, (list, tuple)):
                        if len(node_name) > 0:
                            ranks.append(str(node_name[0]))
                        else:
                            ranks.append(f"node_{i}")
                    else:
                        ranks.append(str(node_name))
                else:
                    ranks.append(f"node_{i}")
        
        # 6. 組織結果
        result = {
            "adj": adj_numpy,
            "node_names": node_names,
            "ranks": ranks  # 現在是字符串列表格式
        }
        
        end_time = time.time()
        print(f"GNN-KAN RCA completed in {end_time - start_time:.2f} seconds")
        print(f"Top 5 root causes: {ranks[:5]}")
        
        return result
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return {"adj": np.array([]), "node_names": [], "ranks": []}
    except Exception as e:
        print(f"Critical error in GNN-KAN RCA: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空結果
        return {
            "adj": np.array([]),
            "node_names": [],
            "ranks": []
        }