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
from .kan_components.kan_layer import (
    OptimizedGNNKANEncoder,
    GradientStabilizer
)

# Import feature processing functions directly
from .feature_processing import (
    simplified_metric_processing,
    enhanced_trace_processing
)

# 使用相對導入
try:
    from ..io.time_series import preprocess, drop_constant
except ImportError:
    print("警告：io.time_series 模組不可用，使用簡化預處理")
    
    def preprocess(data, dataset=None, **kwargs):
        """簡化的數據預處理"""
        if isinstance(data, pd.DataFrame):
            return data.fillna(method='ffill').fillna(0)
        return data
    
    def drop_constant(data):
        """簡化的常數列移除"""
        if isinstance(data, pd.DataFrame):
            return data.loc[:, data.std() > 1e-8]
        return data

# Import from other modules
from .config import GNNKANConfig
from .feature_extractors import MultiModalFeatureExtractor
from .graph_constructors import SimplifiedGraphConstructor
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


# 移除重複的 gnn_kan_rca 函數 - 主實現在 e2e/gnnkan.py 中