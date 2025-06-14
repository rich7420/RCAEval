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

# GNNKANModel 類已移至 models.py 中，避免重複定義
# 主要的 gnn_kan_rca 函數在 e2e/gnnkan.py 中實現