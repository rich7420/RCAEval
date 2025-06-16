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
from .kan_components.kan_layers import (
    OptimizedGNNKANEncoder,
    GradientStabilizer
)

# Import feature processing functions directly
from .feature_processing import (
    simplified_metric_processing,
    enhanced_trace_processing
)

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

print("✅ GNN-KAN Core 模組載入完成 - 無重複定義")