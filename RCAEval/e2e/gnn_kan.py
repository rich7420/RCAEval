"""
GNN-KAN RCA: Graph Neural Network with Kolmogorov-Arnold Networks for Root Cause Analysis
Main implementation file for the RCAEval framework
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

# Import our KAN modules
from RCAEval.kan import (
    KANLayer, GNNKANEncoder,
    sliding_window_alignment, extract_log_features, stl_decomposition,
    kll_feature_processing, compute_topology_features, extract_error_features,
    feature_fusion
)
from RCAEval.io.time_series import preprocess, drop_constant

warnings.filterwarnings("ignore")


class GNNKANConfig:
    """Configuration class for GNN-KAN parameters"""
    
    def __init__(self):
        # Model architecture
        self.input_dim = 64
        self.hidden_dims = [128, 64]
        self.output_dim = 32
        self.kan_grid_size = 5
        self.kan_spline_order = 3
        self.num_gnn_layers = 3
        self.dropout = 0.1
        
        # Training parameters
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.5
        
        # Feature extraction
        self.window_size = 32
        self.step_size = 1
        self.use_dla = False
        self.max_log_features = 500
        self.stl_seasonal = 7
        self.kll_k = 512
        self.fusion_method = 'attention'
        self.target_feature_dim = 64
        
        # Graph construction
        self.similarity_threshold = 0.3
        self.max_edges_per_node = 5
        self.use_self_loops = True
        
        # Hardware
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        
        # Output
        self.top_k_results = 10


class MultiModalFeatureExtractor:
    """多模態特徵提取器"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def extract_features(self, data, inject_time=None):
        """
        提取多模態特徵
        
        Args:
            data: 輸入數據 (dict 或 DataFrame)
            inject_time: 故障注入時間
            
        Returns:
            features: 提取的特徵
            node_names: 節點名稱
        """
        if isinstance(data, dict):
            return self._extract_multimodal_features(data, inject_time)
        else:
            return self._extract_single_modal_features(data, inject_time)
    
    def _extract_multimodal_features(self, data, inject_time):
        """處理多模態數據"""
        all_features = []
        node_names = []
        
        # 處理 metrics 數據
        if 'metric' in data:
            metric_data = data['metric']
            if inject_time is not None:
                metric_data = metric_data.iloc[::15, :]  # 降採樣
                normal_data = metric_data[metric_data['time'] < inject_time]
                anomal_data = metric_data[metric_data['time'] >= inject_time]
                
                normal_processed = preprocess(normal_data, dataset='default')
                anomal_processed = preprocess(anomal_data, dataset='default')
                
                # 對齊列
                intersect = [x for x in normal_processed.columns if x in anomal_processed.columns]
                normal_processed = normal_processed[intersect]
                anomal_processed = anomal_processed[intersect]
                
                metric_data = pd.concat([normal_processed, anomal_processed], axis=0, ignore_index=True)
            else:
                metric_data = preprocess(metric_data, dataset='default')
            
            # STL 分解
            stl_features, stl_names = stl_decomposition(
                metric_data.select_dtypes(include=[np.number]),
                seasonal=self.config.stl_seasonal
            )
            
            if stl_features.size > 0:
                # KLL 處理
                processed_features = kll_feature_processing(
                    stl_features, k=self.config.kll_k
                )
                all_features.append(processed_features)
                node_names.extend(stl_names)
        
        # 處理 log 數據
        if 'logts' in data:
            log_data = data['logts']
            log_data = drop_constant(log_data)
            
            if inject_time is not None:
                normal_log = log_data[log_data['time'] < inject_time].drop(columns=['time'])
                anomal_log = log_data[log_data['time'] >= inject_time].drop(columns=['time'])
                log_combined = pd.concat([normal_log, anomal_log], axis=0, ignore_index=True)
            else:
                log_combined = log_data.drop(columns=['time'], errors='ignore')
            
            # TF-IDF 特徵提取
            log_features, log_names = extract_log_features(
                log_combined,
                use_dla=self.config.use_dla,
                max_features=self.config.max_log_features
            )
            
            if log_features.size > 0:
                all_features.append(log_features)
                node_names.extend(log_names)
        
        # 合併所有特徵
        if all_features:
            # 對齊特徵長度
            min_length = min(f.shape[0] for f in all_features)
            aligned_features = [f[:min_length] for f in all_features]
            
            # 特徵融合
            fused_features = feature_fusion(
                *aligned_features,
                fusion_method=self.config.fusion_method,
                target_dim=self.config.target_feature_dim
            )
            
            return fused_features, node_names
        else:
            return np.array([]), []
    
    def _extract_single_modal_features(self, data, inject_time):
        """處理單一模態數據"""
        # 預處理數據
        processed_data = preprocess(data, dataset='default')
        
        # STL 分解
        stl_features, node_names = stl_decomposition(
            processed_data.select_dtypes(include=[np.number]),
            seasonal=self.config.stl_seasonal
        )
        
        if stl_features.size > 0:
            # KLL 處理
            processed_features = kll_feature_processing(
                stl_features, k=self.config.kll_k
            )
            return processed_features, node_names
        else:
            return np.array([]), []


class GraphConstructor:
    """圖構建器"""
    
    def __init__(self, config):
        self.config = config
    
    def build_graph(self, features, node_names):
        """
        構建圖結構
        
        Args:
            features: 特徵矩陣 [num_samples, num_features]
            node_names: 節點名稱列表
            
        Returns:
            edge_index: 邊索引 [2, num_edges]
            edge_weights: 邊權重 [num_edges]
        """
        if features.size == 0 or len(node_names) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
        
        num_nodes = len(node_names)
        
        # 計算節點特徵 (使用特徵的統計量)
        if features.ndim == 2 and features.shape[0] > 1:
            # 每個節點的特徵是對應列的統計量
            node_features = np.array([
                [
                    np.mean(features[:, i % features.shape[1]]),
                    np.std(features[:, i % features.shape[1]]),
                    np.max(features[:, i % features.shape[1]]),
                    np.min(features[:, i % features.shape[1]])
                ]
                for i in range(num_nodes)
            ])
        else:
            # 使用隨機特徵作為後備
            node_features = np.random.randn(num_nodes, 4)
        
        # 計算相似性矩陣
        similarity_matrix = cosine_similarity(node_features)
        
        # 構建邊
        edge_list = []
        edge_weights = []
        
        for i in range(num_nodes):
            # 找到最相似的節點
            similarities = similarity_matrix[i]
            
            # 排除自己並找到前 k 個相似節點
            similarities[i] = -1  # 排除自己
            top_indices = np.argsort(similarities)[-self.config.max_edges_per_node:]
            
            for j in top_indices:
                if similarities[j] > self.config.similarity_threshold:
                    edge_list.append([i, j])
                    edge_weights.append(similarities[j])
        
        # 添加自環
        if self.config.use_self_loops:
            for i in range(num_nodes):
                edge_list.append([i, i])
                edge_weights.append(1.0)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        else:
            # 創建最小連通圖
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t()
            edge_weights = torch.ones(num_nodes, dtype=torch.float)
        
        return edge_index, edge_weights


class GNNKANModel(nn.Module):
    """GNN-KAN 模型"""
    
    def __init__(self, config, num_nodes):
        super(GNNKANModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # 特徵投影層
        self.feature_projection = KANLayer(
            config.target_feature_dim, 
            config.input_dim,
            grid_size=config.kan_grid_size
        )
        
        # GNN-KAN 編碼器
        self.gnn_encoder = GNNKANEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            grid_size=config.kan_grid_size,
            dropout=config.dropout
        )
        
        # 輸出層
        self.output_kan = KANLayer(
            config.output_dim,
            num_nodes,
            grid_size=config.kan_grid_size
        )
        
        # 圖重建損失
        self.graph_decoder = KANLayer(
            config.output_dim * 2,
            1,
            grid_size=config.kan_grid_size
        )
    
    def forward(self, node_features, edge_index):
        """
        前向傳播
        
        Args:
            node_features: 節點特徵 [num_nodes, feature_dim]
            edge_index: 邊索引 [2, num_edges]
            
        Returns:
            node_embeddings: 節點嵌入 [num_nodes, output_dim]
            adj_scores: 鄰接矩陣分數 [num_nodes, num_nodes]
        """
        # 特徵投影
        projected_features = self.feature_projection(node_features)
        
        # GNN-KAN 編碼
        node_embeddings = self.gnn_encoder(projected_features, edge_index)
        
        # 計算鄰接矩陣分數
        adj_scores = self._compute_adjacency_scores(node_embeddings)
        
        return node_embeddings, adj_scores
    
    def _compute_adjacency_scores(self, embeddings):
        """計算鄰接矩陣分數"""
        num_nodes = embeddings.size(0)
        adj_scores = torch.zeros(num_nodes, num_nodes, device=embeddings.device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                # 使用 KAN 計算邊的存在概率
                edge_features = torch.cat([embeddings[i], embeddings[j]], dim=0)
                score = torch.sigmoid(self.graph_decoder(edge_features.unsqueeze(0)))
                adj_scores[i, j] = score.squeeze()
        
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
        graph_constructor = GraphConstructor(config)
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
        
        if config.use_cuda:
            model = model.cuda()
            edge_index = edge_index.cuda()
            node_features = torch.tensor(node_features, dtype=torch.float).cuda()
        else:
            node_features = torch.tensor(node_features, dtype=torch.float)
        
        # 訓練模型
        model, final_adj = train_gnn_kan_model(
            model, node_features, edge_index, config
        )
        
        # 5. 根因分析
        print("Performing root cause analysis...")
        
        # 使用 PageRank 進行排序
        try:
            # 轉換為 NetworkX 圖
            adj_np = final_adj.cpu().numpy()
            
            # 應用閾值
            adj_binary = (adj_np > 0.3).astype(float)
            
            if np.sum(adj_binary) > 0:
                pagerank = PageRank()
                scores = pagerank.fit_transform(adj_binary)
                
                # 合併分數和節點名稱，按分數排序
                ranks = list(zip(node_names, scores))
                ranks.sort(key=lambda x: x[1], reverse=True)
                ranks = [x[0] for x in ranks]
            else:
                # 如果圖為空，返回原始順序
                ranks = node_names
                
        except Exception as e:
            print(f"PageRank failed: {e}, using node order")
            ranks = node_names
        
        end_time = time.time()
        print(f"GNN-KAN RCA completed in {end_time - start_time:.2f} seconds")
        
        return {
            "adj": adj_np,
            "node_names": node_names,
            "ranks": ranks
        }
        
    except Exception as e:
        print(f"Error in GNN-KAN RCA: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "adj": np.array([]),
            "node_names": [],
            "ranks": []
        }


def train_gnn_kan_model(model, node_features, edge_index, config):
    """
    訓練 GNN-KAN 模型
    
    Args:
        model: GNN-KAN 模型
        node_features: 節點特徵
        edge_index: 邊索引
        config: 配置參數
        
    Returns:
        trained_model: 訓練後的模型
        final_adj: 最終的鄰接矩陣
    """
    model.train()
    
    # 設置優化器
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    scheduler = lr_scheduler.StepLR(optimizer, 
                                   step_size=config.scheduler_step_size,
                                   gamma=config.scheduler_gamma)
    
    # 訓練循環
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        # 前向傳播
        node_embeddings, adj_scores = model(node_features, edge_index)
        
        # 計算損失
        loss = compute_loss(node_embeddings, adj_scores, edge_index, config)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{config.epochs}, Loss: {loss.item():.4f}")
    
    # 獲取最終的鄰接矩陣
    model.eval()
    with torch.no_grad():
        _, final_adj = model(node_features, edge_index)
    
    return model, final_adj


def compute_loss(node_embeddings, adj_scores, edge_index, config):
    """
    計算訓練損失
    
    Args:
        node_embeddings: 節點嵌入
        adj_scores: 鄰接矩陣分數
        edge_index: 邊索引
        config: 配置參數
        
    Returns:
        total_loss: 總損失
    """
    # 1. 圖重建損失
    num_nodes = node_embeddings.size(0)
    
    # 創建真實鄰接矩陣
    true_adj = torch.zeros(num_nodes, num_nodes, device=node_embeddings.device)
    if edge_index.size(1) > 0:
        true_adj[edge_index[0], edge_index[1]] = 1.0
    
    # 重建損失 (二元交叉熵)
    reconstruction_loss = F.binary_cross_entropy(adj_scores, true_adj)
    
    # 2. 嵌入正則化損失
    embedding_reg = torch.norm(node_embeddings, p=2, dim=1).mean()
    
    # 3. 稀疏性損失 (鼓勵稀疏的鄰接矩陣)
    sparsity_loss = torch.norm(adj_scores, p=1) / (num_nodes * num_nodes)
    
    # 總損失
    total_loss = reconstruction_loss + 0.01 * embedding_reg + 0.001 * sparsity_loss
    
    return total_loss


def run_gnn_kan_rca(data, inject_time=None, dataset=None, **kwargs):
    """
    GNN-KAN RCA 的主要入口函數，兼容原有接口
    """
    return gnn_kan_rca(data, inject_time=inject_time, dataset=dataset, **kwargs)
    

# 測試函數
def test_gnn_kan():
    """測試 GNN-KAN 功能"""
    print("Testing GNN-KAN RCA...")
    
    # 創建測試數據
    np.random.seed(42)
    test_data = pd.DataFrame({
        'time': range(100),
        'cpu_usage': np.random.rand(100) * 100,
        'memory_usage': np.random.rand(100) * 100,
        'disk_io': np.random.rand(100) * 1000,
        'network_latency': np.random.rand(100) * 50
    })
    
    # 運行 GNN-KAN RCA
    result = gnn_kan_rca(test_data, inject_time=50, dataset='test')
    
    print(f"Result keys: {list(result.keys())}")
    print(f"Number of nodes: {len(result['node_names'])}")
    print(f"Top 5 ranked nodes: {result['ranks'][:5]}")
    print(f"Adjacency matrix shape: {result['adj'].shape}")
    
    print("GNN-KAN test completed!")
    return result


if __name__ == "__main__":
    test_gnn_kan()