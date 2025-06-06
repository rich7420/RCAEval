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

# Import our optimized KAN modules
from RCAEval.kan import (
    SimplifiedKANLayer, UltraFastKANLayer, OptimizedGNNKANEncoder,
    sliding_window_alignment, extract_log_features, stl_decomposition,
    kll_feature_processing, compute_topology_features,
    feature_fusion, extract_error_features,
    extract_trace_features, build_service_dependency_graph,
    extract_service_topology_features
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
        
        print("Processing multimodal data...")
        
        # 首先應用 sliding window 對齊
        if inject_time is not None:
            print("Applying sliding window alignment...")
            windows, timestamps = sliding_window_alignment(
                data, 
                window_size=self.config.window_size, 
                step_size=self.config.step_size,
                timestamp_col='time'
            )
            print(f"Created {len(windows)} windows for analysis")
        else:
            # 如果沒有 inject_time，處理整個數據集
            windows = [data]
            timestamps = [None]
        
        # 對每個窗口進行特徵提取
        for window_idx, window_data in enumerate(windows):
            window_features = []
            window_node_names = []
            
            # 處理 trace 數據 (新增功能)
            if 'trace' in window_data or 'traces' in window_data:
                trace_key = 'trace' if 'trace' in window_data else 'traces'
                trace_data = window_data[trace_key]
                
                print(f"Extracting trace features from window {window_idx}...")
                trace_features, operation_names, service_graph = extract_trace_features(
                    trace_data, inject_time
                )
                
                if trace_features.size > 0:
                    window_features.append(trace_features)
                    window_node_names.extend([f'w{window_idx}_trace_{name}' for name in operation_names])
                    print(f"✓ Extracted {trace_features.shape[0]} trace operations")
                
                # 提取服務拓樸特徵
                if service_graph is not None:
                    service_topo_features, service_topo_names = extract_service_topology_features(
                        service_graph, list(service_graph.nodes())
                    )
                    if service_topo_features.size > 0:
                        # 廣播服務拓樸特徵到窗口長度
                        min_length = trace_features.shape[0] if trace_features.size > 0 else 1
                        service_topo_expanded = np.tile(service_topo_features, (min_length, 1))
                        window_features.append(service_topo_expanded)
                        window_node_names.extend([f'w{window_idx}_service_topo_{name}' for name in service_topo_names])
                        print(f"✓ Extracted {len(service_topo_names)} service topology features")
            
            # 處理 metric 數據
            if 'metric' in window_data:
                metric_data = window_data['metric']
                
                if inject_time is not None:
                    # 分割正常和異常數據
                    normal_df = metric_data[metric_data['time'] < inject_time]
                    anomal_df = metric_data[metric_data['time'] >= inject_time]
                    
                    if not normal_df.empty and not anomal_df.empty:
                        normal_processed = preprocess(normal_df, dataset='default')
                        anomal_processed = preprocess(anomal_df, dataset='default')
                        
                        # 對齊列
                        intersect = [x for x in normal_processed.columns if x in anomal_processed.columns]
                        normal_processed = normal_processed[intersect]
                        anomal_processed = anomal_processed[intersect]
                        
                        metric_data = pd.concat([normal_processed, anomal_processed], axis=0, ignore_index=True)
                    else:
                        metric_data = preprocess(metric_data, dataset='default')
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
                    window_features.append(processed_features)
                    window_node_names.extend([f'w{window_idx}_{name}' for name in stl_names])
            
            # 處理 log 數據
            if 'log' in window_data:
                log_data = window_data['log']
                log_features, log_names = extract_log_features(
                    log_data,
                    use_dla=self.config.use_dla,
                    max_features=self.config.max_log_features
                )
                
                if log_features.size > 0:
                    window_features.append(log_features)
                    window_node_names.extend([f'w{window_idx}_{name}' for name in log_names])
            
            # 處理單一模態 DataFrame
            if isinstance(window_data, pd.DataFrame):
                # 檢查是否包含 trace 相關的列
                trace_columns = ['serviceName', 'methodName', 'operationName', 'startTime', 'duration']
                if any(col in window_data.columns for col in trace_columns):
                    print(f"Detected trace data in DataFrame format for window {window_idx}")
                    trace_features, operation_names, service_graph = extract_trace_features(
                        window_data, inject_time
                    )
                    
                    if trace_features.size > 0:
                        window_features.append(trace_features)
                        window_node_names.extend([f'w{window_idx}_trace_{name}' for name in operation_names])
                        print(f"✓ Extracted {trace_features.shape[0]} trace operations from DataFrame")
                    
                    # 提取服務拓樸特徵
                    if service_graph is not None:
                        service_topo_features, service_topo_names = extract_service_topology_features(
                            service_graph, list(service_graph.nodes())
                        )
                        if service_topo_features.size > 0:
                            min_length = trace_features.shape[0] if trace_features.size > 0 else 1
                            service_topo_expanded = np.tile(service_topo_features, (min_length, 1))
                            window_features.append(service_topo_expanded)
                            window_node_names.extend([f'w{window_idx}_service_topo_{name}' for name in service_topo_names])
                
                # STL 分解
                stl_features, stl_names = stl_decomposition(
                    window_data.select_dtypes(include=[np.number]),
                    seasonal=self.config.stl_seasonal
                )
                
                if stl_features.size > 0:
                    # KLL 處理
                    processed_features = kll_feature_processing(
                        stl_features, k=self.config.kll_k
                    )
                    window_features.append(processed_features)
                    window_node_names.extend([f'w{window_idx}_{name}' for name in stl_names])
                
                # 提取錯誤特徵
                print("Extracting error features...")
                error_features, error_names = extract_error_features(window_data)
                if error_features.size > 0:
                    window_features.append(error_features.reshape(1, -1) if error_features.ndim == 1 else error_features)
                    window_node_names.extend([f'w{window_idx}_{name}' for name in error_names])
            
            # 收集當前窗口的特徵
            if window_features:
                all_features.extend(window_features)
                node_names.extend(window_node_names)
        
        if not all_features:
            print("No features extracted, returning empty arrays")
            return np.array([]), []
        
        # 對齊所有特徵的長度
        min_length = min(f.shape[0] for f in all_features if f.size > 0)
        if min_length == 0:
            min_length = 1
            
        aligned_features = []
        for features in all_features:
            if features.size == 0:
                continue
            if features.shape[0] > min_length:
                features = features[:min_length]
            elif features.shape[0] < min_length:
                # 重複最後一行
                padding = np.repeat(features[-1:], min_length - features.shape[0], axis=0)
                features = np.vstack([features, padding])
            aligned_features.append(features)
        
        if not aligned_features:
            return np.array([]), []
        
        # 計算拓樸特徵
        print("Computing topology features...")
        # 先構建一個初步的相似性矩陣
        if len(aligned_features) > 1:
            # 計算特徵間的相似性
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 將所有特徵拼接
            combined_features = np.hstack(aligned_features)
            if combined_features.shape[1] > 1:
                similarity_matrix = cosine_similarity(combined_features.T)
                # 轉換為鄰接矩陣
                adj_matrix = (similarity_matrix > 0.5).astype(float)
                
                topology_features, topology_names = compute_topology_features(
                    adj_matrix, node_names[:adj_matrix.shape[0]]
                )
                
                if topology_features.size > 0:
                    # 將拓樸特徵廣播到所有節點
                    topo_features_expanded = np.tile(topology_features, (min_length, 1))
                    aligned_features.append(topo_features_expanded)
                    node_names.extend([f'topology_{name}' for name in topology_names])
        
        # 特徵融合
        print("Performing feature fusion...")
        
        # 分離不同類型的特徵進行融合
        log_feats = None
        metric_feats = None
        topo_feats = None
        error_feats = None
        trace_feats = None
        service_topo_feats = None
        
        combined_features = []
        for i, features in enumerate(aligned_features):
            node_name = node_names[i] if i < len(node_names) else ""
            
            if 'trace_' in node_name:
                if trace_feats is None:
                    trace_feats = features
                else:
                    trace_feats = np.hstack([trace_feats, features])
            elif 'service_topo_' in node_name:
                if service_topo_feats is None:
                    service_topo_feats = features
                else:
                    service_topo_feats = np.hstack([service_topo_feats, features])
            elif 'log' in node_name:
                if log_feats is None:
                    log_feats = features
                else:
                    log_feats = np.hstack([log_feats, features])
            elif 'topology' in node_name:
                if topo_feats is None:
                    topo_feats = features
                else:
                    topo_feats = np.hstack([topo_feats, features])
            elif 'error' in node_name:
                if error_feats is None:
                    error_feats = features
                else:
                    error_feats = np.hstack([error_feats, features])
            else:
                if metric_feats is None:
                    metric_feats = features
                else:
                    metric_feats = np.hstack([metric_feats, features])
        
        # 使用改進的特徵融合，包含 trace 特徵
        fused_features = enhanced_feature_fusion(
            log_feats, metric_feats, topo_feats, error_feats, trace_feats, service_topo_feats,
            fusion_method=self.config.fusion_method,
            target_dim=self.config.target_feature_dim
        )
        
        if fused_features.size == 0:
            # 回退方案：直接拼接
            fused_features = np.hstack(aligned_features)
        
        return fused_features, node_names
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
    """GPU 優化的 GNN-KAN 模型"""
    
    def __init__(self, config, num_nodes):
        super(GNNKANModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # 特徵投影層 (使用優化的 KAN)
        self.feature_projection = UltraFastKANLayer(
            config.target_feature_dim, 
            config.input_dim
        )
        
        # GNN-KAN 編碼器 (使用優化版本)
        self.gnn_encoder = OptimizedGNNKANEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            kan_type='ultra_fast',  # 使用最快的版本
            dropout=config.dropout
        )
        
        # 輸出層 (使用優化的 KAN)
        self.output_kan = UltraFastKANLayer(
            config.output_dim,
            num_nodes
        )
        
        # 圖重建損失 (使用優化的 KAN)
        self.graph_decoder = UltraFastKANLayer(
            config.output_dim * 2,
            1
        )
    
    def forward(self, node_features, edge_index):
        """
        優化的前向傳播
        """
        # 特徵投影
        projected_features = self.feature_projection(node_features)
        
        # GNN-KAN 編碼
        node_embeddings = self.gnn_encoder(projected_features, edge_index)
        
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
    print(f"Training on device: {next(model.parameters()).device}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    
    model.train()
    
    # 設置優化器
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    scheduler = lr_scheduler.StepLR(optimizer, 
                                   step_size=config.scheduler_step_size,
                                   gamma=config.scheduler_gamma)
    
    # 訓練循環
    try:
        for epoch in range(config.epochs):
            optimizer.zero_grad()
            
            # 前向傳播
            try:
                node_embeddings, adj_scores = model(node_features, edge_index)
                
                # 計算損失
                loss = compute_loss(node_embeddings, adj_scores, edge_index, config)
                
                # 檢查 loss 是否為 NaN
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}")
                    break
                
                # 反向傳播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                if epoch % 5 == 0:  # 更頻繁的日誌輸出
                    print(f"Epoch {epoch}/{config.epochs}, Loss: {loss.item():.4f}")
                    
                    # GPU 記憶體監控
                    if torch.cuda.is_available():
                        print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                        
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                break
                
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    
    # 獲取最終的鄰接矩陣
    print("Getting final adjacency matrix...")
    model.eval()
    try:
        with torch.no_grad():
            _, final_adj = model(node_features, edge_index)
    except Exception as e:
        print(f"Error getting final adjacency: {e}")
        # 返回隨機鄰接矩陣作為後備
        num_nodes = node_features.size(0)
        final_adj = torch.rand(num_nodes, num_nodes, device=node_features.device)
    
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
    
    # 檢查 GPU 可用性
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU")

    # 創建測試數據
    np.random.seed(42)
    test_data = pd.DataFrame({
        'time': range(100),
        'cpu_usage': np.random.rand(100) * 100,
        'memory_usage': np.random.rand(100) * 100,
        'disk_io': np.random.rand(100) * 1000,
        'network_latency': np.random.rand(100) * 50
    })

    # 運行 GNN-KAN RCA，使用較少的訓練輪數進行測試
    result = gnn_kan_rca(test_data, inject_time=50, dataset='test', 
                        stl_seasonal=3, epochs=20)

    print(f"Result keys: {list(result.keys())}")
    print(f"Number of nodes: {len(result['node_names'])}")
    print(f"Top 5 ranked nodes: {result['ranks'][:5]}")
    print(f"Adjacency matrix shape: {result['adj'].shape}")
    
    print("GNN-KAN test completed!")
    return result


def enhanced_feature_fusion(log_feats, metric_feats, topo_feats, error_feats, trace_feats, service_topo_feats,
                           fusion_method='attention', target_dim=None):
    """
    增強的多模態特徵融合，包含 trace 特徵
    
    Args:
        log_feats: 日誌特徵
        metric_feats: 度量特徵
        topo_feats: 拓樸特徵
        error_feats: 錯誤特徵
        trace_feats: trace 特徵 (新增)
        service_topo_feats: 服務拓樸特徵 (新增)
        fusion_method: 融合方法
        target_dim: 目標維度
    
    Returns:
        fused_features: 融合後的特徵
    """
    # 收集所有非空特徵
    all_features = []
    feature_weights = []
    
    if log_feats is not None and log_feats.size > 0:
        if log_feats.ndim == 1:
            log_feats = log_feats.reshape(1, -1)
        all_features.append(log_feats)
        feature_weights.append(0.2)  # 日誌特徵權重
    
    if metric_feats is not None and metric_feats.size > 0:
        if metric_feats.ndim == 1:
            metric_feats = metric_feats.reshape(1, -1)
        all_features.append(metric_feats)
        feature_weights.append(0.3)  # 度量特徵權重
    
    if trace_feats is not None and trace_feats.size > 0:
        if trace_feats.ndim == 1:
            trace_feats = trace_feats.reshape(1, -1)
        all_features.append(trace_feats)
        feature_weights.append(0.25)  # trace 特徵權重 (重要)
    
    if service_topo_feats is not None and service_topo_feats.size > 0:
        if service_topo_feats.ndim == 1:
            service_topo_feats = service_topo_feats.reshape(1, -1)
        all_features.append(service_topo_feats)
        feature_weights.append(0.15)  # 服務拓樸特徵權重
    
    if topo_feats is not None and topo_feats.size > 0:
        if topo_feats.ndim == 1:
            topo_feats = topo_feats.reshape(1, -1)
        all_features.append(topo_feats)
        feature_weights.append(0.08)  # 一般拓樸特徵權重
    
    if error_feats is not None and error_feats.size > 0:
        if error_feats.ndim == 1:
            error_feats = error_feats.reshape(1, -1)
        all_features.append(error_feats)
        feature_weights.append(0.02)  # 錯誤特徵權重
    
    if not all_features:
        return np.array([])
    
    # 對齊特徵維度
    max_rows = max(f.shape[0] for f in all_features)
    aligned_features = []
    
    for features in all_features:
        if features.shape[0] < max_rows:
            # 重複最後一行以對齊
            padding = np.repeat(features[-1:], max_rows - features.shape[0], axis=0)
            features = np.vstack([features, padding])
        aligned_features.append(features)
    
    # 特徵融合
    if fusion_method == 'concatenate':
        fused_features = np.hstack(aligned_features)
    
    elif fusion_method == 'weighted':
        # 加權平均（需要特徵維度相同）
        normalized_features = []
        target_cols = min(f.shape[1] for f in aligned_features)
        
        for features in aligned_features:
            if features.shape[1] > target_cols:
                # PCA 降維
                from sklearn.decomposition import PCA
                pca = PCA(n_components=target_cols)
                features = pca.fit_transform(features)
            elif features.shape[1] < target_cols:
                # 填充零
                padding = np.zeros((features.shape[0], target_cols - features.shape[1]))
                features = np.hstack([features, padding])
            
            normalized_features.append(features)
        
        # 加權融合
        fused_features = np.zeros_like(normalized_features[0])
        for features, weight in zip(normalized_features, feature_weights):
            fused_features += weight * features
    
    elif fusion_method == 'attention':
        # 注意力機制融合
        fused_features = attention_fusion_enhanced(aligned_features, feature_weights)
    
    else:
        fused_features = np.hstack(aligned_features)
    
    # 降維到目標維度
    if target_dim is not None and fused_features.shape[1] > target_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim)
        fused_features = pca.fit_transform(fused_features)
    
    return fused_features


def attention_fusion_enhanced(features_list, weights):
    """增強的注意力機制特徵融合"""
    import torch
    from sklearn.preprocessing import MinMaxScaler
    
    # 計算注意力權重
    attention_weights = torch.softmax(torch.tensor(weights), dim=0).numpy()
    
    # 標準化特徵維度
    target_cols = min(f.shape[1] for f in features_list)
    normalized_features = []
    
    for features in features_list:
        if features.shape[1] != target_cols:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
            if features.shape[1] > target_cols:
                features = features[:, :target_cols]
            else:
                padding = np.zeros((features.shape[0], target_cols - features.shape[1]))
                features = np.hstack([features, padding])
        
        normalized_features.append(features)
    
    # 注意力加權
    fused = np.zeros_like(normalized_features[0])
    for features, weight in zip(normalized_features, attention_weights):
        fused += weight * features
    
    return fused