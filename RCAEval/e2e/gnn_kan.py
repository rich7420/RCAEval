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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import traceback

# Import our optimized KAN modules
from RCAEval.kan import (
    OptimizedGNNKANEncoder, UltraFastKANLayer, 
    GradientStabilizer, StabilizedKANLayer,
    sliding_window_alignment, extract_log_features,
    stl_decomposition, kll_feature_processing,
    compute_topology_features, extract_error_features,
    feature_fusion, extract_trace_features,
    build_service_dependency_graph, extract_service_topology_features
)
from RCAEval.io.time_series import preprocess, drop_constant

warnings.filterwarnings("ignore")


class GNNKANConfig:
    """Configuration class for GNN-KAN parameters"""
    
    def __init__(self):
        # 🚀 高容量模型架構 - 恢復並提升原始複雜度
        self.input_dim = 128   # 大幅提升輸入維度
        self.hidden_dims = [256, 192, 128, 96]  # 更深更寬的4層隱藏層
        self.output_dim = 64   # 更大的輸出維度
        
        # 🔑 保持高表達能力的KAN設置 - 不降低
        self.kan_grid_size = 5        # 保持原始 B-spline 網格點數
        self.kan_spline_order = 3     # 保持 3次樣條的表達能力
        self.num_gnn_layers = 3       # 保持 3層 GNN 的深度
        self.dropout = 0.1            # 降低dropout保持更多資訊
        
        # 🛡️ 高級梯度穩定策略 - 核心穩定技術
        self.use_residual_connections = True     # 殘差連接穩定深層梯度
        self.use_layer_norm = True              # 層標準化替代BatchNorm
        self.use_gradient_checkpointing = True  # 梯度檢查點節省記憶體
        self.use_spectral_norm = True           # 譜標準化控制Lipschitz常數
        self.use_warmup_scheduler = True        # 預熱學習率調度
        self.use_orthogonal_init = True         # 正交初始化
        self.use_weight_normalization = True    # 權重標準化
        
        # 🎯 精密調優的訓練參數 - 高容量訓練
        self.epochs = 100             # 增加訓練輪數獲得更好效果
        self.batch_size = 8           # 較小批次更穩定
        self.base_learning_rate = 5e-5  # 保守的基礎學習率
        self.max_learning_rate = 2e-4   # 適中的最大學習率
        self.weight_decay = 5e-7        # 極輕的權重衰減
        self.warmup_epochs = 20         # 更長的預熱階段
        self.scheduler_step_size = 25
        self.scheduler_gamma = 0.8      # 溫和的學習率衰減
        
        # 🛡️ 梯度穩定核心參數
        self.gradient_clip_norm = 1.0    # 嚴格的梯度裁剪
        self.gradient_accumulation_steps = 4  # 梯度累積增加有效批次大小
        self.spectral_norm_power_iterations = 3  # 譜標準化迭代次數
        self.layer_norm_eps = 1e-6      # 層標準化數值穩定參數
        
        # 🧠 智能初始化策略
        self.kan_init_method = 'xavier_uniform'  # KAN層的智能初始化
        self.spline_coeff_init_std = 0.02       # 極保守的B-spline係數初始化
        self.gnn_init_method = 'orthogonal'     # GNN層正交初始化
        self.bias_init_method = 'zeros'         # 偏置初始化為零
        
        # 🔬 數值穩定性檢查
        self.enable_nan_detection = True        # 啟用NaN檢測
        self.enable_inf_detection = True        # 啟用Inf檢測
        self.stability_check_frequency = 10     # 每10個epoch檢查一次穩定性
        self.emergency_fallback = True          # 啟用緊急回退機制
        
        # 🎛️ 自適應正則化
        self.adaptive_l1_lambda = True          # 自適應L1正則化強度
        self.adaptive_entropy_lambda = True     # 自適應熵正則化強度
        self.base_l1_lambda = 1e-6             # 基礎L1正則化
        self.base_entropy_lambda = 1e-6        # 基礎熵正則化
        self.max_l1_lambda = 1e-4              # 最大L1正則化
        self.max_entropy_lambda = 1e-4         # 最大熵正則化
        
        # Feature extraction - 保持豐富特徵
        self.window_size = 30         # 增加窗口大小獲得更多上下文
        self.step_size = 1
        self.use_dla = True           # 啟用DLA獲得更好特徵
        self.max_log_features = 300   # 增加特徵數量
        self.stl_seasonal = 12        # 更長的季節性週期
        self.stl_period = 24          # 更長的週期
        self.stl_robust = True
        self.kll_k = 512             # 增加KLL特徵維度
        self.fusion_method = 'attention'  # 使用注意力機制
        self.target_feature_dim = 128     # 增加目標維度
        
        # PCA settings - 保持更多資訊
        self.use_pca = True
        self.pca_components = 64      # 增加主成分數量
        self.pca_variance_threshold = 0.98  # 保留更多方差
        
        # Graph construction - 更豐富的圖結構
        self.similarity_threshold = 0.25  # 降低閾值增加連接
        self.max_edges_per_node = 8       # 增加每個節點的最大邊數
        self.use_self_loops = True
        self.use_attention_edges = True   # 注意力邊權重
        self.edge_dropout = 0.1          # 邊dropout防止過擬合
        
        # Hardware optimization
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.mixed_precision = True      # 混合精度訓練
        self.compile_model = True        # PyTorch 2.0 模型編譯
        
        # Output
        self.top_k_results = 20
        
        # 🚨 緊急模式設置 (當檢測到梯度問題時自動啟用)
        self.emergency_mode = False
        self.emergency_learning_rate = 1e-5
        self.emergency_gradient_clip = 0.5
        self.emergency_dropout = 0.3


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
    
    def _apply_pca_with_variance_check(self, features, feature_type, target_components=None):
        """
        應用 PCA 降維，包含方差檢查
        
        Args:
            features: 輸入特徵矩陣
            feature_type: 特徵類型標識
            target_components: 目標降維維度
            
        Returns:
            降維後的特徵矩陣
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            if features.size == 0:
                return features
                
            # 確保特徵矩陣有足夠的樣本和特徵
            n_samples, n_features = features.shape
            if n_samples < 2 or n_features < 2:
                print(f"⚠️ {feature_type}: 特徵矩陣太小 ({n_samples}x{n_features})，跳過 PCA")
                return features
            
            # 設置目標組件數
            if target_components is None:
                target_components = min(self.config.pca_components, n_features, n_samples)
            else:
                target_components = min(target_components, n_features, n_samples)
            
            if target_components >= n_features:
                print(f"⚠️ {feature_type}: 目標維度 {target_components} >= 原始維度 {n_features}，跳過 PCA")
                return features
            
            # 標準化
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # 檢查方差
            feature_var = np.var(features_scaled, axis=0)
            valid_features = feature_var > 1e-8
            
            if not np.any(valid_features):
                print(f"⚠️ {feature_type}: 所有特徵方差過小，跳過 PCA")
                return features
            
            # 過濾低方差特徵
            features_filtered = features_scaled[:, valid_features]
            if features_filtered.shape[1] <= target_components:
                print(f"⚠️ {feature_type}: 過濾後特徵數 {features_filtered.shape[1]} <= 目標維度 {target_components}")
                return features_filtered
            
            # 應用 PCA
            pca = PCA(n_components=target_components, random_state=42)
            features_pca = pca.fit_transform(features_filtered)
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"✓ {feature_type}: PCA {features.shape[1]} -> {target_components}, 解釋方差: {explained_variance:.3f}")
            
            return features_pca
            
        except Exception as e:
            print(f"⚠️ {feature_type}: PCA 失敗 {e}，返回原始特徵")
            return features
    
    def _extract_multimodal_features(self, data, inject_time):
        """處理多模態數據 - 參考 traceRCA 的 trace 處理方式"""
        all_features = []
        node_names = []
        
        print("Processing multimodal data...")
        
        # 🔧 1. 首先檢查並處理基本的 metric 數據 (確保有基礎特徵)
        basic_features_extracted = False
        
        # 處理滑動窗口對齊
        if inject_time is not None:
            print("Applying sliding window alignment...")
            try:
                windows, timestamps = sliding_window_alignment(
                    data, 
                    window_size=self.config.window_size, 
                    step_size=self.config.step_size,
                    timestamp_col='time'
                )
                print(f"Created {len(windows)} windows for analysis")
            except Exception as e:
                print(f"⚠️ Sliding window alignment failed: {e}, using raw data")
                windows = [data]
                timestamps = [None]
        else:
            windows = [data]
            timestamps = [None]
        
        # 對每個窗口進行特徵提取
        for window_idx, window_data in enumerate(windows):
            window_features = []
            window_node_names = []
            
            # 🔧 2. 處理 trace 數據 - 參考 traceRCA 的方法
            trace_data_found = False
            if 'trace' in window_data or 'traces' in window_data:
                trace_key = 'trace' if 'trace' in window_data else 'traces'
                trace_data = window_data[trace_key]
                trace_data_found = True
            elif isinstance(window_data, pd.DataFrame):
                # 檢查 DataFrame 是否包含 trace 相關列 - 參考 traceRCA 的列名檢查
                trace_columns = ['serviceName', 'operationName', 'startTime', 'duration', 'traceID', 'spanID']
                if any(col in window_data.columns for col in trace_columns):
                    trace_data = window_data
                    trace_data_found = True
                    print(f"Detected trace data in DataFrame format for window {window_idx}")
            
            if trace_data_found:
                print(f"Extracting trace features from window {window_idx}...")
                try:
                    # 🔧 3. 參考 traceRCA 的 trace 處理邏輯
                    trace_features, operation_names, service_graph = self._extract_trace_features_tracerca_style(
                        trace_data, inject_time, window_idx
                    )
                    
                    if trace_features.size > 0:
                        # PCA降維處理trace特徵
                        pca_trace_features = self._apply_pca_with_variance_check(
                            trace_features, f'trace_window_{window_idx}', target_components=64
                        )
                        window_features.append(pca_trace_features)
                        window_node_names.extend([f'trace_{name}' for name in operation_names[:pca_trace_features.shape[1]]])
                        basic_features_extracted = True
                    
                    # 提取服務拓扑特徵
                    if service_graph is not None:
                        service_topo_features, service_names = extract_service_topology_features(service_graph)
                        if service_topo_features.size > 0:
                            pca_service_features = self._apply_pca_with_variance_check(
                                service_topo_features, f'service_topo_window_{window_idx}', target_components=32
                            )
                            window_features.append(pca_service_features)
                            window_node_names.extend([f'service_topo_{name}' for name in service_names[:pca_service_features.shape[1]]])
                            
                except Exception as e:
                    print(f"⚠️ Trace feature extraction failed: {e}, continuing without trace features")
            
            # 🔧 4. 處理 metric 數據 (確保有基礎特徵)
            metric_data_processed = False
            if 'metric' in window_data or 'metrics' in window_data:
                metric_key = 'metric' if 'metric' in window_data else 'metrics'
                metric_data = window_data[metric_key]
                metric_data_processed = True
            elif isinstance(window_data, pd.DataFrame) and not trace_data_found:
                # 如果沒有找到 trace 數據，將 DataFrame 作為 metric 數據處理
                metric_data = window_data
                metric_data_processed = True
            
            if metric_data_processed:
                try:
                    # 預處理 metric 數據
                    if inject_time is not None and 'time' in metric_data.columns:
                        normal_df = metric_data[metric_data['time'] < inject_time]
                        anomal_df = metric_data[metric_data['time'] >= inject_time]
                        
                        if not normal_df.empty and not anomal_df.empty:
                            normal_processed = preprocess(normal_df, dataset='default')
                            anomal_processed = preprocess(anomal_df, dataset='default')
                            
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
                        processed_features, kll_names = kll_feature_processing(
                            stl_features, sketch_size=self.config.kll_k
                        )
                        
                        if self.config.use_pca and processed_features.shape[1] > self.config.pca_components:
                            processed_features = self._apply_pca_with_variance_check(
                                processed_features, f"metric_window_{window_idx}"
                            )
                        
                        window_features.append(processed_features)
                        window_node_names.extend([f'w{window_idx}_{name}' for name in kll_names])
                        basic_features_extracted = True
                        print(f"✓ Extracted {len(kll_names)} metric features")
                        
                except Exception as e:
                    print(f"⚠️ Metric feature extraction failed: {e}")
            
            # 🔧 5. 處理 log 數據
            if 'log' in window_data or 'logs' in window_data:
                log_key = 'log' if 'log' in window_data else 'logs'
                log_data = window_data[log_key]
                
                try:
                    log_features, log_names = extract_log_features(
                        log_data,
                        use_dla=self.config.use_dla,
                        max_features=self.config.max_log_features
                    )
                    
                    if log_features.size > 0:
                        if self.config.use_pca and log_features.shape[1] > self.config.pca_components:
                            log_features = self._apply_pca_with_variance_check(
                                log_features, f"log_window_{window_idx}"
                            )
                        
                        window_features.append(log_features)
                        window_node_names.extend([f'w{window_idx}_{name}' for name in log_names])
                        print(f"✓ Extracted {len(log_names)} log features")
                        
                except Exception as e:
                    print(f"⚠️ Log feature extraction failed: {e}")
            
            # 收集當前窗口的特徵
            if window_features:
                all_features.extend(window_features)
                node_names.extend(window_node_names)
        
        # 🔧 6. 如果沒有提取到任何基礎特徵，使用合成特徵
        if not basic_features_extracted:
            print("⚠️ No basic features extracted, generating synthetic features...")
            n_samples = 100
            n_features = 20
            synthetic_features = np.random.randn(n_samples, n_features)
            synthetic_names = [f'synthetic_feature_{i}' for i in range(n_features)]
            
            all_features.append(synthetic_features)
            node_names.extend(synthetic_names)
            print(f"✓ Generated {len(synthetic_names)} synthetic features")
        
        if not all_features:
            print("No features extracted, returning empty arrays")
            return np.array([]), []
        
        # 繼續原有的特徵對齊和融合邏輯...
        return self._finalize_features(all_features, node_names)


    def _extract_trace_features_tracerca_style(self, trace_data, inject_time, window_idx):
        """
        參考 traceRCA 的方式提取 trace 特徵
        
        Args:
            trace_data: trace 數據 (DataFrame)
            inject_time: 故障注入時間
            window_idx: 窗口索引
            
        Returns:
            trace_features: 提取的 trace 特徵
            operation_names: 操作名稱列表
            service_graph: 服務依賴圖
        """
        try:
            # 🔧 1. 數據格式檢查和適配 - 參考 traceRCA
            if not isinstance(trace_data, pd.DataFrame):
                print(f"⚠️ Trace data is not DataFrame, attempting conversion...")
                try:
                    trace_data = pd.DataFrame(trace_data)
                except:
                    print(f"⚠️ Cannot convert trace data to DataFrame")
                    return np.array([]), [], None
            
            # 🔧 2. 列名標準化 - 參考 traceRCA 的列名映射
            column_mapping = {
                'service_name': 'serviceName',
                'service': 'serviceName', 
                'operation_name': 'operationName',
                'operation': 'operationName',
                'method_name': 'operationName',
                'method': 'operationName',
                'start_time': 'startTime',
                'timestamp': 'startTime',
                'time': 'startTime',
                'trace_id': 'traceID',
                'span_id': 'spanID'
            }
            
            # 應用列名映射
            for old_col, new_col in column_mapping.items():
                if old_col in trace_data.columns and new_col not in trace_data.columns:
                    trace_data[new_col] = trace_data[old_col]
            
            # 🔧 3. 確保必要的列存在
            required_columns = ['serviceName', 'operationName', 'startTime', 'duration']
            missing_columns = [col for col in required_columns if col not in trace_data.columns]
            
            if missing_columns:
                print(f"Missing trace columns: {missing_columns}, attempting to create defaults...")
                
                # 創建缺失的列
                if 'serviceName' not in trace_data.columns:
                    if 'service' in trace_data.columns:
                        trace_data['serviceName'] = trace_data['service']
                    else:
                        trace_data['serviceName'] = f'service_{window_idx}'
                
                if 'operationName' not in trace_data.columns:
                    if 'methodName' in trace_data.columns:
                        trace_data['operationName'] = trace_data['methodName']
                    elif 'operation' in trace_data.columns:
                        trace_data['operationName'] = trace_data['operation']
                    else:
                        trace_data['operationName'] = 'default_operation'
                
                if 'startTime' not in trace_data.columns:
                    if 'timestamp' in trace_data.columns:
                        trace_data['startTime'] = trace_data['timestamp']
                    else:
                        # 創建時間序列
                        trace_data['startTime'] = pd.date_range('2024-01-01', periods=len(trace_data), freq='1s')
                
                if 'duration' not in trace_data.columns:
                    # 創建合理的duration值
                    trace_data['duration'] = np.random.lognormal(2, 1, len(trace_data))
            
            # 🔧 4. 數據清理和驗證
            # 確保 duration 是數值型
            if 'duration' in trace_data.columns:
                trace_data['duration'] = pd.to_numeric(trace_data['duration'], errors='coerce')
                trace_data['duration'] = trace_data['duration'].fillna(trace_data['duration'].mean())
            
            # 確保 startTime 是時間型
            if 'startTime' in trace_data.columns:
                try:
                    trace_data['startTime'] = pd.to_datetime(trace_data['startTime'])
                except:
                    print("⚠️ Cannot convert startTime to datetime, creating default timestamps")
                    trace_data['startTime'] = pd.date_range('2024-01-01', periods=len(trace_data), freq='1s')
            
            # 🔧 5. 調用原有的 trace 特徵提取函數
            trace_features, operation_names, service_graph = extract_trace_features(
                trace_data, inject_time
            )
            
            # 🔧 6. 特徵驗證和後處理
            if trace_features.size > 0:
                # 確保特徵矩陣不包含 NaN 或 Inf
                trace_features = np.nan_to_num(trace_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # 確保操作名稱列表長度正確
                if len(operation_names) != trace_features.shape[1]:
                    print(f"⚠️ Operation names length {len(operation_names)} != features width {trace_features.shape[1]}")
                    # 調整操作名稱
                    if len(operation_names) < trace_features.shape[1]:
                        operation_names.extend([f'trace_feature_{i}' for i in range(len(operation_names), trace_features.shape[1])])
                    else:
                        operation_names = operation_names[:trace_features.shape[1]]
            
            return trace_features, operation_names, service_graph
                
        except Exception as e:
            print(f"⚠️ TraceRCA-style extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), [], None


    def _finalize_features(self, all_features, node_names):
        """
        最终化特征处理
        
        Args:
            all_features: 所有特徵列表
            node_names: 节点名称列表
            
        Returns:
            fused_features: 融合后的特征
            node_names: 最终的节点名称列表
        """
        if not all_features:
            return np.array([]), []
        
        # 对齐所有特征的长度
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
                # 重复最后一行
                padding = np.repeat(features[-1:], min_length - features.shape[0], axis=0)
                features = np.vstack([features, padding])
            aligned_features.append(features)
        
        if not aligned_features:
            return np.array([]), []
        
        # 计算拓扑特征
        print("Computing topology features...")
        if len(aligned_features) > 1:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                
                combined_features = np.hstack(aligned_features)
                if combined_features.shape[1] > 1:
                    similarity_matrix = cosine_similarity(combined_features.T)
                    adj_matrix = (similarity_matrix > 0.5).astype(float)
                    
                    topology_features, topology_names = compute_topology_features(
                        adj_matrix, node_names[:adj_matrix.shape[0]]
                    )
                    
                    if topology_features.size > 0:
                        topo_features_expanded = np.tile(topology_features, (min_length, 1))
                        
                        if self.config.use_pca and topo_features_expanded.shape[1] > self.config.pca_components:
                            topo_features_expanded = self._apply_pca_with_variance_check(
                                topo_features_expanded, "topology_features"
                            )
                        
                        aligned_features.append(topo_features_expanded)
                        node_names.extend([f'topology_{name}' for name in topology_names])
            except Exception as e:
                print(f"⚠️ Topology feature computation failed: {e}")
        
        # 特征融合
        print("Performing feature fusion...")
        
        # 分离不同类型的特征进行融合
        log_feats = None
        metric_feats = None
        topo_feats = None
        error_feats = None
        trace_feats = None
        service_topo_feats = None
        
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
        
        # 使用改進的特徵融合 - 修正參數順序
        fused_features = enhanced_feature_fusion(
            log_feats, metric_feats, topo_feats, error_feats, trace_feats, service_topo_feats,
            fusion_method=self.config.fusion_method,
            target_dim=self.config.target_feature_dim
        )
        
        if fused_features.size == 0:
            # 回退方案：直接拼接
            fused_features = np.hstack(aligned_features)
        
        # 最终PCA降维确保特征维度符合要求
        if self.config.use_pca and fused_features.shape[1] > self.config.target_feature_dim:
            print(f"Applying final PCA: {fused_features.shape[1]} -> {self.config.target_feature_dim}")
            fused_features = self._apply_pca_with_variance_check(
                fused_features, "final_fusion", target_components=self.config.target_feature_dim
            )
        
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
            processed_features, kll_names = kll_feature_processing(
                stl_features, sketch_size=self.config.kll_k
            )
            return processed_features, node_names
        else:
            return np.array([]), []
    
    def _compute_loss_with_stabilization(self, predictions, targets, epoch=0):
        """
        計算帶穩定化的損失函數
        
        Args:
            predictions: 預測結果
            targets: 目標值
            epoch: 當前訓練輪次
            
        Returns:
            stabilized_loss: 穩定化後的損失值
        """
        try:
            # 基礎損失計算
            base_loss = F.mse_loss(predictions, targets)
            
            # 梯度懲罰項
            gradient_penalty = 0.0
            if predictions.requires_grad:
                gradients = torch.autograd.grad(
                    outputs=predictions.sum(),
                    inputs=predictions,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0] if predictions.requires_grad else torch.zeros_like(predictions)
                
                gradient_penalty = torch.mean(gradients ** 2)
            
            # 動態權重調整
            stability_weight = max(0.1, 1.0 - epoch * 0.01)
            
            # 總損失
            total_loss = base_loss + stability_weight * gradient_penalty
            
            # 數值穩定性檢查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ Unstable loss detected at epoch {epoch}, using base loss")
                return base_loss
            
            return total_loss
            
        except Exception as e:
            print(f"⚠️ Loss computation failed: {e}")
            return F.mse_loss(predictions, targets)


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
        
        # 🔧 設備管理優化
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
        
        # 🔧 關鍵修正：確保最終評估時的設備一致性
        print("Getting final adjacency matrix...")
        model.eval()
        
        # 確保所有張量在相同設備上
        model_device = next(model.parameters()).device
        node_features = node_features.to(model_device)
        edge_index = edge_index.to(model_device)
        
        try:
            with torch.no_grad():
                _, final_adj = model(node_features, edge_index)
                
                # 檢查結果
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
            
            # 🔧 關鍵修正：確保返回字符串列表，處理可能的嵌套列表
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
            
            # 🔧 關鍵修正：確保返回字符串列表，處理可能的嵌套列表
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
        final_adj: 最終鄰接矩陣
    """
    print("Starting GNN-KAN model training...")
    
    # 梯度穩定化器 - 修正參數傳遞
    stabilizer = GradientStabilizer(
        l1_lambda=config.base_l1_lambda,
        entropy_lambda=config.base_entropy_lambda,
        grad_clip_value=config.gradient_clip_norm,
        pruning_threshold=1e-2,
        enable_dynamic_scaling=True,
        stability_check_freq=config.stability_check_frequency
    )
    
    # 優化器設置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.base_learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # 學習率調度器
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.warmup_epochs,
        T_mult=2,
        eta_min=config.base_learning_rate * 0.01
    )
    
    model.train()
    successful_epochs = 0
    loss_history = []
    
    try:
        for epoch in range(config.epochs):
            optimizer.zero_grad()
            
            try:
                # 前向傳播
                node_embeddings, adj_scores = model(node_features, edge_index)
                
                # 檢查輸出是否為 NaN 或 Inf
                if torch.isnan(node_embeddings).any() or torch.isinf(node_embeddings).any():
                    print(f"NaN/Inf in embeddings at epoch {epoch}, skipping...")
                    continue
                
                if torch.isnan(adj_scores).any() or torch.isinf(adj_scores).any():
                    print(f"NaN/Inf in adj_scores at epoch {epoch}, skipping...")
                    continue
                
                # 計算基礎損失
                try:
                    base_loss = compute_loss_stable(node_embeddings, adj_scores, edge_index, config)
                except RuntimeError as e:
                    print(f"Loss computation failed at epoch {epoch}: {e}")
                    continue
                
                # 使用梯度穩定化器計算總損失（包含正則化）
                total_loss = stabilizer.compute_total_regularization_loss(model, base_loss)
                
                # 檢查 loss 是否為 NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Invalid total loss at epoch {epoch}: {total_loss.item()}")
                    continue
                
                # 反向傳播 (添加異常處理)
                try:
                    total_loss.backward()
                except RuntimeError as e:
                    print(f"Backward pass failed at epoch {epoch}: {e}")
                    continue
                
                # 應用梯度裁剪和檢查
                grad_norm = stabilizer.apply_gradient_clipping(model, clip_type='norm')
                
                # 數值穩定性檢查
                if epoch % stabilizer.stability_check_freq == 0:
                    stability_report = stabilizer.check_numerical_stability(model)
                    if stability_report['gradient_nan_count'] > 0 or stability_report['gradient_inf_count'] > 0:
                        print(f"⚠️ Epoch {epoch}: Gradient stability issues detected")
                
                # 梯度爆炸檢查 - 使用更嚴格的閾值
                if grad_norm > 8.0:
                    if epoch % 50 == 0:
                        print(f"Large gradient norm {grad_norm:.3f} at epoch {epoch}, applying stabilization...")
                
                # 優化器更新
                optimizer.step()
                scheduler.step()
                
                successful_epochs += 1
                loss_history.append(total_loss.item())
                
                # 動態調整正則化強度
                if len(loss_history) >= 10:
                    stabilizer.adaptive_regularization_scaling(total_loss.item(), loss_history)
                
                # 動態剪枝 (每50個epoch執行一次)
                if successful_epochs % 50 == 0 and successful_epochs > 0:
                    pruning_ratio = stabilizer.apply_dynamic_pruning(model)
                    if pruning_ratio > 0:
                        print(f"Epoch {epoch}: Applied pruning, ratio: {pruning_ratio:.3f}")
                
                if epoch % 20 == 0 or successful_epochs <= 5:
                    print(f"Epoch {epoch}/{config.epochs}, Base Loss: {base_loss.item():.6f}, "
                          f"Total Loss: {total_loss.item():.6f}, Grad norm: {grad_norm:.6f}")
                    print(f"L1 λ: {stabilizer.l1_lambda:.6f}, Entropy λ: {stabilizer.entropy_lambda:.6f}")
                    
                    # GPU 記憶體監控
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated()/1024**3
                        cached = torch.cuda.memory_reserved()/1024**3
                        print(f"GPU memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
                        
                        # 如果記憶體使用過高，切換到CPU
                        if allocated > 8.0:
                            print("GPU memory usage too high, switching to CPU...")
                            return train_on_cpu_fallback(model, node_features, edge_index, config)
                            
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"CUDA error at epoch {epoch}: {e}")
                    print("Attempting CPU fallback...")
                    return train_on_cpu_fallback(model, node_features, edge_index, config)
                else:
                    print(f"Error in epoch {epoch}: {e}")
                    continue
                    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        if "CUDA" in str(e):
            print("CUDA error detected, falling back to CPU...")
            return train_on_cpu_fallback(model, node_features, edge_index, config)
        import traceback
        traceback.print_exc()
    
    print(f"Training completed with {successful_epochs} successful epochs out of {config.epochs}")
    
    # 打印最終穩定性報告
    final_report = stabilizer.get_stability_report()
    if "message" not in final_report:
        print("=== 梯度穩定性報告 ===")
        print(f"平均梯度範數: {final_report['gradient_statistics']['mean_grad_norm']:.6f}")
        print(f"最大梯度範數: {final_report['gradient_statistics']['max_grad_norm']:.6f}")
        print(f"梯度裁剪次數: {final_report['gradient_statistics']['gradient_clips']}")
        print(f"穩定性違規次數: {final_report['stability_violations']}")
        print(f"最終 L1 λ: {final_report['regularization_config']['l1_lambda']:.6f}")
        print(f"最終熵 λ: {final_report['regularization_config']['entropy_lambda']:.6f}")
    
    # 如果成功訓練的epoch太少，使用簡化策略
    if successful_epochs < 5:
        print("Too few successful epochs, using simplified adjacency calculation...")
        return train_on_cpu_fallback(model, node_features, edge_index, config)
    
    # 獲取最終的鄰接矩陣 (添加安全機制)
    print("Getting final adjacency matrix...")
    model.eval()
    try:
        with torch.no_grad():
            _, final_adj = model(node_features, edge_index)
            
            # 檢查結果
            if torch.isnan(final_adj).any() or torch.isinf(final_adj).any():
                print("NaN/Inf in final adjacency, using fallback...")
                num_nodes = node_features.size(0)
                final_adj = torch.eye(num_nodes, device=node_features.device) * 0.8
                final_adj += torch.rand(num_nodes, num_nodes, device=node_features.device) * 0.2
                
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error in final evaluation: {e}")
            return train_on_cpu_fallback(model, node_features, edge_index, config)
        else:
            print(f"Error getting final adjacency: {e}")
            num_nodes = node_features.size(0)
            final_adj = torch.eye(num_nodes, device=node_features.device)
    
    return model, final_adj


def train_on_cpu_fallback(model, node_features, edge_index, config):
    """CPU 回退訓練函數 - 確保設備一致性"""
    print("=== CPU FALLBACK MODE ===")
    
    # 🔧 確保所有數據都移動到 CPU 並且設備一致
    try:
        # 強制移動到 CPU
        if hasattr(model, 'cpu'):
            model = model.cpu()
        if hasattr(node_features, 'cpu'):
            node_features = node_features.cpu()
        if hasattr(edge_index, 'cpu'):
            edge_index = edge_index.cpu()
        
        # 簡化模型結構以適應 CPU
        print("Using simplified training for CPU...")
        
        # 獲取節點數量
        if hasattr(node_features, 'size'):
            num_nodes = node_features.size(0)
        elif hasattr(node_features, 'shape'):
            num_nodes = node_features.shape[0]
        else:
            num_nodes = len(node_features)
        
        # 基於特徵相似性構建鄰接矩陣
        with torch.no_grad():
            # 確保 node_features 是正確的 tensor 格式
            if not isinstance(node_features, torch.Tensor):
                node_features = torch.tensor(node_features, dtype=torch.float, device='cpu')
            else:
                node_features = node_features.to('cpu')
            
            # 計算餘弦相似性 - 確保所有張量在CPU上
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # 應用閾值和sigmoid
            final_adj = torch.sigmoid(similarity_matrix * 3.0) # 增強對比度
            
            # 確保對角線為高值 (自相似性)
            final_adj.fill_diagonal_(0.9)
            
            # 確保結果在CPU上
            final_adj = final_adj.cpu()
            
    except Exception as e:
        print(f"CPU fallback also failed: {e}")
        # 最終回退：恆等矩陣
        try:
            if hasattr(node_features, 'size'):
                num_nodes = node_features.size(0)
            elif hasattr(node_features, 'shape'):
                num_nodes = node_features.shape[0]
            else:
                num_nodes = len(node_features)
        except:
            num_nodes = 10  # 默認值
            
        final_adj = torch.eye(num_nodes, device='cpu')
    
    print("CPU fallback completed")
    # 確保返回的模型也在 CPU 上
    if hasattr(model, 'cpu'):
        model = model.cpu()
    return model, final_adj


def compute_loss_stable(node_embeddings, adj_scores, edge_index, config):
    """
    數值穩定的損失計算
    
    Args:
        node_embeddings: 節點嵌入
        adj_scores: 鄰接矩陣分數
        edge_index: 邊索引
        config: 配置參數
        
    Returns:
        total_loss: 總損失
    """
    num_nodes = node_embeddings.size(0)
    device = node_embeddings.device
    
    # 1. 圖重建損失 (使用更穩定的版本)
    true_adj = torch.zeros(num_nodes, num_nodes, device=device)
    if edge_index.size(1) > 0:
        # 確保索引在有效範圍內
        valid_indices = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        if valid_indices.any():
            valid_edge_index = edge_index[:, valid_indices]
            true_adj[valid_edge_index[0], valid_edge_index[1]] = 1.0
    
    # 裁剪 adj_scores 以避免數值不穩定
    adj_scores_clipped = torch.clamp(adj_scores, min=1e-7, max=1-1e-7)
    
    # 使用穩定的二元交叉熵
    reconstruction_loss = F.binary_cross_entropy(adj_scores_clipped, true_adj, reduction='mean')
    
    # 2. 嵌入正則化損失 (使用更溫和的正則化)
    embedding_reg = torch.norm(node_embeddings, p=2, dim=1).mean()
    
    # 3. 稀疏性損失 (鼓勵稀疏的鄰接矩陣)
    sparsity_loss = torch.norm(adj_scores, p=1) / (num_nodes * num_nodes)
    
    # 檢查各個損失項
    if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
        reconstruction_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    if torch.isnan(embedding_reg) or torch.isinf(embedding_reg):
        embedding_reg = torch.tensor(0.0, device=device)
    
    if torch.isnan(sparsity_loss) or torch.isinf(sparsity_loss):
        sparsity_loss = torch.tensor(0.0, device=device)
    
    # 總損失 (使用更小的權重)
    total_loss = reconstruction_loss + 0.001 * embedding_reg + 0.0001 * sparsity_loss
    
    return total_loss


def enhanced_feature_fusion(log_feats, metric_feats, topo_feats, error_feats, trace_feats, service_topo_feats, 
                           fusion_method='attention', target_dim=128):
    """
    增強的特徵融合，整合多模態特徵
    
    Args:
        log_feats: 日誌特徵
        metric_feats: 指標特徵  
        topo_feats: 拓撲特徵
        error_feats: 錯誤特徵
        trace_feats: trace 特徵
        service_topo_feats: 服務拓撲特徵
        fusion_method: 融合方法
        target_dim: 目標維度
    
    Returns:
        融合後的特徵
    """
    features_list = []
    feature_names = []
    
    # 收集所有可用的特徵
    if log_feats is not None and log_feats.size > 0:
        features_list.append(log_feats)
        feature_names.append('log')
    
    if metric_feats is not None and metric_feats.size > 0:
        features_list.append(metric_feats)
        feature_names.append('metric')
        
    if topo_feats is not None and topo_feats.size > 0:
        features_list.append(topo_feats)
        feature_names.append('topology')
        
    if error_feats is not None and error_feats.size > 0:
        features_list.append(error_feats)
        feature_names.append('error')
        
    if trace_feats is not None and trace_feats.size > 0:
        features_list.append(trace_feats)
        feature_names.append('trace')
        
    if service_topo_feats is not None and service_topo_feats.size > 0:
        features_list.append(service_topo_feats)
        feature_names.append('service_topology')
    
    if not features_list:
        print("⚠️ 沒有可用的特徵進行融合")
        return np.array([])
    
    # 對齊特徵長度
    min_length = min(f.shape[0] for f in features_list)
    aligned_features = []
    
    for features in features_list:
        if features.shape[0] > min_length:
            aligned_features.append(features[:min_length])
        elif features.shape[0] < min_length:
            # 重複最後一行來填充
            padding = np.repeat(features[-1:], min_length - features.shape[0], axis=0)
            aligned_features.append(np.vstack([features, padding]))
        else:
            aligned_features.append(features)
    
    # 根據融合方法進行融合
    if fusion_method == 'attention':
        try:
            # 注意力機制融合
            weights = [1.0] * len(aligned_features)  # 均等權重
            fused_features = attention_fusion_enhanced(aligned_features, weights)
            print(f"✓ 使用注意力機制融合 {len(feature_names)} 種特徵")
        except Exception as e:
            print(f"⚠️ 注意力融合失敗: {e}，使用簡單拼接")
            fused_features = np.hstack(aligned_features)
    else:
        # 簡單拼接
        fused_features = np.hstack(aligned_features)
        print(f"✓ 簡單拼接融合 {len(feature_names)} 種特徵")
    
    # PCA 降維到目標維度
    if fused_features.shape[1] > target_dim:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim, random_state=42)
            fused_features = pca.fit_transform(fused_features)
            print(f"✓ PCA 降維到目標維度: {target_dim}")
        except Exception as e:
            print(f"⚠️ PCA 降維失敗: {e}")
    
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


def _apply_pca_with_variance_check(features, feature_type, target_components=None, variance_threshold=0.95):
    """
    應用PCA降維並檢查方差保留
    
    Args:
        features: 輸入特徵矩陣
        feature_type: 特徵類型 (用於日誌)
        target_components: 目標主成分數量
        variance_threshold: 最小方差保留比例
        
    Returns:
        pca_features: PCA降維後的特徵
    """
    if target_components is None:
        target_components = 64  # 默認值
        
    try:
        # 確保有足夠的樣本進行PCA
        n_samples, n_features = features.shape
        max_components = min(n_samples, n_features, target_components)
        
        if max_components < 2:
            print(f"⚠️ {feature_type}: Insufficient samples/features for PCA, keeping original")
            return features
        
        # 標準化特徵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 檢查是否有常數特徵
        if np.allclose(features_scaled.var(axis=0), 0):
            print(f"⚠️ {feature_type}: All features are constant, keeping original")
            return features
        
        # 應用PCA
        pca = PCA(n_components=max_components)
        pca_features = pca.fit_transform(features_scaled)
        
        # 檢查保留的方差比例
        variance_ratio = np.sum(pca.explained_variance_ratio_)
        
        if variance_ratio < variance_threshold:
            print(f"⚠️ {feature_type}: PCA variance ratio {variance_ratio:.3f} < threshold {variance_threshold}, keeping original")
            return features
        else:
            print(f"✓ {feature_type}: PCA {n_features} -> {max_components} features, variance ratio: {variance_ratio:.3f}")
            return pca_features
            
    except Exception as e:
        print(f"⚠️ {feature_type}: PCA failed ({e}), keeping original features")
        return features


def _compute_loss_with_stabilization(output, target, epoch):
    """
    計算帶穩定化的損失函數
    大幅改進版：更保守的損失計算，更好的數值穩定性
    """
    try:
        # 確保輸出和目標的數值穩定性
        output = torch.clamp(output, min=-10, max=10)  # 嚴格限制輸出範圍
        target = torch.clamp(target, min=-10, max=10)
        
        # 基礎損失 - 使用更穩定的 Huber 損失
        huber_loss = nn.HuberLoss(delta=0.5)  # 減小 delta 提高穩定性
        base_loss = huber_loss(output, target)
        
        # 檢查基礎損失的有效性
        if torch.isnan(base_loss) or torch.isinf(base_loss):
            print(f"Invalid base loss detected: {base_loss}, using fallback")
            base_loss = torch.tensor(1.0, device=output.device, requires_grad=True)
        
        # 極度保守的正則化
        l1_reg = torch.tensor(0.0, device=output.device)
        entropy_reg = torch.tensor(0.0, device=output.device)
        
        # 漸進式正則化強度
        base_l1_lambda = 0.001  # 更小的基礎值
        base_entropy_lambda = 0.001
        
        # 只在後期階段增加正則化
        if epoch > 15:
            progress = min((epoch - 15) / 50.0, 1.0)
            l1_lambda = base_l1_lambda * progress
            entropy_lambda = base_entropy_lambda * progress
            
            # 計算 L1 正則化 (僅針對可能存在的參數)
            try:
                l1_params = []
                for param in output.view(-1)[:min(100, output.numel())]:  # 限制參數數量
                    if param.requires_grad:
                        l1_params.append(param)
                
                if l1_params:
                    l1_reg = sum(torch.abs(p).sum() for p in l1_params) * l1_lambda
                    l1_reg = torch.clamp(l1_reg, max=base_loss.item())  # 不超過基礎損失
            except:
                l1_reg = torch.tensor(0.0, device=output.device)
            
            # 計算熵正則化 (更安全的實現)
            try:
                if output.numel() > 0:
                    output_prob = torch.softmax(output.view(-1)[:min(100, output.numel())], dim=0)
                    # 使用數值穩定的熵計算
                    log_prob = torch.log(output_prob + 1e-8)
                    entropy_reg = -torch.sum(output_prob * log_prob) * entropy_lambda
                    entropy_reg = torch.clamp(entropy_reg, max=base_loss.item())
            except:
                entropy_reg = torch.tensor(0.0, device=output.device)
        
        # 檢查所有正則化項
        if torch.isnan(l1_reg) or torch.isinf(l1_reg):
            l1_reg = torch.tensor(0.0, device=output.device)
        if torch.isnan(entropy_reg) or torch.isinf(entropy_reg):
            entropy_reg = torch.tensor(0.0, device=output.device)
        
        # 總損失 - 使用非常保守的組合
        total_loss = base_loss + 0.001 * l1_reg + 0.001 * entropy_reg
        
        # 最終檢查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ Final loss check failed at epoch {epoch}, using base loss")
            total_loss = base_loss
        
        # 梯度爆炸保護
        total_loss = torch.clamp(total_loss, max=100.0)
        
        return total_loss
            
    except Exception as e:
        print(f"⚠️ Loss computation failed: {e}")
        # 緊急回退：簡單的MSE損失
        try:
            fallback_loss = F.mse_loss(output, target)
            if torch.isnan(fallback_loss) or torch.isinf(fallback_loss):
                return torch.tensor(1.0, device=output.device, requires_grad=True)
            return fallback_loss
        except:
            return torch.tensor(1.0, device=output.device, requires_grad=True)


# 導出主要函數供外部使用
__all__ = [
    'gnn_kan_rca',
    'GNNKANConfig', 
    'MultiModalFeatureExtractor',
    'GraphConstructor',
    'GNNKANModel',
    'train_gnn_kan_model',
    'enhanced_feature_fusion'
]