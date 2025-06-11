"""
E2E GNN-KAN RCA 入口文件 - 使用完全模組化的實現
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# 🎯 從模組化結構導入所有需要的函數
from RCAEval.gnn_kan_module import (
    SimplifiedGNNKANConfig,
    MultiModalFeatureExtractor,
    GNNKANModel, 
    train_gnn_kan_model,
    SimplifiedGraphConstructor,
    enhanced_feature_fusion,
    compute_service_criticality_weights
)

from RCAEval.gnn_kan_module.feature_processing import (
    simplified_metric_processing,
    enhanced_trace_processing,
    psm_metric_processing,
    gnn_kan_rca as modularized_gnn_kan_rca
)

from RCAEval.io.time_series import preprocess, drop_constant
from sknetwork.ranking import PageRank


def gnn_kan_rca(data, inject_time=None, dataset=None, with_bg=False, **kwargs):
    """
    主要的 GNN-KAN RCA 方法 - 使用完全模組化的實現
    
    Args:
        data: 輸入數據 (multimodal 或 單一模態)
        inject_time: 注入時間點
        dataset: 數據集名稱
        with_bg: 是否包含背景數據
        **kwargs: 其他參數
    
    Returns:
        dict: 包含 adj, node_names, ranks 的結果
    """
    print("🔥 使用完全模組化的 GNN-KAN RCA 實現")
    start_time = time.time()
    
    try:
        # 使用模組化的配置
        config = SimplifiedGNNKANConfig()
        
        # 更新配置參數
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 🎯 1. 特徵提取 - 使用模組化特徵提取器
        print("🔧 使用模組化特徵提取器...")
        feature_extractor = MultiModalFeatureExtractor(config)
        features, node_names = feature_extractor.extract_features(data, inject_time)
        
        if features.size == 0 or len(node_names) == 0:
            print("⚠️ 沒有提取到特徵，返回空結果")
            return {"adj": np.array([]), "node_names": [], "ranks": []}
        
        print(f"✓ 提取特徵形狀: {features.shape}, 節點數: {len(node_names)}")
        
        # 🎯 2. 圖構建 - 使用模組化圖構建器
        print("🔗 使用模組化圖構建器...")
        graph_constructor = SimplifiedGraphConstructor(config)
        edge_index, edge_weights = graph_constructor.build_graph(features, node_names)
        
        print(f"✓ 構建圖：{len(node_names)} 個節點，{edge_index.size(1)} 條邊")
        
        # 🎯 3. 準備節點特徵
        print("🎯 準備節點特徵...")
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
        
        # 🎯 4. 初始化並訓練模組化模型
        print("🤖 使用模組化 GNN-KAN 模型...")
        model = GNNKANModel(config, len(node_names))
        
        # 設備管理
        device = 'cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu'
        print(f"📱 使用設備: {device}")
        
        if device == 'cuda':
            try:
                model = model.cuda()
                edge_index = edge_index.cuda()
                node_features_tensor = torch.tensor(node_features, dtype=torch.float).cuda()
            except RuntimeError as e:
                print(f"CUDA 初始化失敗: {e}，回退到 CPU")
                device = 'cpu'
                model = model.cpu()
                edge_index = edge_index.cpu()
                node_features_tensor = torch.tensor(node_features, dtype=torch.float).cpu()
        else:
            model = model.cpu()
            node_features_tensor = torch.tensor(node_features, dtype=torch.float).cpu()
            edge_index = edge_index.cpu()
        
        # 🎯 5. 訓練模型
        print("🏋️ 訓練模組化模型...")
        model, final_adj = train_gnn_kan_model(
            model, node_features_tensor, edge_index, config
        )
        
        # 🎯 6. 獲取最終鄰接矩陣
        print("📊 獲取最終鄰接矩陣...")
        model.eval()
        
        # 確保所有張量在相同設備上
        model_device = next(model.parameters()).device
        node_features_tensor = node_features_tensor.to(model_device)
        edge_index = edge_index.to(model_device)
        
        try:
            with torch.no_grad():
                _, final_adj = model(node_features_tensor, edge_index)
                
                # 檢查結果有效性
                if torch.isnan(final_adj).any() or torch.isinf(final_adj).any():
                    print("⚠️ 最終鄰接矩陣包含 NaN/Inf，使用回退策略...")
                    num_nodes = node_features_tensor.size(0)
                    final_adj = torch.eye(num_nodes, device=model_device) * 0.8
                    final_adj += torch.rand(num_nodes, num_nodes, device=model_device) * 0.2
                    
        except RuntimeError as e:
            if "CUDA" in str(e) or "device" in str(e).lower():
                print(f"設備錯誤: {e}，強制切換到 CPU")
                model = model.cpu()
                node_features_tensor = node_features_tensor.cpu()
                edge_index = edge_index.cpu()
                
                with torch.no_grad():
                    try:
                        _, final_adj = model(node_features_tensor, edge_index)
                    except:
                        num_nodes = node_features_tensor.size(0)
                        final_adj = torch.eye(num_nodes, device='cpu')
            else:
                print(f"獲取最終鄰接矩陣時出錯: {e}")
                num_nodes = node_features_tensor.size(0)
                final_adj = torch.eye(num_nodes, device=model_device)
        
        # 🎯 7. 計算 PageRank 排名
        print("🎯 計算 PageRank 排名...")
        adj_numpy = final_adj.detach().cpu().numpy()
        
        try:
            # 使用 PageRank 算法
            pagerank = PageRank()
            scores = pagerank.fit_transform(adj_numpy)
            
            # 獲取排名
            ranked_indices = np.argsort(scores)[::-1]
            top_k_indices = ranked_indices[:config.top_k_results]
            
            # 確保返回字符串列表
            ranks = []
            for i in top_k_indices:
                if i < len(node_names):
                    node_name = node_names[i]
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
            print(f"PageRank 計算失敗: {e}，使用度中心性")
            degrees = np.sum(adj_numpy, axis=1)
            ranked_indices = np.argsort(degrees)[::-1]
            top_k_indices = ranked_indices[:config.top_k_results]
            
            ranks = []
            for i in top_k_indices:
                if i < len(node_names):
                    node_name = node_names[i]
                    if isinstance(node_name, (list, tuple)):
                        if len(node_name) > 0:
                            ranks.append(str(node_name[0]))
                        else:
                            ranks.append(f"node_{i}")
                    else:
                        ranks.append(str(node_name))
                else:
                    ranks.append(f"node_{i}")
        
        # 🎯 8. 組織結果
        result = {
            "adj": adj_numpy,
            "node_names": node_names,
            "ranks": ranks
        }
        
        end_time = time.time()
        print(f"✅ 模組化 GNN-KAN RCA 完成，耗時 {end_time - start_time:.2f} 秒")
        print(f"🎯 前 5 個根因分析結果: {ranks[:5]}")
        
        return result
        
    except KeyboardInterrupt:
        print("❌ 用戶中斷訓練")
        return {"adj": np.array([]), "node_names": [], "ranks": []}
        
    except Exception as e:
        print(f"❌ 模組化 GNN-KAN RCA 出現嚴重錯誤: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "adj": np.array([]),
            "node_names": [],
            "ranks": []
        }


# 確保可以被正確導入
__all__ = ['gnn_kan_rca']

print("✅ 模組化 GNN-KAN RCA 入口文件載入成功")