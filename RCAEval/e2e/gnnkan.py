"""
E2E GNN-KAN RCA 主入口點 - 模組化架構，專注於KAN取代MLP
核心目標：證明用KAN取代GNN中MLP層是有效的方法（準確率極高）
確保：保留KAN特性，最小化MLP相關性，功能完整
"""

import time
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# 🎯 從模組化組件導入所需的類和函數
import sys
import os

# 添加 RCAEval 目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 模組化導入 - 確保功能完整性
from RCAEval.gnn_kan_module import (
    SimplifiedGNNKANConfig,
    HighCapacityGNNKANConfig,
    FastGNNKANConfig,
    MultiModalFeatureExtractor,
    SimplifiedGraphConstructor,
    GNNKANModel,
    train_gnn_kan_model
)

# 導入新的特徵處理方法
from RCAEval.gnn_kan_module.feature_processing import (
    ica_metric_processing,
    kpca_metric_processing,
    simplified_metric_processing
)

# 簡化的PageRank實現 - 避免外部依賴
class PageRank:
    def __init__(self, alpha=0.85, max_iter=100, tol=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    def fit_transform(self, adj):
        """簡化但有效的 PageRank 實現"""
        try:
            n = adj.shape[0]
            if n == 0:
                return np.array([])
            
            # 歸一化鄰接矩陣
            row_sums = np.sum(adj, axis=1)
            row_sums[row_sums == 0] = 1  # 避免除零
            adj_norm = adj / row_sums[:, np.newaxis]
            
            # 初始化PageRank值
            pr = np.ones(n) / n
            
            # 迭代計算
            for _ in range(self.max_iter):
                pr_new = (1 - self.alpha) / n + self.alpha * np.dot(adj_norm.T, pr)
                if np.linalg.norm(pr_new - pr, 1) < self.tol:
                    break
                pr = pr_new
            
            return pr
        except Exception as e:
            print(f"PageRank計算失敗: {e}，使用度中心性")
            degrees = np.sum(adj, axis=1)
            return degrees / (np.sum(degrees) + 1e-8)


def gnn_kan_rca(data, inject_time=None, dataset=None, with_bg=False, 
                config_type='simplified', feature_method='ica', **kwargs):
    """
    主要的 GNN-KAN RCA 方法 - 使用純粹KAN實現
    
    Args:
        data: 輸入數據 (multimodal 或 單一模態)
        inject_time: 注入時間點
        dataset: 數據集名稱
        with_bg: 是否包含背景數據
        config_type: 配置類型 ('simplified', 'high_capacity', 'fast')
        feature_method: 特徵處理方法 ('ica', 'kpca', 'pca', 'simplified')
        **kwargs: 其他參數
    
    Returns:
        dict: 包含 adj, node_names, ranks 的結果
    """
    print("🔥 使用純粹KAN模組化架構進行RCA分析")
    print(f"🎯 目標：證明用KAN取代MLP的有效性（高準確率）")
    print(f"🔧 特徵方法：{feature_method}，配置類型：{config_type}")
    start_time = time.time()
    
    try:
        # 🎯 1. 選擇適當的配置 - 根據應用場景
        if config_type == 'high_capacity':
            config = HighCapacityGNNKANConfig()
        elif config_type == 'fast':
            config = FastGNNKANConfig()
        else:
            config = SimplifiedGNNKANConfig()
        
        # 設置特徵處理方法
        config.feature_method = feature_method
        config.use_ica = (feature_method == 'ica')
        config.use_kpca = (feature_method == 'kpca')
        
        # 更新配置參數
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 🔧 最大化KAN純粹性
        config.update_for_kan_purity()
        
        # 驗證配置
        config_issues = config.validate_config()
        if config_issues:
            print(f"⚠️ 配置問題：{config_issues}")
        
        # 🎯 2. 特徵提取 - 使用新的特徵處理方法
        print(f"🔧 使用 {feature_method} 特徵提取方法...")
        feature_extractor = MultiModalFeatureExtractor(config)
        features, node_names = feature_extractor.extract_features(data, inject_time)
        
        if features.size == 0 or len(node_names) == 0:
            print("⚠️ 沒有提取到特徵，返回空結果")
            return {"adj": np.array([]), "node_names": [], "ranks": []}
        
        print(f"✓ 提取特徵形狀: {features.shape}, 節點數: {len(node_names)}")
        
        # 🎯 3. 圖構建 - 使用可學習圖結構
        print("🔗 使用模組化圖構建器（支持可學習圖結構）...")
        graph_constructor = SimplifiedGraphConstructor(config)
        edge_index, edge_weights = graph_constructor.build_graph(features, node_names)
        
        print(f"✓ 構建圖：{len(node_names)} 個節點，{edge_index.size(1)} 條邊")
        
        # 🎯 4. 準備節點特徵 - 優化維度匹配
        print("🎯 準備節點特徵...")
        target_dim = config.input_dim
        
        if features.ndim == 2 and features.shape[1] >= target_dim:
            node_features = features[:len(node_names), :target_dim]
        else:
            # 使用統計特徵作為節點特徵
            if features.ndim == 2 and features.shape[0] > 0:
                # 基本統計特徵
                node_stats = []
                for i in range(len(node_names)):
                    col_idx = i % features.shape[1]
                    col_data = features[:, col_idx]
                    
                    stats = [
                        np.mean(col_data),
                        np.std(col_data),
                        np.max(col_data),
                        np.min(col_data),
                        np.median(col_data),
                        np.percentile(col_data, 25),
                        np.percentile(col_data, 75),
                        np.var(col_data)
                    ]
                    node_stats.append(stats)
                
                node_stats = np.array(node_stats)
                
                # 調整到目標維度
                if node_stats.shape[1] < target_dim:
                    padding = np.zeros((len(node_names), target_dim - node_stats.shape[1]))
                    node_features = np.hstack([node_stats, padding])
                else:
                    node_features = node_stats[:, :target_dim]
            else:
                # 隨機初始化 (最後選項)
                node_features = np.random.randn(len(node_names), target_dim) * 0.1
        
        # 確保數值穩定性
        node_features = np.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🎯 5. 初始化純粹KAN模型
        print("🤖 初始化純粹KAN模型（KAN取代MLP）...")
        model = GNNKANModel(config, len(node_names))
        
        # 設備管理
        device = 'cuda' if hasattr(config, 'use_cuda') and config.use_cuda and torch.cuda.is_available() else 'cpu'
        print(f"📱 使用設備: {device}")
        
        # 確保張量在正確設備上
        try:
            if device == 'cuda':
                model = model.cuda()
                edge_index = edge_index.cuda() if hasattr(edge_index, 'cuda') else torch.tensor(edge_index, device='cuda', dtype=torch.long)
                node_features_tensor = torch.tensor(node_features, dtype=torch.float, device='cuda')
            else:
                model = model.cpu()
                edge_index = edge_index.cpu() if hasattr(edge_index, 'cpu') else torch.tensor(edge_index, device='cpu', dtype=torch.long)
                node_features_tensor = torch.tensor(node_features, dtype=torch.float, device='cpu')
        except RuntimeError as e:
            print(f"設備設置失敗: {e}，回退到 CPU")
            device = 'cpu'
            model = model.cpu()
            edge_index = torch.tensor(edge_index, device='cpu', dtype=torch.long) if not isinstance(edge_index, torch.Tensor) else edge_index.cpu()
            node_features_tensor = torch.tensor(node_features, dtype=torch.float, device='cpu')
        
        # 🎯 6. 訓練純粹KAN模型
        print("🏋️ 訓練純粹KAN模型（證明KAN>MLP）...")
        print(f"🎯 KAN配置：grid_size={config.kan_grid_size}, num_basis={config.kan_num_basis}")
        
        model, final_adj = train_gnn_kan_model(
            model, node_features_tensor, edge_index, config
        )
        
        # 🎯 7. 獲取最終結果
        print("📊 獲取KAN模型預測結果...")
        model.eval()
        
        # 確保所有張量在相同設備上
        model_device = next(model.parameters()).device
        node_features_tensor = node_features_tensor.to(model_device)
        edge_index = edge_index.to(model_device)
        
        try:
            with torch.no_grad():
                _, final_adj = model(node_features_tensor, edge_index)
                
                # 檢查KAN模型輸出的有效性
                if torch.isnan(final_adj).any() or torch.isinf(final_adj).any():
                    print("⚠️ KAN模型輸出包含無效值，進行數值修復...")
                    final_adj = torch.nan_to_num(final_adj, nan=0.0, posinf=1.0, neginf=0.0)
                
                # 轉移到CPU進行後處理
                final_adj_np = final_adj.cpu().numpy()
                
        except Exception as e:
            print(f"KAN模型預測失敗: {e}，使用圖構建結果")
            if edge_weights is not None:
                final_adj_np = np.zeros((len(node_names), len(node_names)))
                edge_index_np = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index
                edge_weights_np = edge_weights.cpu().numpy() if isinstance(edge_weights, torch.Tensor) else edge_weights
                
                for i, (src, dst) in enumerate(edge_index_np.T):
                    if i < len(edge_weights_np):
                        final_adj_np[src, dst] = edge_weights_np[i]
            else:
                final_adj_np = np.eye(len(node_names))
        
        # 🎯 8. 計算PageRank排名
        print("📈 計算PageRank排名...")
        pagerank = PageRank()
        ranks = pagerank.fit_transform(final_adj_np)
        
        # 處理結果
        if len(ranks) == 0:
            ranks = np.ones(len(node_names)) / len(node_names)
        
        # 🎯 9. 準備最終結果
        sorted_indices = np.argsort(ranks)[::-1]
        ranked_nodes = [node_names[i] for i in sorted_indices]
        
        execution_time = time.time() - start_time
        print(f"⏱️ KAN模型執行時間: {execution_time:.2f}秒")
        print(f"🏆 Top 5 根因候選: {ranked_nodes[:5]}")
        
        return {
            "adj": final_adj_np,
            "node_names": node_names,
            "ranks": ranked_nodes,
            "scores": ranks[sorted_indices],
            "execution_time": execution_time,
            "model_type": "GNN-KAN",
            "config_type": config_type,
            "feature_method": feature_method
        }
        
    except Exception as e:
        print(f"❌ GNN-KAN RCA 執行失敗: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空結果
        return {
            "adj": np.array([]),
            "node_names": [],
            "ranks": [],
            "scores": [],
            "execution_time": time.time() - start_time,
            "error": str(e)
        }


class GNNKANEndToEnd:
    """
    GNN-KAN 端到端類 - 提供物件導向接口
    專注於KAN取代MLP的核心價值
    """
    
    def __init__(self, config=None):
        if config is None:
            config = SimplifiedGNNKANConfig()
        self.config = config
        print("✅ 模組化 GNN-KAN RCA 入口文件載入成功")
        
    def run_rca(self, data, inject_time=None, dataset=None, with_bg=False, **kwargs):
        """
        運行 RCA 分析
        
        Args:
            data: 輸入數據
            inject_time: 注入時間點
            dataset: 數據集名稱
            with_bg: 是否包含背景數據
            **kwargs: 其他參數
            
        Returns:
            dict: RCA 結果
        """
        # 合併配置參數
        config_dict = {
            'config_type': getattr(self.config, 'config_type', 'simplified'),
            'feature_method': getattr(self.config, 'feature_method', 'ica')
        }
        config_dict.update(kwargs)
        
        return gnn_kan_rca(data, inject_time, dataset, with_bg, **config_dict)
    
    def configure(self, **kwargs):
        """配置參數"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


# 確保模組正確加載
print("✅ 模組化 GNN-KAN RCA 入口文件載入成功")

# 模組導出
__all__ = ['gnn_kan_rca', 'GNNKANEndToEnd', 'PageRank']