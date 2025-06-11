"""
E2E GNN-KAN RCA 入口文件 - 使用完全整合的 KAN 組件
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

from gnn_kan_module import (
    SimplifiedGNNKANConfig,
    MultiModalFeatureExtractor,
    SimplifiedGraphConstructor,
    GNNKANModel,
    train_gnn_kan_model
)

# 簡化的PageRank實現
class PageRank:
    def __init__(self, alpha=0.85, max_iter=100, tol=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    def fit_transform(self, adj):
        """簡化的 PageRank 實現"""
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

# 備用：從其他模組導入輔助函數
try:
    from io.time_series import preprocess, drop_constant
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


class GNNKANEndToEnd:
    """
    GNN-KAN 端到端根因分析類
    
    這個類封裝了完整的 GNN-KAN RCA 流程，提供統一的接口
    """
    
    def __init__(self, config=None):
        """
        初始化 GNN-KAN End-to-End 系統
        
        Args:
            config: 配置對象，如果為None則使用默認配置
        """
        self.config = config if config is not None else SimplifiedGNNKANConfig()
        print("✅ GNN-KAN End-to-End 系統初始化完成")
    
    def run_rca(self, data, inject_time=None, dataset=None, with_bg=False, **kwargs):
        """
        執行根因分析
        
        Args:
            data: 輸入數據
            inject_time: 故障注入時間
            dataset: 數據集名稱
            with_bg: 是否包含背景數據
            **kwargs: 其他參數
        
        Returns:
            dict: RCA 結果，包含 adj, node_names, ranks
        """
        return gnn_kan_rca(
            data=data,
            inject_time=inject_time,
            dataset=dataset,
            with_bg=with_bg,
            **kwargs
        )
    
    def configure(self, **kwargs):
        """
        更新配置參數
        
        Args:
            **kwargs: 要更新的配置參數
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"✅ 更新配置: {key} = {value}")
            else:
                print(f"⚠️ 未知配置參數: {key}")


# 確保可以被正確導入
__all__ = ['gnn_kan_rca', 'GNNKANEndToEnd', 'PageRank']

print("✅ 模組化 GNN-KAN RCA 入口文件載入成功")

if __name__ == "__main__":
    print("🚀 測試 GNN-KAN 模組化實現")
    
    # 創建測試數據
    test_data = {
        'metrics': pd.DataFrame({
            'cpu_usage': np.random.rand(100),
            'memory_usage': np.random.rand(100),
            'network_io': np.random.rand(100)
        }),
        'traces': pd.DataFrame({
            'serviceName': ['service_a', 'service_b'] * 50,
            'operationName': ['op1', 'op2'] * 50,
            'duration': np.random.lognormal(2, 1, 100),
            'startTime': pd.date_range('2024-01-01', periods=100, freq='1min')
        })
    }
    
    # 創建 E2E 實例並運行測試
    try:
        e2e = GNNKANEndToEnd()
        result = e2e.run_rca(test_data, inject_time=test_data['traces']['startTime'].iloc[50])
        
        print(f"✅ 測試完成！發現 {len(result['ranks'])} 個潛在根因")
        print(f"🎯 前3個根因: {result['ranks'][:3]}")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()