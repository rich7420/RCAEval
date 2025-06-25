import networkx as nx
import numpy as np
from sknetwork.ranking import PageRank


def page_rank_preprocess(adj):
    """
    預處理鄰接矩陣用於PageRank計算
    智能檢測並處理離散值（因果圖）和連續值（GNN-KAN輸出）
    """
    pr_input = np.zeros_like(adj)
    node_num = len(adj)
    
    # 檢測矩陣類型：離散值 vs 連續值
    unique_values = np.unique(adj.flatten())
    is_discrete = all(val in [-1, 0, 1, 2] for val in unique_values)
    
    if is_discrete:
        # 處理傳統離散值鄰接矩陣（因果圖）
        for a in range(node_num):
            for b in range(node_num):
                # case 1 no edge: a b
                if adj[a, b] == adj[b, a] == 0:
                    pass
                # case 2 undirected a -- b
                elif adj[a, b] == adj[b, a] == -1:
                    pr_input[a, b] = pr_input[b, a] = 1
                # case 3 directed a -> b
                elif adj[a, b] == 1 and adj[b, a] == -1:
                    pr_input[a, b] = 1
                # case 4 directed a <- b
                elif adj[a, b] == -1 and adj[b, a] == 1:
                    pr_input[b, a] = 1
                elif adj[a, b] == 0 and adj[b, a] == 1:
                    pr_input[a, b] = 0
                    pr_input[b, a] = 1
                elif adj[a, b] == 1 and adj[b, a] == 0:
                    pr_input[a, b] = 1
                    pr_input[b, a] = 0
                elif adj[a, b] == 1 and adj[b, a] == 1:
                    # a <-> b, in FCI
                    pr_input[a, b] = 1
                    pr_input[b, a] = 1
                elif adj[a, b] == 2 and adj[b, a] == 1:
                    # a o-> b, in FCI
                    pr_input[a, b] = 1
                    pr_input[b, a] = 0
                elif adj[a, b] == 1 and adj[b, a] == 2:
                    # a <-0 b, in FCI
                    pr_input[a, b] = 0
                    pr_input[b, a] = 1
                elif adj[a, b] == 2 and adj[b, a] == 2:
                    # a o-o b, in FCI
                    pr_input[a, b] = 1
                    pr_input[b, a] = 1
    else:
        # 🎯 處理GNN-KAN生成的連續值鄰接矩陣
        print(f"✓ 檢測到GNN-KAN連續值鄰接矩陣，值域: [{np.min(adj):.3f}, {np.max(adj):.3f}]")
        
        # 智能閾值設定：基於數據分佈
        threshold = np.percentile(adj.flatten(), 75)  # 使用75分位數作為閾值
        threshold = max(threshold, 0.5)  # 確保閾值不低於0.5
        
        for a in range(node_num):
            for b in range(node_num):
                if a != b:  # 不處理對角線元素
                    # 對於GNN-KAN輸出，直接使用權重值
                    if adj[a, b] > threshold:
                        pr_input[a, b] = adj[a, b]  # 保留權重信息
                    # 對稱性檢查：如果矩陣近似對稱，視為無向圖
                    elif abs(adj[a, b] - adj[b, a]) < 0.1 and adj[a, b] > 0.3:
                        pr_input[a, b] = pr_input[b, a] = (adj[a, b] + adj[b, a]) / 2
        
        print(f"✓ 連續值預處理完成，閾值: {threshold:.3f}")
    
    return pr_input


def page_rank(adj, node_names=None, damping_factor=0.85, solver="piteration", n_iter=10, tol=1e-6):
    """
    計算 PageRank 分數 - 支持 GNN-KAN 連續值和傳統離散值
    """
    if node_names is None:
        node_names = [f"X{i}" for i in range(len(adj))]

    try:
        pr_input = page_rank_preprocess(adj)
        
        # 使用 sknetwork.ranking.PageRank
        try:
            pr = PageRank(damping_factor=damping_factor, solver=solver, n_iter=n_iter, tol=tol)
            # 檢查可用方法
            if hasattr(pr, 'fit_transform'):
                scores = pr.fit_transform(pr_input)
            elif hasattr(pr, 'fit'):
                pr.fit(pr_input)
                scores = pr.scores_
            else:
                # 回退到 NetworkX 實現
                raise AttributeError("sknetwork PageRank 方法不可用")
                
        except (AttributeError, ImportError) as e:
            print(f"⚠️ sknetwork PageRank 失敗: {e}，使用 NetworkX 實現")
            # 使用 NetworkX 作為後備
            G = nx.DiGraph()
            n = len(pr_input)
            for i in range(n):
                for j in range(n):
                    if pr_input[i, j] > 0:
                        G.add_edge(i, j, weight=pr_input[i, j])
            
            # 如果圖為空，創建基本結構
            if len(G.edges()) == 0:
                for i in range(n):
                    G.add_node(i)
                    if i < n - 1:
                        G.add_edge(i, i + 1, weight=1.0)
            
            nx_pagerank = nx.pagerank(G, alpha=damping_factor, max_iter=n_iter, tol=tol)
            scores = [nx_pagerank.get(i, 0.0) for i in range(n)]

        # 合併分數和節點名稱，按分數排序
        output = list(zip(node_names, scores))
        output.sort(key=lambda x: x[1], reverse=True)
        return output
        
    except Exception as e:
        print(f"⚠️ PageRank計算失敗: {e}，使用簡化排序")
        # 後備方案：基於鄰接矩陣的度中心性
        degrees = np.sum(np.abs(adj), axis=1)
        normalized_degrees = degrees / (np.sum(degrees) + 1e-8)
        
        output = list(zip(node_names, normalized_degrees))
        output.sort(key=lambda x: x[1], reverse=True)
        return output
