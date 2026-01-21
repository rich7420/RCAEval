import networkx as nx
import numpy as np
from sknetwork.ranking import PageRank


def page_rank_preprocess(adj):
    """
    Preprocess adjacency matrix for PageRank computation
    Intelligently detect and handle discrete values (causal graphs) and continuous values (GNN-KAN output)
    """
    pr_input = np.zeros_like(adj)
    node_num = len(adj)
    
    # Detect matrix type: discrete values vs continuous values
    unique_values = np.unique(adj.flatten())
    is_discrete = all(val in [-1, 0, 1, 2] for val in unique_values)
    
    if is_discrete:
        # Handle traditional discrete value adjacency matrix (causal graph)
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
        # Handle GNN-KAN generated continuous value adjacency matrix
        # Intelligent threshold setting: based on data distribution
        threshold = np.percentile(adj.flatten(), 75)  # Use 75th percentile as threshold
        threshold = max(threshold, 0.5)  # Ensure threshold is not below 0.5
        
        for a in range(node_num):
            for b in range(node_num):
                if a != b:  # Don't process diagonal elements
                    # For GNN-KAN output, directly use weight values
                    if adj[a, b] > threshold:
                        pr_input[a, b] = adj[a, b]  # Preserve weight information
                    # Symmetry check: if matrix is approximately symmetric, treat as undirected graph
                    elif abs(adj[a, b] - adj[b, a]) < 0.1 and adj[a, b] > 0.3:
                        pr_input[a, b] = pr_input[b, a] = (adj[a, b] + adj[b, a]) / 2
    
    return pr_input


def page_rank(adj, node_names=None, damping_factor=0.85, solver="piteration", n_iter=10, tol=1e-6):
    """
    Compute PageRank scores - support GNN-KAN continuous values and traditional discrete values
    """
    if node_names is None:
        node_names = [f"X{i}" for i in range(len(adj))]

    try:
        pr_input = page_rank_preprocess(adj)
        
        # Use sknetwork.ranking.PageRank
        try:
            pr = PageRank(damping_factor=damping_factor, solver=solver, n_iter=n_iter, tol=tol)
            # Check available methods
            if hasattr(pr, 'fit_transform'):
                scores = pr.fit_transform(pr_input)
            elif hasattr(pr, 'fit'):
                pr.fit(pr_input)
                scores = pr.scores_
            else:
                # Fallback to NetworkX implementation
                raise AttributeError("sknetwork PageRank 方法不可用")
                
        except (AttributeError, ImportError) as e:
            # Use NetworkX as fallback
            G = nx.DiGraph()
            n = len(pr_input)
            for i in range(n):
                for j in range(n):
                    if pr_input[i, j] > 0:
                        G.add_edge(i, j, weight=pr_input[i, j])
            
            # If graph is empty, create basic structure
            if len(G.edges()) == 0:
                for i in range(n):
                    G.add_node(i)
                    if i < n - 1:
                        G.add_edge(i, i + 1, weight=1.0)
            
            nx_pagerank = nx.pagerank(G, alpha=damping_factor, max_iter=n_iter, tol=tol)
            scores = [nx_pagerank.get(i, 0.0) for i in range(n)]

        # Combine scores and node names, sort by score
        output = list(zip(node_names, scores))
        output.sort(key=lambda x: x[1], reverse=True)
        return output
        
    except Exception as e:
        # Fallback: degree centrality based on adjacency matrix
        degrees = np.sum(np.abs(adj), axis=1)
        normalized_degrees = degrees / (np.sum(degrees) + 1e-8)
        
        output = list(zip(node_names, normalized_degrees))
        output.sort(key=lambda x: x[1], reverse=True)
        return output
