import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from sknetwork.ranking import PageRank

from RCAEval.graph_construction.pc import pc_default
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.io.time_series import preprocess, drop_constant, drop_near_constant
from RCAEval.e2e import rca


def _check_singular_matrix(data):
    """Check if data correlation matrix is singular"""
    try:
        corr_matrix = np.corrcoef(data.T)
        # Check for NaN or infinite values
        if np.isnan(corr_matrix).any() or np.isinf(corr_matrix).any():
            return True
        # Check if matrix is singular (determinant close to 0)
        det = np.linalg.det(corr_matrix)
        return abs(det) < 1e-10
    except:
        return True


def _remove_constant_columns(df):
    """Remove constant columns and near-constant columns, return processed DataFrame and corresponding column names"""
    # Remove completely constant columns
    df = drop_constant(df)
    # Remove near-constant columns (standard deviation close to 0)
    std_threshold = 1e-6
    non_constant_cols = []
    for col in df.columns:
        if df[col].std() > std_threshold:
            non_constant_cols.append(col)
    return df[non_constant_cols], non_constant_cols


def pc_pagerank(
    data, inject_time=None, dataset=None, dk_select_useful=False, with_bg=False, n_iter=10, **kwargs
):
    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    
    # Enhanced data preprocessing: remove constant columns
    data = drop_constant(data)
    data = drop_near_constant(data, threshold=0.01)
    
    # Ensure data has enough columns
    if len(data.columns) < 2:
        # If too few columns, return empty result
        return {
            "adj": np.zeros((len(data.columns), len(data.columns))),
            "node_names": data.columns.to_list(),
            "ranks": data.columns.to_list(),
        }
    
    node_names = data.columns.to_list()
    
    # Remove constant columns (maintain DataFrame format to correctly track column names)
    data_cleaned, valid_cols = _remove_constant_columns(data)
    if len(valid_cols) < 2:
        return {
            "adj": np.zeros((len(node_names), len(node_names))),
            "node_names": node_names,
            "ranks": node_names,
        }
    
    # Update node_names to match processed data
    node_names = valid_cols
    data_np = data_cleaned.to_numpy().astype(float)
    
    # Try different independence tests
    cg = None
    indep_tests = [
        ('fisherz', fisherz),
        ('gsq', gsq),
        ('chisq', chisq),
    ]
    
    for test_name, indep_test in indep_tests:
        try:
            # Check if correlation matrix is singular (only for fisherz)
            if test_name == 'fisherz' and _check_singular_matrix(data_np):
                continue
            
            cg = pc(
                data=data_np,
                node_names=node_names,
                indep_test=indep_test,
                alpha=0.05,
                show_progress=False,
            )
            break
        except (ValueError, np.linalg.LinAlgError) as e:
            if "singular" in str(e).lower() or "correlation matrix" in str(e).lower():
                continue
            else:
                # Other errors, re-raise
                raise
        except Exception as e:
            continue
    
    # If all tests fail, return empty result
    if cg is None:
        return {
            "adj": np.zeros((len(node_names), len(node_names))),
            "node_names": node_names,
            "ranks": node_names,
        }
    
    adj = cg.G.graph
    G = nx.DiGraph()
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] == -1:
                G.add_edge(i, j)
            if adj[i, j] == 1:
                G.add_edge(j, i)
    nodes = sorted(G.nodes())
    adj = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))

    pagerank = PageRank()
    scores = pagerank.fit_transform(adj.T)
    ranks = list(zip(node_names, scores))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def cmlp_pagerank(
    data, inject_time=None, dataset=None, dk_select_useful=False, with_bg=False, n_iter=10, **kwargs
):
    from RCAEval.graph_construction.cmlp import cmlp

    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    node_names = data.columns.to_list()

    adj = cmlp(data, max_iter=20000)

    pagerank = PageRank()
    scores = pagerank.fit_transform(adj.T)
    ranks = list(zip(node_names, scores))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def ntlr_pagerank(
    data, inject_time=None, dataset=None, dk_select_useful=False, with_bg=False, n_iter=10, **kwargs
):
    from RCAEval.graph_construction.dag_gnn import notears_low_rank

    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    node_names = data.columns.to_list()

    adj = notears_low_rank(data)
    pagerank = PageRank()
    scores = pagerank.fit_transform(adj.T)
    ranks = list(zip(node_names, scores))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
