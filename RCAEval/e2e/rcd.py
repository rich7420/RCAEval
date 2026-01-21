import time
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.utils.cit import chisq, CIT
from causallearn.utils.PCUtils import SkeletonDiscovery
from sklearn.preprocessing import KBinsDiscretizer
import sys
import os

# Try to import local_skeleton_discovery from lib directory or installed package
HAS_LOCAL_SKELETON = False
local_skeleton_discovery = None

# First try from lib directory (local implementation)
# We need to ensure lib directory is checked before installed packages
lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'lib')
lib_skeleton_path = os.path.join(lib_path, 'causallearn', 'utils', 'PCUtils', 'SkeletonDiscovery.py')

if os.path.exists(lib_skeleton_path):
    # Import directly from lib directory
    import importlib.util
    # Temporarily remove installed causallearn from path to force using lib version
    import causallearn.utils.PCUtils.SkeletonDiscovery as installed_sd
    installed_sd_path = installed_sd.__file__
    
    # Load lib version
    spec = importlib.util.spec_from_file_location("lib_skeleton_discovery", lib_skeleton_path)
    lib_skeleton_module = importlib.util.module_from_spec(spec)
    # Need to handle imports in the lib module
    import causallearn.graph.GraphClass
    import causallearn.utils.cit
    import causallearn.utils.PCUtils.Helper
    # Import necessary modules for lib_skeleton_module
    lib_skeleton_module.CausalGraph = causallearn.graph.GraphClass.CausalGraph
    lib_skeleton_module.chisq = causallearn.utils.cit.chisq
    lib_skeleton_module.gsq = causallearn.utils.cit.gsq
    try:
        from causallearn.utils.PCUtils import Helper
        lib_skeleton_module.append_value = Helper.append_value
    except:
        # Fallback if Helper doesn't have append_value
        def append_value(d, key, value):
            if key not in d:
                d[key] = []
            d[key].append(value)
        lib_skeleton_module.append_value = append_value
    lib_skeleton_module.np = np
    lib_skeleton_module.combinations = __import__('itertools').combinations
    try:
        lib_skeleton_module.tqdm = __import__('tqdm.auto').tqdm
    except:
        lib_skeleton_module.tqdm = lambda x, **kwargs: x
    
    spec.loader.exec_module(lib_skeleton_module)
    
    if hasattr(lib_skeleton_module, 'local_skeleton_discovery'):
        local_skeleton_discovery = lib_skeleton_module.local_skeleton_discovery
        HAS_LOCAL_SKELETON = True
else:
    # Try from installed package
    try:
        from causallearn.utils.PCUtils.SkeletonDiscovery import local_skeleton_discovery
        HAS_LOCAL_SKELETON = True
    except (ImportError, AttributeError):
        # If local_skeleton_discovery is not available, we'll use skeleton_discovery as fallback
        HAS_LOCAL_SKELETON = False
        local_skeleton_discovery = None

from RCAEval.io.time_series import drop_extra

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")


# =========== UTILS.py ====================
# Note: Some of the functions defined here are only used for data
# from sock-shop or real-world application.
CI_TEST = chisq

START_ALPHA = 0.001
ALPHA_STEP = 0.1
ALPHA_LIMIT = 1

VERBOSE = False
F_NODE = "F-node"


def drop_constant(df):
    return df.loc[:, (df != df.iloc[0]).any()]


# Only used for sock-shop and real outage datasets
def preprocess_sock_shop(n_df, a_df, per, dk_select_useful=False):
    _process = lambda df: _select_lat(_scale_down_mem(_rm_time(df)), per)

    n_df = _process(n_df)
    a_df = _process(a_df)

    n_df = drop_constant(n_df)
    a_df = drop_constant(a_df)

    n_df, a_df = _match_columns(n_df, a_df)

    df = add_fnode_and_concat(n_df, a_df)

    if dk_select_useful is True:
        df = _select_useful_cols(df)

    n_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
    a_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    return (n_df, a_df)


def load_datasets(normal, anomalous):
    normal_df = pd.read_csv(normal)
    anomalous_df = pd.read_csv(anomalous)
    return (normal_df, anomalous_df)


def add_fnode_and_concat(normal_df, anomalous_df):
    normal_df[F_NODE] = "0"
    anomalous_df[F_NODE] = "1"
    return pd.concat([normal_df, anomalous_df])


# Run PC (only the skeleton phase) on the given dataset.
# The last column of the data *must* be the F-node
def run_pc(data, alpha, localized=False, labels={}, mi=[], verbose=VERBOSE):
    if labels == {}:
        labels = {i: name for i, name in enumerate(data.columns)}

    np_data = data.to_numpy()
    if localized:
        f_node = np_data.shape[1] - 1
        # Localized PC
        # Try to use local_skeleton_discovery if available, otherwise fall back to skeleton_discovery
        if HAS_LOCAL_SKELETON and local_skeleton_discovery is not None:
            try:
                # For lib version, use CI_TEST (string) directly
                # For installed version, we'd need CIT object, but lib version uses string
                cg = local_skeleton_discovery(
                    np_data,
                    f_node,
                    alpha,
                    indep_test=CI_TEST,
                    mi=mi,
                    labels=labels,
                    verbose=verbose,
                )
            except (AttributeError, TypeError) as e:
                if verbose:
                    print(f"Warning: local_skeleton_discovery failed, using skeleton_discovery: {e}")
                # Fallback: use regular skeleton_discovery
                import inspect
                sig = inspect.signature(SkeletonDiscovery.skeleton_discovery)
                # Create CIT object for the installed causallearn version
                try:
                    cit_obj = CIT(np_data, method=CI_TEST)
                except:
                    cit_obj = CI_TEST
                    
                if 'labels' in sig.parameters:
                    cg = SkeletonDiscovery.skeleton_discovery(
                        np_data,
                        alpha,
                        indep_test=CI_TEST,
                        background_knowledge=None,
                        stable=False,
                        verbose=verbose,
                        labels=labels,
                        show_progress=False,
                    )
                elif 'node_names' in sig.parameters:
                    # Convert labels dict to list of node names
                    node_names = [labels.get(i, str(i)) for i in range(len(labels))]
                    cg = SkeletonDiscovery.skeleton_discovery(
                        np_data,
                        alpha,
                        indep_test=cit_obj,
                        background_knowledge=None,
                        stable=False,
                        verbose=verbose,
                        node_names=node_names,
                        show_progress=False,
                    )
                else:
                    cg = SkeletonDiscovery.skeleton_discovery(
                        np_data,
                        alpha,
                        indep_test=cit_obj,
                        background_knowledge=None,
                        stable=False,
                        verbose=verbose,
                        show_progress=False,
                    )
        else:
            # Fallback: use regular skeleton_discovery if local_skeleton_discovery is not available
            if verbose:
                print("Warning: local_skeleton_discovery not available, using skeleton_discovery")
            # Check if skeleton_discovery accepts labels or node_names parameter
            import inspect
            sig = inspect.signature(SkeletonDiscovery.skeleton_discovery)
            # Create CIT object for the installed causallearn version
            try:
                cit_obj = CIT(np_data, method=CI_TEST)
            except:
                cit_obj = CI_TEST
                
            if 'labels' in sig.parameters:
                cg = SkeletonDiscovery.skeleton_discovery(
                    np_data,
                    alpha,
                    indep_test=CI_TEST,
                    background_knowledge=None,
                    stable=False,
                    verbose=verbose,
                    labels=labels,
                    show_progress=False,
                )
            elif 'node_names' in sig.parameters:
                # Convert labels dict to list of node names
                node_names = [labels.get(i, str(i)) for i in range(len(labels))]
                cg = SkeletonDiscovery.skeleton_discovery(
                    np_data,
                    alpha,
                    indep_test=cit_obj,
                    background_knowledge=None,
                    stable=False,
                    verbose=verbose,
                    node_names=node_names,
                    show_progress=False,
                )
            else:
                cg = SkeletonDiscovery.skeleton_discovery(
                    np_data,
                    alpha,
                    indep_test=cit_obj,
                    background_knowledge=None,
                    stable=False,
                    verbose=verbose,
                    show_progress=False,
                )
    else:
        # Check if skeleton_discovery accepts labels or node_names parameter
        import inspect
        sig = inspect.signature(SkeletonDiscovery.skeleton_discovery)
        # Create CIT object for the installed causallearn version
        try:
            cit_obj = CIT(np_data, method=CI_TEST)
        except:
            cit_obj = CI_TEST
            
        if 'labels' in sig.parameters:
            cg = SkeletonDiscovery.skeleton_discovery(
                np_data,
                alpha,
                indep_test=CI_TEST,
                background_knowledge=None,
                stable=False,
                verbose=verbose,
                labels=labels,
                show_progress=False,
            )
        elif 'node_names' in sig.parameters:
            # Convert labels dict to list of node names
            node_names = [labels.get(i, str(i)) for i in range(len(labels))]
            cg = SkeletonDiscovery.skeleton_discovery(
                np_data,
                alpha,
                indep_test=cit_obj,
                background_knowledge=None,
                stable=False,
                verbose=verbose,
                node_names=node_names,
                show_progress=False,
            )
        else:
            cg = SkeletonDiscovery.skeleton_discovery(
                np_data,
                alpha,
                indep_test=cit_obj,
                background_knowledge=None,
                stable=False,
                verbose=verbose,
                show_progress=False,
            )

    cg.to_nx_graph()
    return cg


def get_fnode_child(G):
    # F_NODE might be named differently in the graph (e.g., "X1", "X2", etc.)
    # Try to find the F-node by checking all nodes
    if F_NODE in G:
        return [*G.successors(F_NODE)]
    # If F_NODE is not found, try to find it by checking node names
    # The F-node is typically the last node (highest index)
    if len(G.nodes()) > 0:
        # Get the last node (F-node should be the last column)
        nodes = list(G.nodes())
        # Try to find node that matches F_NODE pattern or is the last one
        for node in reversed(nodes):
            if str(node) == F_NODE or (isinstance(node, str) and F_NODE in str(node)):
                return [*G.successors(node)]
        # If still not found, return successors of the last node
        if nodes:
            return [*G.successors(nodes[-1])]
    return []


def save_graph(graph, file):
    nx.draw_networkx(graph)
    plt.savefig(file)


def pc_with_fnode(normal_df, anomalous_df, alpha, bins=None, localized=False, verbose=VERBOSE):
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)
    cg = run_pc(data, alpha, localized=localized, verbose=verbose)
    return cg.nx_graph


# Equivelant to \Psi-PC from the main paper
def run_psi_pc(
    normal_df,
    anomalous_df,
    bins=None,
    mi=None,  # TODO: this is just wrong, refactor it
    localized=False,
    start_alpha=None,
    min_nodes=-1,
    verbose=VERBOSE,
):
    """
    Run \Psi-PC on the given dataset.
    The last column of the data *must* be the F-node

    Parameters
    ----------
    normal_df: pd.DataFrame
        Normal data
    anomalous_df: pd.DataFrame
        Anomalous data
    bins: int
        Number of bins to use for discretization
    mi: list
        List of tuples of mutual information
    localized: bool
        Whether to use localized PC
    start_alpha: float
        Starting alpha value
    min_nodes: int
        Minimum number of nodes to order
    verbose: bool
        Whether to print verbose output

    Returns
    -------
    # TODO: refactor this
    """

    if mi is None:
        mi = []
    if 0 in [len(normal_df.columns), len(anomalous_df.columns)]:
        return ([], None, [], 0)
    data = _preprocess_for_fnode(normal_df, anomalous_df, bins)

    if min_nodes == -1:
        # Order all nodes (if possible) except F-node
        min_nodes = len(data.columns) - 1
    assert min_nodes < len(data)

    G = None
    no_ci = 0
    i_to_labels = {i: name for i, name in enumerate(data.columns)}
    labels_to_i = {name: i for i, name in enumerate(data.columns)}

    _preprocess_mi = lambda l: [labels_to_i.get(i) for i in l]  # noqa
    _postprocess_mi = lambda l: [i_to_labels.get(i) for i in list(filter(None, l))]  # noqa
    processed_mi = _preprocess_mi(mi)
    _run_pc = lambda alpha: run_pc(
        data,
        alpha,
        localized=localized,
        mi=processed_mi,
        labels=i_to_labels,
        verbose=verbose,
    )

    rc = []
    _alpha = START_ALPHA if start_alpha is None else start_alpha
    for i in np.arange(_alpha, ALPHA_LIMIT, ALPHA_STEP):
        cg = _run_pc(i)
        G = cg.nx_graph
        # Handle different versions of CausalGraph
        if hasattr(cg, 'no_ci_tests'):
            no_ci += cg.no_ci_tests
        # If no_ci_tests doesn't exist, we can't track it, so skip

        if G is None:
            continue

        f_neigh = get_fnode_child(G)
        # Convert node objects/indices to column names
        new_neigh = []
        for x in f_neigh:
            # Convert node to column name
            if isinstance(x, (int, np.integer)):
                # If it's an integer index, convert to column name
                col_name = i_to_labels.get(x)
                if col_name is not None and col_name not in rc:
                    new_neigh.append(col_name)
            else:
                # If it's already a string/column name
                col_name = str(x)
                if col_name not in rc:
                    new_neigh.append(col_name)
        
        if len(new_neigh) == 0:
            continue
        else:
            # Handle different versions of CausalGraph
            if hasattr(cg, 'p_values') and cg.p_values is not None and len(new_neigh) > 0:
                try:
                    # Convert node names to indices
                    node_indices = []
                    for key in new_neigh:
                        idx = labels_to_i.get(key)
                        if idx is not None:
                            node_indices.append(idx)
                    
                    if len(node_indices) > 0 and len(cg.p_values) > 0:
                        # Access p_values safely
                        if isinstance(cg.p_values, np.ndarray) and len(cg.p_values.shape) >= 2:
                            # Ensure indices are valid
                            valid_indices = [idx for idx in node_indices if 0 <= idx < cg.p_values.shape[1]]
                            if len(valid_indices) == len(node_indices):
                                f_p_values = cg.p_values[-1][node_indices]
                                rc += _order_neighbors(new_neigh, f_p_values)
                            else:
                                # If some indices are invalid, just add neighbors without ordering
                                rc += new_neigh
                        else:
                            # Try dict-like access
                            try:
                                f_p_values = np.array([cg.p_values.get((key,), 0.5) for key in new_neigh])
                                rc += _order_neighbors(new_neigh, f_p_values)
                            except:
                                rc += new_neigh
                    else:
                        rc += new_neigh
                except (IndexError, KeyError, TypeError, AttributeError) as e:
                    # If p_values access fails, just add neighbors without ordering
                    rc += new_neigh
            else:
                # If p_values doesn't exist, just add neighbors without ordering
                rc += new_neigh

        if len(rc) == min_nodes:
            break

    # Handle different versions of CausalGraph
    if hasattr(cg, 'mi'):
        mi_result = _postprocess_mi(cg.mi)
    else:
        mi_result = []
    return (rc, G, mi_result, no_ci)


def _order_neighbors(neigh, p_values):
    _neigh = neigh.copy()
    _p_values = p_values.copy()
    stack = []

    while len(_neigh) != 0:
        i = np.argmax(_p_values)
        node = _neigh[i]
        stack = [node] + stack
        _neigh.remove(node)
        _p_values = np.delete(_p_values, i)
    return stack


# ==================== Private methods =============================

_rm_time = lambda df: df.loc[:, ~df.columns.isin(["time"])]
_list_intersection = lambda l1, l2: [x for x in l1 if x in l2]


def _preprocess_for_fnode(normal_df, anomalous_df, bins):
    df = add_fnode_and_concat(normal_df, anomalous_df)
    if df is None:
        return None

    return _discretize(df, bins) if bins is not None else df


def _select_useful_cols(df):
    i = df.loc[:, df.columns != F_NODE].std() > 1
    cols = i[i].index.tolist()
    cols.append(F_NODE)
    if len(cols) == 1:
        return None
    elif len(cols) == len(df.columns):
        return df

    print(f"Drop {len(df.columns) - len(cols)} columns, left with {len(cols)}")

    return df[cols]


# Only select the metrics that are in both datasets
def _match_columns(n_df, a_df):
    cols = _list_intersection(n_df.columns, a_df.columns)
    return (n_df[cols], a_df[cols])


# Convert all memeory columns to MBs
def _scale_down_mem(df):
    def update_mem(x):
        if not x.name.endswith("_mem"):
            return x
        x /= 1e6
        x = x.astype(int)
        return x

    return df.apply(update_mem)


# Select all the non-latency columns and only select latecy columns
# with given percentaile
def _select_lat(df, per):
    return df.filter(regex=(".*(?<!lat_\d{2})$|_lat_" + str(per) + "$"))


# NOTE: THIS FUNCTION THROWS WARNGINGS THAT ARE SILENCED!
def _discretize(data, bins):
    d = data.iloc[:, :-1]
    discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
    discretizer.fit(d)
    disc_d = discretizer.transform(d)
    disc_d = pd.DataFrame(disc_d, columns=d.columns.values.tolist())
    disc_d[F_NODE] = data[F_NODE].tolist()

    for c in disc_d:
        disc_d[c] = disc_d[c].astype(int)

    return disc_d


# =========== UTILS.py ====================

# np.random.seed(0)

# LOCAL_ALPHA has an effect on execution time. Too strict alpha will produce a sparse graph
# so we might need to run phase-1 multiple times to get up to k elements. Too relaxed alpha
# will give dense graph so the size of the separating set will increase and phase-1 will
# take more time.
# We tried a few different values and found that 0.01 gives the best result in our case
# (between 0.001 and 0.1).
LOCAL_ALPHA = 0.01
DEFAULT_GAMMA = 5


# Split the dataset into multiple subsets
def create_chunks(df, gamma):
    chunks = list()
    names = np.random.permutation(df.columns)
    for i in range(df.shape[1] // gamma + 1):
        chunks.append(names[i * gamma : (i * gamma) + gamma])

    if len(chunks[-1]) == 0:
        chunks.pop()
    return chunks


def run_level(normal_df, anomalous_df, gamma, localized, bins, verbose):
    """
    Run phase-1 of RCD algorithm

    Parameters
    ----------
    normal_df : pandas.DataFrame
        Normal data
    anomalous_df : pandas.DataFrame
        Anomalous data
    gamma : int
        Number of nodes in each subset
    localized : bool
        Run localized version of PSI-PC
    bins : int
        Number of bins
    verbose : bool
        Verbose mode

    Returns
    -------
    f_child_union : list
        List of nodes in the separating set
    mi_union : list
        List of mutual information values
    ci_tests : int
        Number of conditional independence tests
    """
    ci_tests = 0
    chunks = create_chunks(normal_df, gamma)
    if verbose:
        print(f"Created {len(chunks)} subsets")

    f_child_union = []
    mi_union = []
    f_child = []
    for c in chunks:
        # Try this segment with multiple values of alpha until we find at least one node
        rc, _, mi, ci = run_psi_pc(
            normal_df.loc[:, c],
            anomalous_df.loc[:, c],
            bins=bins,
            localized=localized,
            start_alpha=LOCAL_ALPHA,
            min_nodes=1,
            verbose=verbose,
        )
        f_child_union += rc
        mi_union += mi
        ci_tests += ci
        if verbose:
            f_child.append(rc)

    if verbose:
        print(f"Output of individual chunk {f_child}")
        print(f"Total nodes in mi => {len(mi_union)} | {mi_union}")

    return f_child_union, mi_union, ci_tests


def run_multi_phase(normal_df, anomalous_df, gamma, localized, bins, verbose):
    """
    Run RCD algorithm with two phases (phase-1 and phase-2) to find the root causes of the anomaly in the data.

    Parameters
    ----------
    normal_df : pandas.DataFrame
        Normal data
    anomalous_df : pandas.DataFrame
        Anomalous data
    gamma : int
        Number of nodes in each subset
    localized : bool
        Run localized version of PSI-PC
    bins : int
        Number of bins for discretization
    verbose : bool
        Verbose mode


    Returns
    -------
    rc : list
        List of root causes
    """
    # Convert columns to list to ensure it's a list of column names, not Index
    f_child_union = list(normal_df.columns)
    mi_union = []
    i = 0
    prev = len(f_child_union)

    # Phase-1
    while True:
        start = time.time()
        # Ensure f_child_union is a list of column names
        if not isinstance(f_child_union, list):
            f_child_union = list(f_child_union)
        f_child_union, mi, ci_tests = run_level(
            normal_df.loc[:, f_child_union],
            anomalous_df.loc[:, f_child_union],
            gamma,
            localized,
            bins,
            verbose,
        )
        if verbose:
            print(f"Level-{i}: variables {len(f_child_union)} | time {time.time() - start}")
        i += 1
        mi_union += mi
        # Phase-1 with only one level
        # break

        len_child = len(f_child_union)
        # If found gamma nodes or if running the current level did not remove any node
        if len_child <= gamma or len_child == prev:
            break
        prev = len(f_child_union)

    # Phase-2
    mi_union = []
    new_nodes = f_child_union
    # Ensure new_nodes is a list of column names
    if not isinstance(new_nodes, list):
        new_nodes = list(new_nodes)
    rc, _, mi, ci = run_psi_pc(
        normal_df.loc[:, new_nodes],
        anomalous_df.loc[:, new_nodes],
        bins=bins,
        mi=mi_union,
        localized=localized,
        verbose=verbose,
    )
    ci_tests += ci

    # return rc, ci_tests
    return rc


def rcd(
    data,
    inject_time,
    dk_select_useful=False,
    gamma=5,
    localized=True,
    bins=5,
    verbose=False,
    dataset=None,
    seed=None,
    **kwargs,
):
    normal_df = data[data["time"] < inject_time]
    anomal_df = data[data["time"] >= inject_time]

    if dk_select_useful is True:
        normal_df = drop_extra(normal_df)
        anomal_df = drop_extra(anomal_df)

    # if dataset == real outages:
    if dataset == "sock-shop":
        normal_df, anomal_df = preprocess_sock_shop(normal_df, anomal_df, 90, dk_select_useful)
    elif dataset is not None:
        from RCAEval.io.time_series import convert_mem_mb, drop_constant, drop_time, preprocess

        normal_df = drop_constant(convert_mem_mb(drop_time(normal_df)))
        anomal_df = drop_constant(convert_mem_mb(drop_time(anomal_df)))

        normal_df, anomal_df = _match_columns(normal_df, anomal_df)

        df = add_fnode_and_concat(normal_df, anomal_df)
        if dk_select_useful is True:
            df = _select_useful_cols(df)

        normal_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
        anomal_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    if seed is not None:
        np.random.seed(seed)

    rc = run_multi_phase(normal_df, anomal_df, gamma, localized, bins, verbose)
    # return rc
    return {
        "ranks": rc,
    }
