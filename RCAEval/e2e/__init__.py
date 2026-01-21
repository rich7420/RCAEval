"""
E2E RCA Package - Clean version
Unified import paths, eliminate duplicate definitions
"""

# Simple @rca decorator implementation (for backward compatibility)
def rca(func):
    """
    Simple RCA decorator, mainly used to mark functions as RCA methods
    In actual use, this decorator does not change function behavior
    """
    func.is_rca_method = True
    return func

# Lazy import GNN-KAN related modules to avoid triggering logs when other methods run
def _lazy_import_gnn_kan():
    """Lazy import GNN-KAN module, only triggered when needed"""
    try:
        from .gnnkan import gnn_kan_rca, GNNKANEndToEnd
        
        # Import dependency components from gnn_kan_module
        from ..gnn_kan_module import (
            SimplifiedGNNKANConfig,
            # HighCapacityGNNKANConfig,  # Removed 
            FastGNNKANConfig,
            MultiModalFeatureExtractor,
            SimplifiedGraphConstructor,
            GNNKANModel,
            train_gnn_kan_model
        )
        
        return True, {
            'gnn_kan_rca': gnn_kan_rca,
            'GNNKANEndToEnd': GNNKANEndToEnd,
            'SimplifiedGNNKANConfig': SimplifiedGNNKANConfig,
            'FastGNNKANConfig': FastGNNKANConfig,
            'MultiModalFeatureExtractor': MultiModalFeatureExtractor,
            'SimplifiedGraphConstructor': SimplifiedGraphConstructor,
            'GNNKANModel': GNNKANModel,
            'train_gnn_kan_model': train_gnn_kan_model
        }
    except ImportError as e:
        return False, {}

# Standard RCA method imports (not involving GNN-KAN)
try:
    # Import all standard RCA methods
    from .baro import baro
    from .circa import circa
    from .pc_pagerank import pc_pagerank, cmlp_pagerank, ntlr_pagerank
    from .pc_randomwalk import pc_randomwalk, ntlr_randomwalk, fci_randomwalk, lingam_randomwalk, granger_randomwalk
    from .fci_pagerank import fci_pagerank
    from .ges_pagerank import ges_pagerank
    from .lingam_pagerank import lingam_pagerank
    from .granger_pagerank import granger_pagerank
    from .run import run
    
    # Pure GNN methods
    try:
        from .gnn import gnn_rca
    except ImportError:
        gnn_rca = None
    
    # GAT method (fair baseline)
    try:
        from .gat import gat_rca
    except ImportError:
        gat_rca = None
    
    # GATv2 method (improved GAT)
    try:
        from .gatv2 import gatv2_rca
    except ImportError:
        gatv2_rca = None
    
    # Graph Transformer method (modern GNN)
    try:
        from .graph_transformer import graph_transformer_rca
    except ImportError:
        graph_transformer_rca = None
    
    # Other methods
    try:
        from .causalai import causalai
    except ImportError:
        causalai = None
    
    try:
        from .cloudranger import cloudranger
    except ImportError:
        cloudranger = None
    
    try:
        from .dummy import dummy
    except ImportError:
        dummy = None
    
    try:
        from .e_diagnosis import e_diagnosis
    except ImportError:
        e_diagnosis = None
    
    try:
        from .easyrca import easyrca
    except ImportError:
        easyrca = None
    
    try:
        from .microcause import microcause
    except ImportError:
        microcause = None
    
    try:
        from .microrank import microrank
    except ImportError:
        microrank = None
    
    try:
        from .mscred import mscred
    except ImportError:
        mscred = None
    
    try:
        from .nsigma import nsigma
    except ImportError:
        nsigma = None
    
    try:
        from .tracerca import tracerca
    except ImportError:
        tracerca = None
    
    try:
        from .ht import ht
    except ImportError:
        ht = None
    
    try:
        from .rcd import rcd
    except ImportError:
        rcd = None
    
    try:
        from .mmrcd import mmrcd
    except ImportError:
        mmrcd = None
    
    try:
        from .micro_diag import micro_diag
    except ImportError:
        micro_diag = None
    
except ImportError as e:
    pass

# Create compatibility functions
def get_gnn_kan_rca():
    """Get GNN-KAN RCA function (lazy import)"""
    success, components = _lazy_import_gnn_kan()
    if success:
        return components['gnn_kan_rca']
    else:
        def fallback_gnn_kan_rca(data, inject_time=None, **kwargs):
            return {
                "adj": [],
                "node_names": [], 
                "ranks": [],
                "error": "Import failed, using fallback"
            }
        return fallback_gnn_kan_rca

# Lazy imported GNN-KAN components
def get_gnn_kan_components():
    """Get all GNN-KAN components (lazy import)"""
    success, components = _lazy_import_gnn_kan()
    if success:
        return components
    else:
        # Return empty components
        return {
            'gnn_kan_rca': get_gnn_kan_rca(),
            'GNNKANEndToEnd': None,
            'SimplifiedGNNKANConfig': None,
            'FastGNNKANConfig': None,
            'MultiModalFeatureExtractor': None,
            'SimplifiedGraphConstructor': None,
            'GNNKANModel': None,
            'train_gnn_kan_model': None
        }

# Unified export list
__all__ = [
    # Decorator
    'rca',
    
    # Standard RCA methods
    'baro',
    'circa',
    'pc_pagerank', 'cmlp_pagerank', 'ntlr_pagerank',
    'pc_randomwalk', 'ntlr_randomwalk', 'fci_randomwalk', 'lingam_randomwalk', 'granger_randomwalk',
    'fci_pagerank',
    'ges_pagerank',
    'lingam_pagerank',
    'granger_pagerank',
    'run',
    
    # Optional methods
    'causalai',
    'cloudranger',
    'dummy',
    'e_diagnosis',
    'easyrca',
    'microcause',
    'microrank',
    'mscred',
    'nsigma',
    'tracerca',
    'ht',
    'rcd',
    'mmrcd',
    'micro_diag',
    
    # GNN methods
    'gnn_rca',
    'gat_rca',
    
    # GNN-KAN lazy import functions
    'get_gnn_kan_rca',
    'get_gnn_kan_components',
]




