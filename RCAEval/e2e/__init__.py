"""
E2E RCA Package - 清理版本
統一導入路徑，消除重複定義
"""

# 🔧 簡單的 @rca 裝飾器實現（用於向後兼容）
def rca(func):
    """
    簡單的 RCA 裝飾器，主要用於標記函數為 RCA 方法
    在實際使用中這個裝飾器不會改變函數行為
    """
    func.is_rca_method = True
    return func

# 🎯 延遲導入 GNN-KAN 相關模組，避免在其他方法運行時觸發 log
def _lazy_import_gnn_kan():
    """延遲導入 GNN-KAN 模組，只在需要時觸發"""
    try:
        from .gnnkan import gnn_kan_rca, GNNKANEndToEnd
        
        # 🔧 從gnn_kan_module導入依賴組件
        from ..gnn_kan_module import (
            SimplifiedGNNKANConfig,
            HighCapacityGNNKANConfig, 
            FastGNNKANConfig,
            MultiModalFeatureExtractor,
            SimplifiedGraphConstructor,
            GNNKANModel,
            train_gnn_kan_model
        )
        
        print("✅ E2E GNN-KAN模組載入成功 - 主進入點：gnnkan.py")
        return True, {
            'gnn_kan_rca': gnn_kan_rca,
            'GNNKANEndToEnd': GNNKANEndToEnd,
            'SimplifiedGNNKANConfig': SimplifiedGNNKANConfig,
            'HighCapacityGNNKANConfig': HighCapacityGNNKANConfig,
            'FastGNNKANConfig': FastGNNKANConfig,
            'MultiModalFeatureExtractor': MultiModalFeatureExtractor,
            'SimplifiedGraphConstructor': SimplifiedGraphConstructor,
            'GNNKANModel': GNNKANModel,
            'train_gnn_kan_model': train_gnn_kan_model
        }
    except ImportError as e:
        print(f"❌ GNN-KAN導入失敗: {e}")
        return False, {}

# 🔧 標準 RCA 方法導入（不涉及 GNN-KAN）
try:
    # 導入所有標準 RCA 方法
    from .baro import baro
    from .circa import circa
    from .pc_pagerank import pc_pagerank, cmlp_pagerank, ntlr_pagerank
    from .pc_randomwalk import pc_randomwalk, ntlr_randomwalk, fci_randomwalk, lingam_randomwalk, granger_randomwalk
    from .fci_pagerank import fci_pagerank
    from .ges_pagerank import ges_pagerank
    from .lingam_pagerank import lingam_pagerank
    from .granger_pagerank import granger_pagerank
    from .run import run
    
    # 其他方法
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
    
    print("✅ 標準 RCA 方法導入成功")
    
except ImportError as e:
    print(f"❌ 標準 RCA 方法導入失敗: {e}")

# 🎯 創建兼容性函數
def get_gnn_kan_rca():
    """獲取 GNN-KAN RCA 函數（延遲導入）"""
    success, components = _lazy_import_gnn_kan()
    if success:
        return components['gnn_kan_rca']
    else:
        def fallback_gnn_kan_rca(data, inject_time=None, **kwargs):
            print("⚠️ 使用回退實現 - 請檢查gnn_kan_module安裝")
            return {
                "adj": [],
                "node_names": [], 
                "ranks": [],
                "error": "Import failed, using fallback"
            }
        return fallback_gnn_kan_rca

# 延遲導入的 GNN-KAN 組件
def get_gnn_kan_components():
    """獲取所有 GNN-KAN 組件（延遲導入）"""
    success, components = _lazy_import_gnn_kan()
    if success:
        return components
    else:
        # 返回空的組件
        return {
            'gnn_kan_rca': get_gnn_kan_rca(),
            'GNNKANEndToEnd': None,
            'SimplifiedGNNKANConfig': None,
            'HighCapacityGNNKANConfig': None,
            'FastGNNKANConfig': None,
            'MultiModalFeatureExtractor': None,
            'SimplifiedGraphConstructor': None,
            'GNNKANModel': None,
            'train_gnn_kan_model': None
        }

# 🎯 統一導出清單
__all__ = [
    # 裝飾器
    'rca',
    
    # 標準 RCA 方法
    'baro',
    'circa',
    'pc_pagerank', 'cmlp_pagerank', 'ntlr_pagerank',
    'pc_randomwalk', 'ntlr_randomwalk', 'fci_randomwalk', 'lingam_randomwalk', 'granger_randomwalk',
    'fci_pagerank',
    'ges_pagerank',
    'lingam_pagerank',
    'granger_pagerank',
    'run',
    
    # 可選方法
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
    
    # GNN-KAN 延遲導入函數
    'get_gnn_kan_rca',
    'get_gnn_kan_components',
]




