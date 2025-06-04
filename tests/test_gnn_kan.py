#!/usr/bin/env python3
"""
GNN-KAN RCA 測試套件
"""

import sys
import os
import traceback

# 添加項目根目錄到 Python 路徑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def check_dependencies():
    """檢查所需依賴是否可用"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'torch_geometric', 'networkx', 'sklearn', 
        'pandas', 'numpy', 'statsmodels', 'sknetwork'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        return False
    else:
        print("✓ All dependencies are available")
        return True

def test_kan_modules():
    """測試 KAN 模組"""
    print("Testing KAN modules...")
    try:
        # 從 __init__.py 導入 KAN 模組
        from RCAEval.kan import KANLayer, GNNKANEncoder
        
        import torch
        
        # 測試 KANLayer
        kan = KANLayer(input_dim=5, output_dim=3, grid_size=3)
        x = torch.randn(10, 5)
        output = kan(x)
        assert output.shape == (10, 3)
        print("✓ KANLayer working correctly")
        
        # 測試 GNNKANEncoder
        import torch_geometric
        encoder = GNNKANEncoder(input_dim=8, hidden_dims=[16, 8], output_dim=4)
        node_features = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        output = encoder(node_features, edge_index)
        assert output.shape == (5, 4)
        print("✓ GNNKANEncoder working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ KAN modules test failed: {e}")
        traceback.print_exc()
        return False

def test_feature_extraction():
    """測試特徵提取模組"""
    print("Testing feature extraction...")
    try:
        from RCAEval.kan.feature_extraction import (
            stl_decomposition, 
            extract_log_features, 
            kll_feature_processing,
            feature_fusion
        )
        import pandas as pd
        import numpy as np
        
        # 創建測試數據
        data = pd.DataFrame({
            'metric1': np.random.rand(100),
            'metric2': np.random.rand(100),
            'metric3': np.random.rand(100)
        })
        
        # 測試 STL 分解
        features, names = stl_decomposition(data, seasonal=7)
        print(f"✓ STL decomposition: {features.shape}, {len(names)} features")
        
        # 測試 KLL 處理
        if features.size > 0:
            processed = kll_feature_processing(features, k=50)
            print(f"✓ KLL processing: {processed.shape}")
        
        # 測試特徵融合
        log_features = np.random.rand(50, 10)
        metric_features = np.random.rand(50, 8)
        topology_features = np.random.rand(50, 6)
        error_features = np.random.rand(50, 4)
        
        fused = feature_fusion(
            log_features, 
            metric_features, 
            topology_features, 
            error_features,
            fusion_method='attention', 
            target_dim=16
        )
        print(f"✓ Feature fusion: {fused.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        traceback.print_exc()
        return False

def test_feature_fusion():
    """測試特徵融合功能"""
    print("Testing feature fusion...")
    try:
        from RCAEval.kan.feature_extraction import feature_fusion
        import numpy as np
        
        # 創建樣本特徵
        log_features = np.random.rand(10, 20)
        metric_features = np.random.rand(10, 15)
        topology_features = np.random.rand(10, 8)
        error_features = np.random.rand(10, 5)
        
        # 測試連接融合
        fused = feature_fusion(
            log_features, 
            metric_features, 
            topology_features, 
            error_features,
            fusion_method='concatenate'
        )
        
        assert fused.shape[0] == 10
        assert fused.shape[1] == 20 + 15 + 8 + 5  # 所有特徵維度之和
        
        # 測試部分特徵為 None
        fused_partial = feature_fusion(
            log_features, 
            None, 
            topology_features, 
            None,
            fusion_method='concatenate'
        )
        
        assert fused_partial.shape[0] == 10
        assert fused_partial.shape[1] == 20 + 8  # 只有 log 和 topology 特徵
        
        print("✓ Feature fusion tests passed")
        
    except Exception as e:
        print(f"✗ Feature fusion test failed: {e}")
        traceback.print_exc()

def test_simple_rca():
    """測試簡化的 RCA 功能"""
    print("Testing simplified RCA...")
    try:
        # 創建一個簡化版本，不依賴 RCAEval.io
        import pandas as pd
        import numpy as np
        from RCAEval.kan.kan_layer import KANLayer
        from RCAEval.kan.feature_extraction import stl_decomposition, kll_feature_processing
        
        # 創建測試數據
        np.random.seed(42)
        test_data = pd.DataFrame({
            'cpu_usage': np.random.rand(100) * 100,
            'memory_usage': np.random.rand(100) * 100,
            'disk_io': np.random.rand(100) * 1000,
            'network_latency': np.random.rand(100) * 50
        })
        
        # STL 分解
        features, node_names = stl_decomposition(test_data, seasonal=7)
        print(f"✓ Extracted features: {features.shape}, nodes: {len(node_names)}")
        
        # KLL 處理
        if features.size > 0:
            processed_features = kll_feature_processing(features, k=64)
            print(f"✓ Processed features: {processed_features.shape}")
            
            # 測試 KAN 層
            if processed_features.shape[1] >= 4:
                kan = KANLayer(input_dim=processed_features.shape[1], output_dim=len(node_names))
                import torch
                input_tensor = torch.tensor(processed_features, dtype=torch.float32)
                output = kan(input_tensor)
                print(f"✓ KAN output: {output.shape}")
        
        print("✓ Simplified RCA test passed")
        return True
        
    except Exception as e:
        print(f"✗ Simplified RCA test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("=== GNN-KAN RCA Test Suite ===")
    print()
    
    # 1. 檢查依賴
    print("1. ", end="")
    if not check_dependencies():
        print("❌ Dependency check failed.")
        return
    
    print()
    
    # 2. 測試 KAN 模組
    print("2. ", end="")
    if not test_kan_modules():
        print("❌ KAN modules test failed.")
        return
    
    print()
    
    # 3. 測試特徵提取
    print("3. ", end="")
    if not test_feature_extraction():
        print("❌ Feature extraction test failed.")
        return
    
    print()
    
    # 4. 測試簡化 RCA
    print("4. ", end="")
    if not test_simple_rca():
        print("❌ Simplified RCA test failed.")
        return
    
    print()
    print("🎉 All tests passed! GNN-KAN implementation is working correctly.")

if __name__ == "__main__":
    main()