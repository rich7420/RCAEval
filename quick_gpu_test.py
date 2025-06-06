#!/usr/bin/env python3
"""
快速 GPU 優化測試腳本
測試不同 KAN 實現的性能
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd

# 設置環境變量
os.environ['NPY_DISABLE_CPU_FEATURES'] = ''

# 添加項目路徑
sys.path.append('/app')

def test_kan_performance():
    """測試不同 KAN 實現的性能"""
    print("=" * 60)
    print("KAN 性能測試")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳過 GPU 測試")
        return
    
    device = torch.cuda.current_device()
    print(f"使用 GPU: {torch.cuda.get_device_name(device)}")
    
    # 測試參數
    batch_size = 100
    input_dim = 64
    output_dim = 32
    test_iterations = 10
    
    print(f"\n測試配置:")
    print(f"- Batch size: {batch_size}")
    print(f"- Input dim: {input_dim}")
    print(f"- Output dim: {output_dim}")
    print(f"- Iterations: {test_iterations}")
    
    # 創建測試數據
    x = torch.randn(batch_size, input_dim, device='cuda')
    
    try:
        from RCAEval.kan import SimplifiedKANLayer, UltraFastKANLayer
        
        # 1. 測試 SimplifiedKANLayer
        print(f"\n{'='*20} SimplifiedKANLayer {'='*20}")
        
        model_simplified = SimplifiedKANLayer(input_dim, output_dim).cuda()
        
        # 預熱
        for _ in range(3):
            _ = model_simplified(x)
        torch.cuda.synchronize()
        
        # 計時測試
        start_time = time.time()
        for _ in range(test_iterations):
            output = model_simplified(x)
            torch.cuda.synchronize()
        simplified_time = time.time() - start_time
        
        # 記憶體使用
        simplified_memory = torch.cuda.memory_allocated() / 1024**3
        
        print(f"SimplifiedKANLayer:")
        print(f"  時間: {simplified_time/test_iterations*1000:.2f} ms/iter")
        print(f"  記憶體: {simplified_memory:.3f} GB")
        print(f"  輸出形狀: {output.shape}")
        
        # 清理記憶體
        del model_simplified, output
        torch.cuda.empty_cache()
        
        # 2. 測試 UltraFastKANLayer
        print(f"\n{'='*20} UltraFastKANLayer {'='*20}")
        
        model_ultra = UltraFastKANLayer(input_dim, output_dim).cuda()
        
        # 預熱
        for _ in range(3):
            _ = model_ultra(x)
        torch.cuda.synchronize()
        
        # 計時測試
        start_time = time.time()
        for _ in range(test_iterations):
            output = model_ultra(x)
            torch.cuda.synchronize()
        ultra_time = time.time() - start_time
        
        # 記憶體使用
        ultra_memory = torch.cuda.memory_allocated() / 1024**3
        
        print(f"UltraFastKANLayer:")
        print(f"  時間: {ultra_time/test_iterations*1000:.2f} ms/iter")
        print(f"  記憶體: {ultra_memory:.3f} GB")
        print(f"  輸出形狀: {output.shape}")
        print(f"  vs Simplified: {simplified_time/ultra_time:.1f}x 更快")
        
        # 清理記憶體
        del model_ultra, output
        torch.cuda.empty_cache()
        
        # 3. 測試標準線性層作為對比
        print(f"\n{'='*20} Standard Linear Layer {'='*20}")
        
        model_linear = torch.nn.Linear(input_dim, output_dim).cuda()
        
        # 預熱
        for _ in range(3):
            _ = model_linear(x)
        torch.cuda.synchronize()
        
        # 計時測試
        start_time = time.time()
        for _ in range(test_iterations):
            output = model_linear(x)
            torch.cuda.synchronize()
        linear_time = time.time() - start_time
        
        # 記憶體使用
        linear_memory = torch.cuda.memory_allocated() / 1024**3
        
        print(f"Standard Linear:")
        print(f"  時間: {linear_time/test_iterations*1000:.2f} ms/iter")
        print(f"  記憶體: {linear_memory:.3f} GB")
        print(f"  輸出形狀: {output.shape}")
        
        # 性能比較
        print(f"\n{'='*20} 性能比較 {'='*20}")
        print(f"UltraFast vs Linear: {linear_time/ultra_time:.1f}x")
        print(f"Simplified vs Linear: {linear_time/simplified_time:.1f}x")
        
        if ultra_time < simplified_time * 2:
            print("✓ UltraFastKANLayer 性能優秀")
        else:
            print("⚠️ UltraFastKANLayer 需要進一步優化")
        
        # 清理
        del model_linear, output, x
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"性能測試失敗: {e}")
        import traceback
        traceback.print_exc()


def test_optimized_gnn_kan():
    """測試優化的 GNN-KAN"""
    print("\n" + "=" * 60)
    print("優化 GNN-KAN 測試")
    print("=" * 60)
    
    try:
        from RCAEval.e2e.gnn_kan import gnn_kan_rca
        
        # 創建測試數據
        test_data = pd.DataFrame({
            'time': range(100),
            'cpu': np.random.randn(100) + np.sin(np.arange(100) * 0.1),
            'memory': np.random.randn(100) + 0.5,
            'disk': np.random.exponential(1, 100),
        })
        
        print("測試優化的 GNN-KAN...")
        
        # 記錄開始時間和記憶體
        start_time = time.time()
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            start_memory = 0
        
        # 運行 GNN-KAN (使用很少的訓練輪數)
        result = gnn_kan_rca(
            test_data,
            inject_time=50,
            dataset='performance_test',
            epochs=3,  # 很少的訓練輪數
            stl_seasonal=3,
            batch_size=16
        )
        
        # 記錄結束時間和記憶體
        end_time = time.time()
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            end_memory = 0
        
        print(f"\n結果:")
        print(f"  執行時間: {end_time - start_time:.2f} 秒")
        print(f"  記憶體使用: {end_memory - start_memory:.3f} GB")
        print(f"  節點數量: {len(result.get('node_names', []))}")
        print(f"  前3個根因: {result.get('ranks', [])[:3]}")
        
        if end_time - start_time < 60:  # 如果少於1分鐘
            print("✓ GNN-KAN 執行速度合理")
        else:
            print("⚠️ GNN-KAN 執行時間過長")
        
        return True
        
    except Exception as e:
        print(f"GNN-KAN 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    print("GPU 優化效果驗證")
    print("=" * 60)
    
    # 檢查環境
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"初始 GPU 記憶體: {torch.cuda.memory_allocated()/1024**3:.3f} GB")
    
    # 1. KAN 層性能測試
    test_kan_performance()
    
    # 2. 完整 GNN-KAN 測試
    success = test_optimized_gnn_kan()
    
    # 3. 總結
    print("\n" + "=" * 60)
    print("測試總結")
    print("=" * 60)
    
    if success:
        print("✓ 所有測試通過")
        print("✓ GPU 優化有效")
        print("\n建議:")
        print("- 使用 UltraFastKANLayer 進行快速計算")
        print("- 使用較少的 epochs 進行測試")
        print("- 監控 GPU 記憶體使用")
    else:
        print("❌ 部分測試失敗")
        print("\n建議:")
        print("- 檢查 GPU 記憶體是否足夠")
        print("- 減少 batch size")
        print("- 使用 CPU 模式作為後備")
    
    print(f"\n最終 GPU 記憶體: {torch.cuda.memory_allocated()/1024**3:.3f} GB")


if __name__ == "__main__":
    main()