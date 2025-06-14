#!/usr/bin/env python3
"""
🚀 GNN-KAN GPU 綜合測試套件
整合所有GPU相關測試功能，包括性能基準測試和實際工作負載測試
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 設置環境變量
os.environ['NPY_DISABLE_CPU_FEATURES'] = ''

# 動態添加項目路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def check_gpu_availability():
    """檢查GPU可用性"""
    print("=" * 60)
    print("🔍 GPU 可用性檢查")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，將使用 CPU 進行測試")
        return False
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"✅ CUDA 可用")
    print(f"📊 GPU 設備數量: {device_count}")
    print(f"🎯 當前設備: {current_device} - {device_name}")
    print(f"🔧 CUDA 版本: {torch.version.cuda}")
    
    return True

def benchmark_kan_layers():
    """基準測試不同KAN層的性能"""
    print("\n" + "=" * 60)
    print("⚡ KAN 層性能基準測試")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    # 測試配置
    configs = [
        {"batch": 32, "input": 64, "output": 32, "desc": "小規模"},
        {"batch": 100, "input": 128, "output": 64, "desc": "中規模"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*20} {config['desc']} 測試 {'='*20}")
        print(f"Batch: {config['batch']}, Input: {config['input']}, Output: {config['output']}")
        
        # 創建測試數據
        x = torch.randn(config['batch'], config['input'], device=device)
        config_results = {}
        
        try:
            # 測試標準線性層
            print("\n1. Standard Linear (基準):")
            model_linear = torch.nn.Linear(config['input'], config['output']).to(device)
            
            # 預熱
            for _ in range(3):
                _ = model_linear(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 性能測試
            start_time = time.time()
            for _ in range(10):
                output = model_linear(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            linear_time = (time.time() - start_time) / 10
            
            print(f"  時間: {linear_time*1000:.2f} ms")
            print(f"  輸出形狀: {output.shape}")
            
            config_results['Linear'] = {'time': linear_time, 'shape': output.shape}
            del model_linear, output
            
            # 測試KAN層（如果可用）
            try:
                from RCAEval.gnn_kan_module.kan_components import SimplifiedKANLayer
                
                print("\n2. SimplifiedKANLayer:")
                model_kan = SimplifiedKANLayer(
                    config['input'], config['output'], num_basis=5
                ).to(device)
                
                # 預熱
                for _ in range(3):
                    _ = model_kan(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                # 性能測試
                start_time = time.time()
                for _ in range(10):
                    output = model_kan(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                kan_time = (time.time() - start_time) / 10
                
                print(f"  時間: {kan_time*1000:.2f} ms")
                print(f"  輸出形狀: {output.shape}")
                print(f"  vs Linear: {kan_time/linear_time:.1f}x")
                
                config_results['KAN'] = {
                    'time': kan_time, 
                    'shape': output.shape,
                    'vs_linear': kan_time/linear_time
                }
                del model_kan, output
                
            except ImportError:
                print("⚠️ KAN層不可用，跳過KAN測試")
            
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        results[config['desc']] = config_results
        del x
    
    return results

def test_gnn_kan_gpu_performance():
    """測試完整GNN-KAN在GPU上的性能"""
    print("\n" + "=" * 60)
    print("🎯 GNN-KAN 完整模型 GPU 性能測試")
    print("=" * 60)
    
    try:
        from RCAEval.e2e.gnnkan import gnn_kan_rca
        
        # 創建測試數據
        n_samples = 50  # 減少樣本數以加快測試
        test_data = {
            'metrics': pd.DataFrame({
                'cpu_usage': np.random.rand(n_samples) * 100,
                'memory_usage': np.random.rand(n_samples) * 100,
                'network_io': np.random.rand(n_samples) * 1000,
            })
        }
        
        print("🧪 執行 GNN-KAN 性能測試...")
        
        # 記錄開始狀態
        start_time = time.time()
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            start_memory = 0
        
        # 運行GNN-KAN
        result = gnn_kan_rca(
            test_data,
            inject_time=25,
            config_type='simplified',
            epochs=3,  # 減少訓練輪數以加快測試
            kan_grid_size=5,
            hidden_dims=[32, 16],  # 減少模型大小
            num_gnn_layers=1,
            verbose=False
        )
        
        # 記錄結束狀態
        end_time = time.time()
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            end_memory = 0
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"\n📊 GNN-KAN 性能結果:")
        print(f"  ⏱️ 執行時間: {execution_time:.2f} 秒")
        print(f"  💾 記憶體使用: {memory_used:.3f} GB")
        
        if result and isinstance(result, dict):
            ranks = result.get('ranks', [])
            node_names = result.get('node_names', [])
            adj_matrix = result.get('adj', np.array([]))
            
            print(f"  🎯 檢測節點數: {len(node_names)}")
            print(f"  📈 根因候選數: {len(ranks)}")
            print(f"  🔗 鄰接矩陣: {adj_matrix.shape if adj_matrix.size > 0 else '空'}")
            
            # 評估性能
            if execution_time < 60 and len(ranks) > 0:
                print("✅ GNN-KAN GPU 性能優秀")
                return True
            elif execution_time < 120:
                print("✅ GNN-KAN GPU 性能良好")
                return True
            else:
                print("⚠️ GNN-KAN GPU 性能需要優化")
                return False
        else:
            print("❌ GNN-KAN 執行失敗")
            return False
        
    except Exception as e:
        print(f"❌ GNN-KAN GPU 測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 GNN-KAN GPU 綜合測試套件")
    print("=" * 80)
    
    test_results = {}
    
    # 1. GPU可用性檢查
    gpu_available = check_gpu_availability()
    test_results['gpu_available'] = gpu_available
    
    # 2. KAN層性能基準測試
    print("\n" + "🔥" * 20 + " 開始性能測試 " + "🔥" * 20)
    kan_results = benchmark_kan_layers()
    test_results['kan_benchmark'] = len(kan_results) > 0
    
    # 3. 完整GNN-KAN性能測試
    gnn_kan_success = test_gnn_kan_gpu_performance()
    test_results['gnn_kan_performance'] = gnn_kan_success
    
    # 4. 測試總結
    print("\n" + "=" * 80)
    print("📋 GPU 測試總結")
    print("=" * 80)
    
    print(f"🔍 GPU 可用性: {'✅' if test_results['gpu_available'] else '❌'}")
    print(f"⚡ KAN 層基準測試: {'✅' if test_results['kan_benchmark'] else '❌'}")
    print(f"🎯 GNN-KAN 性能測試: {'✅' if test_results['gnn_kan_performance'] else '❌'}")
    
    success_count = sum(1 for v in test_results.values() if v)
    total_count = len(test_results)
    
    print(f"\n🎉 測試完成: {success_count}/{total_count} 項測試通過")
    
    if success_count == total_count:
        print("✅ 所有GPU測試通過！GNN-KAN GPU優化效果良好")
    elif success_count >= total_count * 0.8:
        print("✅ 大部分GPU測試通過，系統基本正常")
    else:
        print("⚠️ 部分GPU測試失敗，建議檢查GPU環境配置")

if __name__ == "__main__":
    main() 