#!/usr/bin/env python3
"""
快速 GPU 性能測試 - 驗證 KAN 優化效果
"""

import os
import sys
import time
import torch
import numpy as np

# 設置環境變量
os.environ['NPY_DISABLE_CPU_FEATURES'] = ''
sys.path.append('/Users/user/RCAEval')

def benchmark_kan_layers():
    """基準測試三種 KAN 層"""
    print("=" * 60)
    print("KAN 層性能基準測試")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU 測試")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    
    # 測試配置
    configs = [
        {"batch": 32, "input": 64, "output": 32, "desc": "小規模"},
        {"batch": 100, "input": 128, "output": 64, "desc": "中規模"},
        {"batch": 200, "input": 256, "output": 128, "desc": "大規模"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*20} {config['desc']} 測試 {'='*20}")
        print(f"Batch: {config['batch']}, Input: {config['input']}, Output: {config['output']}")
        
        # 創建測試數據
        x = torch.randn(config['batch'], config['input'], device=device)
        
        # 測試結果存儲
        config_results = {}
        
        try:
            from RCAEval.kan import SimplifiedKANLayer, UltraFastKANLayer
            
            # 1. 測試 SimplifiedKANLayer
            print("\n1. SimplifiedKANLayer:")
            model_simple = SimplifiedKANLayer(
                config['input'], config['output'], num_basis=5
            ).to(device)
            
            # 預熱
            for _ in range(3):
                _ = model_simple(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 基準測試
            start_time = time.time()
            for _ in range(10):
                output = model_simple(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            simple_time = (time.time() - start_time) / 10
            
            # 記憶體使用
            if device == 'cuda':
                simple_memory = torch.cuda.memory_allocated() / 1024**3
            else:
                simple_memory = 0
            
            print(f"  時間: {simple_time*1000:.2f} ms")
            print(f"  記憶體: {simple_memory:.3f} GB")
            print(f"  輸出形狀: {output.shape}")
            
            config_results['SimplifiedKAN'] = {
                'time': simple_time, 
                'memory': simple_memory,
                'shape': output.shape
            }
            
            del model_simple, output
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # 2. 測試 UltraFastKANLayer
            print("\n2. UltraFastKANLayer:")
            model_ultra = UltraFastKANLayer(
                config['input'], config['output'], table_size=256
            ).to(device)
            
            # 預熱
            for _ in range(3):
                _ = model_ultra(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 基準測試
            start_time = time.time()
            for _ in range(10):
                output = model_ultra(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            ultra_time = (time.time() - start_time) / 10
            
            # 記憶體使用
            if device == 'cuda':
                ultra_memory = torch.cuda.memory_allocated() / 1024**3
            else:
                ultra_memory = 0
            
            print(f"  時間: {ultra_time*1000:.2f} ms")
            print(f"  記憶體: {ultra_memory:.3f} GB")
            print(f"  輸出形狀: {output.shape}")
            print(f"  vs SimplifiedKAN: {simple_time/ultra_time:.1f}x 更快")
            
            config_results['UltraFastKAN'] = {
                'time': ultra_time, 
                'memory': ultra_memory,
                'shape': output.shape,
                'speedup_vs_simple': simple_time/ultra_time
            }
            
            del model_ultra, output
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # 3. 對比標準線性層
            print("\n3. Standard Linear (基準):")
            model_linear = torch.nn.Linear(config['input'], config['output']).to(device)
            
            # 預熱
            for _ in range(3):
                _ = model_linear(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 基準測試
            start_time = time.time()
            for _ in range(10):
                output = model_linear(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            linear_time = (time.time() - start_time) / 10
            
            print(f"  時間: {linear_time*1000:.2f} ms")
            print(f"  UltraFast vs Linear: {linear_time/ultra_time:.1f}x")
            print(f"  Simplified vs Linear: {linear_time/simple_time:.1f}x")
            
            config_results['Linear'] = {
                'time': linear_time,
                'ultra_speedup': linear_time/ultra_time,
                'simple_speedup': linear_time/simple_time
            }
            
            del model_linear, output
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"測試失敗: {e}")
            import traceback
            traceback.print_exc()
        
        results[config['desc']] = config_results
        del x
    
    # 總結性能
    print(f"\n{'='*60}")
    print("性能總結")
    print("="*60)
    
    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        if 'UltraFastKAN' in config_results and 'SimplifiedKAN' in config_results:
            ultra = config_results['UltraFastKAN']
            simple = config_results['SimplifiedKAN']
            
            print(f"  UltraFastKAN: {ultra['time']*1000:.2f}ms")
            print(f"  SimplifiedKAN: {simple['time']*1000:.2f}ms")
            print(f"  性能提升: {ultra['speedup_vs_simple']:.1f}x")
            
            if 'Linear' in config_results:
                linear = config_results['Linear']
                print(f"  vs Linear: {linear['ultra_speedup']:.1f}x")
    
    return results


def test_real_gnn_kan_performance():
    """測試真實 GNN-KAN 性能"""
    print(f"\n{'='*60}")
    print("真實 GNN-KAN 性能測試")
    print("="*60)
    
    try:
        import pandas as pd
        from RCAEval.e2e.gnn_kan import gnn_kan_rca
        
        # 創建測試數據 (適中大小)
        test_data = pd.DataFrame({
            'time': range(150),
            'cpu_util': np.random.randn(150) + np.sin(np.arange(150) * 0.1),
            'memory_util': np.random.randn(150) + 0.5,
            'disk_io': np.random.exponential(1, 150),
            'network_latency': np.random.gamma(2, 1, 150),
        })
        
        print("執行 GNN-KAN 分析...")
        print("- 數據點: 150")
        print("- 特徵維度: 4") 
        print("- 訓練輪數: 5 (快速測試)")
        
        # 記錄執行時間
        start_time = time.time()
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            start_memory = 0
        
        # 執行 GNN-KAN
        result = gnn_kan_rca(
            test_data,
            inject_time=75,
            dataset='performance_benchmark',
            epochs=5,  # 快速測試
            stl_seasonal=3,
            batch_size=32
        )
        
        # 記錄結果
        end_time = time.time()
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            end_memory = 0
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"\n結果:")
        print(f"  ✓ 執行時間: {execution_time:.2f} 秒")
        print(f"  ✓ GPU 記憶體: {memory_used:.3f} GB")
        print(f"  ✓ 節點數量: {len(result.get('node_names', []))}")
        print(f"  ✓ 前3名根因: {result.get('ranks', [])[:3]}")
        
        # 性能評估
        if execution_time < 30:
            print(f"  🚀 性能優秀: 小於30秒")
        elif execution_time < 60:
            print(f"  ✅ 性能良好: 小於1分鐘")
        else:
            print(f"  ⚠️ 性能需改進: 超過1分鐘")
        
        if memory_used > 0.1:
            print(f"  ✅ GPU 有效使用: {memory_used:.3f} GB")
        else:
            print(f"  ⚠️ GPU 使用較少: {memory_used:.3f} GB")
        
        return True
        
    except Exception as e:
        print(f"GNN-KAN 性能測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    print("快速 GPU 性能驗證")
    print("=" * 60)
    
    # 基本環境檢查
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    
    # 1. KAN 層基準測試
    layer_results = benchmark_kan_layers()
    
    # 2. 完整 GNN-KAN 測試
    gnn_success = test_real_gnn_kan_performance()
    
    # 3. 總結與建議
    print(f"\n{'='*60}")
    print("最終總結與建議")
    print("="*60)
    
    if gnn_success:
        print("✅ 所有性能測試通過")
        print("\n🚀 優化效果:")
        if layer_results:
            # 找出最佳性能配置
            best_config = None
            best_speedup = 0
            for config, results in layer_results.items():
                if 'UltraFastKAN' in results and 'speedup_vs_simple' in results['UltraFastKAN']:
                    speedup = results['UltraFastKAN']['speedup_vs_simple']
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_config = config
            
            if best_config:
                print(f"- 最佳性能配置: {best_config}")
                print(f"- 最大加速比: {best_speedup:.1f}x")
        
        print("\n💡 使用建議:")
        print("- 生產環境使用 UltraFastKANLayer")
        print("- 研究環境使用 SimplifiedKANLayer") 
        print("- 大規模數據優先使用 GPU")
        print("- 調整 batch_size 以最大化 GPU 利用率")
        
    else:
        print("❌ 部分測試失敗")
        print("\n🔧 故障排除:")
        print("- 檢查 GPU 記憶體是否充足")
        print("- 嘗試減少 batch_size")
        print("- 驗證 CUDA 環境配置")
        print("- 考慮使用 CPU 模式")
    
    print(f"\n📊 下一步測試命令:")
    print("# 完整功能測試")
    print("python main.py --dataset online-boutique --method gnn_kan --test")
    print("\n# 性能對比測試") 
    print("python main.py --dataset online-boutique --method gnn_kan")
    print("python main.py --dataset online-boutique --method causalrca")


if __name__ == "__main__":
    main()