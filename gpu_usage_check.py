#!/usr/bin/env python3
"""
GPU 使用情況詳細檢測工具
"""

import os
import torch
import numpy as np
import time
import psutil

def detailed_gpu_check():
    """詳細檢查 GPU 使用情況"""
    print("=" * 60)
    print("詳細 GPU 使用情況檢測")
    print("=" * 60)
    
    # 1. 基本 CUDA 信息
    print("\n1. CUDA 基本信息:")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，將使用 CPU")
        return False
    
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 設備數量: {torch.cuda.device_count()}")
    
    # 2. GPU 設備詳情
    print("\n2. GPU 設備詳情:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  - 計算能力: {props.major}.{props.minor}")
        print(f"  - 總記憶體: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - 多處理器數量: {props.multi_processor_count}")
    
    # 3. 當前 GPU 記憶體使用
    print("\n3. GPU 記憶體使用情況:")
    device = torch.cuda.current_device()
    print(f"當前設備: {device}")
    
    # 獲取記憶體信息（更準確的方法）
    try:
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        
        print(f"已分配記憶體: {allocated / 1024**3:.3f} GB")
        print(f"已保留記憶體: {reserved / 1024**3:.3f} GB")
        print(f"最大已分配: {max_allocated / 1024**3:.3f} GB")
        print(f"最大已保留: {max_reserved / 1024**3:.3f} GB")
        
    except Exception as e:
        print(f"獲取記憶體信息失敗: {e}")
    
    # 4. 實際 GPU 計算測試
    print("\n4. GPU 計算能力測試:")
    
    # 小規模測試
    try:
        print("執行小規模矩陣運算...")
        start_time = time.time()
        
        # 創建 GPU 張量
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        
        # 記錄使用前的記憶體
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"張量創建後記憶體: {mem_before:.3f} GB")
        
        # 執行矩陣乘法
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # 確保計算完成
        
        end_time = time.time()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        
        print(f"計算完成後記憶體: {mem_after:.3f} GB")
        print(f"GPU 矩陣運算耗時: {(end_time - start_time)*1000:.2f} ms")
        print("✓ GPU 計算正常工作")
        
        # 清理記憶體
        del a, b, c
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ GPU 計算測試失敗: {e}")
        return False
    
    # 5. 大規模測試 (模擬 GNN-KAN 工作負載)
    print("\n5. 模擬 GNN-KAN 工作負載:")
    
    try:
        from RCAEval.kan import KANLayer
        
        print("創建 KAN 層...")
        kan_layer = KANLayer(input_dim=64, output_dim=32, grid_size=5).cuda()
        
        print("執行前向傳播...")
        x = torch.randn(100, 64, device='cuda')
        
        mem_before = torch.cuda.memory_allocated() / 1024**3
        print(f"KAN 層創建後記憶體: {mem_before:.3f} GB")
        
        start_time = time.time()
        output = kan_layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(f"前向傳播後記憶體: {mem_after:.3f} GB")
        print(f"KAN 層前向傳播耗時: {(end_time - start_time)*1000:.2f} ms")
        
        # 測試反向傳播
        print("執行反向傳播...")
        loss = output.sum()
        
        start_time = time.time()
        loss.backward()
        torch.cuda.synchronize()
        end_time = time.time()
        
        mem_final = torch.cuda.memory_allocated() / 1024**3
        print(f"反向傳播後記憶體: {mem_final:.3f} GB")
        print(f"反向傳播耗時: {(end_time - start_time)*1000:.2f} ms")
        print("✓ KAN 層 GPU 計算正常")
        
        # 清理
        del kan_layer, x, output, loss
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ KAN 層 GPU 測試失敗: {e}")
        return False
    
    # 6. CPU vs GPU 性能比較
    print("\n6. CPU vs GPU 性能比較:")
    
    try:
        size = 2000
        
        # CPU 測試
        print(f"CPU 矩陣運算 ({size}x{size})...")
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU 耗時: {cpu_time*1000:.2f} ms")
        
        # GPU 測試
        print(f"GPU 矩陣運算 ({size}x{size})...")
        a_gpu = torch.randn(size, size, device='cuda')
        b_gpu = torch.randn(size, size, device='cuda')
        
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU 耗時: {gpu_time*1000:.2f} ms")
        
        speedup = cpu_time / gpu_time
        print(f"GPU 加速比: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✓ GPU 顯著加速了計算")
        else:
            print("⚠️ GPU 加速效果不明顯，可能數據量太小")
        
        # 清理
        del a_cpu, b_cpu, c_cpu, a_gpu, b_gpu, c_gpu
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"性能比較測試失敗: {e}")
    
    # 7. 最終記憶體狀態
    print("\n7. 最終 GPU 記憶體狀態:")
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"當前已分配: {allocated:.3f} GB")
        print(f"當前已保留: {reserved:.3f} GB")
        
        # 清空緩存
        torch.cuda.empty_cache()
        
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        reserved_after = torch.cuda.memory_reserved() / 1024**3
        print(f"清空緩存後已分配: {allocated_after:.3f} GB")
        print(f"清空緩存後已保留: {reserved_after:.3f} GB")
        
    except Exception as e:
        print(f"獲取最終記憶體狀態失敗: {e}")
    
    return True


def test_gnn_kan_gpu_usage():
    """測試 GNN-KAN 實際 GPU 使用情況"""
    print("\n" + "=" * 60)
    print("GNN-KAN GPU 使用測試")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳過 GPU 測試")
        return
    
    try:
        from RCAEval.e2e.gnn_kan import GNNKANModel, GNNKANConfig
        
        # 創建配置
        config = GNNKANConfig()
        config.epochs = 5  # 減少訓練輪數
        
        print("\n1. 創建 GNN-KAN 模型...")
        model = GNNKANModel(config, num_nodes=50)
        
        # 檢查模型是否在 GPU 上
        device = next(model.parameters()).device
        print(f"模型設備: {device}")
        
        if device.type != 'cuda':
            print("將模型移動到 GPU...")
            model = model.cuda()
            device = next(model.parameters()).device
            print(f"移動後模型設備: {device}")
        
        # 記錄模型創建後的記憶體
        mem_model = torch.cuda.memory_allocated() / 1024**3
        print(f"模型創建後 GPU 記憶體: {mem_model:.3f} GB")
        
        print("\n2. 創建測試數據...")
        # 創建較大的測試數據
        node_features = torch.randn(50, config.target_feature_dim, device='cuda')
        edge_index = torch.randint(0, 50, (2, 200), device='cuda')
        
        mem_data = torch.cuda.memory_allocated() / 1024**3
        print(f"數據創建後 GPU 記憶體: {mem_data:.3f} GB")
        
        print("\n3. 執行前向傳播...")
        model.train()
        
        start_time = time.time()
        with torch.cuda.profiler.profile():
            embeddings, adj_scores = model(node_features, edge_index)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        
        mem_forward = torch.cuda.memory_allocated() / 1024**3
        print(f"前向傳播後 GPU 記憶體: {mem_forward:.3f} GB")
        print(f"前向傳播耗時: {forward_time*1000:.2f} ms")
        
        print("\n4. 執行反向傳播...")
        # 創建假損失
        loss = adj_scores.sum() + embeddings.sum()
        
        start_time = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - start_time
        
        mem_backward = torch.cuda.memory_allocated() / 1024**3
        print(f"反向傳播後 GPU 記憶體: {mem_backward:.3f} GB")
        print(f"反向傳播耗時: {backward_time*1000:.2f} ms")
        
        print("\n5. GPU 使用總結:")
        print(f"總 GPU 記憶體使用: {mem_backward:.3f} GB")
        print(f"模型參數記憶體: {mem_model:.3f} GB")
        print(f"數據記憶體: {(mem_data - mem_model):.3f} GB")
        print(f"計算緩存記憶體: {(mem_backward - mem_data):.3f} GB")
        
        if mem_backward > 0.01:  # 如果使用超過 10MB
            print("✓ GNN-KAN 確實在使用 GPU")
        else:
            print("⚠️ GPU 記憶體使用很少，可能模型太小")
        
        # 清理
        del model, node_features, edge_index, embeddings, adj_scores, loss
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"GNN-KAN GPU 測試失敗: {e}")
        import traceback
        traceback.print_exc()


def monitor_real_training():
    """監控實際訓練過程中的 GPU 使用"""
    print("\n" + "=" * 60)
    print("實際訓練 GPU 使用監控")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳過監控")
        return
    
    try:
        from RCAEval.e2e.gnn_kan import gnn_kan_rca
        import pandas as pd
        
        print("創建測試數據...")
        # 創建較大的測試數據集
        test_data = pd.DataFrame({
            'time': range(300),
            'metric1': np.random.randn(300) + np.sin(np.arange(300) * 0.1),
            'metric2': np.random.randn(300) + np.cos(np.arange(300) * 0.05),
            'metric3': np.random.randn(300) * 2,
            'metric4': np.random.exponential(1, 300),
        })
        
        # 監控訓練過程
        print("開始監控 GNN-KAN 訓練...")
        
        def memory_monitor():
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                return allocated, reserved
            return 0, 0
        
        # 記錄訓練前的記憶體
        initial_alloc, initial_reserved = memory_monitor()
        print(f"訓練前 GPU 記憶體 - 分配: {initial_alloc:.3f} GB, 保留: {initial_reserved:.3f} GB")
        
        # 執行訓練
        start_time = time.time()
        result = gnn_kan_rca(
            test_data,
            inject_time=150,
            dataset='gpu_test',
            epochs=3,  # 少量訓練輪數用於測試
            batch_size=16
        )
        end_time = time.time()
        
        # 記錄訓練後的記憶體
        final_alloc, final_reserved = memory_monitor()
        print(f"訓練後 GPU 記憶體 - 分配: {final_alloc:.3f} GB, 保留: {final_reserved:.3f} GB")
        
        print(f"\n訓練結果:")
        print(f"總耗時: {end_time - start_time:.2f} 秒")
        print(f"記憶體增量: {final_alloc - initial_alloc:.3f} GB")
        print(f"節點數量: {len(result.get('node_names', []))}")
        
        if final_alloc > initial_alloc + 0.01:
            print("✓ 確認 GPU 被用於訓練")
        else:
            print("⚠️ GPU 記憶體使用變化很小")
        
    except Exception as e:
        print(f"訓練監控失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主檢測函數"""
    print("GPU 使用情況全面檢測")
    print("=" * 60)
    
    # 1. 詳細 GPU 檢查
    gpu_available = detailed_gpu_check()
    
    if not gpu_available:
        print("\n❌ GPU 不可用或有問題，結束檢測")
        return
    
    # 2. GNN-KAN GPU 使用測試
    test_gnn_kan_gpu_usage()
    
    # 3. 實際訓練監控
    monitor_real_training()
    
    print("\n" + "=" * 60)
    print("GPU 檢測完成")
    print("如果看到大量 '✓' 標記，說明 GPU 正在正常工作")
    print("如果記憶體使用很少，可能是模型規模太小導致的")
    print("=" * 60)


if __name__ == "__main__":
    main()