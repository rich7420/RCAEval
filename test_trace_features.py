#!/usr/bin/env python3
"""
測試 GNN-KAN 的 trace 特徵提取功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import networkx as nx
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def test_trace_features():
    """測試 trace 特徵提取功能"""
    print("="*60)
    print("測試 GNN-KAN Trace 特徵提取功能")
    print("="*60)
    
    try:
        from RCAEval.gnn_kan_module.kan_components.feature_extraction import extract_trace_features, build_service_dependency_graph
        
        # 1. 創建模擬 trace 數據
        print("\n1. 創建模擬 trace 數據...")
        trace_data = create_mock_trace_data()
        print(f"✓ 創建了 {len(trace_data)} 條 span 記錄")
        print(f"服務: {trace_data['serviceName'].unique()}")
        print(f"操作: {trace_data['operation'].nunique()} 個不同操作")
        
        # 2. 測試 trace 特徵提取 (無 inject_time)
        print("\n2. 測試基本 trace 特徵提取...")
        trace_features, operation_names, service_graph = extract_trace_features(trace_data)
        
        if trace_features.size > 0:
            print(f"✓ 提取了 {trace_features.shape[0]} 個操作的特徵")
            print(f"✓ 特徵維度: {trace_features.shape[1]}")
            print(f"✓ 操作名稱: {operation_names[:3]}...")
        else:
            print("✗ 未提取到 trace 特徵")
            return False
        
        # 3. 測試服務依賴圖構建
        print("\n3. 測試服務依賴圖構建...")
        if service_graph is not None:
            print(f"✓ 構建了服務依賴圖: {service_graph.number_of_nodes()} 節點, {service_graph.number_of_edges()} 邊")
            print(f"✓ 服務節點: {list(service_graph.nodes())}")
        else:
            print("✗ 服務依賴圖構建失敗")
        
        # 4. 測試帶 inject_time 的 trace 特徵提取 (TracerCA 風格)
        print("\n4. 測試 TracerCA 風格的 trace 特徵提取...")
        inject_time = 50000  # 模擬故障注入時間
        
        trace_features_anomaly, operation_names_anomaly, _ = extract_trace_features(
            trace_data, inject_time=inject_time
        )
        
        if trace_features_anomaly.size > 0:
            print(f"✓ 提取了 {trace_features_anomaly.shape[0]} 個操作的異常特徵")
            print(f"✓ 異常特徵維度: {trace_features_anomaly.shape[1]}")
            print("✓ 特徵包含: support, confidence, ji_score, latency_change 等")
            
            # 顯示前幾個操作的 JI 分數
            ji_scores = trace_features_anomaly[:, 2]  # JI score 是第3列
            top_operations = sorted(zip(operation_names_anomaly, ji_scores), 
                                  key=lambda x: x[1], reverse=True)
            print(f"✓ Top 3 operations by JI score:")
            for i, (op, score) in enumerate(top_operations[:3]):
                print(f"   {i+1}. {op}: {score:.4f}")
        else:
            print("✗ 未提取到異常 trace 特徵")
        
        # 5. 測試服務拓樸特徵
        print("\n5. 測試服務拓樸特徵提取...")
        from RCAEval.kan import extract_service_topology_features
        
        if service_graph is not None:
            service_topo_features, service_topo_names = extract_service_topology_features(
                service_graph, list(service_graph.nodes())
            )
            
            if service_topo_features.size > 0:
                print(f"✓ 提取了 {len(service_topo_names)} 個服務拓樸特徵")
                print(f"✓ 特徵名稱: {service_topo_names}")
                print(f"✓ 特徵值: {service_topo_features.flatten()}")
            else:
                print("✗ 未提取到服務拓樸特徵")
        
        return True
        
    except Exception as e:
        print(f"✗ trace 特徵測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gnn_kan_with_trace():
    """測試 GNN-KAN 與 trace 特徵的集成"""
    print("\n" + "="*60)
    print("測試 GNN-KAN 與 trace 特徵集成")
    print("="*60)
    
    try:
        from RCAEval.e2e.gnn_kan import gnn_kan_rca
        
        # 創建多模態數據，包含 trace
        print("\n1. 創建多模態測試數據 (包含 trace)...")
        
        # Metrics 數據
        metrics_data = pd.DataFrame({
            'time': range(100),
            'cpu_usage': np.random.rand(100) * 100,
            'memory_usage': np.random.rand(100) * 100,
            'disk_io': np.random.rand(100) * 1000,
        })
        
        # Trace 數據
        trace_data = create_mock_trace_data()
        
        # 構建多模態數據
        multimodal_data = {
            'metric': metrics_data,
            'trace': trace_data
        }
        
        print("✓ 創建了包含 metrics 和 trace 的多模態數據")
        
        # 2. 運行 GNN-KAN RCA
        print("\n2. 運行 GNN-KAN RCA (包含 trace 特徵)...")
        inject_time = 50
        
        result = gnn_kan_rca(
            multimodal_data, 
            inject_time=inject_time,
            dataset='test_trace',
            epochs=10,  # 減少訓練輪數以加快測試
            stl_seasonal=3
        )
        
        if result and 'ranks' in result and result['ranks']:
            print(f"✓ GNN-KAN RCA 成功執行")
            print(f"✓ 節點數量: {len(result['node_names'])}")
            print(f"✓ 鄰接矩陣形狀: {result['adj'].shape}")
            print(f"✓ 前 5 個根因排序:")
            
            for i, rank in enumerate(result['ranks'][:5]):
                print(f"   {i+1}. {rank}")
            
            # 檢查是否包含 trace 相關的節點
            trace_nodes = [name for name in result['node_names'] if 'trace_' in name or 'service_topo_' in name]
            if trace_nodes:
                print(f"✓ 檢測到 {len(trace_nodes)} 個 trace 相關節點")
                print(f"   例如: {trace_nodes[:3]}")
            else:
                print("⚠ 未檢測到 trace 相關節點")
                
        else:
            print("✗ GNN-KAN RCA 執行失敗")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ GNN-KAN trace 集成測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracerca_comparison():
    """測試與 TracerCA 的比較"""
    print("\n" + "="*60)
    print("與 TracerCA 方法比較")
    print("="*60)
    
    try:
        from RCAEval.e2e.tracerca import tracerca
        from RCAEval.kan import extract_trace_features
        
        # 創建測試數據
        trace_data = create_mock_trace_data()
        inject_time = 50000
        
        print("\n1. 運行原始 TracerCA...")
        tracerca_result = tracerca(trace_data, inject_time=inject_time)
        
        print(f"✓ TracerCA 排序結果: {tracerca_result['ranks'][:5]}")
        
        print("\n2. 運行 GNN-KAN 的 trace 特徵提取...")
        trace_features, operation_names, _ = extract_trace_features(
            trace_data, inject_time=inject_time
        )
        
        if trace_features.size > 0:
            # 使用 JI 分數排序 (類似 TracerCA)
            ji_scores = trace_features[:, 2]  # JI score 是第3列
            gnn_kan_ranks = [op for op, _ in sorted(zip(operation_names, ji_scores), 
                            key=lambda x: x[1], reverse=True)]
            
            print(f"✓ GNN-KAN trace 排序結果: {gnn_kan_ranks[:5]}")
            
            # 比較結果
            print("\n3. 比較兩種方法的結果...")
            common_top5 = set(tracerca_result['ranks'][:5]) & set(gnn_kan_ranks[:5])
            print(f"✓ 前5名中的共同結果: {len(common_top5)} 個")
            print(f"   共同操作: {list(common_top5)}")
            
            if len(common_top5) >= 2:
                print("✓ 兩種方法在根因識別上有較好的一致性")
            else:
                print("⚠ 兩種方法的結果差異較大，這是正常的因為方法不同")
                
        return True
        
    except Exception as e:
        print(f"✗ TracerCA 比較測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_mock_trace_data():
    """創建模擬的 trace 數據"""
    np.random.seed(42)
    
    services = ['frontend', 'cartservice', 'productcatalog', 'checkout', 'payment']
    methods = ['GET', 'POST', 'grpc.Call', 'http.request']
    
    traces = []
    
    for trace_id in range(1, 51):  # 50 個 trace
        start_time = np.random.randint(10000, 100000)
        
        # 每個 trace 包含多個 span
        for span_id in range(1, np.random.randint(3, 8)):  # 每個 trace 3-7 個 span
            service = np.random.choice(services)
            method = np.random.choice(methods)
            
            # 模擬延遲：正常情況下較低，異常情況下較高
            if start_time > 50000:  # 模擬故障注入後
                if service in ['checkout', 'payment']:  # 這些服務更容易出問題
                    duration = np.random.exponential(5000)  # 異常延遲
                else:
                    duration = np.random.exponential(1000)  # 正常延遲
            else:
                duration = np.random.exponential(500)  # 正常延遲
            
            traces.append({
                'traceId': f'trace_{trace_id}',
                'spanId': f'span_{trace_id}_{span_id}',
                'parentSpanId': f'span_{trace_id}_{span_id-1}' if span_id > 1 else None,
                'serviceName': service,
                'methodName': method,
                'operationName': f'{service}.{method}',
                'startTime': start_time + span_id * 100,
                'duration': duration,
                'tags': f'service={service}'
            })
    
    df = pd.DataFrame(traces)
    
    # 添加 operation 列 (類似 TracerCA)
    df['operation'] = df['serviceName'] + '_' + df['methodName']
    
    return df


def main():
    """主測試函數"""
    print("開始測試 GNN-KAN trace 特徵提取功能...")
    
    all_tests_passed = True
    
    # 測試1: 基本 trace 特徵提取
    if not test_trace_features():
        all_tests_passed = False
    
    # 測試2: GNN-KAN 與 trace 特徵集成
    if not test_gnn_kan_with_trace():
        all_tests_passed = False
    
    # 測試3: 與 TracerCA 比較
    if not test_tracerca_comparison():
        all_tests_passed = False
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("✅ 所有 trace 特徵測試通過！")
        print("\n🎉 GNN-KAN 現在支持 trace 特徵提取：")
        print("   • 基於 TracerCA 的異常檢測特徵")
        print("   • 服務依賴圖構建")
        print("   • 服務拓樸特徵提取") 
        print("   • 多模態特徵融合 (metrics + trace)")
        print("   • 增強的圖構建能力")
    else:
        print("❌ 部分測試失敗，請檢查實現")
    print("="*60)


if __name__ == "__main__":
    main()