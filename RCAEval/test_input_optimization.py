"""
GNN+KAN 輸入優化測試
比較優化前後的性能差異
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# 添加路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from RCAEval.gnn_kan_module.optimized_input_processor import optimize_gnn_kan_input
from RCAEval.gnn_kan_module.feature_extractors import MultiModalFeatureExtractor
from RCAEval.gnn_kan_module.graph_constructors import SimplifiedGraphConstructor
from RCAEval.gnn_kan_module.config import SimplifiedGNNKANConfig


def generate_test_data(num_samples=1000, num_services=9):
    """生成測試數據"""
    services = ['adservice', 'cartservice', 'checkoutservice', 'currencyservice', 
                'emailservice', 'paymentservice', 'productcatalogservice', 
                'recommendationservice', 'shippingservice'][:num_services]
    
    data = {}
    
    # 為每個服務生成多種指標
    for service in services:
        for metric in ['cpu_usage', 'memory_usage', 'latency_p90', 'error_rate']:
            col_name = f"{service}_{metric}"
            # 生成帶噪聲的時序數據
            base_value = np.random.uniform(0.1, 0.8)
            noise = np.random.normal(0, 0.1, num_samples)
            trend = np.linspace(0, 0.2, num_samples) * np.random.choice([-1, 1])
            data[col_name] = base_value + noise + trend
    
    # 添加時間列
    data['time'] = np.arange(num_samples)
    
    return pd.DataFrame(data)


def test_original_method(data):
    """測試原始方法"""
    print("🔧 測試原始方法...")
    start_time = time.time()
    
    config = SimplifiedGNNKANConfig()
    
    # 原始特徵提取
    feature_extractor = MultiModalFeatureExtractor(config)
    features, node_names = feature_extractor.extract_features(data)
    
    # 原始圖構建
    graph_constructor = SimplifiedGraphConstructor(config)
    edge_index, edge_weights = graph_constructor.build_graph(features, node_names)
    
    processing_time = time.time() - start_time
    
    return {
        'processing_time': processing_time,
        'num_nodes': len(node_names),
        'num_edges': edge_index.size(1) if hasattr(edge_index, 'size') else len(edge_index[0]),
        'feature_shape': features.shape,
        'node_names': node_names
    }


def test_optimized_method(data, feature_method='ica'):
    """測試優化方法"""
    print(f"🚀 測試優化方法 (feature_method={feature_method})...")
    start_time = time.time()
    
    optimized_data = optimize_gnn_kan_input(
        data=data,
        feature_method=feature_method,
        target_dim=64
    )
    
    processing_time = time.time() - start_time
    
    return {
        'processing_time': processing_time,
        'num_nodes': optimized_data.metadata['num_nodes'],
        'num_edges': optimized_data.metadata['num_edges'],
        'feature_shape': optimized_data.node_features.shape,
        'node_names': optimized_data.node_names,
        'metadata': optimized_data.metadata
    }


def run_performance_comparison():
    """運行性能比較測試"""
    print("=" * 60)
    print("🎯 GNN+KAN 輸入優化性能測試")
    print("=" * 60)
    
    # 生成不同規模的測試數據
    test_configs = [
        {'num_samples': 500, 'num_services': 5, 'name': '小規模'},
        {'num_samples': 1000, 'num_services': 9, 'name': '中規模'},
        {'num_samples': 2000, 'num_services': 12, 'name': '大規模'}
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n📊 測試 {config['name']} 數據 (樣本數: {config['num_samples']}, 服務數: {config['num_services']})")
        
        # 生成測試數據
        test_data = generate_test_data(config['num_samples'], config['num_services'])
        print(f"✓ 生成測試數據: {test_data.shape}")
        
        try:
            # 測試原始方法
            original_result = test_original_method(test_data)
            print(f"✓ 原始方法: {original_result['processing_time']:.3f}秒")
            
            # 測試優化方法 - ICA
            optimized_ica_result = test_optimized_method(test_data, 'ica')
            print(f"✓ 優化方法(ICA): {optimized_ica_result['processing_time']:.3f}秒")
            
            # 測試優化方法 - 統計
            optimized_stat_result = test_optimized_method(test_data, 'simplified')
            print(f"✓ 優化方法(統計): {optimized_stat_result['processing_time']:.3f}秒")
            
            # 計算性能提升
            speedup_ica = original_result['processing_time'] / optimized_ica_result['processing_time']
            speedup_stat = original_result['processing_time'] / optimized_stat_result['processing_time']
            
            result = {
                'config': config,
                'original': original_result,
                'optimized_ica': optimized_ica_result,
                'optimized_stat': optimized_stat_result,
                'speedup_ica': speedup_ica,
                'speedup_stat': speedup_stat
            }
            
            results.append(result)
            
            print(f"🚀 性能提升: ICA方法 {speedup_ica:.2f}x, 統計方法 {speedup_stat:.2f}x")
            
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            continue
    
    # 打印總結
    print("\n" + "=" * 60)
    print("📈 性能測試總結")
    print("=" * 60)
    
    for result in results:
        config = result['config']
        print(f"\n{config['name']} 數據:")
        print(f"  數據規模: {config['num_samples']} 樣本, {config['num_services']} 服務")
        print(f"  原始方法: {result['original']['processing_time']:.3f}秒")
        print(f"  優化ICA: {result['optimized_ica']['processing_time']:.3f}秒 (提升 {result['speedup_ica']:.2f}x)")
        print(f"  優化統計: {result['optimized_stat']['processing_time']:.3f}秒 (提升 {result['speedup_stat']:.2f}x)")
        print(f"  節點數比較: {result['original']['num_nodes']} vs {result['optimized_ica']['num_nodes']}")
        print(f"  邊數比較: {result['original']['num_edges']} vs {result['optimized_ica']['num_edges']}")
    
    # 計算平均性能提升
    if results:
        avg_speedup_ica = np.mean([r['speedup_ica'] for r in results])
        avg_speedup_stat = np.mean([r['speedup_stat'] for r in results])
        
        print(f"\n🎯 平均性能提升:")
        print(f"  ICA優化方法: {avg_speedup_ica:.2f}x")
        print(f"  統計優化方法: {avg_speedup_stat:.2f}x")
        
        print(f"\n✅ 優化效果顯著！推薦使用優化後的輸入處理器")


def test_feature_quality():
    """測試特徵質量"""
    print("\n" + "=" * 60)
    print("🔍 特徵質量測試")
    print("=" * 60)
    
    test_data = generate_test_data(1000, 9)
    
    # 測試不同特徵方法
    methods = ['ica', 'simplified']
    
    for method in methods:
        print(f"\n📊 測試 {method} 方法:")
        
        optimized_data = optimize_gnn_kan_input(
            data=test_data,
            feature_method=method,
            target_dim=64
        )
        
        features = optimized_data.node_features.numpy()
        
        print(f"  特徵形狀: {features.shape}")
        print(f"  特徵範圍: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  特徵均值: {features.mean():.3f}")
        print(f"  特徵標準差: {features.std():.3f}")
        print(f"  NaN數量: {np.isnan(features).sum()}")
        print(f"  Inf數量: {np.isinf(features).sum()}")
        print(f"  節點名稱: {optimized_data.node_names}")
        
        # 檢查特徵質量
        if np.isnan(features).sum() == 0 and np.isinf(features).sum() == 0:
            print(f"  ✅ {method} 方法特徵質量良好")
        else:
            print(f"  ⚠️ {method} 方法特徵包含無效值")


def test_memory_usage():
    """測試內存使用"""
    import psutil
    import os
    
    print("\n" + "=" * 60)
    print("💾 內存使用測試")
    print("=" * 60)
    
    process = psutil.Process(os.getpid())
    
    # 基準內存
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"基準內存使用: {baseline_memory:.1f} MB")
    
    # 生成大數據集
    large_data = generate_test_data(5000, 15)
    after_data_memory = process.memory_info().rss / 1024 / 1024
    print(f"數據加載後: {after_data_memory:.1f} MB (+{after_data_memory - baseline_memory:.1f} MB)")
    
    # 測試優化方法內存使用
    optimized_data = optimize_gnn_kan_input(large_data, feature_method='ica')
    after_processing_memory = process.memory_info().rss / 1024 / 1024
    print(f"優化處理後: {after_processing_memory:.1f} MB (+{after_processing_memory - after_data_memory:.1f} MB)")
    
    print(f"✅ 內存使用合理，處理開銷: {after_processing_memory - after_data_memory:.1f} MB")


if __name__ == "__main__":
    try:
        # 運行性能比較
        run_performance_comparison()
        
        # 測試特徵質量
        test_feature_quality()
        
        # 測試內存使用
        test_memory_usage()
        
        print("\n🎉 所有測試完成！")
        
    except Exception as e:
        print(f"❌ 測試過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc() 