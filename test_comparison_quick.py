#!/usr/bin/env python3
"""
快速測試 GNN-KAN vs BARO 比較功能
================================

這是一個簡化的測試腳本，用於驗證比較功能是否正常工作
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加項目路徑
sys.path.insert(0, '.')

def create_test_data():
    """創建測試數據"""
    np.random.seed(42)
    
    # 創建模擬的微服務指標數據
    num_samples = 100
    services = ['frontend', 'cartservice', 'productservice', 'recommendationservice']
    metrics = ['cpu', 'memory', 'latency']
    
    data = {'time': range(num_samples)}
    
    # 為每個服務和指標創建時間序列
    for service in services:
        for metric in metrics:
            # 正常期間的數據
            normal_data = np.random.normal(50, 10, num_samples // 2)
            # 異常期間的數據（某些服務有異常）
            if service == 'cartservice' and metric == 'cpu':
                # cartservice的CPU有異常
                anomaly_data = np.random.normal(90, 15, num_samples // 2)
            else:
                anomaly_data = np.random.normal(55, 12, num_samples // 2)
            
            # 合併數據
            full_data = np.concatenate([normal_data, anomaly_data])
            data[f'{service}_{metric}'] = full_data
    
    return pd.DataFrame(data)

def test_gnn_kan_import():
    """測試GNN-KAN導入"""
    try:
        from RCAEval.e2e.gnnkan import gnn_kan_rca
        print("✅ GNN-KAN導入成功")
        return True
    except ImportError as e:
        print(f"❌ GNN-KAN導入失敗: {e}")
        return False

def test_baro_import():
    """測試BARO導入"""
    try:
        from RCAEval.e2e.baro import baro
        print("✅ BARO導入成功")
        return True
    except ImportError as e:
        print(f"❌ BARO導入失敗: {e}")
        return False

def test_gnn_kan_function():
    """測試GNN-KAN功能"""
    try:
        from RCAEval.e2e.gnnkan import gnn_kan_rca
        
        # 創建測試數據
        data = create_test_data()
        inject_time = 50
        
        print("🤖 測試GNN-KAN功能...")
        result = gnn_kan_rca(
            data={'metrics': data},
            inject_time=inject_time,
            config_type='simplified',
            feature_method='simplified'
        )
        
        if 'ranks' in result and len(result['ranks']) > 0:
            print(f"✅ GNN-KAN功能測試成功 - 返回 {len(result['ranks'])} 個排名")
            print(f"   前5個根因: {result['ranks'][:5]}")
            return True
        else:
            print("❌ GNN-KAN功能測試失敗 - 沒有返回有效排名")
            return False
            
    except Exception as e:
        print(f"❌ GNN-KAN功能測試失敗: {e}")
        return False

def test_baro_function():
    """測試BARO功能"""
    try:
        from RCAEval.e2e.baro import baro
        
        # 創建測試數據
        data = create_test_data()
        inject_time = 50
        
        print("🔧 測試BARO功能...")
        result = baro(
            data=data,
            inject_time=inject_time
        )
        
        if 'ranks' in result and len(result['ranks']) > 0:
            print(f"✅ BARO功能測試成功 - 返回 {len(result['ranks'])} 個排名")
            print(f"   前5個根因: {result['ranks'][:5]}")
            return True
        else:
            print("❌ BARO功能測試失敗 - 沒有返回有效排名")
            return False
            
    except Exception as e:
        print(f"❌ BARO功能測試失敗: {e}")
        return False

def test_comparison_class():
    """測試比較類"""
    try:
        from gnn_kan_vs_baro_comparison import GNNKANvsBAROComparator
        
        print("📊 測試比較類...")
        comparator = GNNKANvsBAROComparator(output_dir="test_comparison_output")
        
        # 測試指標計算
        predicted = ['cartservice_cpu', 'frontend_memory', 'productservice_latency']
        ground_truth = ['cartservice_cpu', 'cartservice_memory']
        
        metrics = comparator.calculate_metrics(predicted, ground_truth)
        
        print(f"✅ 比較類測試成功")
        print(f"   Precision@1: {metrics['precision@1']:.3f}")
        print(f"   Precision@3: {metrics['precision@3']:.3f}")
        print(f"   Avg@5: {metrics['avg@5']:.3f}")
        print(f"   MRR: {metrics['mrr']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 比較類測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🧪 開始快速測試 GNN-KAN vs BARO 比較功能")
    print("=" * 60)
    
    tests = [
        ("GNN-KAN導入", test_gnn_kan_import),
        ("BARO導入", test_baro_import),
        ("GNN-KAN功能", test_gnn_kan_function),
        ("BARO功能", test_baro_function),
        ("比較類", test_comparison_class),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 測試: {test_name}")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 測試 {test_name} 出現異常: {e}")
            results.append((test_name, False))
    
    # 總結
    print("\n" + "=" * 60)
    print("📋 測試總結:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"  {test_name:15s}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 總體結果: {passed}/{total} 測試通過 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有測試通過！比較功能準備就緒。")
        print("\n🚀 可以運行完整比較測試:")
        print("   python gnn_kan_vs_baro_comparison.py --limit 1")
    else:
        print("⚠️ 部分測試失敗，請檢查相關問題。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 