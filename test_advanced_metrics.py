#!/usr/bin/env python3
"""
測試高級指標功能
驗證參數效率、可解釋性、計算效率等新增指標的計算
"""

import sys
import os
import numpy as np
import pandas as pd

# 添加項目路徑
sys.path.insert(0, '.')

def test_advanced_metrics_calculation():
    """測試高級指標計算功能"""
    print("🧪 測試高級指標計算功能")
    print("=" * 50)
    
    try:
        from gnn_kan_vs_baro_comparison import GNNKANvsBAROComparator
        
        # 創建比較器實例
        comparator = GNNKANvsBAROComparator()
        
        # 1. 測試參數效率計算
        print("\n📊 測試參數效率計算...")
        
        # 模擬GNN-KAN模型信息
        gnn_kan_model_info = {
            'model_parameters': {
                'total': 50000,
                'trainable': 45000
            },
            'num_nodes': 10,
            'hidden_dim': 64
        }
        
        param_eff = comparator.calculate_parameter_efficiency("gnn_kan", gnn_kan_model_info)
        print(f"  GNN-KAN參數效率: {param_eff}")
        
        param_eff_baro = comparator.calculate_parameter_efficiency("baro")
        print(f"  BARO參數效率: {param_eff_baro}")
        
        # 2. 測試可解釋性計算
        print("\n🔍 測試可解釋性計算...")
        
        # 模擬稀疏性信息
        gnn_kan_model_info['sparsity_info'] = {
            'sparsity_ratio': 0.25,
            'active_connections': 750,
            'pruned_connections': 250,
            'total_connections': 1000
        }
        
        interp_metrics = comparator.calculate_interpretability_metrics("gnn_kan", gnn_kan_model_info)
        print(f"  GNN-KAN可解釋性: {interp_metrics}")
        
        interp_metrics_baro = comparator.calculate_interpretability_metrics("baro")
        print(f"  BARO可解釋性: {interp_metrics_baro}")
        
        # 3. 測試計算效率
        print("\n⚡ 測試計算效率計算...")
        
        # 模擬記憶體使用
        gnn_kan_model_info['memory_usage'] = 512  # MB
        
        comp_eff = comparator.calculate_computational_efficiency("gnn_kan", 120.5, gnn_kan_model_info)
        print(f"  GNN-KAN計算效率: {comp_eff}")
        
        comp_eff_baro = comparator.calculate_computational_efficiency("baro", 45.2)
        print(f"  BARO計算效率: {comp_eff_baro}")
        
        # 4. 測試綜合高級指標
        print("\n🎯 測試綜合高級指標計算...")
        
        # 模擬結果
        mock_result = {
            'model_info': gnn_kan_model_info,
            'ranks': ['service_a', 'service_b', 'service_c']
        }
        
        advanced_metrics = comparator.calculate_advanced_metrics("gnn_kan", mock_result, 120.5)
        print(f"  GNN-KAN綜合高級指標:")
        for category, metrics in advanced_metrics.items():
            if isinstance(metrics, dict):
                print(f"    {category}:")
                for key, value in metrics.items():
                    print(f"      {key}: {value}")
            else:
                print(f"    {category}: {metrics}")
        
        print("\n✅ 高級指標計算測試完成")
        return True
        
    except Exception as e:
        print(f"❌ 高級指標測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_with_advanced_metrics():
    """測試包含高級指標的比較功能"""
    print("\n🔬 測試包含高級指標的比較功能")
    print("=" * 50)
    
    try:
        from gnn_kan_vs_baro_comparison import GNNKANvsBAROComparator
        
        # 創建比較器
        comparator = GNNKANvsBAROComparator(output_dir="test_advanced_results")
        
        # 創建模擬數據
        n_samples = 30
        test_data = pd.DataFrame({
            'time': range(n_samples),
            'cpu_usage': np.random.rand(n_samples) * 100,
            'memory_usage': np.random.rand(n_samples) * 100,
            'network_io': np.random.rand(n_samples) * 1000,
        })
        
        print(f"📊 創建測試數據: {test_data.shape}")
        
        # 模擬運行方法（簡化版）
        print("\n🧪 模擬方法運行...")
        
        # 模擬BARO結果
        baro_result = {
            "success": True,
            "result": {
                "ranks": ["service_a", "service_b"],
                "model_info": {}
            },
            "execution_time": 25.3
        }
        
        # 模擬GNN-KAN結果
        gnn_kan_result = {
            "success": True,
            "result": {
                "ranks": ["service_b", "service_a", "service_c"],
                "model_info": {
                    'model_parameters': {'total': 35000, 'trainable': 32000},
                    'num_nodes': 8,
                    'hidden_dim': 48,
                    'sparsity_info': {
                        'sparsity_ratio': 0.3,
                        'active_connections': 700,
                        'pruned_connections': 300,
                        'total_connections': 1000
                    },
                    'memory_usage': 256
                }
            },
            "execution_time": 95.7
        }
        
        # 計算高級指標
        print("📈 計算BARO高級指標...")
        baro_advanced = comparator.calculate_advanced_metrics("baro", baro_result["result"], baro_result["execution_time"])
        
        print("📈 計算GNN-KAN高級指標...")
        gnn_kan_advanced = comparator.calculate_advanced_metrics("gnn_kan", gnn_kan_result["result"], gnn_kan_result["execution_time"])
        
        # 顯示結果
        print(f"\n📊 BARO高級指標總結:")
        print(f"  綜合評分: {baro_advanced['overall_score']:.3f}")
        print(f"  參數效率比: {baro_advanced['parameter_efficiency']['efficiency_ratio']:.2f}")
        print(f"  可解釋性評分: {baro_advanced['interpretability']['interpretability_score']:.3f}")
        print(f"  計算效率評分: {baro_advanced['computational_efficiency']['efficiency_score']:.3f}")
        
        print(f"\n📊 GNN-KAN高級指標總結:")
        print(f"  綜合評分: {gnn_kan_advanced['overall_score']:.3f}")
        print(f"  參數效率比: {gnn_kan_advanced['parameter_efficiency']['efficiency_ratio']:.2f}")
        print(f"  可解釋性評分: {gnn_kan_advanced['interpretability']['interpretability_score']:.3f}")
        print(f"  計算效率評分: {gnn_kan_advanced['computational_efficiency']['efficiency_score']:.3f}")
        
        # 比較結果
        if gnn_kan_advanced['overall_score'] > baro_advanced['overall_score']:
            print(f"\n🏆 GNN-KAN在高級指標上勝出!")
        elif baro_advanced['overall_score'] > gnn_kan_advanced['overall_score']:
            print(f"\n🏆 BARO在高級指標上勝出!")
        else:
            print(f"\n🤝 高級指標評分平手!")
        
        print("\n✅ 高級指標比較測試完成")
        return True
        
    except Exception as e:
        print(f"❌ 高級指標比較測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("🚀 高級指標功能測試套件")
    print("=" * 80)
    
    test_results = []
    
    # 測試1: 高級指標計算
    result1 = test_advanced_metrics_calculation()
    test_results.append(("高級指標計算", result1))
    
    # 測試2: 包含高級指標的比較
    result2 = test_comparison_with_advanced_metrics()
    test_results.append(("高級指標比較", result2))
    
    # 總結
    print("\n" + "=" * 80)
    print("📋 測試結果總結")
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎉 測試完成: {passed}/{total} 項測試通過")
    
    if passed == total:
        print("✅ 所有高級指標功能測試通過！")
        print("🎯 新增功能包括:")
        print("  📊 參數效率 (Parameter Efficiency)")
        print("  🔍 可解釋性 (Interpretability)")
        print("  ⚡ 計算效率 (Computational Efficiency)")
        print("  🎯 綜合評分 (Overall Score)")
    else:
        print("⚠️ 部分測試失敗，請檢查實現")

if __name__ == "__main__":
    main() 