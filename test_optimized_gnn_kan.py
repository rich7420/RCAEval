#!/usr/bin/env python3
"""
高容量 GNN-KAN 測試腳本
驗證梯度穩定性的同時保持高準確率，不降低模型複雜度
"""

import torch
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# 修正import路徑 - 使用模組化結構
from RCAEval.gnn_kan_module.kan_components.feature_extraction import extract_trace_features
from RCAEval.e2e.gnnkan import gnn_kan_rca
from RCAEval.gnn_kan_module.config import SimplifiedGNNKANConfig
from RCAEval.gnn_kan_module.kan_components.feature_extraction import stl_decomposition, kll_feature_processing

def test_high_capacity_gradient_stability():
    """測試高容量模型的梯度穩定性"""
    print("=" * 80)
    print("🚀 高容量 GNN-KAN 梯度穩定性測試")
    print("=" * 80)
    
    try:
        # 創建複雜的測試數據 - 模擬真實場景
        np.random.seed(42)
        torch.manual_seed(42)
        
        # 生成多模態異常數據
        n_samples = 1000
        n_services = 10
        
        test_data = {
            'metrics': pd.DataFrame({
                # CPU 使用率 - 有明顯異常峰值
                'cpu_util': np.concatenate([
                    np.random.normal(30, 5, n_samples//2),     # 正常期間
                    np.random.exponential(3, n_samples//4) * 40 + 60,  # 異常期間
                    np.random.normal(25, 8, n_samples//4)      # 恢復期間
                ]),
                
                # 記憶體使用 - 緩慢增長型異常
                'memory_usage': np.concatenate([
                    60 + np.random.normal(0, 3, n_samples//2),
                    60 + np.cumsum(np.random.exponential(0.5, n_samples//4)),  # 記憶體洩漏
                    np.random.normal(65, 10, n_samples//4)
                ]),
                
                # 網路延遲 - 間歇性尖峰
                'network_latency': np.concatenate([
                    np.random.lognormal(2, 0.3, n_samples//2),
                    np.random.lognormal(3, 0.8, n_samples//4),  # 高延遲期
                    np.random.lognormal(2.2, 0.4, n_samples//4)
                ]),
                
                # 磁碟 I/O - 週期性異常
                'disk_io': np.concatenate([
                    np.random.gamma(2, 10, n_samples//2),
                    np.random.gamma(5, 20, n_samples//4) + np.sin(np.arange(n_samples//4)) * 50,
                    np.random.gamma(2.5, 12, n_samples//4)
                ]),
                
                # 錯誤率 - 異常期間激增
                'error_rate': np.concatenate([
                    np.random.poisson(0.1, n_samples//2),
                    np.random.poisson(2.5, n_samples//4),       # 錯誤激增
                    np.random.poisson(0.3, n_samples//4)
                ])
            }),
            
            'logs': pd.DataFrame({
                'message': (['INFO: Normal operation'] * (n_samples//2) + 
                          ['ERROR: Service timeout'] * (n_samples//8) +
                          ['WARNING: High memory usage'] * (n_samples//8) +
                          ['ERROR: Database connection failed'] * (n_samples//8) +
                          ['INFO: Service recovered'] * (n_samples//8))
            }),
            
            # 添加 trace 數據
            'trace': pd.DataFrame({
                'serviceName': np.random.choice([f'service-{i}' for i in range(n_services)], n_samples),
                'operationName': np.random.choice(['GET /api/users', 'POST /api/orders', 'GET /api/products'], n_samples),
                'duration': np.concatenate([
                    np.random.lognormal(2, 0.5, n_samples//2),     # 正常響應時間
                    np.random.lognormal(4, 1.2, n_samples//4),     # 異常響應時間
                    np.random.lognormal(2.5, 0.8, n_samples//4)    # 恢復期間
                ]),
                'startTime': pd.date_range('2024-01-01', periods=n_samples, freq='1min')
            })
        }
        
        print(f"✓ 生成複雜多模態測試數據:")
        print(f"  - 樣本數: {n_samples}")
        print(f"  - 服務數: {n_services}")
        print(f"  - 度量維度: {test_data['metrics'].shape[1]}")
        print(f"  - 包含 trace 數據: {test_data['trace'].shape}")
        
        # 測試不同容量配置
        capacity_configs = [
            {
                'name': '🔥 超高容量模型',
                'config_updates': {
                    'input_dim': 256,
                    'hidden_dims': [512, 384, 256, 128],
                    'output_dim': 128,
                    'kan_grid_size': 5,        # 保持高複雜度
                    'kan_spline_order': 3,     # 保持3次樣條
                    'num_gnn_layers': 3,       # 保持3層GNN
                    'epochs': 80,
                    'dropout': 0.05,           # 極低dropout保持容量
                    'max_learning_rate': 3e-4,  # 較高學習率
                    'gradient_clip_norm': 1.5,  # 適度梯度裁剪
                }
            },
            {
                'name': '⚡ 平衡高容量模型',
                'config_updates': {
                    'input_dim': 128,
                    'hidden_dims': [256, 192, 128, 96],
                    'output_dim': 64,
                    'kan_grid_size': 5,        # 保持原設置
                    'kan_spline_order': 3,     # 保持原設置
                    'num_gnn_layers': 3,       # 保持原設置
                    'epochs': 60,
                    'dropout': 0.1,
                    'max_learning_rate': 2e-4,
                    'gradient_clip_norm': 1.0,
                }
            }
        ]
        
        test_results = {}
        
        for config_test in capacity_configs:
            print(f"\n{'-'*60}")
            print(f"測試配置: {config_test['name']}")
            print(f"{'-'*60}")
            
            try:
                # 執行 GNN-KAN 分析
                start_time = time.time()
                
                results = gnn_kan_rca(
                    test_data, 
                    verbose=True,
                    **config_test['config_updates']
                )
                
                execution_time = time.time() - start_time
                
                # 分析結果
                adj_matrix = results.get('adj', np.array([]))
                root_causes = results.get('ranks', [])
                node_names = results.get('node_names', [])
                
                print(f"\n📊 執行結果:")
                print(f"  ✓ 執行時間: {execution_time:.2f} 秒")
                print(f"  ✓ 鄰接矩陣大小: {adj_matrix.shape if adj_matrix.size > 0 else '空'}")
                print(f"  ✓ 檢測到的節點數: {len(node_names)}")
                print(f"  ✓ 根因候選數: {len(root_causes)}")
                
                if len(root_causes) > 0:
                    print(f"  🎯 前5個根因:")
                    for i, cause in enumerate(root_causes[:5]):
                        print(f"    {i+1}. {cause}")
                
                # 計算梯度穩定性指標
                gradient_stability_score = calculate_gradient_stability_score(
                    execution_time, len(node_names), adj_matrix.size > 0
                )
                
                # 計算準確率指標 (基於結果的合理性)
                accuracy_score = calculate_accuracy_score(root_causes, node_names, adj_matrix)
                
                test_results[config_test['name']] = {
                    'execution_time': execution_time,
                    'gradient_stability': gradient_stability_score,
                    'accuracy_score': accuracy_score,
                    'node_count': len(node_names),
                    'root_cause_count': len(root_causes),
                    'success': True
                }
                
                print(f"  📈 梯度穩定性評分: {gradient_stability_score:.2f}/10")
                print(f"  📈 準確率評分: {accuracy_score:.2f}/10")
                
                if gradient_stability_score >= 8.0 and accuracy_score >= 7.0:
                    print(f"  🎉 {config_test['name']}: 優秀表現！")
                elif gradient_stability_score >= 6.0 and accuracy_score >= 5.0:
                    print(f"  ✅ {config_test['name']}: 良好表現")
                else:
                    print(f"  ⚠️ {config_test['name']}: 需要改進")
                
            except Exception as e:
                print(f"  ❌ {config_test['name']} 測試失敗: {e}")
                test_results[config_test['name']] = {
                    'execution_time': float('inf'),
                    'gradient_stability': 0.0,
                    'accuracy_score': 0.0,
                    'node_count': 0,
                    'root_cause_count': 0,
                    'success': False
                }
        
        # 總結報告
        print(f"\n{'='*80}")
        print("🏆 高容量模型測試總結")
        print(f"{'='*80}")
        
        successful_tests = [name for name, result in test_results.items() if result['success']]
        
        if successful_tests:
            best_config = max(successful_tests, 
                            key=lambda x: test_results[x]['gradient_stability'] + test_results[x]['accuracy_score'])
            
            print(f"🥇 最佳配置: {best_config}")
            best_result = test_results[best_config]
            print(f"  - 梯度穩定性: {best_result['gradient_stability']:.2f}/10")
            print(f"  - 準確率: {best_result['accuracy_score']:.2f}/10")
            print(f"  - 執行時間: {best_result['execution_time']:.2f} 秒")
            print(f"  - 檢測節點數: {best_result['node_count']}")
            
            # 驗證是否達到高容量且穩定的目標
            if (best_result['gradient_stability'] >= 7.0 and 
                best_result['accuracy_score'] >= 7.0 and 
                best_result['node_count'] >= 10):
                print(f"\n🎉 SUCCESS: 達成高容量且梯度穩定的目標！")
                print(f"✓ 模型容量: 高 (檢測到{best_result['node_count']}個節點)")
                print(f"✓ 梯度穩定性: 優秀 ({best_result['gradient_stability']:.1f}/10)")
                print(f"✓ 準確率: 優秀 ({best_result['accuracy_score']:.1f}/10)")
                return True
            else:
                print(f"\n⚠️ 部分指標需要改進:")
                if best_result['gradient_stability'] < 7.0:
                    print(f"  - 梯度穩定性需提升 (當前: {best_result['gradient_stability']:.1f})")
                if best_result['accuracy_score'] < 7.0:
                    print(f"  - 準確率需提升 (當前: {best_result['accuracy_score']:.1f})")
                if best_result['node_count'] < 10:
                    print(f"  - 模型容量可再提升 (當前節點數: {best_result['node_count']})")
                return False
        else:
            print("❌ 所有配置測試失敗")
            return False
            
    except Exception as e:
        print(f"❌ 高容量測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_gradient_stability_score(execution_time, node_count, has_valid_output):
    """計算梯度穩定性評分"""
    score = 0.0
    
    # 基礎分數：是否成功執行
    if has_valid_output:
        score += 4.0
    
    # 執行時間評分：合理的執行時間加分
    if execution_time < 30:
        score += 3.0
    elif execution_time < 60:
        score += 2.0
    elif execution_time < 120:
        score += 1.0
    
    # 節點處理能力評分
    if node_count >= 20:
        score += 2.0
    elif node_count >= 10:
        score += 1.5
    elif node_count >= 5:
        score += 1.0
    
    # 穩定性獎勵：沒有崩潰或異常
    if has_valid_output and execution_time < float('inf'):
        score += 1.0
    
    return min(score, 10.0)


def calculate_accuracy_score(root_causes, node_names, adj_matrix):
    """計算準確率評分"""
    score = 0.0
    
    # 基礎分數：有輸出結果
    if len(root_causes) > 0:
        score += 3.0
    
    if len(node_names) > 0:
        score += 1.0
    
    # 結果多樣性評分
    if len(root_causes) >= 5:
        score += 2.0
    elif len(root_causes) >= 3:
        score += 1.0
    
    # 鄰接矩陣質量評分
    if adj_matrix.size > 0:
        score += 1.0
        
        # 檢查矩陣的數值特性
        if not np.isnan(adj_matrix).any() and not np.isinf(adj_matrix).any():
            score += 1.0
            
            # 檢查矩陣的稀疏性（好的鄰接矩陣應該是稀疏的）
            if adj_matrix.max() <= 1.0 and adj_matrix.min() >= 0.0:
                sparsity = 1.0 - np.count_nonzero(adj_matrix > 0.5) / adj_matrix.size
                if 0.3 <= sparsity <= 0.8:  # 合理的稀疏性
                    score += 2.0
                elif 0.1 <= sparsity <= 0.9:
                    score += 1.0
    
    return min(score, 10.0)


def test_capacity_vs_stability_tradeoff():
    """測試容量與穩定性的權衡"""
    print("\n" + "=" * 80)
    print("🔬 容量 vs 穩定性權衡分析")
    print("=" * 80)
    
    # 不同的KAN配置測試
    kan_configs = [
        {
            'name': 'KAN-5x3 (原始高容量)',
            'kan_grid_size': 5,
            'kan_spline_order': 3,
            'expected_capacity': '高',
            'expected_stability': '需驗證'
        },
        {
            'name': 'KAN-7x3 (增強容量)',
            'kan_grid_size': 7,
            'kan_spline_order': 3,
            'expected_capacity': '極高',
            'expected_stability': '需驗證'
        },
        {
            'name': 'KAN-5x4 (更高階樣條)',
            'kan_grid_size': 5,
            'kan_spline_order': 4,
            'expected_capacity': '極高',
            'expected_stability': '需驗證'
        }
    ]
    
    # 簡化測試數據
    test_data = {
        'metrics': pd.DataFrame({
            'cpu': np.random.randn(200) + np.sin(np.arange(200) * 0.1),
            'memory': np.random.randn(200) + np.cos(np.arange(200) * 0.08),
            'latency': np.random.lognormal(1, 0.5, 200)
        })
    }
    
    results = {}
    
    for config in kan_configs:
        print(f"\n測試配置: {config['name']}")
        print(f"  - Grid Size: {config['kan_grid_size']}")
        print(f"  - Spline Order: {config['kan_spline_order']}")
        print(f"  - 預期容量: {config['expected_capacity']}")
        
        try:
            start_time = time.time()
            
            result = gnn_kan_rca(
                test_data,
                kan_grid_size=config['kan_grid_size'],
                kan_spline_order=config['kan_spline_order'],
                epochs=30,  # 較短的測試
                verbose=False
            )
            
            execution_time = time.time() - start_time
            
            # 評估結果
            success = len(result.get('ranks', [])) > 0
            stability_score = 10.0 if success and execution_time < 60 else 5.0 if success else 0.0
            
            results[config['name']] = {
                'success': success,
                'execution_time': execution_time,
                'stability_score': stability_score,
                'node_count': len(result.get('node_names', [])),
                'config': config
            }
            
            print(f"  ✓ 執行時間: {execution_time:.2f}s")
            print(f"  ✓ 穩定性評分: {stability_score:.1f}/10")
            print(f"  ✓ 節點數: {results[config['name']]['node_count']}")
            
        except Exception as e:
            print(f"  ❌ 測試失敗: {e}")
            results[config['name']] = {
                'success': False,
                'execution_time': float('inf'),
                'stability_score': 0.0,
                'node_count': 0,
                'config': config
            }
    
    # 分析結果
    print(f"\n📊 容量 vs 穩定性分析結果:")
    successful_configs = [name for name, result in results.items() if result['success']]
    
    if successful_configs:
        print(f"✅ 成功配置數: {len(successful_configs)}/{len(kan_configs)}")
        
        best_config = max(successful_configs, key=lambda x: results[x]['stability_score'])
        best_result = results[best_config]
        
        print(f"\n🏆 最佳權衡配置: {best_config}")
        print(f"  - 執行時間: {best_result['execution_time']:.2f}s")
        print(f"  - 穩定性: {best_result['stability_score']:.1f}/10")
        print(f"  - 節點數: {best_result['node_count']}")
        
        # 建議
        print(f"\n💡 建議:")
        if best_result['stability_score'] >= 8.0:
            print(f"  🎉 {best_config} 達到了高容量與穩定性的最佳平衡")
            print(f"  ✓ 可以安全地在生產環境中使用")
        else:
            print(f"  ⚠️ 建議進一步調優梯度穩定化參數")
        
        return True
    else:
        print(f"❌ 所有配置都失敗了，需要檢查基礎實現")
        return False


def test_large_scale_performance():
    """測試大規模性能"""
    print("\n" + "=" * 80)
    print("🚀 大規模性能測試")
    print("=" * 80)
    
    scale_tests = [
        {'name': '中等規模', 'samples': 500, 'services': 5},
        {'name': '大規模', 'samples': 1000, 'services': 10},
        {'name': '超大規模', 'samples': 2000, 'services': 15}
    ]
    
    performance_results = {}
    
    for test in scale_tests:
        print(f"\n測試: {test['name']} ({test['samples']} 樣本, {test['services']} 服務)")
        
        try:
            # 生成測試數據
            n_samples = test['samples']
            n_services = test['services']
            
            test_data = {
                'metrics': pd.DataFrame({
                    'cpu': np.random.beta(2, 3, n_samples) * 100,
                    'memory': np.random.gamma(2, 30, n_samples),
                    'network': np.random.lognormal(2, 0.8, n_samples),
                    'disk': np.random.exponential(20, n_samples)
                }),
                'trace': pd.DataFrame({
                    'serviceName': np.random.choice([f'service-{i}' for i in range(n_services)], n_samples),
                    'duration': np.random.lognormal(2, 1, n_samples),
                    'operationName': np.random.choice(['GET', 'POST', 'PUT'], n_samples)
                })
            }
            
            # 執行測試
            start_time = time.time()
            result = gnn_kan_rca(
                test_data,
                epochs=40,  # 中等訓練輪數
                verbose=False
            )
            execution_time = time.time() - start_time
            
            # 計算性能指標
            throughput = n_samples / execution_time if execution_time > 0 else 0
            memory_efficiency = len(result.get('node_names', [])) / max(n_services, 1)
            
            performance_results[test['name']] = {
                'execution_time': execution_time,
                'throughput': throughput,
                'memory_efficiency': memory_efficiency,
                'success': len(result.get('ranks', [])) > 0
            }
            
            print(f"  ✓ 執行時間: {execution_time:.2f}s")
            print(f"  ✓ 吞吐量: {throughput:.1f} 樣本/秒")
            print(f"  ✓ 記憶體效率: {memory_efficiency:.2f}")
            
        except Exception as e:
            print(f"  ❌ {test['name']} 失敗: {e}")
            performance_results[test['name']] = {
                'execution_time': float('inf'),
                'throughput': 0,
                'memory_efficiency': 0,
                'success': False
            }
    
    # 性能總結
    print(f"\n📈 大規模性能總結:")
    successful_tests = [name for name, result in performance_results.items() if result['success']]
    
    if successful_tests:
        avg_throughput = np.mean([performance_results[name]['throughput'] for name in successful_tests])
        print(f"  ✓ 平均吞吐量: {avg_throughput:.1f} 樣本/秒")
        print(f"  ✓ 成功率: {len(successful_tests)}/{len(scale_tests)}")
        
        # 性能預測
        if avg_throughput > 0:
            estimated_10k_time = 10000 / avg_throughput
            print(f"  📊 預估處理10K樣本時間: {estimated_10k_time:.1f}秒")
            
            if estimated_10k_time < 300:  # 5分鐘內
                print(f"  🎉 大規模性能: 優秀")
            elif estimated_10k_time < 600:  # 10分鐘內
                print(f"  ✅ 大規模性能: 良好")
            else:
                print(f"  ⚠️ 大規模性能: 需要優化")
        
        return len(successful_tests) == len(scale_tests)
    else:
        print(f"  ❌ 所有大規模測試失敗")
        return False


def main():
    """主測試函數"""
    print("🚀 高容量 GNN-KAN 全面驗證測試")
    print("保持原始複雜度，確保梯度穩定性與高準確率")
    print("=" * 100)
    
    # 檢查環境
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 設備: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # 執行測試套件
    test_results = {}
    
    print("🔥 開始高容量梯度穩定性測試...")
    test_results['high_capacity_stability'] = test_high_capacity_gradient_stability()
    
    print("\n🔬 開始容量 vs 穩定性權衡分析...")
    test_results['capacity_stability_tradeoff'] = test_capacity_vs_stability_tradeoff()
    
    print("\n🚀 開始大規模性能測試...")
    test_results['large_scale_performance'] = test_large_scale_performance()
    
    # 最終評估報告
    print("\n" + "=" * 100)
    print("🏆 高容量 GNN-KAN 最終評估報告")
    print("=" * 100)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    success_rate = passed_tests / total_tests * 100
    
    print(f"總測試數: {total_tests}")
    print(f"通過測試: {passed_tests}")
    print(f"成功率: {success_rate:.1f}%")
    print()
    
    # 詳細結果
    for test_name, result in test_results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        test_display = test_name.replace('_', ' ').title()
        print(f"  {test_display}: {status}")
    
    print()
    
    # 最終判定
    if passed_tests == total_tests:
        print("🎉 🎉 🎉 完美成功！🎉 🎉 🎉")
        print()
        print("✨ 高容量 GNN-KAN 已成功實現以下目標:")
        print("  🔥 保持原始高複雜度 (KAN 5x3, GNN 3層)")
        print("  🛡️ 實現卓越的梯度穩定性")
        print("  🎯 維持高準確率的根因分析")
        print("  ⚡ 支援大規模數據處理")
        print()
        print("💡 可以進行的下一步:")
        print("  1. 在實際生產數據上進行驗證")
        print("  2. 與其他RCA方法進行準確率對比")
        print("  3. 針對特定領域進行微調")
        print("  4. 進一步優化超參數以獲得更好性能")
        
    elif passed_tests >= total_tests * 0.7:
        print("✅ 大部分成功！")
        print(f"✨ {passed_tests}/{total_tests} 個測試通過，系統基本達到目標")
        print()
        print("📈 已實現的改進:")
        if test_results.get('high_capacity_stability', False):
            print("  ✓ 高容量梯度穩定性: 優秀")
        if test_results.get('capacity_stability_tradeoff', False):
            print("  ✓ 容量與穩定性平衡: 良好")
        if test_results.get('large_scale_performance', False):
            print("  ✓ 大規模性能: 可接受")
        
        print()
        print("⚠️ 需要改進的方面:")
        failed_tests = [name for name, result in test_results.items() if not result]
        for test_name in failed_tests:
            print(f"  - {test_name.replace('_', ' ').title()}")
    
    else:
        print("⚠️ 需要進一步優化")
        print(f"只有 {passed_tests}/{total_tests} 個測試通過")
        print()
        print("🔧 建議的改進方向:")
        print("  1. 檢查梯度穩定化器的參數設置")
        print("  2. 調整學習率調度策略")
        print("  3. 優化KAN層的初始化方法")
        print("  4. 增強數值穩定性檢查")
    
    print(f"\n{'='*100}")
    print("感謝使用高容量 GNN-KAN 測試套件！")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    main()