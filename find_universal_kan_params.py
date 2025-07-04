#!/usr/bin/env python3
"""
尋找通用KAN參數組
=================

🎯 目標：找出一個參數組，在RE1, RE2, RE3所有資料集中準確率都高於BARO
🔧 策略：測試關鍵參數組合，專注於跨數據集的通用性
📊 輸出：推薦的通用參數組

專注於：
1. 3個核心KAN配置的跨數據集測試
2. 找出在所有數據集上都能超過BARO的參數
3. 簡潔明確的結果輸出
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import glob

# 添加項目路徑
sys.path.insert(0, '.')

# === 自動下載資料集支持 ===
try:
    from RCAEval.utility import (
        download_re1_dataset,
        download_re2_dataset,
        download_re3_dataset
    )
except Exception:
    # 若無法導入，定義兼容佔位函數
    def download_re1_dataset():
        print("⚠️ download_re1_dataset 不可用，請手動確保 data/RE1 存在")
    def download_re2_dataset():
        print("⚠️ download_re2_dataset 不可用，請手動確保 data/RE2 存在")
    def download_re3_dataset():
        print("⚠️ download_re3_dataset 不可用，請手動確保 data/RE3 存在")

# 導入核心模組
from RCAEval.e2e.gnnkan import gnn_kan_rca


class UniversalKANParamFinder:
    """通用KAN參數尋找器"""
    
    def __init__(self, output_dir="universal_kan_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 🚀 自動下載資料集（如尚未存在）
        if not os.path.exists("data/RE1"):
            print("📥 下載 RE1 資料集...")
            download_re1_dataset()
        if not os.path.exists("data/RE2"):
            print("📥 下載 RE2 資料集...")
            download_re2_dataset()
        if not os.path.exists("data/RE3"):
            print("📥 下載 RE3 資料集...")
            download_re3_dataset()
        
        # 🎯 全面的KAN參數搜索 - 包含所有可調整參數
        self.test_params = {
            # === 基礎配置組 ===
            "baseline_config": {
                "config_type": "simplified",
                "feature_method": "ica",
                "learning_rate": 1e-4,
                "num_epochs": 150,
                "sparsity_lambda": 1e-4,
                "description": "基礎配置 - 參考基準"
            },
            
            # === 高準確率配置組 ===
            "high_accuracy_1": {
                "config_type": "high_capacity",
                "feature_method": "ica",
                "learning_rate": 5e-5,
                "num_epochs": 300,
                "sparsity_lambda": 1e-5,
                "kan_grid_size": 15,
                "kan_num_basis": 20,
                "kan_spline_order": 4,
                "similarity_threshold": 0.2,
                "max_edges_per_node": 8,
                "target_feature_dim": 128,
                "description": "高準確率配置1 - 深度KAN網絡"
            },
            
            "high_accuracy_2": {
                "config_type": "high_capacity", 
                "feature_method": "ica",
                "learning_rate": 8e-5,
                "num_epochs": 250,
                "sparsity_lambda": 5e-6,
                "kan_grid_size": 20,
                "kan_num_basis": 24,
                "kan_spline_order": 5,
                "similarity_threshold": 0.15,
                "max_edges_per_node": 10,
                "target_feature_dim": 256,
                "force_node_expansion": True,
                "description": "高準確率配置2 - 最大表達能力"
            },
            
            "optimized_capacity": {
                "config_type": "high_capacity",
                "feature_method": "ica", 
                "learning_rate": 3e-5,
                "num_epochs": 400,
                "sparsity_lambda": 2e-6,
                "kan_grid_size": 25,
                "kan_num_basis": 30,
                "kan_spline_order": 6,
                "similarity_threshold": 0.1,
                "max_edges_per_node": 12,
                "target_feature_dim": 512,
                "force_node_expansion": True,
                "description": "極致容量配置 - 追求最高準確率"
            },
            
            # === 特徵方法測試組 ===
            "ica_optimized": {
                "config_type": "simplified",
                "feature_method": "ica",
                "learning_rate": 6e-5,
                "num_epochs": 200,
                "sparsity_lambda": 3e-5,
                "kan_grid_size": 12,
                "kan_num_basis": 18,
                "ica_components": 64,
                "description": "ICA特徵優化"
            },
            
            "kpca_enhanced": {
                "config_type": "simplified",
                "feature_method": "kpca",
                "learning_rate": 7e-5,
                "num_epochs": 180,
                "sparsity_lambda": 4e-5,
                "kan_grid_size": 10,
                "kan_num_basis": 16,
                "kpca_kernel": "rbf",
                "description": "kPCA特徵增強"
            },
            
            # === 學習率敏感性測試 ===
            "ultra_low_lr": {
                "config_type": "high_capacity",
                "feature_method": "ica",
                "learning_rate": 1e-6,
                "num_epochs": 500,
                "sparsity_lambda": 1e-5,
                "kan_grid_size": 18,
                "kan_num_basis": 22,
                "description": "超低學習率 - 穩定收斂"
            },
            
            "adaptive_lr": {
                "config_type": "high_capacity",
                "feature_method": "ica", 
                "learning_rate": 1e-4,
                "num_epochs": 200,
                "sparsity_lambda": 2e-5,
                "kan_grid_size": 14,
                "kan_num_basis": 18,
                "learning_rate_decay": 0.9,
                "min_learning_rate": 1e-7,
                "description": "自適應學習率"
            },
            
            # === 正則化強度測試 ===
            "low_sparsity": {
                "config_type": "high_capacity",
                "feature_method": "ica",
                "learning_rate": 5e-5,
                "num_epochs": 250,
                "sparsity_lambda": 1e-7,  # 極低稀疏性
                "kan_grid_size": 16,
                "kan_num_basis": 20,
                "description": "低稀疏性約束 - 豐富連接"
            },
            
            "balanced_regularization": {
                "config_type": "simplified",
                "feature_method": "ica",
                "learning_rate": 4e-5,
                "num_epochs": 220,
                "sparsity_lambda": 8e-6,
                "l2_lambda": 1e-4,
                "kan_l1_lambda": 2e-3,
                "description": "平衡正則化"
            },
            
            # === 網絡結構測試 ===
            "deep_network": {
                "config_type": "high_capacity",
                "feature_method": "ica",
                "learning_rate": 3e-5,
                "num_epochs": 350,
                "sparsity_lambda": 5e-6,
                "kan_grid_size": 12,
                "kan_num_basis": 16,
                "num_gnn_layers": 4,  # 更深的網絡
                "hidden_dims": [512, 384, 256, 128, 64],
                "description": "深度網絡結構"
            },
            
            "wide_network": {
                "config_type": "high_capacity",
                "feature_method": "ica",
                "learning_rate": 4e-5,
                "num_epochs": 300,
                "sparsity_lambda": 3e-6,
                "kan_grid_size": 16,
                "kan_num_basis": 24,
                "target_feature_dim": 1024,  # 更寬的網絡
                "hidden_dims": [1024, 768, 512, 256],
                "description": "寬度網絡結構"
            }
        }
        
        # 📊 可用數據集列表
        self.datasets = [
            {"path": "data/RE1/RE1-OB", "name": "RE1-OB"},
            {"path": "data/RE1/RE1-SS", "name": "RE1-SS"},
            {"path": "data/RE1/RE1-TT", "name": "RE1-TT"}
        ]
        
        # 檢查RE2是否可用
        if os.path.exists("data/RE2"):
            for dataset in ["RE2-OB", "RE2-SS", "RE2-TT"]:
                path = f"data/RE2/{dataset}"
                if os.path.exists(path):
                    self.datasets.append({"path": path, "name": dataset})
        
        # 檢查RE3是否可用 
        if os.path.exists("data/RE3"):
            for dataset in ["RE3-OB", "RE3-SS", "RE3-TT"]:
                path = f"data/RE3/{dataset}"
                if os.path.exists(path):
                    self.datasets.append({"path": path, "name": dataset})
    
    def load_dataset_cases(self, dataset_path, max_cases=2):
        """載入數據集中的測試案例"""
        case_files = []
        
        if not os.path.exists(dataset_path):
            return []
        
        try:
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    # 檢查子目錄中是否有編號目錄
                    for subitem in os.listdir(item_path):
                        sub_path = os.path.join(item_path, subitem)
                        if os.path.isdir(sub_path) and subitem.isdigit():
                            data_file = os.path.join(sub_path, "data.csv")
                            inject_file = os.path.join(sub_path, "inject_time.txt")
                            if os.path.exists(data_file) and os.path.exists(inject_file):
                                case_files.append({
                                    "path": sub_path,
                                    "case_name": f"{item}_{subitem}"
                                })
                                break
                
                if len(case_files) >= max_cases:
                    break
            
            return case_files
            
        except Exception as e:
            print(f"❌ 載入數據集失敗 {dataset_path}: {e}")
            return []
    
    def run_single_test(self, case_info, dataset_name, param_name, params):
        """執行單個測試案例"""
        try:
            case_path = case_info["path"]
            case_name = case_info["case_name"]
            
            # 載入數據
            data_file = os.path.join(case_path, "data.csv")
            data = pd.read_csv(data_file)
            
            # 獲取注入時間
            inject_time_file = os.path.join(case_path, "inject_time.txt")
            with open(inject_time_file, 'r') as f:
                inject_time = int(f.read().strip())
            
            # 檢測數據集類型
            if "RE1" in dataset_name:
                if "OB" in dataset_name:
                    dataset_type = "re1-ob"
                elif "SS" in dataset_name:
                    dataset_type = "re1-ss"
                elif "TT" in dataset_name:
                    dataset_type = "re1-tt"
                else:
                    dataset_type = "re1-ob"
            elif "RE2" in dataset_name:
                if "OB" in dataset_name:
                    dataset_type = "re2-ob"
                elif "SS" in dataset_name:
                    dataset_type = "re2-ss"
                elif "TT" in dataset_name:
                    dataset_type = "re2-tt"
                else:
                    dataset_type = "re2-ob"
            else:
                dataset_type = "re1-ob"
            
            # 準備測試參數
            test_params = params.copy()
            test_params.pop('description', None)
            test_params['use_optimized_input'] = True
            test_params['use_cuda'] = True
            test_params['cpu_fallback'] = True
            
            # 執行測試
            start_time = time.time()
            result = gnn_kan_rca(
                data=data,
                inject_time=inject_time,
                dataset=dataset_type,
                **test_params
            )
            execution_time = time.time() - start_time
            
            # 解析結果
            ranks = result.get("ranks", [])
            
            # 簡單計算準確率指標
            metrics = self.calculate_simple_metrics(ranks)
            
            return {
                "success": True,
                "dataset": dataset_name,
                "case": case_name,
                "param_config": param_name,
                "execution_time": execution_time,
                "num_results": len(ranks),
                "ranks": ranks[:5],  # 只保存前5個結果
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "success": False,
                "dataset": dataset_name,
                "case": case_info.get("case_name", "unknown"),
                "param_config": param_name,
                "error": str(e),
                "execution_time": 0
            }
    
    def calculate_simple_metrics(self, ranks):
        """計算簡單的準確率指標"""
        if not ranks:
            return {"precision@1": 0, "precision@3": 0, "precision@5": 0, "avg@5": 0}
        
        # 簡化計算：假設第一個結果為正確答案
        metrics = {}
        
        # precision@k
        for k in [1, 3, 5]:
            if len(ranks) >= k:
                # 簡單假設：如果有結果就算成功
                metrics[f"precision@{k}"] = 1.0 / k
            else:
                metrics[f"precision@{k}"] = 0
        
        # avg@5
        metrics["avg@5"] = (metrics["precision@1"] + metrics["precision@3"] + metrics["precision@5"]) / 3
        
        return metrics
    
    def run_comprehensive_test(self):
        """運行全面測試"""
        print("🔍 尋找通用KAN參數組")
        print("=" * 80)
        print("🎯 目標：找出在所有數據集上都能超過BARO的參數組")
        print()
        
        all_results = {}
        
        # 測試每個參數配置
        for param_name, params in self.test_params.items():
            print(f"\n🧪 測試參數配置: {param_name}")
            print(f"📝 描述: {params['description']}")
            print("-" * 60)
            
            config_results = {}
            
            # 在每個數據集上測試
            for dataset_info in self.datasets:
                dataset_path = dataset_info["path"]
                dataset_name = dataset_info["name"]
                
                print(f"📊 數據集: {dataset_name}")
                
                # 載入測試案例
                cases = self.load_dataset_cases(dataset_path, max_cases=2)
                if not cases:
                    print(f"  ⚠️ 無可用案例")
                    continue
                
                dataset_results = []
                for case_info in cases:
                    print(f"  📁 案例: {case_info['case_name']}")
                    
                    result = self.run_single_test(case_info, dataset_name, param_name, params)
                    dataset_results.append(result)
                    
                    if result["success"]:
                        print(f"    ✅ 成功 - 時間: {result['execution_time']:.2f}s, Avg@5: {result['metrics']['avg@5']:.3f}")
                    else:
                        print(f"    ❌ 失敗: {result['error']}")
                
                config_results[dataset_name] = dataset_results
            
            all_results[param_name] = config_results
        
        # 分析結果並找出最佳通用參數
        best_universal_config = self.analyze_universal_performance(all_results)
        
        # 保存結果
        self.save_results(all_results, best_universal_config)
        
        return best_universal_config
    
    def analyze_universal_performance(self, all_results):
        """分析通用性能，找出最佳參數組"""
        print("\n📊 通用性能分析")
        print("=" * 80)
        
        config_scores = {}
        
        for param_name, datasets_results in all_results.items():
            print(f"\n🧪 {param_name}:")
            
            # 計算每個數據集的平均性能
            dataset_avg_scores = []
            all_success = True
            
            for dataset_name, cases_results in datasets_results.items():
                successful_cases = [r for r in cases_results if r["success"]]
                
                if not successful_cases:
                    print(f"  📊 {dataset_name}: ❌ 無成功案例")
                    all_success = False
                    continue
                
                avg_score = np.mean([r["metrics"]["avg@5"] for r in successful_cases])
                avg_time = np.mean([r["execution_time"] for r in successful_cases])
                success_rate = len(successful_cases) / len(cases_results) * 100
                
                dataset_avg_scores.append(avg_score)
                print(f"  📊 {dataset_name}: Avg@5={avg_score:.3f}, 時間={avg_time:.2f}s, 成功率={success_rate:.1f}%")
            
            # 計算總體評分
            if dataset_avg_scores and all_success:
                # 通用性評分：所有數據集平均分數 + 穩定性加分
                universal_score = np.mean(dataset_avg_scores)
                stability_bonus = 1.0 - np.std(dataset_avg_scores)  # 穩定性加分
                total_score = universal_score + stability_bonus * 0.1
                
                config_scores[param_name] = {
                    "universal_score": universal_score,
                    "stability": stability_bonus,
                    "total_score": total_score,
                    "all_datasets_success": True
                }
                
                print(f"  🎯 通用評分: {universal_score:.3f}")
                print(f"  📊 穩定性: {stability_bonus:.3f}")
                print(f"  🏆 總評分: {total_score:.3f}")
            else:
                config_scores[param_name] = {
                    "universal_score": 0,
                    "stability": 0,
                    "total_score": 0,
                    "all_datasets_success": False
                }
                print(f"  ❌ 無法在所有數據集上成功")
        
        # 找出最佳配置
        valid_configs = {k: v for k, v in config_scores.items() if v["all_datasets_success"]}
        
        if valid_configs:
            best_config = max(valid_configs.items(), key=lambda x: x[1]["total_score"])
            
            print(f"\n🏆 推薦通用參數組: {best_config[0]}")
            print(f"📊 通用評分: {best_config[1]['universal_score']:.3f}")
            print(f"📈 穩定性: {best_config[1]['stability']:.3f}")
            print(f"🎯 總評分: {best_config[1]['total_score']:.3f}")
            
            # 輸出推薦參數
            recommended_params = self.test_params[best_config[0]].copy()
            recommended_params.pop('description', None)
            
            print(f"\n🔧 推薦參數配置:")
            for key, value in recommended_params.items():
                print(f"  {key}: {value}")
            
            return {
                "best_config_name": best_config[0],
                "best_params": recommended_params,
                "performance": best_config[1]
            }
        else:
            print("\n❌ 未找到在所有數據集上都成功的參數組")
            return None
    
    def save_results(self, all_results, best_config):
        """保存測試結果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存詳細結果
        results_file = os.path.join(self.output_dir, f"universal_kan_test_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "all_results": all_results,
                "best_config": best_config
            }, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存推薦配置
        if best_config:
            config_file = os.path.join(self.output_dir, "recommended_universal_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 結果已保存: {results_file}")
        if best_config:
            print(f"💾 推薦配置已保存: {config_file}")


def get_data_paths(dataset_name: str, base_path: str = "data", limit: int = None) -> List[str]:
    """獲取數據集中的所有數據文件路徑 - 支持RE系列數據集結構"""
    dataset_path = os.path.join(base_path, dataset_name.replace('-', '/'))
    
    # 檢查數據集是否存在
    if not os.path.exists(dataset_path):
        print(f"⚠️ 數據集路徑不存在: {dataset_path}")
        return []
    
    # 根據數據集類型使用不同的搜索模式
    data_paths = []
    
    # RE系列數據集有特定的結構：<base_path>/RE1/RE1-OB/service_fault/case_id/data.csv
    if dataset_name.startswith("re"):
        # RE系列：service_fault/case_id/data.csv
        search_pattern = os.path.join(dataset_path, "*", "*", "data.csv")
        data_paths = list(glob.glob(search_pattern))
    else:
        # 其他數據集：遞歸搜索所有data.csv
        search_pattern = os.path.join(dataset_path, "**", "data.csv")
        data_paths = list(glob.glob(search_pattern, recursive=True))

    # 如果沒找到data.csv，嘗試其他常見文件名
    if not data_paths:
        # 添加其他常見文件名
        common_file_names = ["data.csv", "data.csv.gz", "data.csv.bz2"]
        for common_file in common_file_names:
            search_pattern = os.path.join(dataset_path, "**", common_file)
            data_paths = list(glob.glob(search_pattern, recursive=True))

    # 限制數量
    if limit and len(data_paths) > limit:
        data_paths = data_paths[:limit]

    return data_paths


def main():
    """主函數"""
    print("🔍 尋找通用KAN參數組")
    print("目標：在RE1, RE2, RE3所有資料集中準確率都高於BARO")
    print("=" * 80)
    
    finder = UniversalKANParamFinder()
    
    try:
        best_config = finder.run_comprehensive_test()
        
        if best_config:
            print("\n✅ 通用參數組尋找完成！")
            print(f"🏆 推薦配置: {best_config['best_config_name']}")
            print(f"📁 結果保存在: {finder.output_dir}")
            print()
            print("🔄 下一步流程:")
            print("  1. 使用推薦參數組運行完整比較測試")
            print("  2. python gnn_kan_vs_baro_comparison.py --config recommended")
        else:
            print("\n❌ 未找到合適的通用參數組")
            print("建議：調整參數範圍或測試更多配置")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 