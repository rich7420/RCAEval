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
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# 添加項目路徑
sys.path.insert(0, '.')

# --- 統一模組導入 ---
try:
    from RCAEval.e2e.gnnkan import gnn_kan_rca
    from RCAEval.e2e.baro import baro
    from RCAEval.benchmark.evaluation import Evaluator
    from RCAEval.utility import (
        download_re1_dataset,
        download_re2_dataset,
        download_re3_dataset,
        load_json,
        dump_json
    )
    print("✅ 核心模組與工具導入成功")
except ImportError as e:
    print(f"❌ 關鍵模組導入錯誤: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


# --- 全域配置 ---
OUTPUT_DIR = "universal_kan_results"

DATASETS = {
    "RE1-OB": {"path": "data/RE1/RE1-OB", "download_func": download_re1_dataset, "series": "RE1"},
    "RE1-SS": {"path": "data/RE1/RE1-SS", "download_func": download_re1_dataset, "series": "RE1"},
    "RE1-TT": {"path": "data/RE1/RE1-TT", "download_func": download_re1_dataset, "series": "RE1"},
    "RE2-OB": {"path": "data/RE2/RE2-OB", "download_func": download_re2_dataset, "series": "RE2"},
    "RE2-SS": {"path": "data/RE2/RE2-SS", "download_func": download_re2_dataset, "series": "RE2"},
    "RE2-TT": {"path": "data/RE2/RE2-TT", "download_func": download_re2_dataset, "series": "RE2"},
    "RE3-OB": {"path": "data/RE3/RE3-OB", "download_func": download_re3_dataset, "series": "RE3"},
    "RE3-SS": {"path": "data/RE3/RE3-SS", "download_func": download_re3_dataset, "series": "RE3"},
    "RE3-TT": {"path": "data/RE3/RE3-TT", "download_func": download_re3_dataset, "series": "RE3"},
}

TEST_PARAMS = {
    "baseline": {
        "config_type": "simplified", "feature_method": "ica", "learning_rate": 1e-4, 
        "num_epochs": 150, "sparsity_lambda": 1e-4, "description": "基礎配置 - 參考基準"
    },
    "high_accuracy": {
        "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 5e-5,
        "num_epochs": 300, "sparsity_lambda": 1e-5, "kan_grid_size": 15,
        "target_feature_dim": 128, "description": "高準確率配置 - 深度KAN網絡"
    },
    "max_capacity": {
        "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 8e-5,
        "num_epochs": 250, "sparsity_lambda": 5e-6, "kan_grid_size": 20,
        "max_edges_per_node": 10, "target_feature_dim": 256,
        "force_node_expansion": True, "description": "最大表達能力配置"
    },
    "low_sparsity": {
        "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 5e-5,
        "num_epochs": 250, "sparsity_lambda": 1e-7, "description": "低稀疏性約束 - 豐富連接"
    },
    "kpca_focused": {
        "config_type": "simplified", "feature_method": "kpca", "kpca_kernel": "rbf", 
        "learning_rate": 9e-5, "num_epochs": 200, "sparsity_lambda": 1e-4, 
        "description": "特徵提取測試 - kPCA(rbf)"
    },
    "stable_convergence": {
        "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 1e-6,
        "num_epochs": 500, "sparsity_lambda": 1e-5,
        "description": "學習率測試 - 超低學習率穩定收斂"
    },
    "deep_network": {
        "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 4e-5,
        "num_epochs": 350, "sparsity_lambda": 5e-6, "num_gnn_layers": 4,
        "hidden_dims": [256, 128, 64, 32], "description": "網絡結構測試 - 更深的GNN網絡"
    },
    "dense_graph": {
        "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 8e-5,
        "num_epochs": 250, "sparsity_lambda": 1e-6, "similarity_threshold": 0.1,
        "max_edges_per_node": 15, "description": "圖構建測試 - 更稠密的圖"
    }
}

# --- 核心功能函式 ---

def check_and_download_datasets(dataset_names: List[str]):
    """檢查數據集是否存在，如果不存在則下載"""
    print("📥 檢查並下載數據集...")
    for name in dataset_names:
        if name in DATASETS:
            dataset_info = DATASETS[name]
            # 檢查頂層目錄，例如 data/RE1
            top_level_path = os.path.join("data", dataset_info["series"])
            if not os.path.exists(top_level_path):
                print(f"  ⚠️ 數據集系列 {dataset_info['series']} 不存在，開始下載...")
                try:
                    dataset_info["download_func"]()
                    print(f"  ✅ {dataset_info['series']} 下載完成")
                except Exception as e:
                    print(f"  ❌ {dataset_info['series']} 下載失敗: {e}")
            else:
                 # 即使頂層目錄存在，還是檢查具體路徑
                 if not os.path.exists(dataset_info["path"]):
                     print(f"  ⚠️ 數據集 {name} 路徑不完整，嘗試重新下載...")
                     try:
                         dataset_info["download_func"]()
                         print(f"  ✅ {name} 下載完成")
                     except Exception as e:
                         print(f"  ❌ {name} 下載失敗: {e}")
                 else:
                     print(f"  ✅ 數據集 {name} 已存在")
        else:
            print(f"  ⚠️ 未知的數據集定義: {name}")


def get_all_case_paths(dataset_names: List[str], limit_per_dataset: int) -> List[Tuple[str, str]]:
    """
    獲取所有指定數據集的案例路徑（強健版本）
    返回一個元組列表 (案例路徑, 數據集名稱)
    """
    print(f"🔍 正在從 {dataset_names} 收集最多 {limit_per_dataset} 個案例...")
    all_paths = []
    
    for name in dataset_names:
        if name not in DATASETS:
            print(f"  ⚠️ 跳過未知數據集: {name}")
            continue
        
        dataset_path = DATASETS[name]["path"]
        if not os.path.exists(dataset_path):
            print(f"  ⚠️ 數據集路徑不存在: {dataset_path}，跳過")
            continue
            
        # 使用 glob 遞歸搜索 data.csv 文件，能處理 RE 系列的巢狀結構
        case_files = sorted(glob.glob(os.path.join(dataset_path, "**", "data.csv"), recursive=True))
        
        if not case_files:
            print(f"  ⚠️ 在 {name} 中未找到任何 'data.csv' 案例文件")
            continue
            
        # 限制每個數據集的案例數量
        limited_cases = case_files[:limit_per_dataset]
        all_paths.extend([(path, name) for path in limited_cases])
        print(f"  ✅ 從 {name} 收集了 {len(limited_cases)} 個案例")

    if not all_paths:
        print("❌ 嚴重錯誤：未找到任何可用案例！請檢查 'data' 目錄結構或下載腳本。")
        sys.exit(1)
        
    return all_paths


def get_ground_truth_from_path(case_path: str) -> List[str]:
    """從案例路徑中提取真實根因"""
    try:
        # 路徑格式: .../data/RE1/RE1-OB/adservice_cpu/1/data.csv
        parts = case_path.split(os.sep)
        service_fault = parts[-3] # 'adservice_cpu'
        service = service_fault.split('_')[0]
        return [service]
    except Exception:
        return []

def run_single_test(case_path: str, dataset_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """運行單一案例測試並返回結果"""
    try:
        data = pd.read_csv(case_path)
        inject_time_file = os.path.join(os.path.dirname(case_path), "inject_time.txt")
        if os.path.exists(inject_time_file):
            with open(inject_time_file, 'r') as f:
                inject_time = int(f.read().strip())
        else:
            inject_time = len(data) // 2
            
        # 運行 GNN-KAN
        gnn_kan_result = gnn_kan_rca(
            data=data,
            inject_time=inject_time,
            dataset=dataset_name,
            use_cuda=True,
            cpu_fallback=True,
            **params
        )
        
        # 運行 BARO 作為基準
        baro_result = baro(data, inject_time, dataset_name)
        
        # 計算分數
        ground_truth = get_ground_truth_from_path(case_path)
        evaluator = Evaluator(ground_truth)
        
        kan_score = evaluator.eval(gnn_kan_result.get("ranks", [])).get('avg@5', 0.0)
        baro_score = evaluator.eval(baro_result.get("ranks", [])).get('avg@5', 0.0)
        
        return {
            "success": True,
            "kan_score": kan_score,
            "baro_score": baro_score,
            "kan_beats_baro": kan_score > baro_score,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "kan_score": 0.0,
            "baro_score": 0.0,
            "kan_beats_baro": False,
            "error": str(e)
        }

def analyze_and_save_results(all_results: Dict, best_config: Dict):
    """分析並保存最終結果"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存詳細結果
    detailed_file = os.path.join(OUTPUT_DIR, f"detailed_results_{timestamp}.json")
    try:
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📄 詳細結果已保存至: {detailed_file}")
    except Exception as e:
        print(f"❌ 保存詳細結果失敗: {e}")

    # 2. 生成並保存總結報告
    report = []
    report.append("="*80)
    report.append("🏆 通用KAN參數尋找報告 🏆")
    report.append("="*80)
    report.append(f"📅 日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if best_config:
        report.append("\n🎉 找到最佳通用參數組! 🎉")
        report.append(f"配置名稱: {best_config['name']}")
        report.append(f"平均分數 (Avg@5): {best_config['avg_score']:.3f}")
        report.append(f"擊敗BARO的比例: {best_config['win_rate']:.1%}")
        report.append("\n--- 參數詳情 ---")
        best_params = best_config['params'].copy()
        description = best_params.pop('description', 'N/A')
        report.append(f"描述: {description}")
        report.append(json.dumps(best_params, indent=2))
        report.append("---")
    else:
        report.append("\n❌ 未能找到在所有數據集上都表現優於BARO的通用參數組。")
        report.append("建議調整 TEST_PARAMS 中的參數範圍或增加測試案例數量。")

    report.append("\n--- 各配置詳細表現 ---")
    for name, data in all_results.items():
        report.append(f"\n🔧 配置: {name}")
        report.append(f"  - 平均分數 (Avg@5): {data['avg_score']:.3f}")
        report.append(f"  - 擊敗BARO比例: {data['win_rate']:.1%}")
        report.append(f"  - 總案例數: {data['total_cases']}, 成功案例數: {data['successful_cases']}")
        report.append(f"  - 描述: {TEST_PARAMS.get(name, {}).get('description')}")

    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    report_file = os.path.join(OUTPUT_DIR, f"summary_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"📄 總結報告已保存至: {report_file}")
    
    # 3. 保存最佳參數到獨立文件
    if best_config:
        best_params_file = os.path.join(OUTPUT_DIR, "best_universal_params.json")
        with open(best_params_file, 'w') as f:
            dump_json(best_config['params'], best_params_file, indent=2)
        print(f"✅ 最佳參數已保存至: {best_params_file}")

def main():
    """主執行函式"""
    # --- 動態添加額外的測試參數 ---
    # 這樣做可以避免直接修改複雜的字典結構，提高修改成功率
    additional_params = {
        "kpca_focused": {
            "config_type": "simplified", "feature_method": "kpca", "kpca_kernel": "rbf",
            "learning_rate": 9e-5, "num_epochs": 200, "sparsity_lambda": 1e-4,
            "description": "特徵提取測試 - kPCA(rbf)"
        },
        "stable_convergence": {
            "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 1e-6,
            "num_epochs": 500, "sparsity_lambda": 1e-5,
            "description": "學習率測試 - 超低學習率穩定收斂"
        },
        "deep_network": {
            "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 4e-5,
            "num_epochs": 350, "sparsity_lambda": 5e-6, "num_gnn_layers": 4,
            "hidden_dims": [256, 128, 64, 32], "description": "網絡結構測試 - 更深的GNN網絡"
        },
        "dense_graph": {
            "config_type": "high_capacity", "feature_method": "ica", "learning_rate": 8e-5,
            "num_epochs": 250, "sparsity_lambda": 1e-6, "similarity_threshold": 0.1,
            "max_edges_per_node": 15, "description": "圖構建測試 - 更稠密的圖"
        }
    }
    TEST_PARAMS.update(additional_params)
    # --- 參數添加完成 ---

    parser = argparse.ArgumentParser(description="尋找通用GNN-KAN參數")
    parser.add_argument(
        "--datasets", nargs="+", default=["RE1-OB", "RE2-OB", "RE3-OB", "RE1-SS", "RE2-SS"],
        choices=list(DATASETS.keys()), help="要測試的數據集列表"
    )
    parser.add_argument("--limit", type=int, default=10, help="每個數據集使用的案例數量上限")
    args = parser.parse_args()
    
    print("="*50)
    print("🔍 尋找通用KAN參數組")
    print("="*50)
    print(f"🎯 目標：在 {args.datasets} 中找到最佳通用參數")
    print(f"⏱️  日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 準備數據：檢查並下載數據集，然後收集所有案例路徑
    check_and_download_datasets(args.datasets)
    all_case_paths = get_all_case_paths(args.datasets, args.limit)
    
    # 2. 遍歷所有參數配置進行測試
    all_results = {}
    
    for param_name, params in TEST_PARAMS.items():
        print(f"\n--- 測試配置: {param_name} ---")
        print(f"    描述: {params.get('description', 'N/A')}")
        
        case_results = []
        
        progress_bar = tqdm(all_case_paths, desc=f"測試 {param_name}")
        for case_path, dataset_name in progress_bar:
            result = run_single_test(case_path, dataset_name, params)
            if result["success"]:
                case_results.append(result)
            else:
                print(f"⚠️ 案例 {os.path.basename(os.path.dirname(case_path))} 失敗: {result['error']}")

        # 3. 統計當前參數配置的表現
        total_cases = len(case_results)
        if total_cases == 0:
            print("  ❌ 此配置下沒有成功運行的案例，跳過。")
            continue
            
        successful_cases = len([r for r in case_results if r['success']])
        avg_kan_score = np.mean([r['kan_score'] for r in case_results])
        win_count = sum(r['kan_beats_baro'] for r in case_results)
        win_rate = win_count / total_cases
        
        all_results[param_name] = {
            "avg_score": avg_kan_score,
            "win_rate": win_rate,
            "total_cases": len(all_case_paths),
            "successful_cases": successful_cases,
            "details": case_results
        }
        
        print(f"  📊 結果: 平均分數={avg_kan_score:.3f}, 擊敗BARO比例={win_rate:.1%}")

    # 4. 分析所有配置的結果，找出最佳通用參數
    best_config = None
    best_score = -1.0
    
    for name, data in all_results.items():
        # 標準：勝率超過60% 且 平均分最高
        if data['win_rate'] > 0.6 and data['avg_score'] > best_score:
            best_score = data['avg_score']
            best_config = {
                "name": name,
                "params": TEST_PARAMS[name],
                "avg_score": data['avg_score'],
                "win_rate": data['win_rate']
            }
            
    # 5. 保存報告和最佳參數
    analyze_and_save_results(all_results, best_config)


if __name__ == "__main__":
    main() 