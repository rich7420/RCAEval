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
import argparse
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
    from RCAEval.utility import (
        download_re1_dataset,
        download_re2_dataset,
        download_re3_dataset,
        load_json,
        dump_json
    )
    # 針對性導入具體下載函式
    from RCAEval.utility import (
        download_re1ob_dataset, download_re1ss_dataset, download_re1tt_dataset,
        download_re2ob_dataset, download_re2ss_dataset, download_re2tt_dataset,
        download_re3ob_dataset, download_re3ss_dataset, download_re3tt_dataset
    )
    print("✅ 核心模組與工具導入成功")
except ImportError as e:
    print(f"❌ 關鍵模組導入錯誤: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")


# --- 全域配置 ---
OUTPUT_DIR = "universal_kan_results"

DATASETS = {
    "RE1-OB": {"path": "data/RE1/RE1-OB", "download_func": download_re1ob_dataset, "series": "RE1"},
    "RE1-SS": {"path": "data/RE1/RE1-SS", "download_func": download_re1ss_dataset, "series": "RE1"},
    "RE1-TT": {"path": "data/RE1/RE1-TT", "download_func": download_re1tt_dataset, "series": "RE1"},
    "RE2-OB": {"path": "data/RE2/RE2-OB", "download_func": download_re2ob_dataset, "series": "RE2"},
    "RE2-SS": {"path": "data/RE2/RE2-SS", "download_func": download_re2ss_dataset, "series": "RE2"},
    "RE2-TT": {"path": "data/RE2/RE2-TT", "download_func": download_re2tt_dataset, "series": "RE2"},
    "RE3-OB": {"path": "data/RE3/RE3-OB", "download_func": download_re3ob_dataset, "series": "RE3"},
    "RE3-SS": {"path": "data/RE3/RE3-SS", "download_func": download_re3ss_dataset, "series": "RE3"},
    "RE3-TT": {"path": "data/RE3/RE3-TT", "download_func": download_re3tt_dataset, "series": "RE3"},
}

TEST_PARAMS = {
    # 1. 稠密對照組
    "dense_baseline": {
        "graph_head": "pagerank", "config_type": "simplified", "feature_method": "ica",
        "learning_rate": 1e-4, "num_epochs": 150,
        "sparsity_lambda": 1e-6, "similarity_threshold": 0.1,
        "max_edges_per_node": 15,
        "description": "稠密對照 - 幾乎無稀疏性約束"
    },
    # 2. 中度稀疏
    "moderate_sparsity": {
        "graph_head": "pagerank", "config_type": "simplified", "feature_method": "ica",
        "learning_rate": 8e-5, "num_epochs": 200,
        "sparsity_lambda": 1e-2, "similarity_threshold": 0.4,
        "max_edges_per_node": 10,
        "description": "中度稀疏 - 兼顧準確率與稀疏度"
    },
    # 3. 高稀疏
    "high_sparsity": {
        "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
        "learning_rate": 5e-5, "num_epochs": 250,
        "sparsity_lambda": 5e-2, "similarity_threshold": 0.6,
        "max_edges_per_node": 5,
        "description": "高稀疏 - 強稀疏性懲罰"
    },
    # 4. 極端稀疏
    "extreme_sparsity": {
        "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
        "learning_rate": 5e-5, "num_epochs": 300,
        "sparsity_lambda": 1e-1, "similarity_threshold": 0.8,
        "max_edges_per_node": 3,
        "description": "極端稀疏 - 驗證稀疏極限效應"
    },
    # 5. 高容量 + 中學習率
    "high_capacity_mid_lr": {
        "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
        "learning_rate": 5e-5, "num_epochs": 300,
        "sparsity_lambda": 5e-3, "kan_grid_size": 15,
        "target_feature_dim": 128,
        "description": "高容量網絡 - 中等學習率"
    },
    # 6. 高容量 + 低學習率
    "high_capacity_low_lr": {
        "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
        "learning_rate": 1e-5, "num_epochs": 400,
        "sparsity_lambda": 1e-3, "kan_grid_size": 20,
        "target_feature_dim": 256,
        "description": "高容量網絡 - 極低學習率"
    },
    # 7. 深層 GNN
    "deep_gnn": {
        "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
        "learning_rate": 4e-5, "num_epochs": 350,
        "sparsity_lambda": 5e-4, "num_gnn_layers": 4,
        "hidden_dims": [256, 128, 64, 32],
        "description": "更深的 GNN 結構"
    },
    # 8. kPCA(rbf) 特徵
    "kpca_rbf": {
        "graph_head": "pagerank", "config_type": "simplified", "feature_method": "kpca", "kpca_kernel": "rbf",
        "learning_rate": 9e-5, "num_epochs": 200,
        "sparsity_lambda": 5e-3, "similarity_threshold": 0.4,
        "description": "特徵提取 - kPCA(rbf)"
    },
    # 9. kPCA(poly) + 高稀疏
    "kpca_poly_sparse": {
        "graph_head": "pagerank", "config_type": "simplified", "feature_method": "kpca", "kpca_kernel": "poly",
        "learning_rate": 1e-4, "num_epochs": 250,
        "sparsity_lambda": 5e-2, "similarity_threshold": 0.6,
        "max_edges_per_node": 6,
        "description": "kPCA(poly) 與高稀疏"
    },
    # 10. 快速收斂對照
    "fast_convergence": {
        "graph_head": "pagerank", "config_type": "simplified", "feature_method": "ica",
        "learning_rate": 5e-4, "num_epochs": 100,
        "sparsity_lambda": 5e-3, "similarity_threshold": 0.3,
        "description": "快速收斂 - 高學習率低輪數"
    }
}
        
# --- 核心功能函式 ---

def check_and_download_datasets(dataset_names: List[str]):
    """檢查數據集是否存在，如果不存在則下載（增強版）"""
    print("📥 檢查並下載數據集...")
    os.makedirs("data", exist_ok=True)

    for name in dataset_names:
        if name in DATASETS:
            dataset_info = DATASETS[name]
            dataset_path = dataset_info["path"]
            download_func = dataset_info["download_func"]
            
            # 確保目標父目錄存在
            parent_dir = os.path.dirname(dataset_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
        
            # 直接檢查最終路徑是否存在且非空
            if os.path.exists(dataset_path) and os.listdir(dataset_path):
                print(f"  ✅ 數據集 {name} 已存在且非空")
                continue

            print(f"  ⚠️ 數據集 {name} 不存在或為空，開始下載...")
            try:
                # 呼叫精確的下載函式，傳遞正確的目標父目錄
                download_func(local_path=parent_dir)
                
                # 下載後再次驗證
                if os.path.exists(dataset_path) and os.listdir(dataset_path):
                    print(f"  ✅ {name} 下載並驗證成功")
                else:
                    print(f"  ❌ {name} 下載後驗證失敗！目標路徑 {dataset_path} 仍不存在或為空。")
            except Exception as e:
                print(f"  ❌ {name} 下載過程中發生嚴重錯誤: {e}")
                traceback.print_exc()
        else:
            print(f"  ⚠️ 未知的數據集定義: {name}")


def get_all_case_paths(dataset_names: List[str], limit_per_dataset: int) -> List[Tuple[str, str]]:
    """
    獲取所有指定數據集的案例路徑（os.walk 強健版本）
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
            print(f"  ⚠️ 數據集路徑不存在: {dataset_path}，跳過。請檢查下載過程。")
            continue

        found_cases = []
        for root, _, files in os.walk(dataset_path):
            if "data.csv" in files:
                # 確保是案例目錄，而不是其他地方的 data.csv
                if "inject_time.txt" in files:
                    found_cases.append(os.path.join(root, "data.csv"))
        
        found_cases.sort() # 確保順序一致

        if not found_cases:
            print(f"  ⚠️ 在 {name} ({dataset_path}) 中未找到任何 'data.csv' 案例文件")
            continue

        # 限制每個數據集的案例數量
        if limit_per_dataset > 0:
            limited_cases = found_cases[:limit_per_dataset]
            print(f"  ✅ 從 {name} 限制為 {len(limited_cases)} 個案例")
        else:
            limited_cases = found_cases
            print(f"  ✅ 從 {name} 收集了 {len(limited_cases)} 個案例 (無限制)")
        
        all_paths.extend([(path, name) for path in limited_cases])

    if not all_paths:
        print("❌ 嚴重錯誤：未找到任何可用案例！請檢查 'data' 目錄結構或下載腳本。")
        sys.exit(1)

    return all_paths


def get_ground_truth_from_path(case_path: str) -> List[str]:
    """從案例路徑中提取真實根因 (服務名稱)

    規則：
    1. 從 data.csv 所在目錄往上走，尋找第一個 *非純數字* 目錄（即 fault 目錄）。
    2. 該目錄格式通常為 "<service>_<metric>"，例如
       - productcatalogservice_cpu
       - ts-auth-service_mem
    3. 取 "_" 之前的部分作為服務名稱。
    若解析失敗，回傳空列表。"""
    try:
        parts = case_path.split(os.sep)
        # 從倒數第二層開始（倒數第一層是 data.csv 所在的數字目錄）
        for part in reversed(parts[:-1]):
            # 跳過純數字目錄（案例序號）
            if part.isdigit():
                continue
            # 找到包含 "_" 的 fault 目錄
            if "_" in part:
                service = part.split("_")[0]
                if service:  # 非空
                    return [service]
            # 若沒有 "_"，但也不是純數字，直接用整個目錄名
            return [part]
        return []
    except Exception:
        return []
    
def run_single_test(case_path: str, dataset_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """執行單一案例並回傳結果字典"""
    try:
        data = pd.read_csv(case_path)
            
        # 取得注入時間
        inject_time_file = os.path.join(os.path.dirname(case_path), "inject_time.txt")
        if os.path.exists(inject_time_file):
            with open(inject_time_file, "r") as f:
                inject_time = int(f.read().strip())
        else:
            inject_time = len(data) // 2

        # 執行 GNN-KAN 與 BARO
        gnn_kan_result = gnn_kan_rca(
                data=data,
                inject_time=inject_time,
            dataset=dataset_name,
            use_cuda=True,
            cpu_fallback=True,
            **params,
        )
        baro_result = baro(data, inject_time, dataset_name)

        # 取得真實根因並評分
        ground_truth = get_ground_truth_from_path(case_path)

        def avg_at_5(pred, truth):
            hit_sum = 0
            for k in range(1, 6):
                hit = any(t in pred[:k] for t in truth)
                hit_sum += int(hit)
            return hit_sum / 5

        kan_score = avg_at_5(gnn_kan_result.get("ranks", []), ground_truth)
        baro_score = avg_at_5(baro_result.get("ranks", []), ground_truth)
        
        # 提取稀疏性信息
        training_info = gnn_kan_result.get("training_info", {})
        final_sparsity = training_info.get("final_graph_sparsity", 0.0)
        final_adj_probs = training_info.get("final_adj_probs", {"min": 0.0, "max": 0.0, "mean": 0.0})
        sparsity_metrics = training_info.get("sparsity_metrics", {"0.1": 0.0, "0.3": 0.0, "0.5": 0.0})
        
        return {
            "success": True,
            "kan_score": kan_score,
            "baro_score": baro_score,
            "kan_beats_baro": kan_score > baro_score,
            "error": None,
            "graph_sparsity": final_sparsity,
            "adj_probs": final_adj_probs,
            "sparsity_metrics": sparsity_metrics,
        }
    except Exception as e:
        return {
            "success": False,
            "kan_score": 0.0,
            "baro_score": 0.0,
            "kan_beats_baro": False,
            "error": str(e),
            "graph_sparsity": 0.0,
            "adj_probs": {"min": 0.0, "max": 0.0, "mean": 0.0},
            "sparsity_metrics": {"0.1": 0.0, "0.3": 0.0, "0.5": 0.0},
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
        
        # 添加稀疏性統計
        sparsity_stats = data.get('sparsity_stats', {})
        if sparsity_stats:
            avg_sparsity = sparsity_stats.get('avg_graph_sparsity', 0.0)
            adj_probs = sparsity_stats.get('avg_adj_probs', {})
            sparsity_metrics = sparsity_stats.get('avg_sparsity_metrics', {})
            
            report.append(f"  - 圖稀疏度: {avg_sparsity:.4f}")
            report.append(f"  - 鄰接矩陣概率: min={adj_probs.get('min', 0.0):.4f}, max={adj_probs.get('max', 0.0):.4f}, mean={adj_probs.get('mean', 0.0):.4f}")
            report.append(f"  - 多閾值稀疏度: 0.1={sparsity_metrics.get('0.1', 0.0):.3f}, 0.3={sparsity_metrics.get('0.3', 0.0):.3f}, 0.5={sparsity_metrics.get('0.5', 0.0):.3f}")
        
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
        dump_json(best_params_file, best_config['params'])
        print(f"✅ 最佳參數已保存至: {best_params_file}")

def main():
    """主執行函式"""
    # --- 動態添加額外的測試參數 ---
    # 這樣做可以避免直接修改複雜的字典結構，提高修改成功率
    additional_params = {
        "fast_conv": {
            "graph_head": "pagerank", "config_type": "simplified", "feature_method": "ica",
            "learning_rate": 5e-4, "num_epochs": 100, "sparsity_lambda": 5e-3,
            "description": "快速收斂測試 - 高學習率, 少輪數 (加強稀疏性)"
        },
        "high_sparsity": {
            "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
            "learning_rate": 5e-5, "num_epochs": 300, "sparsity_lambda": 5e-2,
            "description": "高稀疏性約束 - 尋找最關鍵鏈接 (極強稀疏性)"
        },
        "sparse_graph": {
            "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
            "learning_rate": 5e-5, "num_epochs": 250, "sparsity_lambda": 1e-2,
            "similarity_threshold": 0.5, "max_edges_per_node": 8,
            "description": "稀疏圖構建 - 高閾值 + 強稀疏性"
        },
        "ultra_sparse": {
            "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
            "learning_rate": 5e-5, "num_epochs": 300, "sparsity_lambda": 1e-1,
            "similarity_threshold": 0.6, "max_edges_per_node": 5,
            "description": "超稀疏配置 - 極強稀疏性 + 極高閾值"
        },
        "ultra_strong_sparsity": {
            "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
            "learning_rate": 5e-5, "num_epochs": 400, "sparsity_lambda": 5e-1,
            "similarity_threshold": 0.7, "max_edges_per_node": 3,
            "description": "超強稀疏性配置 - 極端稀疏化"
        },
        "extreme_sparsity": {
            "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
            "learning_rate": 5e-5, "num_epochs": 500, "sparsity_lambda": 1e0,
            "similarity_threshold": 0.8, "max_edges_per_node": 2,
            "description": "極端稀疏配置 - 最強稀疏性 + 最高閾值"
        },
        "kpca_poly": {
            "graph_head": "pagerank", "config_type": "simplified", "feature_method": "kpca", "kpca_kernel": "poly",
            "learning_rate": 1e-4, "num_epochs": 200, "sparsity_lambda": 5e-3,
            "description": "特徵提取測試 - kPCA(poly) (加強稀疏性)"
        },
        "ica_and_dense": {
            "graph_head": "pagerank", "config_type": "high_capacity", "feature_method": "ica",
            "learning_rate": 8e-5, "num_epochs": 250, "sparsity_lambda": 1e-5,
            "similarity_threshold": 0.1, "max_edges_per_node": 15,
            "description": "組合測試 - ICA特徵 + 稠密圖 (對照組)"
        }
    }
    TEST_PARAMS.update(additional_params)
    # --- 參數添加完成 ---

    parser = argparse.ArgumentParser(description="尋找通用GNN-KAN參數")
    parser.add_argument(
        "--datasets", nargs="+", default=["RE1-OB", "RE2-OB", "RE3-OB", "RE1-SS", "RE2-SS"],
        choices=list(DATASETS.keys()), help="要測試的數據集列表"
    )
    parser.add_argument("--limit", type=int, default=0, help="每個數據集使用的案例數量上限 (設為0則不限制)")
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
        
        # 計算稀疏性統計
        successful_results = [r for r in case_results if r['success']]
        if successful_results:
            avg_graph_sparsity = np.mean([r['graph_sparsity'] for r in successful_results])
            avg_adj_min = np.mean([r['adj_probs']['min'] for r in successful_results])
            avg_adj_max = np.mean([r['adj_probs']['max'] for r in successful_results])
            avg_adj_mean = np.mean([r['adj_probs']['mean'] for r in successful_results])
            avg_sparsity_01 = np.mean([r['sparsity_metrics']['0.1'] for r in successful_results])
            avg_sparsity_03 = np.mean([r['sparsity_metrics']['0.3'] for r in successful_results])
            avg_sparsity_05 = np.mean([r['sparsity_metrics']['0.5'] for r in successful_results])
        else:
            avg_graph_sparsity = 0.0
            avg_adj_min = avg_adj_max = avg_adj_mean = 0.0
            avg_sparsity_01 = avg_sparsity_03 = avg_sparsity_05 = 0.0
        
        all_results[param_name] = {
            "avg_score": avg_kan_score,
            "win_rate": win_rate,
            "total_cases": len(all_case_paths),
            "successful_cases": successful_cases,
            "details": case_results,
            "sparsity_stats": {
                "avg_graph_sparsity": avg_graph_sparsity,
                "avg_adj_probs": {"min": avg_adj_min, "max": avg_adj_max, "mean": avg_adj_mean},
                "avg_sparsity_metrics": {"0.1": avg_sparsity_01, "0.3": avg_sparsity_03, "0.5": avg_sparsity_05}
            }
        }
        
        print(f"  📊 結果: 平均分數={avg_kan_score:.3f}, 擊敗BARO比例={win_rate:.1%}")
        print(f"  📈 稀疏性: Graph Sparsity={avg_graph_sparsity:.4f}, Adj Probs(min/max/mean)={avg_adj_min:.4f}/{avg_adj_max:.4f}/{avg_adj_mean:.4f}")
        print(f"  📊 多閾值稀疏度: 0.1:{avg_sparsity_01:.3f}, 0.3:{avg_sparsity_03:.3f}, 0.5:{avg_sparsity_05:.3f}")
        
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