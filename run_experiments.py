#!/usr/bin/env python3
"""
實驗腳本：重複執行不同方法、資料集和基礎函數的組合
"""

import os
import subprocess
import csv
import re
import time
import argparse
from pathlib import Path

# 配置參數
# 使用快速的 metrics 方法
# baro: 最快（~5秒），統計方法
# pc_pagerank: 較快（~40秒），PC算法 + PageRank
# gnn_rca: 純GNN方法（~60秒），使用標準MLP的GNN
# gat_rca: GAT方法（~90秒），作為 GNN-KAN 的公平基線
# gatv2_rca: GATv2方法（改進的 GAT，使用動態注意力）
# graph_transformer_rca: Graph Transformer方法（現代 GNN，使用自注意力機制）
# causalrca: 因果推論方法，基於因果結構學習
METHODS = ['causalrca']
# METHODS = ['baro', 'pc_pagerank', 'gnn_rca', 'gat_rca', 'gatv2_rca', 'graph_transformer_rca', 'gnn_kan_rca']
# DATASETS = [
#     're1-ob', 're1-ss', 're1-tt',
# ]
DATASETS = [
    're1-tt','re2-ss', 're2-tt'
]
# DATASETS = [
#     # 基礎數據集
#     'online-boutique', 'sock-shop-1', 'sock-shop-2', 'train-ticket',
#     # RE1 系列
#     're1-ob', 're1-ss', 're1-tt',
#     # RE2 系列
#     're2-ob', 're2-ss', 're2-tt',
#     # RE3 系列
#     're3-ob', 're3-ss', 're3-tt',
#     # 合成數據集
#     'syn_circa', 'syn_rcd', 'syn_causil',
#     # 多源數據集
#     'multi-source'
# ]
BASIS_FUNCTIONS = ['chebyshev', 'b_spline', 'fourier', 'pqc', 'pqc_gpu']

# 路徑配置
WORKSPACE_PATH = '/Users/rich/RCAEval'
OUTPUT_CSV_PATH = os.path.join(WORKSPACE_PATH, 'output', 'output.csv')
RESULTS_DIR = os.path.join(WORKSPACE_PATH, 'output', 'results')

def clean_results_directory():
    """清空 ./output/results 目錄"""
    print("🧹 清空 results 目錄...")
    if os.path.exists(RESULTS_DIR):
        for file in os.listdir(RESULTS_DIR):
            file_path = os.path.join(RESULTS_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_performance_metrics(output_text):
    """從輸出文本中解析性能指標"""
    metrics = {}
    
    if not output_text:
        return metrics
    
    # 解析整體性能指標 - 修正正則表達式以匹配實際輸出格式
    # 實際格式是每行一個指標，使用更靈活的正則表達式
    overall_pattern = r'📊 Overall Performance Metrics:.*?precision@1:\s*([\d.]+).*?precision@3:\s*([\d.]+).*?precision@5:\s*([\d.]+).*?avg@5:\s*([\d.]+)'
    overall_match = re.search(overall_pattern, output_text, re.DOTALL | re.MULTILINE)
    
    # 如果上面的正則失敗，嘗試逐行解析
    if not overall_match:
        lines = output_text.split('\n')
        overall_metrics = {}
        in_overall_section = False
        for line in lines:
            if '📊 Overall Performance Metrics:' in line:
                in_overall_section = True
                continue
            if in_overall_section:
                if 'precision@1:' in line:
                    match = re.search(r'precision@1:\s*([\d.]+)', line)
                    if match:
                        overall_metrics['precision@1'] = float(match.group(1))
                elif 'precision@3:' in line:
                    match = re.search(r'precision@3:\s*([\d.]+)', line)
                    if match:
                        overall_metrics['precision@3'] = float(match.group(1))
                elif 'precision@5:' in line:
                    match = re.search(r'precision@5:\s*([\d.]+)', line)
                    if match:
                        overall_metrics['precision@5'] = float(match.group(1))
                elif 'avg@5:' in line:
                    match = re.search(r'avg@5:\s*([\d.]+)', line)
                    if match:
                        overall_metrics['avg@5'] = float(match.group(1))
                elif line.strip() == '' or '📊' in line or '🎯' in line:
                    # 遇到空行或新的區塊，結束解析
                    if len(overall_metrics) >= 4:
                        break
                    if '🎯' in line:
                        break
        
        if len(overall_metrics) >= 4:
            metrics['overall'] = overall_metrics
    elif overall_match:
        # 使用正則表達式匹配的結果
        metrics['overall'] = {
            'precision@1': float(overall_match.group(1)),
            'precision@3': float(overall_match.group(2)),
            'precision@5': float(overall_match.group(3)),
            'avg@5': float(overall_match.group(4))
        }
    
    # 解析各故障類型的性能指標
    fault_types = ['CPU', 'MEM', 'DISK', 'SOCKET', 'DELAY', 'LOSS']
    
    for fault_type in fault_types:
        # 嘗試多行匹配
        pattern = rf'📊 {fault_type} Faults:.*?precision@1:\s*([\d.]+).*?precision@3:\s*([\d.]+).*?precision@5:\s*([\d.]+).*?avg@5:\s*([\d.]+)'
        match = re.search(pattern, output_text, re.DOTALL | re.MULTILINE)
        
        if match:
            metrics[fault_type.lower()] = {
                'precision@1': float(match.group(1)),
                'precision@3': float(match.group(2)),
                'precision@5': float(match.group(3)),
                'avg@5': float(match.group(4))
            }
        else:
            # 如果多行匹配失敗，嘗試逐行解析
            lines = output_text.split('\n')
            fault_metrics = {}
            in_fault_section = False
            for line in lines:
                if f'📊 {fault_type} Faults:' in line:
                    in_fault_section = True
                    continue
                if in_fault_section:
                    if 'precision@1:' in line:
                        match = re.search(r'precision@1:\s*([\d.]+)', line)
                        if match:
                            fault_metrics['precision@1'] = float(match.group(1))
                    elif 'precision@3:' in line:
                        match = re.search(r'precision@3:\s*([\d.]+)', line)
                        if match:
                            fault_metrics['precision@3'] = float(match.group(1))
                    elif 'precision@5:' in line:
                        match = re.search(r'precision@5:\s*([\d.]+)', line)
                        if match:
                            fault_metrics['precision@5'] = float(match.group(1))
                    elif 'avg@5:' in line:
                        match = re.search(r'avg@5:\s*([\d.]+)', line)
                        if match:
                            fault_metrics['avg@5'] = float(match.group(1))
                    elif line.strip() == '' or '📊' in line or '🎯' in line:
                        # 遇到空行或新的區塊，結束解析
                        if len(fault_metrics) >= 4:
                            break
                        if '📊' in line and f'{fault_type} Faults:' not in line:
                            break
            
            if len(fault_metrics) >= 4:
                metrics[fault_type.lower()] = fault_metrics
    
    return metrics

def write_to_csv(method, dataset, basis_function, metrics):
    """將結果寫入CSV文件"""
    csv_exists = os.path.exists(OUTPUT_CSV_PATH)
    
    with open(OUTPUT_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['method', 'dataset', 'basis_function', 'fault_type', 'precision@1', 'precision@3', 'precision@5', 'avg@5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果是新文件，寫入標題行
        if not csv_exists:
            writer.writeheader()
        
        # 寫入整體結果
        if 'overall' in metrics:
            writer.writerow({
                'method': method,
                'dataset': dataset,
                'basis_function': basis_function,
                'fault_type': 'overall',
                'precision@1': metrics['overall']['precision@1'],
                'precision@3': metrics['overall']['precision@3'],
                'precision@5': metrics['overall']['precision@5'],
                'avg@5': metrics['overall']['avg@5']
            })
        
        # 寫入各故障類型的結果
        fault_types = ['cpu', 'mem', 'disk', 'socket', 'delay', 'loss']
        for fault_type in fault_types:
            if fault_type in metrics:
                writer.writerow({
                    'method': method,
                    'dataset': dataset,
                    'basis_function': basis_function,
                    'fault_type': fault_type,
                    'precision@1': metrics[fault_type]['precision@1'],
                    'precision@3': metrics[fault_type]['precision@3'],
                    'precision@5': metrics[fault_type]['precision@5'],
                    'avg@5': metrics[fault_type]['avg@5']
                })

def run_experiment(method, dataset, basis_function=None, test_mode=False):
    """執行單個實驗"""
    print(f"\n🚀 執行實驗: method={method}, dataset={dataset}, basis_function={basis_function}, test_mode={test_mode}")
    
    # 根據方法選擇 Python 解釋器
    # rcd 方法需要使用 Python 3.10
    # gnn_kan_rca 方法使用 Python 3.9（已安装最新版本的 torch）
    # 其他方法使用 Python 3.9
    if method == 'rcd':
        python_cmd = os.path.join(WORKSPACE_PATH, '.venv310', 'bin', 'python')
    else:
        python_cmd = os.path.join(WORKSPACE_PATH, '.venv39', 'bin', 'python')
    
    # 構建命令
    cmd = [
        python_cmd, 'originTest.py',
        '--method', method,
        '--dataset', dataset,
        '--feature_method', 'rca_aware'
    ]
    
    if test_mode:
        cmd.append('--test')
    
    if basis_function:
        cmd.extend(['--basis_function', basis_function])
    
    # 設置環境變量以允許使用 GPU（如果可用）
    env = os.environ.copy()
    # 不設置 CUDA_VISIBLE_DEVICES，讓 PyTorch 自動檢測 GPU
    # 如果系統有 GPU，PyTorch 會自動使用
    
    # 執行命令
    try:
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_PATH,
            capture_output=True,
            text=True,
            timeout=1800000 if test_mode else 1440000,  # 測試模式30分鐘，正常模式2小時超時
            env=env  # 使用環境變量（允許 GPU）
        )
        
        if result.returncode == 0:
            print("✅ 實驗執行成功")
            return result.stdout
        else:
            # 顯示更詳細的錯誤信息
            error_msg = result.stderr if result.stderr else result.stdout
            # 只顯示最後幾行錯誤信息，避免輸出過長
            error_lines = error_msg.split('\n')
            if len(error_lines) > 20:
                print(f"❌ 實驗執行失敗 (顯示最後20行):")
                for line in error_lines[-20:]:
                    if line.strip():
                        print(f"   {line}")
            else:
                print(f"❌ 實驗執行失敗:")
                for line in error_lines:
                    if line.strip():
                        print(f"   {line}")
            return None
            
    except subprocess.TimeoutExpired:
        print("⏰ 實驗執行超時")
        return None
    except Exception as e:
        print(f"❌ 實驗執行出錯: {e}")
        return None

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='執行 RCA 實驗腳本')
    parser.add_argument('--test', action='store_true', help='運行測試模式（快速驗證所有方法）')
    args = parser.parse_args()
    
    test_mode = args.test
    
    print("🎯 開始執行實驗腳本")
    if test_mode:
        print("🧪 測試模式：將快速驗證所有方法是否能正常運行")
    print(f"📁 工作目錄: {WORKSPACE_PATH}")
    print(f"📊 結果將保存至: {OUTPUT_CSV_PATH}")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    total_experiments = 0
    completed_experiments = 0
    
    # 計算總實驗數
    for dataset in DATASETS:
        for method in METHODS:
            # rcd, pc_pagerank, baro, gnn_rca, gat_rca, gatv2_rca, graph_transformer_rca, causalrca 沒有 basis_function
            if method in ['rcd', 'pc_pagerank', 'baro', 'gnn_rca', 'gat_rca', 'gatv2_rca', 'graph_transformer_rca', 'causalrca']:
                total_experiments += 1
            else:
                total_experiments += len(BASIS_FUNCTIONS)
    
    print(f"📈 總共需要執行 {total_experiments} 個實驗")
    
    # 執行實驗
    for dataset in DATASETS:
        print(f"\n📂 處理資料集: {dataset}")
        
        for method in METHODS:
            # 在測試模式下，跳過耗時的方法或只運行第一個數據集
            if test_mode and method == 'rcd' and dataset != DATASETS[0]:
                print(f"⏭️  測試模式：跳過 {method} 在 {dataset}（RCD 方法較慢，只測試第一個數據集）")
                continue
            # rcd, mscred, pc_pagerank, baro, gnn_rca, gat_rca, gatv2_rca, graph_transformer_rca, causalrca 方法不需要 basis_function
            if method in ['rcd', 'mscred', 'pc_pagerank', 'baro', 'gnn_rca', 'gat_rca', 'gatv2_rca', 'graph_transformer_rca', 'causalrca']:
                print(f"\n🔧 執行 {method.upper()} 方法")
                clean_results_directory()
                
                output = run_experiment(method, dataset, test_mode=test_mode)
                if output:
                    metrics = parse_performance_metrics(output)
                    if metrics:
                        # 調試信息：檢查解析結果
                        if 'overall' in metrics:
                            print(f"  📊 解析到的 overall 指標: precision@1={metrics['overall'].get('precision@1', 'N/A')}")
                        else:
                            print(f"  ⚠️ 警告：未解析到 overall 指標，metrics={metrics}")
                        write_to_csv(method, dataset, '', metrics)
                        print("📝 結果已記錄到CSV")
                    else:
                        print("⚠️ 無法解析性能指標（metrics 為空）")
                        # 調試：顯示輸出的最後幾行
                        if output:
                            lines = output.split('\n')
                            print(f"  輸出最後 30 行:")
                            for line in lines[-30:]:
                                if 'precision' in line.lower() or 'Overall' in line or 'Performance' in line:
                                    print(f"    {line}")
                else:
                    print("❌ 實驗失敗，跳過記錄")
                
                completed_experiments += 1
                print(f"📊 進度: {completed_experiments}/{total_experiments}")
                
            else:
                # GNN-KAN-RCA 方法
                for basis_function in BASIS_FUNCTIONS:
                    print(f"\n🔧 執行 GNN-KAN-RCA 方法，基礎函數: {basis_function}")
                    # 添加隨機延遲避免同時清空
                    import time
                    import random
                    time.sleep(random.uniform(0.1, 0.5))
                    clean_results_directory()
                    
                    output = run_experiment(method, dataset, basis_function, test_mode=test_mode)
                    if output:
                        metrics = parse_performance_metrics(output)
                        if metrics:
                            write_to_csv(method, dataset, basis_function, metrics)
                            print("📝 結果已記錄到CSV")
                        else:
                            print("⚠️ 無法解析性能指標")
                    else:
                        print("❌ 實驗失敗，跳過記錄")
                    
                    completed_experiments += 1
                    print(f"📊 進度: {completed_experiments}/{total_experiments}")
                    
                    # 短暫休息
                    time.sleep(2)
    
    print(f"\n🎉 所有實驗完成！結果已保存至 {OUTPUT_CSV_PATH}")
    print(f"📈 總共完成了 {completed_experiments}/{total_experiments} 個實驗")

if __name__ == "__main__":
    main()

