#!/usr/bin/env python3
"""
實驗腳本：重複執行不同方法、資料集和基礎函數的組合
"""

import os
import subprocess
import csv
import re
import time
from pathlib import Path

# 配置參數
METHODS = ['gnn_kan_rca', 'baro']
DATASETS = ['online-boutique', 'sock-shop-2', 'train-ticket', 're1-ob', 're1-ss', 're1-tt', 're2-ob', 're2-ss', 're2-tt']
BASIS_FUNCTIONS = ['chebyshev', 'b_spline', 'fourier', 'pqc', 'pqc_gpu']

# 路徑配置
WORKSPACE_PATH = '/Users/user/RCAEval'
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
    
    # 解析整體性能指標
    overall_pattern = r'📊 Overall Performance Metrics:\s*precision@1:\s*([\d.]+)\s*precision@3:\s*([\d.]+)\s*precision@5:\s*([\d.]+)\s*avg@5:\s*([\d.]+)'
    overall_match = re.search(overall_pattern, output_text, re.DOTALL)
    
    if overall_match:
        metrics['overall'] = {
            'precision@1': float(overall_match.group(1)),
            'precision@3': float(overall_match.group(2)),
            'precision@5': float(overall_match.group(3)),
            'avg@5': float(overall_match.group(4))
        }
    
    # 解析各故障類型的性能指標
    fault_types = ['CPU', 'MEM', 'DISK', 'SOCKET', 'DELAY', 'LOSS']
    
    for fault_type in fault_types:
        pattern = rf'📊 {fault_type} Faults:\s*precision@1:\s*([\d.]+)\s*precision@3:\s*([\d.]+)\s*precision@5:\s*([\d.]+)\s*avg@5:\s*([\d.]+)'
        match = re.search(pattern, output_text, re.DOTALL)
        
        if match:
            metrics[fault_type.lower()] = {
                'precision@1': float(match.group(1)),
                'precision@3': float(match.group(2)),
                'precision@5': float(match.group(3)),
                'avg@5': float(match.group(4))
            }
    
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

def run_experiment(method, dataset, basis_function=None):
    """執行單個實驗"""
    print(f"\n🚀 執行實驗: method={method}, dataset={dataset}, basis_function={basis_function}")
    
    # 構建命令
    cmd = [
        'python', 'originTest.py',
        '--method', method,
        '--dataset', dataset,
        '--feature_method', 'rca_aware'
    ]
    
    if basis_function:
        cmd.extend(['--basis_function', basis_function])
    
    # 執行命令
    try:
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE_PATH,
            capture_output=True,
            text=True,
            timeout=3600  # 1小時超時
        )
        
        if result.returncode == 0:
            print("✅ 實驗執行成功")
            return result.stdout
        else:
            print(f"❌ 實驗執行失敗: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("⏰ 實驗執行超時")
        return None
    except Exception as e:
        print(f"❌ 實驗執行出錯: {e}")
        return None

def main():
    """主函數"""
    print("🎯 開始執行實驗腳本")
    print(f"📁 工作目錄: {WORKSPACE_PATH}")
    print(f"📊 結果將保存至: {OUTPUT_CSV_PATH}")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    total_experiments = 0
    completed_experiments = 0
    
    # 計算總實驗數
    for dataset in DATASETS:
        for method in METHODS:
            if method == 'baro':
                total_experiments += 1  # BARO 沒有 basis_function
            else:
                total_experiments += len(BASIS_FUNCTIONS)
    
    print(f"📈 總共需要執行 {total_experiments} 個實驗")
    
    # 執行實驗
    for dataset in DATASETS:
        print(f"\n📂 處理資料集: {dataset}")
        
        for method in METHODS:
            if method == 'baro':
                # BARO 方法
                print(f"\n🔧 執行 BARO 方法")
                clean_results_directory()
                
                output = run_experiment(method, dataset)
                if output:
                    metrics = parse_performance_metrics(output)
                    if metrics:
                        write_to_csv(method, dataset, '', metrics)
                        print("📝 結果已記錄到CSV")
                    else:
                        print("⚠️ 無法解析性能指標")
                else:
                    print("❌ 實驗失敗，跳過記錄")
                
                completed_experiments += 1
                print(f"📊 進度: {completed_experiments}/{total_experiments}")
                
            else:
                # GNN-KAN-RCA 方法
                for basis_function in BASIS_FUNCTIONS:
                    print(f"\n🔧 執行 GNN-KAN-RCA 方法，基礎函數: {basis_function}")
                    clean_results_directory()
                    
                    output = run_experiment(method, dataset, basis_function)
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

