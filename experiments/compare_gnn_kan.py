"""
Comparison script for GNN-KAN RCA vs other methods in RCAEval
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Use the correct evaluation class
from RCAEval.benchmark.evaluation import Evaluator


class RCAComparator:
    """RCA 方法比較器"""
    
    def __init__(self):
        self.methods = {
            'gnn_kan': self._run_gnn_kan,
            'causalrca': self._run_causalrca,
            'nsigma': self._run_nsigma,
            'microcause': self._run_microcause,
            'dummy': self._run_dummy
        }
        self.results = defaultdict(list)
        
    def _run_gnn_kan(self, data, inject_time, dataset):
        """執行 GNN-KAN 方法"""
        try:
            # 修正import路徑 - 使用正確的主入口點
            from RCAEval.e2e.gnnkan import gnn_kan_rca
            start_time = time.time()
            result = gnn_kan_rca(data, inject_time=inject_time, dataset=dataset)
            end_time = time.time()
            return result, end_time - start_time
        except Exception as e:
            print(f"GNN-KAN failed: {e}")
            return {"adj": np.array([]), "node_names": [], "ranks": []}, 0
    
    def _run_causalrca(self, data, inject_time, dataset):
        """執行 CausalRCA 方法"""
        try:
            from RCAEval.e2e.causalrca import causalrca
            start_time = time.time()
            result = causalrca(data, inject_time=inject_time, dataset=dataset)
            end_time = time.time()
            return result, end_time - start_time
        except Exception as e:
            print(f"CausalRCA failed: {e}")
            return {"adj": np.array([]), "node_names": [], "ranks": []}, 0
    
    def _run_nsigma(self, data, inject_time, dataset):
        """執行 NSigma 方法"""
        try:
            from RCAEval.e2e import nsigma
            start_time = time.time()
            result = nsigma(data, inject_time=inject_time, dataset=dataset)
            end_time = time.time()
            return result, end_time - start_time
        except Exception as e:
            print(f"NSigma failed: {e}")
            return {"node_names": [], "ranks": []}, 0
    
    def _run_microcause(self, data, inject_time, dataset):
        """執行 MicroCause 方法"""
        try:
            from RCAEval.e2e.microcause import microcause
            start_time = time.time()
            result = microcause(data, inject_time=inject_time, dataset=dataset)
            end_time = time.time()
            return result, end_time - start_time
        except Exception as e:
            print(f"MicroCause failed: {e}")
            return {"adj": np.array([]), "node_names": [], "ranks": []}, 0
    
    def _run_dummy(self, data, inject_time, dataset):
        """執行 Dummy 方法（基線）"""
        try:
            from RCAEval.e2e import dummy
            start_time = time.time()
            result = dummy(data, inject_time=inject_time, dataset=dataset)
            end_time = time.time()
            return result, end_time - start_time
        except Exception as e:
            print(f"Dummy failed: {e}")
            return {"adj": np.array([]), "node_names": [], "ranks": []}, 0
    
    def run_comparison(self, data, inject_time, dataset, ground_truth=None, methods=None):
        """
        運行方法比較
        
        Args:
            data: 測試數據
            inject_time: 故障注入時間
            dataset: 數據集名稱
            ground_truth: 真實根因（可選）
            methods: 要比較的方法列表（可選）
        
        Returns:
            comparison_results: 比較結果
        """
        if methods is None:
            methods = list(self.methods.keys())
        
        results = {}
        
        print(f"Running comparison with {len(methods)} methods...")
        
        for method_name in methods:
            if method_name in self.methods:
                print(f"\nRunning {method_name}...")
                
                try:
                    result, execution_time = self.methods[method_name](
                        data, inject_time, dataset
                    )
                    
                    results[method_name] = {
                        'result': result,
                        'execution_time': execution_time,
                        'success': True,
                        'error': None
                    }
                    
                    print(f"  ✓ {method_name} completed in {execution_time:.2f}s")
                    print(f"    - Nodes: {len(result.get('node_names', []))}")
                    print(f"    - Rankings: {len(result.get('ranks', []))}")
                    
                except Exception as e:
                    results[method_name] = {
                        'result': {"adj": np.array([]), "node_names": [], "ranks": []},
                        'execution_time': 0,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  ✗ {method_name} failed: {e}")
            else:
                print(f"  ⚠️ Unknown method: {method_name}")
        
        # 計算度量指標（如果有真實根因）
        if ground_truth is not None:
            for method_name, method_result in results.items():
                if method_result['success']:
                    ranks = method_result['result'].get('ranks', [])
                    if ranks:
                        # 計算 precision@k, recall@k 等指標
                        metrics = self._calculate_evaluation_metrics(ranks, ground_truth)
                        method_result['metrics'] = metrics
        
        return results
    
    def _calculate_evaluation_metrics(self, predicted_ranks, ground_truth):
        """計算評估指標"""
        metrics = {}
        
        # Precision@k 和 Recall@k
        for k in [1, 3, 5, 10]:
            if len(predicted_ranks) >= k:
                top_k = predicted_ranks[:k]
                true_positives = len(set(top_k) & set(ground_truth))
                
                precision_k = true_positives / k if k > 0 else 0
                recall_k = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
                
                metrics[f'precision@{k}'] = precision_k
                metrics[f'recall@{k}'] = recall_k
        
        # Average Precision
        average_precision = 0
        if ground_truth:
            for i, node in enumerate(predicted_ranks):
                if node in ground_truth:
                    precision_at_i = len(set(predicted_ranks[:i+1]) & set(ground_truth)) / (i+1)
                    average_precision += precision_at_i
            average_precision /= len(ground_truth)
        
        metrics['average_precision'] = average_precision
        
        # Mean Reciprocal Rank
        mrr = 0
        for i, node in enumerate(predicted_ranks):
            if node in ground_truth:
                mrr = 1.0 / (i + 1)
                break
        
        metrics['mrr'] = mrr
        
        return metrics
    
    def generate_report(self, comparison_results, save_path=None):
        """生成比較報告"""
        report = []
        report.append("=" * 60)
        report.append("GNN-KAN RCA Comparison Report")
        report.append("=" * 60)
        report.append("")
        
        # 執行時間比較
        report.append("Execution Time Comparison:")
        report.append("-" * 30)
        for method, result in comparison_results.items():
            status = "✓" if result['success'] else "✗"
            time_str = f"{result['execution_time']:.2f}s" if result['success'] else "Failed"
            report.append(f"{status} {method:15} {time_str:>10}")
        report.append("")
        
        # 結果統計
        report.append("Results Summary:")
        report.append("-" * 30)
        for method, result in comparison_results.items():
            if result['success']:
                res = result['result']
                nodes = len(res.get('node_names', []))
                ranks = len(res.get('ranks', []))
                adj_shape = res.get('adj', np.array([])).shape if hasattr(res.get('adj', []), 'shape') else 'N/A'
                report.append(f"{method:15} Nodes: {nodes:3d}, Rankings: {ranks:3d}, Adj: {adj_shape}")
            else:
                report.append(f"{method:15} Failed: {result['error']}")
        report.append("")
        
        # 性能指標（如果有）
        metrics_available = any('metrics' in result for result in comparison_results.values())
        if metrics_available:
            report.append("Performance Metrics:")
            report.append("-" * 30)
            
            metric_names = ['precision@1', 'precision@3', 'precision@5', 'mrr', 'average_precision']
            
            # 表頭
            header = "Method".ljust(15)
            for metric in metric_names:
                header += metric.ljust(12)
            report.append(header)
            report.append("-" * len(header))
            
            # 數據行
            for method, result in comparison_results.items():
                if result['success'] and 'metrics' in result:
                    line = method.ljust(15)
                    for metric in metric_names:
                        value = result['metrics'].get(metric, 0)
                        line += f"{value:.3f}".ljust(12)
                    report.append(line)
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {save_path}")
        
        return report_text


def create_synthetic_data(num_samples=500, num_metrics=10, inject_point=0.7):
    """創建合成測試數據"""
    np.random.seed(42)
    
    # 生成時間序列
    time_points = np.arange(num_samples)
    inject_time = int(num_samples * inject_point)
    
    data = {'time': time_points}
    
    # 生成正常行為的指標
    for i in range(num_metrics):
        # 基本時間序列 + 噪聲
        base_signal = np.sin(time_points * 0.1) + np.random.normal(0, 0.1, num_samples)
        
        # 在故障點後添加異常
        if i < 3:  # 前3個指標是真正的根因
            anomaly = np.zeros(num_samples)
            anomaly[inject_time:] = np.random.normal(2, 0.5, num_samples - inject_time)
            base_signal += anomaly
        
        data[f'metric_{i}'] = base_signal * 100  # 縮放到合理範圍
    
    df = pd.DataFrame(data)
    ground_truth = [f'metric_{i}' for i in range(3)]  # 前3個是真正的根因
    
    return df, inject_time, ground_truth


def run_online_boutique_comparison(data_path):
    """運行 Online Boutique 數據集比較"""
    try:
        # 嘗試載入 Online Boutique 數據
        if os.path.exists(data_path):
            # 假設數據路徑包含 metrics.csv
            metrics_file = os.path.join(data_path, 'metrics.csv')
            if os.path.exists(metrics_file):
                data = pd.read_csv(metrics_file)
                inject_time = len(data) * 0.7  # 假設故障在70%處
                
                comparator = RCAComparator()
                results = comparator.run_comparison(
                    data, inject_time, 'online-boutique'
                )
                
                report = comparator.generate_report(results)
                return results, report
            else:
                print(f"Metrics file not found: {metrics_file}")
                return None, None
        else:
            print(f"Data path not found: {data_path}")
            return None, None
            
    except Exception as e:
        print(f"Failed to run Online Boutique comparison: {e}")
        return None, None


def main():
    """主函數"""
    print("GNN-KAN RCA Comparison Tool")
    print("=" * 40)
    
    # 1. 合成數據測試
    print("\n1. Running synthetic data comparison...")
    synthetic_data, inject_time, ground_truth = create_synthetic_data()
    
    comparator = RCAComparator()
    
    # 選擇要比較的方法
    methods_to_compare = ['gnn_kan', 'nsigma', 'dummy']
    
    results = comparator.run_comparison(
        synthetic_data, 
        inject_time, 
        'synthetic',
        ground_truth=ground_truth,
        methods=methods_to_compare
    )
    
    # 生成報告
    report = comparator.generate_report(results, 'gnn_kan_comparison_report.txt')
    
    # 2. Online Boutique 數據測試（如果可用）
    print("\n2. Checking for Online Boutique data...")
    ob_data_path = 'data/online-boutique/cartservice_mem/1/'
    
    if os.path.exists(ob_data_path):
        print("Running Online Boutique comparison...")
        ob_results, ob_report = run_online_boutique_comparison(ob_data_path)
        if ob_results:
            with open('gnn_kan_ob_comparison.txt', 'w') as f:
                f.write(ob_report)
    else:
        print("Online Boutique data not found, skipping...")
    
    # 3. 創建可視化（如果有matplotlib）
    try:
        create_performance_visualization(results)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("\n" + "=" * 40)
    print("Comparison completed!")
    print("Reports saved to:")
    print("- gnn_kan_comparison_report.txt")
    if os.path.exists('gnn_kan_ob_comparison.txt'):
        print("- gnn_kan_ob_comparison.txt")


def create_performance_visualization(results):
    """創建性能可視化圖表"""
    methods = list(results.keys())
    execution_times = [results[m]['execution_time'] for m in methods]
    success_rates = [1 if results[m]['success'] else 0 for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 執行時間比較
    ax1.bar(methods, execution_times)
    ax1.set_title('Execution Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 成功率比較
    ax2.bar(methods, success_rates)
    ax2.set_title('Success Rate')
    ax2.set_ylabel('Success (1=Success, 0=Failure)')
    ax2.set_ylim(0, 1.2)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('gnn_kan_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performance visualization saved to: gnn_kan_performance_comparison.png")


if __name__ == "__main__":
    main()