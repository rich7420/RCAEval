import os
import sys
import time
import json
import warnings
import argparse
import importlib
import traceback
from datetime import datetime
from typing import Dict, List, Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

warnings.filterwarnings("ignore")

# --- Constants ---
DEFAULT_DATA_DIR = "data"
DEFAULT_MODELS_DIR = "RCAEval/e2e"
DEFAULT_OUTPUT_DIR = "comparison_results"
DEFAULT_CSV_FILENAME = "comprehensive_comparison_results.csv"

def discover_models(models_dir: str) -> Dict[str, Callable]:
    """
    Dynamically discovers and imports RCA models from the specified directory.
    Assumes each file corresponds to a model and has a main function with the same name.
    e.g., 'baro.py' should have a function 'baro(...)'.
    """
    models = {}
    print(f"🔍 Discovering models in '{models_dir}'...")
    if not os.path.isdir(models_dir):
        print(f"⚠️ Models directory not found: {models_dir}")
        return {}

    for filename in os.listdir(models_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            model_name = filename[:-3]
            module_path = f"{os.path.basename(models_dir)}.{model_name}"
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, model_name):
                    models[model_name] = getattr(module, model_name)
                    print(f"  ✅ Found model: {model_name}")
                else:
                    print(f"  ⚠️ Warning: No function named '{model_name}' found in {filename}")
            except ImportError as e:
                print(f"  ❌ Error importing {module_path}: {e}")
    return models

def discover_datasets(data_dir: str, limit_per_dataset: int = None) -> List[Dict[str, str]]:
    """
    Discovering all test cases (data.csv files) within the data directory.
    """
    print(f"📂 Discovering datasets in '{data_dir}'...")
    test_cases = []
    if not os.path.isdir(data_dir):
        print(f"⚠️ Data directory not found: {data_dir}")
        return []

    # Find all subdirectories that might contain data
    dataset_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(data_dir, dataset_name)
        
        # Find all data.csv or simple_metrics.csv files
        case_paths = list(pd.read_csv(os.path.join(dataset_path, "**/data.csv"), recursive=True))
        if not case_paths:
            case_paths = list(pd.read_csv(os.path.join(dataset_path, "**/simple_metrics.csv"), recursive=True))

        if limit_per_dataset:
            case_paths = case_paths[:limit_per_dataset]

        for case_path in case_paths:
            path_parts = case_path.split(os.sep)
            try:
                # Expected format: .../dataset_name/service_fault/case_id/data.csv
                case_id = path_parts[-2]
                service_fault = path_parts[-3]
                service, fault = "unknown", "unknown"
                if "_" in service_fault:
                    service, fault = service_fault.split("_", 1)

                test_cases.append({
                    "dataset": dataset_name,
                    "service": service,
                    "fault_type": fault,
                    "case_id": case_id,
                    "path": case_path
                })
            except IndexError:
                print(f"  ⚠️ Could not parse path structure for: {case_path}")

    print(f"  📄 Found {len(test_cases)} total test cases.")
    return test_cases

def get_ground_truth(case_info: Dict[str, str]) -> List[str]:
    """
    Generates a list of possible ground truth names based on service and fault type.
    This is a simplified version of the logic in the original script.
    """
    service = case_info.get('service', 'unknown')
    if service == "unknown":
        return []
    
    # Basic ground truth is the service name itself
    ground_truth = [service]
    
    # Add common variations
    variations = [
        service.replace('-', '_'),
        f"ts-{service}",
        f"{service}-service"
    ]
    ground_truth.extend(variations)
    
    return list(set(ground_truth))


def calculate_metrics(predicted_ranks: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    Calculates evaluation metrics: P@k, R@k, F1@k, and MRR.
    """
    metrics = {}
    k_values = [1, 3, 5]

    if not predicted_ranks or not ground_truth:
        for k in k_values:
            metrics[f'precision@{k}'] = 0.0
            metrics[f'recall@{k}'] = 0.0
            metrics[f'f1@{k}'] = 0.0
        metrics['mrr'] = 0.0
        return metrics

    # Normalize names for better matching
    norm_gt = {name.lower().replace('_', '-') for name in ground_truth}

    for k in k_values:
        top_k_preds = predicted_ranks[:k]
        norm_top_k = {p.lower().replace('_', '-') for p in top_k_preds}
        
        true_positives = len(norm_top_k.intersection(norm_gt))
        
        precision = true_positives / k
        recall = true_positives / len(norm_gt)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'precision@{k}'] = precision
        metrics[f'recall@{k}'] = recall
        metrics[f'f1@{k}'] = f1

    # Calculate MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, rank in enumerate(predicted_ranks):
        norm_rank = rank.lower().replace('_', '-')
        if norm_rank in norm_gt:
            mrr = 1.0 / (i + 1)
            break
    metrics['mrr'] = mrr
    
    return metrics

def run_single_model(model_name: str, model_func: Callable, case_info: Dict[str, str]) -> Dict[str, Any]:
    """
    Runs a single model on a single test case and captures its results and metrics.
    """
    start_time = time.time()
    result_data = {
        "model_name": model_name,
        "dataset": case_info["dataset"],
        "case_id": case_info["case_id"],
        "execution_time": 0.0,
        "success": False,
        "error_message": "",
        "predicted_ranks_count": 0,
        "predicted_top_3": [],
    }
    # Initialize metrics
    for k in [1, 3, 5]:
        result_data[f'precision@{k}'] = 0.0
        result_data[f'recall@{k}'] = 0.0
        result_data[f'f1@{k}'] = 0.0
    result_data['mrr'] = 0.0

    try:
        # Load data
        data = pd.read_csv(case_info["path"])
        inject_time = len(data) // 2

        # Run the model
        # Note: We assume a unified interface for all models.
        # Some models might need more specific arguments. This is a simplification.
        # The gnnkan model is particularly complex. We'll use its default settings for now.
        if model_name == 'gnn_kan_rca':
             # Simplified call for gnn_kan, using its internal defaults
            model_output = model_func(data=data, inject_time=inject_time, dataset=case_info["dataset"])
        else:
            model_output = model_func(data=data, inject_time=inject_time, dataset=case_info["dataset"])

        execution_time = time.time() - start_time
        
        predicted_ranks = model_output.get("ranks", [])
        ground_truth = get_ground_truth(case_info)
        
        metrics = calculate_metrics(predicted_ranks, ground_truth)
        
        result_data.update({
            "execution_time": execution_time,
            "success": True,
            "predicted_ranks_count": len(predicted_ranks),
            "predicted_top_3": predicted_ranks[:3],
            **metrics
        })

    except Exception as e:
        result_data["execution_time"] = time.time() - start_time
        result_data["error_message"] = str(e).replace('\n', ' ').strip()
        traceback.print_exc()

    return result_data


def run_evaluation(models: Dict[str, Callable], test_cases: List[Dict[str, str]], output_path: str):
    """
    Main evaluation loop. Iterates through models and test cases, saving results incrementally.
    """
    results = []
    total_runs = len(models) * len(test_cases)
    
    print(f"\n🚀 Starting comprehensive evaluation for {len(models)} models across {len(test_cases)} cases ({total_runs} total runs).")
    
    # Prepare CSV file
    fieldnames = [
        "model_name", "dataset", "case_id", "execution_time", "success", 
        "precision@1", "recall@1", "f1@1",
        "precision@3", "recall@3", "f1@3",
        "precision@5", "recall@5", "f1@5",
        "mrr", "predicted_ranks_count", "predicted_top_3", "error_message"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = pd.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    with tqdm(total=total_runs, desc="Overall Progress") as pbar:
        for model_name, model_func in models.items():
            for case in test_cases:
                pbar.set_description(f"Running {model_name} on {case['dataset']}/{case['case_id']}")
                
                result = run_single_model(model_name, model_func, case)
                results.append(result)
                
                # Save incrementally to CSV
                with open(output_path, 'a', newline='') as f:
                    writer = pd.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)
                
                pbar.update(1)
    
    print(f"\n✅ Evaluation complete. Results saved to {output_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Comprehensive RCA Model Comparison Framework")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the datasets."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing the model implementation .py files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the results."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of test cases per dataset for a quick run."
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=None,
        help="Specify a subset of models to run (e.g., baro gnn_kan_rca)."
    )
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv_path = os.path.join(args.output_dir, DEFAULT_CSV_FILENAME)
    
    # --- Discovery ---
    all_models = discover_models(args.models_dir)
    if not all_models:
        print("No models found. Exiting.")
        return

    if args.models:
        models_to_run = {name: func for name, func in all_models.items() if name in args.models}
        if not models_to_run:
            print(f"Specified models not found: {args.models}. Available: {list(all_models.keys())}")
            return
    else:
        models_to_run = all_models

    test_cases = discover_datasets(args.data_dir, args.limit)
    if not test_cases:
        print("No test cases found. Exiting.")
        return

    # --- Execution ---
    run_evaluation(models_to_run, test_cases, output_csv_path)

    # --- Summary ---
    try:
        df = pd.read_csv(output_csv_path)
        print("\n--- 📊 Results Summary ---")
        
        # Overall summary
        summary = df.groupby('model_name').agg(
            num_runs=('case_id', 'count'),
            num_success=('success', lambda x: x.sum()),
            avg_time=('execution_time', 'mean'),
            avg_p1=('precision@1', 'mean'),
            avg_p3=('precision@3', 'mean'),
            avg_mrr=('mrr', 'mean')
        ).reset_index()
        
        summary['success_rate'] = (summary['num_success'] / summary['num_runs']) * 100
        
        print("Overall Performance:")
        print(summary[['model_name', 'success_rate', 'avg_time', 'avg_p1', 'avg_p3', 'avg_mrr']].round(3))
        
    except Exception as e:
        print(f"\nCould not generate summary: {e}")


if __name__ == "__main__":
    main()
