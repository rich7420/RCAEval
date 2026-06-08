#!/usr/bin/env python3
"""
Systems Overhead Measurement Script
Measures training time, inference latency, and parameter count for different methods
"""

import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RCAEval.e2e.gnnkan import gnn_kan_rca
from RCAEval.e2e.gnn import gnn_rca
from RCAEval.e2e.gat import gat_rca
from RCAEval.e2e.baro import baro_rca

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def measure_training_time(method_func, data, inject_time, dataset, basis_function=None, num_epochs=1):
    """
    Measure training time (average time per epoch)
    
    Args:
        method_func: Method function
        data: Data
        inject_time: Injection time
        dataset: Dataset name
        basis_function: Basis function (only for GNN_KAN)
        num_epochs: Number of training epochs (for averaging)
    
    Returns:
        Average training time per epoch (seconds)
    """
    times = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        try:
            if basis_function:
                result = method_func(data, inject_time=inject_time, dataset=dataset, 
                                    basis_function=basis_function)
            else:
                result = method_func(data, inject_time=inject_time, dataset=dataset)
            
            end_time = time.time()
            epoch_time = end_time - start_time
            times.append(epoch_time)
            
        except Exception as e:
            print(f"  ⚠️ Training failed: {e}")
            return None
    
    return np.mean(times) if times else None

def measure_inference_latency(method_func, data, inject_time, dataset, basis_function=None, 
                             batch_size=32, num_runs=10):
    """
    Measure inference latency (average time per sample)
    
    Args:
        method_func: Method function
        data: Data
        inject_time: Injection time
        dataset: Dataset name
        basis_function: Basis function (only for GNN_KAN)
        batch_size: Batch size
        num_runs: Number of runs (for averaging)
    
    Returns:
        Average inference latency per sample (milliseconds)
    """
    latencies = []
    
    # Warm-up (to avoid first-run overhead)
    try:
        if basis_function:
            _ = method_func(data, inject_time=inject_time, dataset=dataset, 
                          basis_function=basis_function)
        else:
            _ = method_func(data, inject_time=inject_time, dataset=dataset)
    except:
        pass
    
    # Actual measurement
    for run in range(num_runs):
        start_time = time.time()
        
        try:
            if basis_function:
                result = method_func(data, inject_time=inject_time, dataset=dataset, 
                                    basis_function=basis_function)
            else:
                result = method_func(data, inject_time=inject_time, dataset=dataset)
            
            end_time = time.time()
            run_time = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(run_time)
            
        except Exception as e:
            print(f"  ⚠️ Inference failed: {e}")
            return None
    
    avg_latency = np.mean(latencies) if latencies else None
    # Convert to per-sample latency (assuming processing one sample)
    per_sample_latency = avg_latency / batch_size if avg_latency else None
    
    return per_sample_latency

def get_model_parameters(method_func, data, inject_time, dataset, basis_function=None):
    """
    Get model parameter count
    
    Args:
        method_func: Method function
        data: Data
        inject_time: Injection time
        dataset: Dataset name
        basis_function: Basis function (only for GNN_KAN)
    
    Returns:
        Parameter count (in thousands)
    """
    try:
        if basis_function:
            result = method_func(data, inject_time=inject_time, dataset=dataset, 
                               basis_function=basis_function)
        else:
            result = method_func(data, inject_time=inject_time, dataset=dataset)
        
        if result and isinstance(result, dict):
            # Try to extract model info from result
            model_info = result.get('model_info', {})
            if model_info:
                params = model_info.get('model_parameters', {})
                if params:
                    total_params = params.get('total', 0)
                    return total_params / 1000  # Convert to thousands
                
                # Fallback: get from total_parameters
                total_params = model_info.get('total_parameters', 0)
                if total_params:
                    return total_params / 1000
        
        # If unable to get from result, return None
        return None
        
    except Exception as e:
        print(f"  ⚠️ Failed to get parameter count: {e}")
        return None

def find_train_ticket_data():
    """Find Train-Ticket dataset data files"""
    data_paths = []
    
    # Possible paths
    possible_paths = [
        'data/train-ticket',
        'data/train_ticket',
        'data/mm-tt',
        'data/collected/train-ticket',
    ]
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            # Find CSV files
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.csv') and 'inject_time.txt' in os.listdir(root):
                        data_paths.append(os.path.join(root, file))
                        break  # Only take one file per directory
    
    return data_paths[:5]  # Return first 5 files for testing

def load_data(data_path):
    """Load data"""
    try:
        data = pd.read_csv(data_path)
        
        # Get injection time
        data_dir = os.path.dirname(data_path)
        inject_time_file = os.path.join(data_dir, 'inject_time.txt')
        
        if os.path.exists(inject_time_file):
            with open(inject_time_file, 'r') as f:
                inject_time = int(f.readlines()[0].strip())
        else:
            # If no injection time file, use data midpoint
            inject_time = data['time'].median() if 'time' in data.columns else len(data) // 2
        
        return data, inject_time
    except Exception as e:
        print(f"  ⚠️ Failed to load data: {e}")
        return None, None

def measure_all_methods():
    """Measure systems overhead for all methods"""
    
    print("🔬 Starting systems overhead measurement...")
    print("=" * 60)
    
    # Find Train-Ticket data
    data_paths = find_train_ticket_data()
    
    if not data_paths:
        print("❌ Train-Ticket dataset not found, please check data paths")
        return None
    
    print(f"📂 Found {len(data_paths)} data files")
    
    # Load first data file
    data, inject_time = load_data(data_paths[0])
    if data is None:
        print("❌ Unable to load data")
        return None
    
    print(f"✅ Data loaded successfully: {len(data)} rows, injection time: {inject_time}")
    print()
    
    results = {}
    
    # Define methods to test
    methods = [
        {
            'name': 'BARO',
            'func': baro_rca,
            'basis_function': None
        },
        {
            'name': 'GNN (MLP)',
            'func': gnn_rca,
            'basis_function': None
        },
        {
            'name': 'GAT',
            'func': gat_rca,
            'basis_function': None
        },
        {
            'name': 'GNN_KAN (Cheb)',
            'func': gnn_kan_rca,
            'basis_function': 'chebyshev'
        },
        {
            'name': 'GNN_KAN (PQC)',
            'func': gnn_kan_rca,
            'basis_function': 'pqc'
        }
    ]
    
    for method in methods:
        print(f"📊 Measuring {method['name']}...")
        
        method_results = {
            'training_time': None,
            'inference_latency': None,
            'parameters': None
        }
        
        # Measure training time
        print("  ⏱️  Measuring training time...")
        training_time = measure_training_time(
            method['func'], data, inject_time, 'train-ticket',
            basis_function=method['basis_function'],
            num_epochs=3  # Run 3 times for averaging
        )
        method_results['training_time'] = training_time
        if training_time:
            print(f"    Training time: {training_time:.2f} s/epoch")
        
        # Measure inference latency
        print("  ⏱️  Measuring inference latency...")
        inference_latency = measure_inference_latency(
            method['func'], data, inject_time, 'train-ticket',
            basis_function=method['basis_function'],
            batch_size=32,
            num_runs=10
        )
        method_results['inference_latency'] = inference_latency
        if inference_latency:
            print(f"    Inference latency: {inference_latency:.2f} ms/sample")
        
        # Get parameter count
        print("  📏 Getting parameter count...")
        parameters = get_model_parameters(
            method['func'], data, inject_time, 'train-ticket',
            basis_function=method['basis_function']
        )
        method_results['parameters'] = parameters
        if parameters:
            print(f"    Parameters: {parameters:.1f} K")
        
        results[method['name']] = method_results
        print()
    
    return results

def save_results(results, output_path='output/systems_overhead.json'):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")

def print_results_table(results):
    """Print results table"""
    print("\n" + "=" * 60)
    print("📊 Systems Overhead Measurement Results")
    print("=" * 60)
    print(f"{'Method':<20} {'Training Time':<15} {'Inference Latency':<18} {'Parameters':<12}")
    print(f"{'':<20} {'(s/epoch)':<15} {'(ms/sample)':<18} {'(K)':<12}")
    print("-" * 60)
    
    for method_name, method_results in results.items():
        training_time = f"{method_results['training_time']:.2f}" if method_results['training_time'] else "-"
        inference_latency = f"{method_results['inference_latency']:.2f}" if method_results['inference_latency'] else "-"
        parameters = f"{method_results['parameters']:.1f}" if method_results['parameters'] else "-"
        
        print(f"{method_name:<20} {training_time:<15} {inference_latency:<18} {parameters:<12}")
    
    print("=" * 60)

if __name__ == "__main__":
    print("🚀 Systems Overhead Measurement Tool")
    print("=" * 60)
    print("This script will measure the following metrics:")
    print("  1. Training time (seconds per epoch)")
    print("  2. Inference latency (milliseconds per sample)")
    print("  3. Parameter count (in thousands)")
    print()
    
    results = measure_all_methods()
    
    if results:
        print_results_table(results)
        save_results(results)
        
        print("\n✅ Measurement completed!")
        print("💡 Tip: Please fill the results into Table 2 of the paper")
    else:
        print("\n❌ Measurement failed, please check data paths and dependencies")

