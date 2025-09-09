#!/usr/bin/env python3
"""
Single Experiment Framework for RCA Methods
==========================================

A neutral, single-method experimental framework for evaluating Root Cause Analysis (RCA) methods.
This framework is designed to provide fair and unbiased evaluation of different RCA approaches.

Features:
- Single method evaluation per run
- Neutral evaluation metrics
- Consistent data preprocessing
- Fair performance measurement
- Standardized output format

Usage:
    python experiment.py --method baro --dataset online-boutique --limit 10
    python experiment.py --method gnn_kan --dataset train-ticket --limit 5
    python experiment.py --method circa --dataset re2-ob --test
"""

import argparse
import glob
import json
import os
import time
import warnings
from datetime import datetime
from os.path import basename, dirname, join
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project path
import sys
sys.path.insert(0, '.')

# Import evaluation utilities
try:
    from RCAEval.benchmark.evaluation import Evaluator
    from RCAEval.classes.graph import Node
    from RCAEval.utility import (
        load_json, dump_json,
        download_online_boutique_dataset,
        download_sock_shop_1_dataset,
        download_sock_shop_2_dataset,
        download_train_ticket_dataset,
        download_re1_dataset,
        download_re2_dataset,
        download_re3_dataset,
        is_py38, is_py310
    )
    print("✅ Core utilities imported successfully")
except ImportError as e:
    print(f"⚠️ Some utilities not available: {e}")
    
    # Fallback implementations
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def dump_json(filename, data):
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_data = convert_numpy(data)
        with open(filename, 'w') as f:
            json.dump(converted_data, f, indent=2)
    
    class Node:
        def __init__(self, name, type_):
            self.name = name
            self.type = type_
        
        def __eq__(self, other):
            return self.name == other.name
        
        def __hash__(self):
            return hash(self.name)

# Import available RCA methods based on Python version
AVAILABLE_METHODS = {}

if is_py310() if 'is_py310' in globals() else False:
    try:
        from RCAEval.e2e import (
            baro, circa, cloudranger, dummy, e_diagnosis, easyrca,
            granger_pagerank, lingam_pagerank, micro_diag, microcause,
            microrank, mscred, nsigma, pc_pagerank, tracerca
        )
        # Import GNN-KAN separately to avoid bias
        try:
            from RCAEval.e2e.gnnkan import gnn_kan_rca
            AVAILABLE_METHODS['gnn_kan'] = gnn_kan_rca
        except ImportError:
            pass
        
        # Add other methods
        method_mapping = {
            'baro': baro,
            'circa': circa,
            'cloudranger': cloudranger,
            'dummy': dummy,
            'e_diagnosis': e_diagnosis,
            'easyrca': easyrca,
            'granger_pagerank': granger_pagerank,
            'lingam_pagerank': lingam_pagerank,
            'micro_diag': micro_diag,
            'microcause': microcause,
            'microrank': microrank,
            'mscred': mscred,
            'nsigma': nsigma,
            'pc_pagerank': pc_pagerank,
            'tracerca': tracerca
        }
        AVAILABLE_METHODS.update(method_mapping)
        
    except ImportError as e:
        print(f"⚠️ Python 3.10+ methods not available: {e}")

elif is_py38() if 'is_py38' in globals() else False:
    try:
        from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd
        AVAILABLE_METHODS.update({
            'dummy': dummy,
            'e_diagnosis': e_diagnosis,
            'ht': ht,
            'rcd': rcd,
            'mmrcd': mmrcd
        })
    except ImportError as e:
        print(f"⚠️ Python 3.8 methods not available: {e}")

# Dataset configuration
DATASET_CONFIG = {
    "online-boutique": {
        "path": "data/online-boutique",
        "download_func": download_online_boutique_dataset if 'download_online_boutique_dataset' in globals() else None,
        "description": "Online Boutique microservices system"
    },
    "sock-shop-1": {
        "path": "data/sock-shop-1",
        "download_func": download_sock_shop_1_dataset if 'download_sock_shop_1_dataset' in globals() else None,
        "description": "Sock Shop microservices system v1"
    },
    "sock-shop-2": {
        "path": "data/sock-shop-2",
        "download_func": download_sock_shop_2_dataset if 'download_sock_shop_2_dataset' in globals() else None,
        "description": "Sock Shop microservices system v2"
    },
    "train-ticket": {
        "path": "data/train-ticket",
        "download_func": download_train_ticket_dataset if 'download_train_ticket_dataset' in globals() else None,
        "description": "Train Ticket microservices system"
    },
    "re1-ob": {
        "path": "data/RE1/RE1-OB",
        "download_func": download_re1_dataset if 'download_re1_dataset' in globals() else None,
        "description": "RE1 Online Boutique dataset"
    },
    "re1-ss": {
        "path": "data/RE1/RE1-SS",
        "download_func": download_re1_dataset if 'download_re1_dataset' in globals() else None,
        "description": "RE1 Sock Shop dataset"
    },
    "re1-tt": {
        "path": "data/RE1/RE1-TT",
        "download_func": download_re1_dataset if 'download_re1_dataset' in globals() else None,
        "description": "RE1 Train Ticket dataset"
    },
    "re2-ob": {
        "path": "data/RE2/RE2-OB",
        "download_func": download_re2_dataset if 'download_re2_dataset' in globals() else None,
        "description": "RE2 Online Boutique dataset"
    },
    "re2-ss": {
        "path": "data/RE2/RE2-SS",
        "download_func": download_re2_dataset if 'download_re2_dataset' in globals() else None,
        "description": "RE2 Sock Shop dataset"
    },
    "re2-tt": {
        "path": "data/RE2/RE2-TT",
        "download_func": download_re2_dataset if 'download_re2_dataset' in globals() else None,
        "description": "RE2 Train Ticket dataset"
    },
    "re3-ob": {
        "path": "data/RE3/RE3-OB",
        "download_func": download_re3_dataset if 'download_re3_dataset' in globals() else None,
        "description": "RE3 Online Boutique dataset"
    },
    "re3-ss": {
        "path": "data/RE3/RE3-SS",
        "download_func": download_re3_dataset if 'download_re3_dataset' in globals() else None,
        "description": "RE3 Sock Shop dataset"
    },
    "re3-tt": {
        "path": "data/RE3/RE3-TT",
        "download_func": download_re3_dataset if 'download_re3_dataset' in globals() else None,
        "description": "RE3 Train Ticket dataset"
    }
}


class SingleMethodExperiment:
    """
    Single Method Experiment Framework
    
    A neutral framework for evaluating individual RCA methods without bias.
    Each experiment focuses on one method with consistent evaluation criteria.
    """
    
    def __init__(self, method_name: str, dataset_name: str, output_dir: str = "experiment_output"):
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.dataset_config = DATASET_CONFIG.get(dataset_name)
        
        if not self.dataset_config:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if method_name not in AVAILABLE_METHODS:
            raise ValueError(f"Method {method_name} not available. Available: {list(AVAILABLE_METHODS.keys())}")
        
        self.method_func = AVAILABLE_METHODS[method_name]
        
        # Check if method is actually callable
        if self.method_func is None:
            raise ValueError(f"Method {method_name} is not properly implemented (None)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.results = {
            "experiment_info": {
                "method": method_name,
                "dataset": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "dataset_description": self.dataset_config["description"]
            },
            "cases": [],
            "summary": {}
        }
    
    def download_dataset(self):
        """Download the dataset if download function is available"""
        if self.dataset_config.get("download_func"):
            print(f"📥 Downloading dataset: {self.dataset_name}")
            try:
                self.dataset_config["download_func"]()
                print(f"✅ Dataset {self.dataset_name} downloaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to download {self.dataset_name}: {e}")
        else:
            print(f"📁 Using existing dataset: {self.dataset_name}")
    
    def get_data_paths(self, limit: Optional[int] = None) -> List[str]:
        """Get data file paths from the dataset"""
        dataset_path = self.dataset_config["path"]
        
        if not os.path.exists(dataset_path):
            print(f"⚠️ Dataset path does not exist: {dataset_path}")
            return []
        
        # Search for data files
        patterns = ["**/data.csv", "**/simple_metrics.csv", "**/metrics.csv"]
        data_paths = []
        
        for pattern in patterns:
            paths = list(glob.glob(os.path.join(dataset_path, pattern), recursive=True))
            if paths:
                data_paths = paths
                break
        
        if not data_paths:
            print(f"⚠️ No data files found in {dataset_path}")
            return []
        
        data_paths = sorted(data_paths)
        
        if limit and len(data_paths) > limit:
            # Even sampling across different fault types
            from collections import defaultdict
            paths_by_fault = defaultdict(list)
            
            for path in data_paths:
                parts = path.split(os.sep)
                if len(parts) >= 3:
                    service_fault = parts[-3]
                    if "_" in service_fault:
                        fault_type = service_fault.split("_")[-1]
                        paths_by_fault[fault_type].append(path)
            
            # Sample evenly
            selected_paths = []
            fault_types = list(paths_by_fault.keys())
            if fault_types:
                paths_per_type = max(1, limit // len(fault_types))
                for fault_type in fault_types:
                    selected_paths.extend(paths_by_fault[fault_type][:paths_per_type])
                
                # Add remaining if needed
                remaining = limit - len(selected_paths)
                if remaining > 0:
                    all_remaining = [p for p in data_paths if p not in selected_paths]
                    selected_paths.extend(all_remaining[:remaining])
                
                data_paths = selected_paths[:limit]
        
        print(f"📊 Found {len(data_paths)} data files for {self.dataset_name}")
        return data_paths
    
    def extract_case_info(self, data_path: str) -> Dict[str, str]:
        """Extract case information from data path"""
        path_parts = data_path.split(os.sep)
        
        service = "unknown"
        fault_type = "unknown"
        case_id = "unknown"
        
        try:
            if len(path_parts) >= 3:
                case_folder = path_parts[-2]
                service_fault_folder = path_parts[-3]
                
                if "_" in service_fault_folder:
                    parts = service_fault_folder.split("_")
                    service = parts[0]
                    fault_type = "_".join(parts[1:])
                else:
                    service = service_fault_folder
                
                case_id = case_folder
        except Exception as e:
            print(f"⚠️ Failed to parse path info: {data_path}, error: {e}")
        
        return {
            "service": service,
            "fault_type": fault_type,
            "case_id": case_id,
            "path": data_path
        }
    
    def preprocess_data(self, data: pd.DataFrame, data_path: str) -> pd.DataFrame:
        """
        Neutral data preprocessing
        Applied consistently regardless of method
        """
        processed_data = data.copy()
        
        # Remove latency-50, keep latency-90
        processed_data = processed_data.loc[:, ~processed_data.columns.str.endswith("_latency-50")]
        
        # Handle special dataset formats
        if "mm-tt" in data_path:
            if "time" in processed_data.columns:
                time_col = processed_data["time"]
                processed_data = processed_data.loc[:, processed_data.columns.str.startswith("ts-")]
                processed_data["time"] = time_col
        
        # Handle infinite values
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        processed_data = processed_data.fillna(method="ffill")
        processed_data = processed_data.fillna(0)
        
        # Rename latency columns
        processed_data = processed_data.rename(
            columns={
                c: c.replace("_latency-90", "_latency")
                for c in processed_data.columns
                if c.endswith("_latency-90")
            }
        )
        
        return processed_data
    
    def get_inject_time(self, data_path: str, data: pd.DataFrame) -> int:
        """Get injection time from various sources"""
        data_dir = dirname(data_path)
        
        # Try to read inject_time.txt file
        inject_time_file = join(data_dir, "inject_time.txt")
        if os.path.exists(inject_time_file):
            try:
                with open(inject_time_file, 'r') as f:
                    inject_time = int(f.read().strip())
                print(f"  📅 Injection time from file: {inject_time}")
                return inject_time
            except Exception as e:
                print(f"  ⚠️ Failed to read inject_time.txt: {e}")
        
        # Fallback to data-based estimation
        if 'time' in data.columns:
            inject_time = int(data['time'].median())
            print(f"  📅 Injection time estimated (median): {inject_time}")
        else:
            inject_time = len(data) // 2
            print(f"  📅 Injection time estimated (midpoint): {inject_time}")
        
        return inject_time
    
    def get_sli(self, data: pd.DataFrame, data_path: str, case_info: Dict[str, str]) -> Optional[str]:
        """Determine Service Level Indicator (SLI) based on dataset and service"""
        service = case_info["service"]
        
        # Dataset-specific SLI determination
        if "sock-shop" in data_path:
            sli = "front-end_cpu"
            if f"{service}_latency" in data.columns:
                sli = f"{service}_latency"
            elif f"{service}_lat_90" in data.columns:
                sli = f"{service}_lat_90"
        elif "train-ticket" in data_path or "RE2-TT" in data_path:
            sli = "ts-ui-dashboard_latency"
            if f"{service}_latency" in data.columns:
                sli = f"{service}_latency"
        elif "online-boutique" in data_path or "RE2-OB" in data_path or "RE2-SS" in data_path:
            sli = "frontend_latency"
            if f"{service}_latency" in data.columns:
                sli = f"{service}_latency"
            elif "frontend_1" in data.columns:
                sli = "frontend_1"
        else:
            # Default fallback
            sli = None
            latency_cols = [col for col in data.columns if 'latency' in col.lower()]
            if latency_cols:
                sli = latency_cols[0]
        
        return sli
    
    def run_method(self, data: pd.DataFrame, inject_time: int, case_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Run the RCA method with neutral configuration
        No method-specific optimizations or bias
        """
        start_time = time.time()
        
        try:
            # Get SLI
            sli = self.get_sli(data, case_info["path"], case_info)
            
            # Basic parameters - neutral for all methods
            base_kwargs = {
                "dataset": self.dataset_name,
                "sli": sli,
                "verbose": False
            }
            
            # Method-specific parameter handling (minimal, neutral adjustments)
            if self.method_name == "gnn_kan":
                # Use default configuration without bias
                method_kwargs = {
                    **base_kwargs,
                    "config_type": "simplified",  # Use standard config
                    "feature_method": "simplified",  # Use standard features
                    "use_cuda": False,  # Neutral - don't assume GPU availability
                    "num_epochs": 100,  # Standard training
                    "learning_rate": 1e-4,  # Standard learning rate
                }
            else:
                # For other methods, use defaults
                method_kwargs = base_kwargs
            
            # Execute method
            result = self.method_func(
                data=data,
                inject_time=inject_time,
                **method_kwargs
            )
            
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "error": None,
                "parameters": method_kwargs
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "result": None,
                "execution_time": execution_time,
                "error": str(e),
                "parameters": {}
            }
    
    def get_ground_truth(self, case_info: Dict[str, str]) -> List[str]:
        """Generate ground truth based on case information"""
        service = case_info["service"]
        fault_type = case_info["fault_type"]
        
        if service == "unknown":
            return []
        
        ground_truth = [service]
        
        # Add service name variations
        variations = [
            service.replace('-', '_'),
            service.replace('_', '-'),
            service.lower()
        ]
        
        # Add train-ticket specific format
        if "train-ticket" in self.dataset_name or "tt" in self.dataset_name:
            variations.append(f"ts-{service}")
        
        # Add service suffix
        variations.append(f"{service}-service")
        
        # Add fault-specific variations
        fault_mappings = {
            "cpu": ["cpu"],
            "mem": ["memory", "mem"],
            "memory": ["memory", "mem"],
            "disk": ["disk", "io"],
            "io": ["io", "disk"],
            "latency": ["latency", "delay"],
            "delay": ["latency", "delay"],
            "loss": ["loss"],
            "network": ["network", "net"]
        }
        
        fault_indicators = fault_mappings.get(fault_type.lower(), [fault_type])
        
        # Generate service_metric combinations
        for indicator in fault_indicators:
            variations.extend([
                f"{service}_{indicator}",
                f"{service}-{indicator}"
            ])
        
        # Remove duplicates and add to ground truth
        for var in variations:
            if var != service and var not in ground_truth:
                ground_truth.append(var)
        
        return ground_truth[:10]  # Limit to reasonable size
    
    def calculate_metrics(self, predicted_ranks: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Calculate evaluation metrics neutrally
        Standard metrics without bias toward any method
        """
        if not predicted_ranks or not ground_truth:
            return {metric: 0.0 for metric in [
                'precision@1', 'precision@3', 'precision@5',
                'recall@1', 'recall@3', 'recall@5',
                'f1@1', 'f1@3', 'f1@5',
                'avg@5', 'mrr', 'hit_rate@1', 'hit_rate@3', 'hit_rate@5'
            ]}
        
        def normalize_name(name):
            """Normalize service names for comparison"""
            if not name:
                return ""
            normalized = str(name).lower().strip()
            normalized = normalized.replace('_', '-').replace('.', '-')
            # Remove common prefixes/suffixes
            prefixes = ['ts-', 'service-', 'app-']
            suffixes = ['-service', '-app', '-server']
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            for suffix in suffixes:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
            return normalized
        
        def is_match(pred_name, truth_names):
            """Check if prediction matches any ground truth"""
            pred_norm = normalize_name(pred_name)
            
            for truth_name in truth_names:
                truth_norm = normalize_name(truth_name)
                
                # Exact match
                if pred_norm == truth_norm:
                    return True
                
                # Substring match (both directions)
                if len(pred_norm) >= 3 and len(truth_norm) >= 3:
                    if pred_norm in truth_norm or truth_norm in pred_norm:
                        return True
            
            return False
        
        metrics = {}
        
        # Ensure ground_truth is list
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        # Calculate metrics for different k values
        k_values = [1, 3, 5]
        
        for k in k_values:
            top_k = predicted_ranks[:k]
            
            # Count matches
            matches = sum(1 for pred in top_k if is_match(pred, ground_truth))
            
            # Precision@k
            metrics[f'precision@{k}'] = matches / k if k > 0 else 0.0
            
            # Recall@k
            metrics[f'recall@{k}'] = matches / len(ground_truth) if len(ground_truth) > 0 else 0.0
            
            # F1@k
            p_k = metrics[f'precision@{k}']
            r_k = metrics[f'recall@{k}']
            if p_k + r_k > 0:
                metrics[f'f1@{k}'] = 2 * p_k * r_k / (p_k + r_k)
            else:
                metrics[f'f1@{k}'] = 0.0
            
            # Hit Rate@k
            metrics[f'hit_rate@{k}'] = 1.0 if matches > 0 else 0.0
        
        # Avg@5 (mean precision at positions 1-5)
        precisions = []
        for i, pred in enumerate(predicted_ranks[:5], 1):
            if is_match(pred, ground_truth):
                precisions.append(i)
        
        if precisions:
            metrics['avg@5'] = sum(1.0/pos for pos in precisions) / min(5, len(predicted_ranks))
        else:
            metrics['avg@5'] = 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, pred in enumerate(predicted_ranks, 1):
            if is_match(pred, ground_truth):
                mrr = 1.0 / i
                break
        metrics['mrr'] = mrr
        
        return metrics
    
    def run_experiment(self, limit: Optional[int] = None, test_mode: bool = False, num_cases: int = 5) -> Dict[str, Any]:
        """
        Run the complete experiment for the specified method and dataset
        """
        print(f"🧪 Starting experiment: {self.method_name} on {self.dataset_name}")
        print(f"📁 Dataset: {self.dataset_config['description']}")
        
        # Download dataset if needed
        self.download_dataset()
        
        # Get data paths - 優先使用 limit，然後是 test_mode，最後是 num_cases
        if test_mode:
            case_limit = 2
        elif limit is not None:
            case_limit = limit
        else:
            case_limit = min(num_cases, 20)  # 限制最多20個案例
        
        data_paths = self.get_data_paths(limit=case_limit)
        
        if not data_paths:
            print(f"❌ No data files found for {self.dataset_name}")
            return self.results
        
        print(f"📊 Processing {len(data_paths)} cases")
        
        # Process each case
        for i, data_path in enumerate(tqdm(data_paths, desc=f"Running {self.method_name}")):
            case_info = self.extract_case_info(data_path)
            print(f"\n🔍 Case {i+1}/{len(data_paths)}: {case_info['service']}_{case_info['fault_type']}")
            
            try:
                # Load and preprocess data
                data = pd.read_csv(data_path)
                data = self.preprocess_data(data, data_path)
                
                # Get injection time
                inject_time = self.get_inject_time(data_path, data)
                
                # Get ground truth
                ground_truth = self.get_ground_truth(case_info)
                
                # Run method
                method_result = self.run_method(data, inject_time, case_info)
                
                # Calculate metrics if successful
                if method_result["success"]:
                    ranks = method_result["result"].get("ranks", [])
                    metrics = self.calculate_metrics(ranks, ground_truth)
                    
                    print(f"  ✅ Success - Time: {method_result['execution_time']:.2f}s, Avg@5: {metrics['avg@5']:.3f}")
                else:
                    metrics = {}
                    print(f"  ❌ Failed: {method_result['error']}")
                
                # Store case result
                case_result = {
                    "case_info": case_info,
                    "ground_truth": ground_truth,
                    "inject_time": inject_time,
                    "method_result": method_result,
                    "metrics": metrics
                }
                
                self.results["cases"].append(case_result)
                
            except Exception as e:
                print(f"  💥 Case processing failed: {e}")
                continue
        
        # Calculate summary
        self.calculate_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def calculate_summary(self):
        """Calculate experiment summary statistics with fault type grouping"""
        cases = self.results["cases"]
        
        if not cases:
            self.results["summary"] = {
                "total_cases": 0,
                "successful_cases": 0,
                "success_rate": 0.0,
                "average_metrics": {},
                "average_execution_time": 0.0,
                "metrics_by_fault_type": {}
            }
            return
        
        successful_cases = [case for case in cases if case["method_result"]["success"]]
        
        summary = {
            "total_cases": len(cases),
            "successful_cases": len(successful_cases),
            "success_rate": len(successful_cases) / len(cases) if cases else 0.0,
            "average_execution_time": 0.0,
            "average_metrics": {},
            "metrics_by_fault_type": {}  # 🆕 新增故障類型分組
        }
        
        if successful_cases:
            # Calculate average execution time
            times = [case["method_result"]["execution_time"] for case in successful_cases]
            summary["average_execution_time"] = np.mean(times)
            
            # Calculate average metrics
            metric_names = ['precision@1', 'precision@3', 'precision@5', 'avg@5', 'mrr']
            avg_metrics = {}
            
            for metric in metric_names:
                values = [case["metrics"].get(metric, 0.0) for case in successful_cases]
                avg_metrics[metric] = np.mean(values) if values else 0.0
            
            summary["average_metrics"] = avg_metrics
            
            # 🆕 計算按故障類型分組的指標
            summary["metrics_by_fault_type"] = self._calculate_metrics_by_fault_type(successful_cases)
        
        self.results["summary"] = summary
    
    def _calculate_metrics_by_fault_type(self, successful_cases):
        """計算按故障類型分組的指標"""
        fault_type_metrics = {}
        
        # 按故障類型分組
        for case in successful_cases:
            case_info = case.get('case_info', {})
            fault_type = case_info.get('fault_type', 'unknown')
            
            if fault_type not in fault_type_metrics:
                fault_type_metrics[fault_type] = []
            
            fault_type_metrics[fault_type].append(case['metrics'])
        
        # 計算每個故障類型的平均指標
        avg_by_fault = {}
        for fault_type, metrics_list in fault_type_metrics.items():
            if not metrics_list:
                continue
                
            fault_avg = {}
            metric_names = ['precision@1', 'precision@3', 'precision@5', 'avg@5', 'mrr']
            
            for metric in metric_names:
                values = [m.get(metric, 0.0) for m in metrics_list]
                fault_avg[metric] = np.mean(values) if values else 0.0
            
            avg_by_fault[fault_type] = fault_avg
        
        return avg_by_fault
    
    def save_results(self):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{self.method_name}_{self.dataset_name}_{timestamp}.json"
        filepath = join(self.output_dir, filename)
        
        try:
            dump_json(filepath, self.results)
            print(f"📄 Results saved: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def print_summary(self):
        """Print experiment summary with fault type grouping"""
        summary = self.results["summary"]
        experiment_info = self.results["experiment_info"]
        
        print("\n" + "="*60)
        print(f"📊 Experiment Summary: {experiment_info['method']} on {experiment_info['dataset']}")
        print("="*60)
        
        print(f"📁 Dataset: {experiment_info['dataset_description']}")
        print(f"🧪 Method: {experiment_info['method']}")
        print(f"📅 Timestamp: {experiment_info['timestamp']}")
        print()
        
        print(f"📈 Results:")
        print(f"  Total cases: {summary['total_cases']}")
        print(f"  Successful cases: {summary['successful_cases']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  Average execution time: {summary['average_execution_time']:.2f}s")
        print()
        
        # 🆕 顯示整體性能指標
        if summary["average_metrics"]:
            print(f"📊 Overall Performance Metrics:")
            metrics = summary["average_metrics"]
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
            print()
        
        # 🆕 顯示按故障類型分組的指標
        if summary.get("metrics_by_fault_type"):
            print(f"🎯 Performance Metrics by Fault Type:")
            fault_metrics = summary["metrics_by_fault_type"]
            
            # 定義故障類型顯示名稱
            fault_type_display = {
                'cpu': 'CPU',
                'mem': 'MEM', 
                'memory': 'MEM',
                'disk': 'DISK',
                'io': 'DISK',
                'latency': 'DELAY',
                'delay': 'DELAY',
                'loss': 'LOSS',
                'network': 'NET',
                'unknown': 'UNKNOWN'
            }
            
            for fault_type, metrics in fault_metrics.items():
                display_name = fault_type_display.get(fault_type.lower(), fault_type.upper())
                print(f"\n  📊 {display_name} Faults:")
                
                # 顯示關鍵指標
                key_metrics = ['precision@1', 'precision@3', 'precision@5', 'avg@5', 'mrr']
                for metric in key_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        print(f"    {metric}: {value:.3f}")
        
        print("="*60)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Single Method RCA Experiment Framework")
    
    parser.add_argument("--method", type=str, required=True,
                       choices=list(AVAILABLE_METHODS.keys()),
                       help=f"RCA method to evaluate. Available: {list(AVAILABLE_METHODS.keys())}")
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIG.keys()),
                       help=f"Dataset to use. Available: {list(DATASET_CONFIG.keys())}")
    
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of test cases")
    
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode (only 2 cases)")
    
    parser.add_argument("--cases", type=int, default=5,
                       help="Number of test cases to run (default: 5, max: 20)")
    
    parser.add_argument("--output-dir", type=str, default="experiment_output",
                       help="Output directory for results")
    
    return parser.parse_args()


def main():
    """Main experiment execution"""
    args = parse_args()
    
    print("🧪 Single Method RCA Experiment Framework")
    print("="*50)
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Test mode: {args.test}")
    if args.limit:
        print(f"Case limit: {args.limit}")
    elif not args.test:
        print(f"Number of cases: {args.cases}")
    print("="*50)
    
    try:
        # Create and run experiment
        experiment = SingleMethodExperiment(
            method_name=args.method,
            dataset_name=args.dataset,
            output_dir=args.output_dir
        )
        
        # Run experiment
        start_time = time.time()
        results = experiment.run_experiment(limit=args.limit, test_mode=args.test, num_cases=args.cases)
        total_time = time.time() - start_time
        
        # Print summary
        experiment.print_summary()
        
        print(f"\n⏱️ Total experiment time: {total_time:.2f}s")
        print(f"📁 Results saved in: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
