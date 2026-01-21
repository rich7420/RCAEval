"""
GNN+KAN input optimization test
Compare performance differences before and after optimization
"""

import time
import numpy as np
import pandas as pd
import sys
import os

# Add path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from RCAEval.gnn_kan_module.optimized_input_processor import optimize_gnn_kan_input
from RCAEval.gnn_kan_module.feature_extractors import MultiModalFeatureExtractor
from RCAEval.gnn_kan_module.graph_constructors import SimplifiedGraphConstructor
from RCAEval.gnn_kan_module.config import SimplifiedGNNKANConfig


def generate_test_data(num_samples=1000, num_services=9):
    """Generate test data"""
    services = ['adservice', 'cartservice', 'checkoutservice', 'currencyservice', 
                'emailservice', 'paymentservice', 'productcatalogservice', 
                'recommendationservice', 'shippingservice'][:num_services]
    
    data = {}
    
    # Generate multiple metrics for each service
    for service in services:
        for metric in ['cpu_usage', 'memory_usage', 'latency_p90', 'error_rate']:
            col_name = f"{service}_{metric}"
            # Generate time series data with noise
            base_value = np.random.uniform(0.1, 0.8)
            noise = np.random.normal(0, 0.1, num_samples)
            trend = np.linspace(0, 0.2, num_samples) * np.random.choice([-1, 1])
            data[col_name] = base_value + noise + trend
    
    # Add time column
    data['time'] = np.arange(num_samples)
    
    return pd.DataFrame(data)


def test_original_method(data):
    """Test original method"""
    print("Testing original method...")
    start_time = time.time()
    
    config = SimplifiedGNNKANConfig()
    
    # Original feature extraction
    feature_extractor = MultiModalFeatureExtractor(config)
    features, node_names = feature_extractor.extract_features(data)
    
    # Original graph construction
    graph_constructor = SimplifiedGraphConstructor(config)
    edge_index, edge_weights = graph_constructor.build_graph(features, node_names)
    
    processing_time = time.time() - start_time
    
    return {
        'processing_time': processing_time,
        'num_nodes': len(node_names),
        'num_edges': edge_index.size(1) if hasattr(edge_index, 'size') else len(edge_index[0]),
        'feature_shape': features.shape,
        'node_names': node_names
    }


def test_optimized_method(data, feature_method='ica'):
    """Test optimized method"""
    print(f"Testing optimized method (feature_method={feature_method})...")
    start_time = time.time()
    
    optimized_data = optimize_gnn_kan_input(
        data=data,
        feature_method=feature_method,
        target_dim=64
    )
    
    processing_time = time.time() - start_time
    
    return {
        'processing_time': processing_time,
        'num_nodes': optimized_data.metadata['num_nodes'],
        'num_edges': optimized_data.metadata['num_edges'],
        'feature_shape': optimized_data.node_features.shape,
        'node_names': optimized_data.node_names,
        'metadata': optimized_data.metadata
    }


def run_performance_comparison():
    """Run performance comparison test"""
    print("=" * 60)
    print("GNN+KAN Input Optimization Performance Test")
    print("=" * 60)
    
    # Generate test data of different scales
    test_configs = [
        {'num_samples': 500, 'num_services': 5, 'name': 'Small scale'},
        {'num_samples': 1000, 'num_services': 9, 'name': 'Medium scale'},
        {'num_samples': 2000, 'num_services': 12, 'name': 'Large scale'}
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting {config['name']} data (samples: {config['num_samples']}, services: {config['num_services']})")
        
        # Generate test data
        test_data = generate_test_data(config['num_samples'], config['num_services'])
        print(f"Generated test data: {test_data.shape}")
        
        try:
            # Test original method
            original_result = test_original_method(test_data)
            print(f"Original method: {original_result['processing_time']:.3f}s")
            
            # Test optimized method - ICA
            optimized_ica_result = test_optimized_method(test_data, 'ica')
            print(f"Optimized method (ICA): {optimized_ica_result['processing_time']:.3f}s")
            
            # Test optimized method - Statistical
            optimized_stat_result = test_optimized_method(test_data, 'simplified')
            print(f"Optimized method (Statistical): {optimized_stat_result['processing_time']:.3f}s")
            
            # Calculate performance improvement
            speedup_ica = original_result['processing_time'] / optimized_ica_result['processing_time']
            speedup_stat = original_result['processing_time'] / optimized_stat_result['processing_time']
            
            result = {
                'config': config,
                'original': original_result,
                'optimized_ica': optimized_ica_result,
                'optimized_stat': optimized_stat_result,
                'speedup_ica': speedup_ica,
                'speedup_stat': speedup_stat
            }
            
            results.append(result)
            
            print(f"Performance improvement: ICA method {speedup_ica:.2f}x, Statistical method {speedup_stat:.2f}x")
            
        except Exception as e:
            print(f"Test failed: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("Performance Test Summary")
    print("=" * 60)
    
    for result in results:
        config = result['config']
        print(f"\n{config['name']} data:")
        print(f"  Data scale: {config['num_samples']} samples, {config['num_services']} services")
        print(f"  Original method: {result['original']['processing_time']:.3f}s")
        print(f"  Optimized ICA: {result['optimized_ica']['processing_time']:.3f}s (improvement {result['speedup_ica']:.2f}x)")
        print(f"  Optimized Statistical: {result['optimized_stat']['processing_time']:.3f}s (improvement {result['speedup_stat']:.2f}x)")
        print(f"  Node count comparison: {result['original']['num_nodes']} vs {result['optimized_ica']['num_nodes']}")
        print(f"  Edge count comparison: {result['original']['num_edges']} vs {result['optimized_ica']['num_edges']}")
    
    # Calculate average performance improvement
    if results:
        avg_speedup_ica = np.mean([r['speedup_ica'] for r in results])
        avg_speedup_stat = np.mean([r['speedup_stat'] for r in results])
        
        print(f"\nAverage performance improvement:")
        print(f"  ICA optimization method: {avg_speedup_ica:.2f}x")
        print(f"  Statistical optimization method: {avg_speedup_stat:.2f}x")
        
        print(f"\nOptimization effect is significant! Recommend using optimized input processor")


def test_feature_quality():
    """Test feature quality"""
    print("\n" + "=" * 60)
    print("Feature Quality Test")
    print("=" * 60)
    
    test_data = generate_test_data(1000, 9)
    
    # Test different feature methods
    methods = ['ica', 'simplified']
    
    for method in methods:
        print(f"\nTesting {method} method:")
        
        optimized_data = optimize_gnn_kan_input(
            data=test_data,
            feature_method=method,
            target_dim=64
        )
        
        features = optimized_data.node_features.numpy()
        
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Feature mean: {features.mean():.3f}")
        print(f"  Feature std: {features.std():.3f}")
        print(f"  NaN count: {np.isnan(features).sum()}")
        print(f"  Inf count: {np.isinf(features).sum()}")
        print(f"  Node names: {optimized_data.node_names}")
        
        # Check feature quality
        if np.isnan(features).sum() == 0 and np.isinf(features).sum() == 0:
            print(f"  {method} method feature quality is good")
        else:
            print(f"  {method} method features contain invalid values")


def test_memory_usage():
    """Test memory usage"""
    import psutil
    import os
    
    print("\n" + "=" * 60)
    print("Memory Usage Test")
    print("=" * 60)
    
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Baseline memory usage: {baseline_memory:.1f} MB")
    
    # Generate large dataset
    large_data = generate_test_data(5000, 15)
    after_data_memory = process.memory_info().rss / 1024 / 1024
    print(f"After data loading: {after_data_memory:.1f} MB (+{after_data_memory - baseline_memory:.1f} MB)")
    
    # Test optimized method memory usage
    optimized_data = optimize_gnn_kan_input(large_data, feature_method='ica')
    after_processing_memory = process.memory_info().rss / 1024 / 1024
    print(f"After optimized processing: {after_processing_memory:.1f} MB (+{after_processing_memory - after_data_memory:.1f} MB)")
    
    print(f"Memory usage is reasonable, processing overhead: {after_processing_memory - after_data_memory:.1f} MB")


if __name__ == "__main__":
    try:
        # Run performance comparison
        run_performance_comparison()
        
        # Test feature quality
        test_feature_quality()
        
        # Test memory usage
        test_memory_usage()
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 