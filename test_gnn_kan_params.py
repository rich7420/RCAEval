#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RCAEval.gnn_kan_module import create_config
from RCAEval.gnn_kan_module.models import create_model_with_config
from RCAEval.gnn_kan_module.optimized_input_processor import GNNKANInputOptimizer
import pandas as pd

# Load data
data = pd.read_csv('data/train-ticket/ts-train-service_mem/1/data.csv')
with open('data/train-ticket/ts-train-service_mem/1/inject_time.txt', 'r') as f:
    inject_time = int(f.readlines()[0].strip())

# Prepare data
config = create_config(config_type='simplified')
config.feature_method = 'enhanced_ica'
config.target_feature_dim = 64

processor = GNNKANInputOptimizer(
    feature_method='enhanced_ica',
    target_dim=64,
    similarity_threshold=0.3,
    max_edges_per_node=15
)

optimized_data = processor.optimize_input(data, inject_time)

# Test Cheb
config.input_dim = optimized_data.node_features.shape[1]
config.basis_function = 'chebyshev'
if not hasattr(config, 'output_dim') or config.output_dim is None:
    config.output_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 64
if not hasattr(config, 'kan_config') or config.kan_config is None:
    config.kan_config = {}
config.kan_config['basis_function'] = 'chebyshev'

try:
    model = create_model_with_config(config)
    total_params = sum(p.numel() for p in model.parameters())
except Exception as e:
    import traceback
    traceback.print_exc()

# Test PQC
config.basis_function = 'pqc'
config.kan_config['basis_function'] = 'pqc'

try:
    model = create_model_with_config(config)
    total_params = sum(p.numel() for p in model.parameters())
except Exception as e:
    import traceback
    traceback.print_exc()
