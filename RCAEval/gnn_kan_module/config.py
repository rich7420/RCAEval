"""
Simplified GNN-KAN configuration module
Remove excessive options, focus on core functionality
"""

import torch
import numpy as np


class GNNKANConfig:
    """
    Simplified GNN-KAN configuration - remove excessive options
    Focus on core functionality, ensure clear method structure
    """
    
    def __init__(self):
        # Significantly enhance KAN expressiveness
        self.kan_grid_size = 20              # Increased from 8 to 20
        self.kan_spline_order = 5            # Increased from 3 to 5
        self.kan_num_basis = 12              # Increased from 4 to 12
        self.learnable_activation = True
        
        # Basis function selection configuration - single source of truth
        self.basis_function = 'chebyshev'    # Default to chebyshev for backward compatibility
        self.basis_kwargs = {}               # Additional parameters for specific basis functions
        self.kernel = 'naive'                # Ch5: 'naive' | 'sparsefuse' B-spline kernel
        
        # Increase model capacity and depth
        self.input_dim = 128                 # Increased from 64 to 128
        self.hidden_dims = [128, 96, 64]     # Increased from [32,16] to [128,96,64]
        self.output_dim = 96                 # Increased from 16 to 96
        self.num_gnn_layers = 4              # Increased from 2 to 4
        self.dropout = 0.1                   # Reduced from 0.2 to 0.1
        
        # Optimize feature processing
        self.feature_method = 'enhanced_ica' # Use stronger feature extraction
        self.target_feature_dim = 128        # Increased from 64 to 128
        
        # Adjust training strategy
        self.learning_rate = 2e-4            # Adjusted from 1e-3 to 2e-4
        self.weight_decay = 5e-6             # Reduced from 1e-4 to 5e-6
        self.num_epochs = 400                # Significantly increased from 100 to 400
        self.batch_size = 8                  # Reduced from 32 to 8 for better stability
        self.patience = 60                   # Increased from 25 to 60
        self.min_delta = 1e-6                # Reduced from 1e-5 to 1e-6
        self.min_epochs = 30                 # New: minimum training epochs guarantee
        
        # Optimize graph construction
        self.similarity_threshold = 0.2      # Significantly reduced from 0.5 to 0.2
        self.max_edges_per_node = 15         # Significantly increased from 4 to 15
        
        # Stability configuration
        self.gradient_clip_norm = 0.5        # Reduced from 1.0 to 0.5
        # Support CUDA, MPS (Apple Silicon), CPU
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.use_cuda = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
            self.use_cuda = True  # MPS also treated as GPU acceleration
        else:
            self.device = 'cpu'
            self.use_cuda = False
    
    def update_for_kan_purity(self):
        """Update configuration to ensure KAN purity"""
        pass  # Simplified version doesn't need additional configuration
        
    def get_device(self):
        """Get computation device"""
        return self.device


# Convenient configuration creation function
def create_config(**kwargs):
    """Create configuration and apply custom parameters"""
    config = GNNKANConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
    

# Compatibility alias
SimplifiedGNNKANConfig = GNNKANConfig