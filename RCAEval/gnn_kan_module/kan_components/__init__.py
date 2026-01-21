"""
KAN Components: Core components for pure KAN implementation
Focus on core value of replacing MLP with KAN
"""

# Base KAN layers
from .kan_layers import (
    SimplifiedKANLayer,
    OptimizedGNNKANEncoder,
    AdvancedKANLayer,
    CompatibleSimplifiedKANLayer,
    create_compatible_kan_layer
)

# Ensure backward compatibility
KANLayer = SimplifiedKANLayer

# High capacity stable KAN - removed, simplified version
HighCapacityGNNKANEncoder = None
HighCapacityStableKANLayer = None

# Gradient stabilizer - removed, use simplified version
GradientStabilizer = None

# Ensure all components can be imported
__all__ = [
    # Base KAN layers
    'AdvancedKANLayer',
    'SimplifiedKANLayer',
    'CompatibleSimplifiedKANLayer',
    'OptimizedGNNKANEncoder',
    'KANLayer',
    
    # High capacity KAN
    'HighCapacityGNNKANEncoder',
    'HighCapacityStableKANLayer',
    
    # Stability components  
    'GradientStabilizer',
    
    # Utility functions
    'create_high_capacity_stable_model',
    'create_compatible_kan_layer'
]
