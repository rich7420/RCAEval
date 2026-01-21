"""
Base processor classes - unified feature processing interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from .data_interface import StandardizedData, UnifiedDataInterface


class BaseFeatureProcessor(ABC):
    """Base feature processor - unified interface for all feature processing"""
    
    def __init__(self, target_dim: int = 64, method: str = 'auto', **kwargs):
        """
        Initialize feature processor
        
        Args:
            target_dim: Target feature dimension
            method: Processing method ('ica', 'kpca', 'pca', 'simplified', 'auto')
            **kwargs: Other parameters
        """
        self.target_dim = target_dim
        self.method = method
        self.config = kwargs
        self.is_fitted = False
        self.feature_names_ = []
        self.scaler_ = None
    
    @abstractmethod
    def fit(self, data: Union[StandardizedData, Any], **kwargs) -> 'BaseFeatureProcessor':
        """Train feature processor"""
        pass
    
    @abstractmethod
    def transform(self, data: Union[StandardizedData, Any]) -> Tuple[np.ndarray, List[str]]:
        """Transform data to features"""
        pass
    
    def fit_transform(self, data: Union[StandardizedData, Any], **kwargs) -> Tuple[np.ndarray, List[str]]:
        """Train and transform data"""
        return self.fit(data, **kwargs).transform(data)
    
    def process(self, data: Any, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        Unified processing interface
        
        Args:
            data: Input data in any format
            **kwargs: Other parameters
            
        Returns:
            features: Processed feature matrix
            feature_names: List of feature names
        """
        # 1. Data standardization
        if not isinstance(data, StandardizedData):
            data = UnifiedDataInterface.standardize_input(data, **kwargs)
        
        # 2. Feature extraction and transformation
        features, feature_names = self.fit_transform(data, **kwargs)
        
        # 3. Dimension adaptation
        features = self._adapt_dimensions(features)
        
        # 4. Numerical stabilization
        features = self._stabilize_features(features)
        
        # Update feature names
        if len(feature_names) != features.shape[1]:
            feature_names = [f'{self.method}_feature_{i}' for i in range(features.shape[1])]
        
        return features, feature_names
    
    def _adapt_dimensions(self, features: np.ndarray) -> np.ndarray:
        """Adapt dimensions to target dimension"""
        if features.size == 0:
            return np.zeros((1, self.target_dim))
        
        # Ensure it's a 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim == 0:
            features = features.reshape(1, 1)
        
        current_dim = features.shape[1]
        
        if current_dim == self.target_dim:
            return features
        elif current_dim > self.target_dim:
            # PCA dimensionality reduction
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.target_dim, random_state=42)
                features = pca.fit_transform(features)
            except:
                # Simple truncation
                features = features[:, :self.target_dim]
        else:
            # Zero padding or feature duplication
            if current_dim > 0:
                # Repeat features for padding
                repeat_times = (self.target_dim + current_dim - 1) // current_dim
                repeated_features = np.tile(features, (1, repeat_times))
                features = repeated_features[:, :self.target_dim]
            else:
                # Zero padding
                features = np.zeros((features.shape[0], self.target_dim))
        
        return features
    
    def _stabilize_features(self, features: np.ndarray) -> np.ndarray:
        """Numerical stabilization processing"""
        # Handle invalid values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Avoid values that are too large or too small
        features = np.clip(features, -1e6, 1e6)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names_


class BaseGraphConstructor(ABC):
    """Base graph constructor"""
    
    def __init__(self, config: Any = None):
        """
        Initialize graph constructor
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.is_learnable = getattr(config, 'learnable_graph', False) if config else False
    
    @abstractmethod
    def build_graph(self, features: np.ndarray, node_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph structure
        
        Args:
            features: Node feature matrix
            node_names: List of node names
            
        Returns:
            edge_index: Edge indices [2, num_edges]
            edge_weights: Edge weights [num_edges]
        """
        pass
    
    def _create_fully_connected_graph(self, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create fully connected graph"""
        edges = []
        weights = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    edges.append([i, j])
                    weights.append(1.0)
        
        if not edges:
            # Single node case
            edges = [[0, 0]]
            weights = [1.0]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(weights, dtype=torch.float32)
        
        return edge_index, edge_weights
    
    def _create_similarity_graph(self, features: np.ndarray, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create graph based on similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(features)
            
            # Apply threshold
            edges = []
            weights = []
            
            num_nodes = similarity_matrix.shape[0]
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and similarity_matrix[i, j] > threshold:
                        edges.append([i, j])
                        weights.append(similarity_matrix[i, j])
            
            if not edges:
                # If no edges, fallback to fully connected
                return self._create_fully_connected_graph(num_nodes)
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(weights, dtype=torch.float32)
            
            return edge_index, edge_weights
            
        except Exception as e:
            return self._create_fully_connected_graph(features.shape[0])


class BaseKANProcessor(ABC):
    """Base KAN processor"""
    
    def __init__(self, config: Any = None):
        """
        Initialize KAN processor
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.kan_config = self._extract_kan_config(config)
    
    def _extract_kan_config(self, config: Any) -> Dict[str, Any]:
        """Extract KAN-related configuration"""
        kan_config = {
            'num_basis': getattr(config, 'kan_grid_size', 8),
            'grid_size': getattr(config, 'kan_grid_size', 8),
            'spline_order': getattr(config, 'kan_spline_order', 3),
            'adaptive_spline_order': getattr(config, 'adaptive_spline_order', True),
            'learnable_activation': getattr(config, 'learnable_activation', True)
        }
        return kan_config
    
    @abstractmethod
    def create_kan_layer(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create KAN layer"""
        pass
    
    @abstractmethod  
    def process_with_kan(self, data: torch.Tensor) -> torch.Tensor:
        """Process data with KAN"""
        pass
    
    def get_kan_parameters(self) -> Dict[str, Any]:
        """Get KAN parameter configuration"""
        return self.kan_config.copy()


# Utility functions
def create_processor_factory(processor_type: str) -> BaseFeatureProcessor:
    """
    Processor factory function
    
    Args:
        processor_type: Processor type ('metrics', 'logs', 'traces', 'multimodal')
        
    Returns:
        Corresponding processor instance
    """
    if processor_type == 'metrics':
        from ..feature_processing import MetricProcessor
        return MetricProcessor()
    elif processor_type == 'logs':
        from ..feature_processing import LogProcessor  
        return LogProcessor()
    elif processor_type == 'traces':
        from ..feature_processing import TraceProcessor
        return TraceProcessor()
    elif processor_type == 'multimodal':
        from ..feature_processing import MultiModalProcessor
        return MultiModalProcessor()
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


def validate_processor_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate processor configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of issues (empty list means no issues)
    """
    issues = []
    
    # Check required configuration items
    required_keys = ['target_dim', 'method']
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required configuration item: {key}")
    
    # Check numeric ranges
    if 'target_dim' in config:
        target_dim = config['target_dim']
        if not isinstance(target_dim, int) or target_dim <= 0:
            issues.append(f"target_dim must be a positive integer, current value: {target_dim}")
    
    # Check method name
    if 'method' in config:
        method = config['method']
        valid_methods = ['ica', 'kpca', 'pca', 'simplified', 'auto']
        if method not in valid_methods:
            issues.append(f"Invalid method name: {method}, supported methods: {valid_methods}")
    
    return issues 