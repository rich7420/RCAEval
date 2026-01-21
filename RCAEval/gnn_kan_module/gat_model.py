"""
GAT (Graph Attention Network) Model - Fair Baseline for GNN-KAN Comparison

Ensures fair comparison:
- Same number of layers: 4 layers
- Same hidden dimensions: [128, 96, 64]
- Same output dimension: 96
- Same input features: RCA-aware features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch_geometric.nn import GATConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available, GAT will use fallback implementation")


class GATLayer(nn.Module):
    """GAT Layer - Standard Graph Attention"""
    
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1, concat=True):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.gat_conv = GATConv(
                in_channels=in_dim,
                out_channels=out_dim // num_heads if concat else out_dim,
                heads=num_heads,
                dropout=dropout,
                concat=concat
            )
        else:
            # Fallback: use simple linear layer
            self.linear = nn.Linear(in_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index):
        """GAT forward propagation"""
        if TORCH_GEOMETRIC_AVAILABLE:
            return self.gat_conv(x, edge_index)
        else:
            # Fallback implementation: use linear layer + ReLU
            x = self.linear(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.layer_norm(x)
            return x


class GATEncoder(nn.Module):
    """
    GAT Encoder - Same architecture depth and dimensions as GNN-KAN
    Ensures fair comparison:
    - 4-layer GAT
    - Hidden dimensions: [128, 96, 64]
    - Output dimension: 96
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=4, num_heads=8, dropout=0.1, **kwargs):
        super(GATEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # Build layer structure, consistent with GNN-KAN
        dims = [input_dim] + hidden_dims + [output_dim]
        
        self.gat_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            # Last layer uses average instead of concatenation
            concat = (i < len(dims) - 2)
            heads = num_heads if concat else 1
            
            self.gat_layers.append(
                GATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_heads=heads,
                    dropout=dropout,
                    concat=concat
                )
            )
    
    def forward(self, node_features, edge_index):
        """
        GAT forward propagation
        
        Args:
            node_features: [N, D] node features
            edge_index: [2, E] edge indices
            
        Returns:
            embeddings: [N, output_dim] node embeddings
        """
        x = node_features
        
        # Multi-layer GAT
        for i, gat_layer in enumerate(self.gat_layers):
            # GAT layer
            if TORCH_GEOMETRIC_AVAILABLE:
                x = gat_layer(x, edge_index)
            else:
                # Fallback: simple message passing
                x = gat_layer(x, edge_index)
                # Aggregate neighbor information (simplified version)
                if edge_index.shape[1] > 0:
                    row, col = edge_index
                    # Simple average aggregation
                    num_nodes = x.size(0)
                    aggregated = torch.zeros_like(x)
                    aggregated.index_add_(0, row, x[col])
                    # Calculate degrees
                    degrees = torch.zeros(num_nodes, device=x.device)
                    degrees.index_add_(0, row, torch.ones(edge_index.shape[1], device=x.device))
                    degrees = degrees.clamp(min=1)
                    aggregated = aggregated / degrees.unsqueeze(1)
                    # Residual connection
                    x = x + aggregated * 0.5
            
            # Dropout and activation (except last layer)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class GATGraphDecoder(nn.Module):
    """
    GAT Graph Decoder - Uses MLP for graph structure learning
    Similar structure to GNN-KAN decoder, but uses standard MLP
    """
    
    def __init__(self, embed_dim, num_nodes, hidden_dim=64):
        super(GATGraphDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        
        # Use MLP for graph structure prediction
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, embeddings):
        """
        Predict adjacency matrix from embeddings
        
        Args:
            embeddings: [N, D] node embeddings
            
        Returns:
            adj_scores: [N, N] adjacency matrix scores
        """
        num_nodes = embeddings.size(0)
        
        # Build features for all node pairs
        embeddings_i = embeddings.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        embeddings_j = embeddings.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        
        # Concatenate node pair features: [N, N, 2D]
        pair_features = torch.cat([embeddings_i, embeddings_j], dim=-1)
        
        # Flatten: [N*N, 2D]
        pair_features_flat = pair_features.view(-1, self.embed_dim * 2)
        
        # MLP prediction: [N*N, 1]
        adj_scores_flat = self.decoder(pair_features_flat)
        
        # Reshape: [N, N]
        adj_scores = adj_scores_flat.view(num_nodes, num_nodes)
        
        return adj_scores


class GATModel(nn.Module):
    """
    GAT Model - Complete Graph Attention Network
    Corresponds to GNN-KAN model structure to ensure fair comparison
    """
    
    def __init__(self, config, num_nodes):
        super(GATModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # Feature projection layer (same as GNN-KAN)
        input_feature_dim = getattr(config, 'target_feature_dim', config.input_dim)
        self.feature_projection = nn.Linear(input_feature_dim, config.input_dim)
        
        # GAT encoder (same number of layers and dimensions as GNN-KAN)
        self.gat_encoder = GATEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            num_heads=8,  # Standard GAT uses 8 attention heads
            dropout=config.dropout
        )
        
        # Graph decoder (corresponds to GNN-KAN)
        self.graph_decoder = GATGraphDecoder(
            embed_dim=config.output_dim,
            num_nodes=num_nodes,
            hidden_dim=64
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, node_features, edge_index, fault_type=None):
        """
        GAT forward propagation
        
        Args:
            node_features: [N, D] node features
            edge_index: [2, E] edge indices
            fault_type: Fault type (optional, for interface compatibility)
            
        Returns:
            embeddings: [N, output_dim] node embeddings
            adj_scores: [N, N] adjacency matrix scores
        """
        try:
            # Feature projection
            x = self.feature_projection(node_features)
            
            # GAT feature extraction
            embeddings = self.gat_encoder(x, edge_index)
            
            # Graph structure learning
            adj_scores = self.graph_decoder(embeddings)
            
            return embeddings, adj_scores
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Warning: GPU memory insufficient, using simplified result")
                embeddings = self.gat_encoder(node_features, edge_index) 
                return embeddings, None
            else:
                raise e
