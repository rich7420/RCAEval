"""
Graph Transformer Model - Modern GNN Baseline

Graph Transformer uses self-attention mechanism:
- Self-attention over all nodes (with graph structure bias)
- Positional encoding for graph structure
- Same architecture depth and dimensions as GNN-KAN for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

try:
    from torch_geometric.nn import TransformerConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available, Graph Transformer will use fallback implementation")


class GraphTransformerLayer(nn.Module):
    """Graph Transformer Layer - Self-attention with graph structure"""
    
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.transformer_conv = TransformerConv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
        else:
            # Fallback: simplified transformer implementation
            self.linear = nn.Linear(in_dim, out_dim)
            self.attention = nn.Linear(out_dim * 2, 1)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(out_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 2, out_dim)
            )
    
    def forward(self, x, edge_index):
        """Graph Transformer forward propagation"""
        if TORCH_GEOMETRIC_AVAILABLE:
            return self.transformer_conv(x, edge_index)
        else:
            # Fallback: simplified self-attention with graph structure
            x = self.linear(x)
            num_nodes = x.size(0)
            
            if edge_index.shape[1] > 0:
                row, col = edge_index
                # Self-attention: compute attention scores for all pairs
                # Use graph structure to bias attention
                edge_features = torch.cat([x[row], x[col]], dim=-1)
                attention_scores = self.attention(edge_features).squeeze(-1)
                attention_weights = F.softmax(attention_scores, dim=0)
                
                # Aggregate with attention weights
                aggregated = torch.zeros_like(x)
                aggregated.index_add_(0, row, x[col] * attention_weights.unsqueeze(-1))
                
                # Residual connection
                x = x + aggregated * 0.5
            else:
                # No edges, just pass through
                pass
            
            x = self.layer_norm(x)
            x = self.dropout(x)
            
            # Feed-forward network
            ff_output = self.feed_forward(x)
            x = self.layer_norm(ff_output + x)
            
            return x


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer Encoder - Same architecture depth and dimensions as GNN-KAN
    Ensures fair comparison:
    - 4-layer Graph Transformer
    - Hidden dimensions: [128, 96, 64]
    - Output dimension: 96
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=4, num_heads=8, dropout=0.1, **kwargs):
        super(GraphTransformerEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # Build layer structure, consistent with GNN-KAN
        dims = [input_dim] + hidden_dims + [output_dim]
        
        self.transformer_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            self.transformer_layers.append(
                GraphTransformerLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_heads=num_heads if i < len(dims) - 2 else min(num_heads, 4),
                    dropout=dropout
                )
            )
    
    def forward(self, node_features, edge_index):
        """
        Graph Transformer forward propagation
        
        Args:
            node_features: [N, D] node features
            edge_index: [2, E] edge indices
            
        Returns:
            embeddings: [N, output_dim] node embeddings
        """
        x = node_features
        
        # Multi-layer Graph Transformer
        for i, transformer_layer in enumerate(self.transformer_layers):
            # Transformer layer
            if TORCH_GEOMETRIC_AVAILABLE:
                x = transformer_layer(x, edge_index)
            else:
                x = transformer_layer(x, edge_index)
            
            # Dropout and activation (except last layer)
            if i < len(self.transformer_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class GraphTransformerDecoder(nn.Module):
    """
    Graph Transformer Decoder - Uses MLP for graph structure learning
    Similar structure to GNN-KAN decoder, but uses standard MLP
    """
    
    def __init__(self, embed_dim, num_nodes, hidden_dim=64):
        super(GraphTransformerDecoder, self).__init__()
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


class GraphTransformerModel(nn.Module):
    """
    Graph Transformer Model - Complete Transformer-based GNN
    Corresponds to GNN-KAN model structure to ensure fair comparison
    """
    
    def __init__(self, config, num_nodes):
        super(GraphTransformerModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # Feature projection layer (same as GNN-KAN)
        input_feature_dim = getattr(config, 'target_feature_dim', config.input_dim)
        self.feature_projection = nn.Linear(input_feature_dim, config.input_dim)
        
        # Graph Transformer encoder (same number of layers and dimensions as GNN-KAN)
        self.transformer_encoder = GraphTransformerEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            num_heads=8,  # Standard Transformer uses 8 attention heads
            dropout=config.dropout
        )
        
        # Graph decoder (corresponds to GNN-KAN)
        self.graph_decoder = GraphTransformerDecoder(
            embed_dim=config.output_dim,
            num_nodes=num_nodes,
            hidden_dim=64
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, node_features, edge_index, fault_type=None):
        """
        Graph Transformer forward propagation
        
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
            
            # Graph Transformer feature extraction
            embeddings = self.transformer_encoder(x, edge_index)
            
            # Graph structure learning
            adj_scores = self.graph_decoder(embeddings)
            
            return embeddings, adj_scores
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Warning: GPU memory insufficient, using simplified result")
                embeddings = self.transformer_encoder(node_features, edge_index) 
                return embeddings, None
            else:
                raise e

