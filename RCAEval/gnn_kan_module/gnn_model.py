"""
Pure GNN Model - Standard GNN with MLP layers (without KAN)
Pure GNN model using standard MLP layers (without KAN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StandardMLPLayer(nn.Module):
    """Standard MLP layer (for comparison with KAN)"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(StandardMLPLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class PureGNNEncoder(nn.Module):
    """
    Pure GNN encoder - uses standard MLP layers
    Compared to GNN-KAN, here standard ReLU + Linear replaces KAN
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=2, dropout=0.1, **kwargs):
        super(PureGNNEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # Use standard MLP layers
        mlp_layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            mlp_layers.append(StandardMLPLayer(
                dims[i], dims[i + 1], 
                dropout=dropout
            ))
        
        self.mlp_layers = nn.ModuleList(mlp_layers)
        
        # Message passing processors
        self.message_processors = nn.ModuleList([
            StandardMLPLayer(dims[i + 1], dims[i + 1], dropout=dropout)
            for i in range(len(dims) - 1)
        ])
    
    def forward(self, node_features, edge_index):
        """
        Forward propagation - standard GNN message passing
        
        Args:
            node_features: [N, D] node features
            edge_index: [2, E] edge indices
            
        Returns:
            embeddings: [N, output_dim] node embeddings
        """
        x = node_features
        
        # Multi-layer GNN
        for i, (mlp_layer, msg_processor) in enumerate(zip(self.mlp_layers, self.message_processors)):
            # 1. Node feature transformation
            x = mlp_layer(x)
            
            # 2. Message passing (simplified version)
            if edge_index.shape[1] > 0:
                # Aggregate neighbor information
                row, col = edge_index
                
                # Compute messages
                messages = x[col]  # [E, D]
                
                # Aggregate to target nodes
                num_nodes = x.size(0)
                aggregated = torch.zeros_like(x)
                
                # Use scatter_add for aggregation
                aggregated.index_add_(0, row, messages)
                
                # Message processing
                aggregated = msg_processor(aggregated)
                
                # Residual connection
                x = x + aggregated
        
        return x


class StandardGraphDecoder(nn.Module):
    """
    Standard graph decoder - uses MLP for graph structure learning
    """
    
    def __init__(self, embed_dim, num_nodes, hidden_dim=64):
        super(StandardGraphDecoder, self).__init__()
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
        # embeddings_i: [N, 1, D]
        # embeddings_j: [1, N, D]
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


class PureGNNModel(nn.Module):
    """Pure GNN model - combines standard GNN encoder and decoder"""
    
    def __init__(self, config, num_nodes):
        super(PureGNNModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # Feature projection layer
        input_feature_dim = getattr(config, 'target_feature_dim', config.input_dim)
        self.feature_projection = nn.Linear(input_feature_dim, config.input_dim)
        
        # Use pure GNN encoder (standard MLP)
        self.gnn_encoder = PureGNNEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            dropout=config.dropout
        )
        
        # Standard graph decoder
        self.graph_decoder = StandardGraphDecoder(
            embed_dim=config.output_dim,
            num_nodes=num_nodes,
            hidden_dim=64
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, node_features, edge_index, fault_type=None):
        """
        GNN forward propagation
        
        Args:
            node_features: [N, D] node features
            edge_index: [2, E] edge indices
            fault_type: Fault type (optional, maintain interface compatibility)
            
        Returns:
            embeddings: [N, output_dim] node embeddings
            adj_scores: [N, N] adjacency matrix scores
        """
        try:
            # Feature projection
            x = self.feature_projection(node_features)
            
            # GNN feature extraction
            embeddings = self.gnn_encoder(x, edge_index)
            
            # Graph structure learning
            adj_scores = self.graph_decoder(embeddings)
            
            return embeddings, adj_scores
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                embeddings = self.gnn_encoder(node_features, edge_index) 
                return embeddings, None
            else:
                raise e



