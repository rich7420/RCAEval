"""
KAN Layer implementation for GNN-KAN RCA
KAN (Kolmogorov-Arnold Networks) replaces traditional MLP layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class BSpline(nn.Module):
    """B-spline basis functions for KAN layer"""
    
    def __init__(self, grid_size=5, spline_order=3):
        super(BSpline, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Create grid points
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size + 2 * spline_order + 1))
        
    def forward(self, x):
        """
        Compute B-spline basis functions
        Args:
            x: input tensor of shape [batch_size, input_dim]
        Returns:
            basis: B-spline basis values [batch_size, input_dim, grid_size + spline_order]
        """
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        
        # Expand x to compare with grid
        x_expanded = x.expand(-1, -1, len(self.grid) - 1)
        grid_expanded = self.grid[:-1].expand(x.size(0), x.size(1), -1)
        
        # Find the interval
        interval = torch.sum(x_expanded >= grid_expanded, dim=-1) - 1
        interval = torch.clamp(interval, 0, len(self.grid) - self.spline_order - 2)
        
        # Initialize basis
        basis = torch.zeros(x.size(0), x.size(1), self.grid_size + self.spline_order, 
                           device=x.device, dtype=x.dtype)
        
        # Compute B-spline basis (simplified version for order 3)
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                idx = interval[i, j]
                t = (x[i, j, 0] - self.grid[idx]) / (self.grid[idx + 1] - self.grid[idx] + 1e-8)
                
                # Cubic B-spline basis functions
                if idx < basis.size(-1) - 3:
                    basis[i, j, idx] = (1 - t) ** 3 / 6
                    basis[i, j, idx + 1] = (3 * t**3 - 6 * t**2 + 4) / 6
                    basis[i, j, idx + 2] = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6
                    basis[i, j, idx + 3] = t**3 / 6
        
        return basis


class KANLayer(nn.Module):
    """
    KAN Layer that replaces traditional MLP layer
    Uses learnable spline functions instead of weight matrices and activation functions
    """
    
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # B-spline basis
        self.bspline = BSpline(grid_size, spline_order)
        
        # Learnable coefficients for each spline function
        # Shape: [output_dim, input_dim, grid_size + spline_order]
        self.coefficients = nn.Parameter(
            torch.randn(output_dim, input_dim, grid_size + spline_order) * 0.1
        )
        
        # Optional bias term
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Learnable scale and shift for input normalization
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.shift = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x):
        """
        Forward pass of KAN layer
        Args:
            x: input tensor [batch_size, input_dim]
        Returns:
            output: [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Normalize input
        x_norm = (x - self.shift) / (self.scale + 1e-8)
        
        # Compute B-spline basis
        basis = self.bspline(x_norm)  # [batch_size, input_dim, grid_size + spline_order]
        
        # Apply learnable coefficients
        # basis: [batch_size, input_dim, n_basis]
        # coefficients: [output_dim, input_dim, n_basis]
        output = torch.einsum('bin,oin->bo', basis, self.coefficients)
        
        # Add bias
        output = output + self.bias
        
        return output


class KANLinear(nn.Module):
    """Simplified KAN layer for comparison"""
    
    def __init__(self, input_dim, output_dim, num_functions=4):
        super(KANLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_functions = num_functions
        
        # Learnable coefficients for polynomial basis
        self.coefficients = nn.Parameter(torch.randn(output_dim, input_dim, num_functions) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        """Use polynomial basis functions"""
        batch_size = x.size(0)
        
        # Create polynomial basis: [1, x, x^2, x^3, ...]
        basis_functions = []
        for i in range(self.num_functions):
            basis_functions.append(torch.pow(x, i))
        
        basis = torch.stack(basis_functions, dim=-1)  # [batch_size, input_dim, num_functions]
        
        # Apply coefficients
        output = torch.einsum('bin,oin->bo', basis, self.coefficients)
        output = output + self.bias
        
        return output


class GNNKANConv(MessagePassing):
    """Graph convolution layer using KAN instead of MLP"""
    
    def __init__(self, input_dim, output_dim, kan_hidden_dim=64, grid_size=5):
        super(GNNKANConv, self).__init__(aggr='add')
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # KAN layers for message and update functions
        self.message_kan = KANLayer(2 * input_dim, kan_hidden_dim, grid_size)
        self.update_kan = KANLayer(input_dim + kan_hidden_dim, output_dim, grid_size)
        
        # Optional linear transformation for residual connection
        if input_dim != output_dim:
            self.residual_transform = nn.Linear(input_dim, output_dim)
        else:
            self.residual_transform = None
            
    def forward(self, x, edge_index):
        """
        Forward pass of GNN-KAN convolution
        Args:
            x: node features [num_nodes, input_dim]
            edge_index: edge connectivity [2, num_edges]
        Returns:
            out: updated node features [num_nodes, output_dim]
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Start propagating messages
        out = self.propagate(edge_index, x=x)
        
        # Residual connection
        if self.residual_transform is not None:
            residual = self.residual_transform(x)
        else:
            residual = x
            
        return out + residual
    
    def message(self, x_i, x_j):
        """
        Create messages between nodes using KAN
        Args:
            x_i: features of target nodes [num_edges, input_dim]
            x_j: features of source nodes [num_edges, input_dim]
        Returns:
            messages: [num_edges, kan_hidden_dim]
        """
        # Concatenate source and target features
        edge_features = torch.cat([x_i, x_j], dim=-1)
        
        # Apply KAN to compute messages
        messages = self.message_kan(edge_features)
        
        return messages
    
    def update(self, aggr_out, x):
        """
        Update node features using aggregated messages
        Args:
            aggr_out: aggregated messages [num_nodes, kan_hidden_dim]
            x: original node features [num_nodes, input_dim]
        Returns:
            updated features [num_nodes, output_dim]
        """
        # Concatenate original features with aggregated messages
        combined = torch.cat([x, aggr_out], dim=-1)
        
        # Apply KAN for update
        updated = self.update_kan(combined)
        
        return updated


class GNNKANEncoder(nn.Module):
    """
    Multi-layer GNN encoder using KAN layers
    Replaces traditional GNN with MLP layers
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers=3, grid_size=5, dropout=0.1):
        super(GNNKANEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GNNKANConv(dims[i], dims[i+1], grid_size=grid_size))
            self.batch_norms.append(nn.BatchNorm1d(dims[i+1]))
            
        # Final output layer
        self.output_kan = KANLayer(dims[-1], output_dim, grid_size)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through GNN-KAN encoder
        Args:
            x: node features [num_nodes, input_dim]
            edge_index: edge connectivity [2, num_edges]
            batch: batch indices for batched graphs
        Returns:
            node_embeddings: [num_nodes, output_dim]
            graph_embedding: [batch_size, output_dim] if batch is provided
        """
        # Apply GNN-KAN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)  # Keep ReLU for stability
            
            if i < len(self.convs) - 1:  # Don't apply dropout to last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final KAN transformation
        node_embeddings = self.output_kan(x)
        
        # Global pooling for graph-level representation
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            graph_embedding = global_mean_pool(node_embeddings, batch)
            return node_embeddings, graph_embedding
        
        return node_embeddings


# Utility functions for testing KAN layers
def test_kan_layer():
    """Test KAN layer functionality"""
    batch_size, input_dim, output_dim = 32, 10, 5
    
    # Create test data
    x = torch.randn(batch_size, input_dim)
    
    # Test KANLayer
    kan = KANLayer(input_dim, output_dim)
    output = kan(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"KAN parameters: {sum(p.numel() for p in kan.parameters())}")
    
    # Compare with traditional linear layer
    linear = nn.Linear(input_dim, output_dim)
    linear_output = linear(x)
    
    print(f"Linear parameters: {sum(p.numel() for p in linear.parameters())}")
    print("KAN test passed!")
    
    return kan, linear


if __name__ == "__main__":
    test_kan_layer()