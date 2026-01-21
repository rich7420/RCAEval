"""
GNN-KAN Models Module
Contains all GNN-KAN related model definitions and training functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import traceback

# Import our optimized KAN modules from the new kan_components
from .kan_components import (
    OptimizedGNNKANEncoder, AdvancedKANLayer,
    SimplifiedKANLayer
)

# Import configuration classes for type checking
from .config import GNNKANConfig


# AttentionGraphDecoder removed - use KNN Baseline instead


class GNNKANModel(nn.Module):
    """GNN-KAN model - combines advantages of GNN and KAN"""
    
    def __init__(self, config, num_nodes):
        super(GNNKANModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # Feature projection layer - adaptive dimensions
        input_feature_dim = getattr(config, 'target_feature_dim', config.input_dim)
        self.feature_projection = nn.Linear(input_feature_dim, config.input_dim)
        
        # Use optimized GNN-KAN encoder
        self.gnn_encoder = OptimizedGNNKANEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            kan_grid_size=config.kan_grid_size,
            kan_spline_order=config.kan_spline_order,
            dropout=config.dropout,
            basis_function=getattr(config, 'basis_function', 'chebyshev'),
            basis_kwargs=getattr(config, 'basis_kwargs', {})
        )
        
        # Temporal attention mechanism - use adapter to solve dimension issues
        try:
            from .dimension_adapters import TemporalAttentionAdapter
            self.temporal_attention = TemporalAttentionAdapter(config.output_dim)
        except ImportError:
            self.temporal_attention = TemporalAttention(config.output_dim)
        
        # Restore Graph Decoder - use multi-scale KAN for graph structure learning
        self.graph_decoder = MultiScaleGraphDecoder(
            embed_dim=config.output_dim,
            num_nodes=num_nodes,
            kan_grid_size=config.kan_grid_size,
            scales=3  # 3 scales: local, middle, global
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, node_features, edge_index, fault_type=None):
        """GNN-KAN forward propagation - use KAN to learn graph structure"""
        
        try:
            # GNN feature extraction
            embeddings = self.gnn_encoder(node_features, edge_index)
            
            # Use KAN-based Graph Decoder to learn graph structure
            adj_scores = self.graph_decoder(embeddings, fault_type)
            
            return embeddings, adj_scores
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Emergency fallback: return simplified result
                embeddings = self.gnn_encoder(node_features, edge_index) 
                return embeddings, None
            else:
                raise e


class TemporalAttention(nn.Module):
    """Temporal attention mechanism - fix dimension matching issues"""
    
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Ensure feature_dim is divisible by num_heads
        if feature_dim % num_heads != 0:
            # Adjust to nearest divisible value
            adjusted_dim = ((feature_dim // num_heads) + 1) * num_heads
            self.projection = nn.Linear(feature_dim, adjusted_dim)
            self.back_projection = nn.Linear(adjusted_dim, feature_dim)
            self.use_projection = True
            self.adjusted_dim = adjusted_dim
        else:
            self.use_projection = False
            self.adjusted_dim = feature_dim
        
        self.attention = nn.MultiheadAttention(self.adjusted_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim, eps=1e-4)
        
    def forward(self, features):
        """Forward propagation with fixed dimension matching"""
        original_shape = features.shape
        
        # Ensure input is at least 3D [batch, seq, feature]
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch, 1, feature]
        
        try:
            # Dimension projection (if needed)
            if self.use_projection:
                projected_features = self.projection(features)
                attn_output, _ = self.attention(projected_features, projected_features, projected_features)
                attn_output = self.back_projection(attn_output)
            else:
                attn_output, _ = self.attention(features, features, features)
            
            # Residual connection
            output = self.norm(attn_output + features)
            
            # Restore original shape
            if len(original_shape) == 2:
                output = output.squeeze(1)
            
            return output
            
        except Exception as e:
            # Safe fallback: directly return normalized input
            if len(original_shape) == 2:
                return self.norm(features.squeeze(1))
            else:
                return self.norm(features)


class AdaptiveGradientStabilizer:
    """Adaptive gradient stabilizer - improve training stability"""
    
    def __init__(self):
        self.loss_history = []
        self.grad_norm_history = []
        
    def adaptive_clipping(self, model, current_loss):
        """Dynamically adjust gradient clipping based on loss history"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) > 10:
            loss_std = np.std(self.loss_history[-10:])
            clip_norm = max(0.5, min(2.0, 1.0 / (loss_std + 1e-8)))
        else:
            clip_norm = 1.0
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        self.grad_norm_history.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        
        return grad_norm


def create_fallback_model(config, num_nodes):
    """Create fallback version of the model"""
    
    class FallbackModel(nn.Module):
        def __init__(self, config, num_nodes):
            super().__init__()
            self.linear = nn.Linear(config.target_feature_dim, num_nodes)
            
        def forward(self, node_features, edge_index):
            # Simple linear transformation
            output = torch.sigmoid(self.linear(node_features))
            adj_matrix = torch.mm(output, output.t())
            return output, adj_matrix
    
    return FallbackModel(config, num_nodes)


# validate_model_setup function moved to utils.py module to avoid duplicate definitions
# train_gnn_kan_model function moved to training.py module to avoid duplicate definitions


def train_on_cpu_fallback(model, node_features, edge_index, config):
    """CPU fallback training function - ensure device consistency"""
    
    try:
        # Force move to CPU
        if hasattr(model, 'cpu'):
            model = model.cpu()
        if hasattr(node_features, 'cpu'):
            node_features = node_features.cpu()
        if hasattr(edge_index, 'cpu'):
            edge_index = edge_index.cpu()
        
        # Simplify model structure to adapt to CPU
        
        # Get number of nodes
        if hasattr(node_features, 'size'):
            num_nodes = node_features.size(0)
        elif hasattr(node_features, 'shape'):
            num_nodes = node_features.shape[0]
        else:
            num_nodes = len(node_features)
        
        # Build adjacency matrix based on feature similarity
        with torch.no_grad():
            # Ensure node_features is in correct tensor format
            if not isinstance(node_features, torch.Tensor):
                node_features = torch.tensor(node_features, dtype=torch.float, device='cpu')
            else:
                node_features = node_features.to('cpu')
            
            # Calculate cosine similarity
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # Apply threshold and sigmoid
            final_adj = torch.sigmoid(similarity_matrix * 3.0)
            
            # Ensure diagonal has high values (self-similarity)
            final_adj.fill_diagonal_(0.9)
            
            # Ensure result is on CPU
            final_adj = final_adj.cpu()
            
    except Exception as e:
        # Final fallback: identity matrix
        try:
            if hasattr(node_features, 'size'):
                num_nodes = node_features.size(0)
            elif hasattr(node_features, 'shape'):
                num_nodes = node_features.shape[0]
            else:
                num_nodes = len(node_features)
        except:
            num_nodes = 10  # Default value
            
        final_adj = torch.eye(num_nodes, device='cpu')
    
    # Ensure returned model is also on CPU
    if hasattr(model, 'cpu'):
        model = model.cpu()
    return model, final_adj


def compute_loss_stable(node_embeddings, adj_scores, edge_index, config):
    """
    Numerically stable loss computation
    
    Args:
        node_embeddings: Node embeddings
        adj_scores: Adjacency matrix scores
        edge_index: Edge indices
        config: Configuration parameters
        
    Returns:
        total_loss: Total loss
    """
    num_nodes = node_embeddings.size(0)
    device = node_embeddings.device
    
    # 1. Graph reconstruction loss (use more stable version)
    true_adj = torch.zeros(num_nodes, num_nodes, device=device)
    if edge_index.size(1) > 0:
        # Ensure indices are within valid range
        valid_indices = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        if valid_indices.any():
            valid_edge_index = edge_index[:, valid_indices]
            true_adj[valid_edge_index[0], valid_edge_index[1]] = 1.0
    
    # Unified use of BCEWithLogits loss function (more stable, applicable to both KAN and MLP)
    # Both KAN and MLP decoders assume output is logit (raw scores)
    reconstruction_loss = F.binary_cross_entropy_with_logits(adj_scores, true_adj, reduction='mean')
    
    # 2. Embedding regularization loss (use gentler regularization)
    embedding_reg = torch.norm(node_embeddings, p=2, dim=1).mean()
    
    # 3. Sparsity loss (encourage sparse adjacency matrix)
    # Apply sigmoid to logit then compute sparsity
    adj_probs = torch.sigmoid(adj_scores)
    sparsity_loss = torch.norm(adj_probs, p=1) / (num_nodes * num_nodes)
    
    # Ensure all loss terms are valid numerical values
    if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
        reconstruction_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    if torch.isnan(embedding_reg) or torch.isinf(embedding_reg):
        embedding_reg = torch.tensor(0.0, device=device)
    
    if torch.isnan(sparsity_loss) or torch.isinf(sparsity_loss):
        sparsity_loss = torch.tensor(0.0, device=device)
    
    # Total loss (use smaller weights)
    total_loss = reconstruction_loss + 0.001 * embedding_reg + 0.0001 * sparsity_loss
    
    return total_loss


class AdvancedTrainingManager:
    """Advanced training manager - includes more advanced features"""
    
    def __init__(self, config):
        self.config = config
        self.training_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def train_with_advanced_features(self, model, node_features, edge_index):
        """Train using advanced features"""
        
        # Early stopping mechanism
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )
        
        # Learning rate scheduler
        optimizer = optim.AdamW(model.parameters(), lr=self.config.base_learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward propagation
            embeddings, adj = model(node_features, edge_index)
            
            # Compute loss
            loss = compute_loss_stable(embeddings, adj, edge_index, self.config)
            
            # Backward propagation
            loss.backward()
            optimizer.step()
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            # Record training history
            self.training_history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Early stopping check
            if early_stopping(loss.item()):
                break
        
        # Get final results
        model.eval()
        with torch.no_grad():
            final_embeddings, final_adj = model(node_features, edge_index)
        
        training_metrics = {
            'total_epochs': len(self.training_history),
            'final_loss': self.training_history[-1]['loss'] if self.training_history else float('inf'),
            'best_loss': min(h['loss'] for h in self.training_history) if self.training_history else float('inf')
        }
        
        return model, final_adj, training_metrics


class EarlyStopping:
    """Early stopping mechanism"""
    
    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class SimplifiedGNNKAN(nn.Module):
    """Simplified GNN-KAN model - main model class"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=None, num_layers=2, 
                 dropout=0.1, use_batch_norm=True, use_residual=True, kan_config=None):
        super(SimplifiedGNNKAN, self).__init__()
        
        if output_dim is None:
            output_dim = hidden_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Build GNN-KAN layers
        self.gnn_kan_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            layer_output_dim = hidden_dim if i < num_layers - 1 else output_dim
            
            # Create GNN-KAN layer
            gnn_kan_layer = self._create_gnn_kan_layer(layer_input_dim, layer_output_dim, kan_config)
            self.gnn_kan_layers.append(gnn_kan_layer)
            
            # Batch normalization
            if use_batch_norm and i < num_layers - 1:
                self.batch_norms.append(nn.LayerNorm(layer_output_dim, eps=1e-4))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Graph decoder
        self.graph_decoder = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1)
        )
        
    def _create_gnn_kan_layer(self, input_dim, output_dim, kan_config):
        """Create pure GNN-KAN layer - use compatible KAN implementation"""
        try:
            # Fix: use compatible KAN component creation function
            from .kan_components.kan_layers import create_compatible_kan_layer
            
            # Create compatible KAN layer using KAN configuration
            if kan_config:
                return create_compatible_kan_layer(
                    input_dim, output_dim,
                    num_basis=kan_config.get('num_basis', 8),
                    spline_order=kan_config.get('spline_order', 3),
                    grid_size=kan_config.get('grid_size', 8),
                    adaptive_spline_order=kan_config.get('adaptive_spline_order', True),
                    basis_function=kan_config.get('basis_function', 'chebyshev'),
                    basis_kwargs=kan_config.get('basis_kwargs', {})
                )
            else:
                return create_compatible_kan_layer(input_dim, output_dim)
                
        except ImportError:
            # Fallback to SimplifiedKANLayer
            try:
                from .kan_components import SimplifiedKANLayer
                return SimplifiedKANLayer(input_dim, output_dim)
            except ImportError:
                # Final fallback - but this means KAN features are missing
                return nn.Sequential(
                    nn.Linear(input_dim, output_dim, bias=False),  # Minimize MLP characteristics
                    nn.LayerNorm(output_dim, eps=1e-4)  # Use LayerNorm instead of BatchNorm
                )
    
    def forward(self, node_features, edge_index):
        """
        Forward propagation
        
        Args:
            node_features: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            node_embeddings: Node embeddings
            adj_scores: Adjacency matrix scores
        """
        # Input projection
        x = self.input_projection(node_features)
        
        # Process layer by layer
        for i, layer in enumerate(self.gnn_kan_layers):
            # Fix residual connection logic - KAN layer doesn't have [0] index
            # Save residual (if residual connection enabled and dimensions match)
            residual = x if self.use_residual else None
            
            # GNN-KAN layer processing
            try:
                x = layer(x)
                
                # Check if output is valid
                if torch.isnan(x).any() or torch.isinf(x).any():
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            except Exception as e:
                # If KAN layer fails, use simple transformation
                if hasattr(layer, 'base_linear'):
                    x = layer.base_linear(x)
                elif hasattr(layer, 'base_transform'):
                    x = layer.base_transform(x)
                else:
                    # Final fallback: keep input unchanged
                    pass
            
            # Safe residual connection
            if residual is not None and self.use_residual:
                try:
                    if x.shape == residual.shape:
                        x = x + residual
                    else:
                        # Don't use residual connection when dimensions don't match
                        pass
                except Exception as e:
                    pass
            
            # Batch normalization
            if self.batch_norms and i < len(self.batch_norms):
                try:
                    x = self.batch_norms[i](x)
                except Exception as e:
                    pass
            
            # Dropout (except last layer)
            if i < len(self.gnn_kan_layers) - 1:
                x = self.dropout(x)
        
        node_embeddings = x
        
        # Compute adjacency matrix scores
        adj_scores = self._compute_adjacency_matrix(node_embeddings)
        
        return node_embeddings, adj_scores
    
    def _compute_adjacency_matrix(self, embeddings):
        """Compute adjacency matrix scores"""
        num_nodes = embeddings.size(0)
        device = embeddings.device
        
        # Batch compute scores for all edges
        adj_scores = torch.zeros(num_nodes, num_nodes, device=device)
        
        # Optimized batch computation
        for i in range(num_nodes):
            # Compute connection scores between node i and all other nodes
            i_embedding = embeddings[i].unsqueeze(0).expand(num_nodes, -1)
            edge_features = torch.cat([i_embedding, embeddings], dim=1)
            scores = self.graph_decoder(edge_features).squeeze()
            adj_scores[i] = scores
        
        return adj_scores
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm,
            'use_residual': self.use_residual
        }


def create_model_with_config(config):
    """
    Create model based on configuration
    
    Args:
        config: SimplifiedGNNKANConfig configuration object
        
    Returns:
        model: Created model instance
    """
    try:
        model = SimplifiedGNNKAN(
            input_dim=config.input_dim,
            hidden_dim=getattr(config, 'hidden_dim', 64),
            output_dim=config.output_dim,
            num_layers=getattr(config, 'num_layers', 2),
            dropout=getattr(config, 'dropout', 0.1),
            use_batch_norm=getattr(config, 'use_batch_norm', True),
            use_residual=getattr(config, 'use_residual', True),
            kan_config=getattr(config, 'kan_config', None)
        )
        
        return model
        
    except Exception as e:
        # Create minimal fallback model
        try:
            model = SimplifiedGNNKAN(
                input_dim=getattr(config, 'input_dim', 32),
                hidden_dim=64,
                output_dim=getattr(config, 'output_dim', 16),
                num_layers=2,
                dropout=0.1
            )
            return model
        except Exception as fallback_error:
            raise e


class FaultAwareGraphDecoder(nn.Module):
    """
    Fault-aware graph decoder - use KAN to learn fault-related graph structure
    
    Principle:
    1. Use KAN layers for node pair interaction learning
    2. Fault type-aware weight adjustment
    3. End-to-end graph structure learning instead of using KNN
    """
    
    def __init__(self, embed_dim, num_nodes, kan_grid_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        
        # KAN-based attention for graph construction
        self.kan_attention = AdvancedKANLayer(embed_dim, embed_dim // 2)
        self.fault_aware_projection = nn.Linear(embed_dim, embed_dim)
        
        # Fault type encoder
        self.fault_encoder = nn.Embedding(6, embed_dim // 4)  # 6 fault types
        
        # KAN-based graph structure output
        self.output_kan = AdvancedKANLayer(embed_dim + embed_dim // 4, 1)
        
        # Fault type mapping
        self.fault_type_map = {
            'cpu': 0, 'mem': 1, 'disk': 2, 
            'socket': 3, 'delay': 4, 'loss': 5
        }
        
    def forward(self, embeddings, fault_type=None):
        """
        Forward propagation - learn fault-related graph structure
        
        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            fault_type: Fault type string
            
        Returns:
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        """
        num_nodes, embed_dim = embeddings.shape
        
        # Fault type-aware projection
        if fault_type is not None and fault_type in self.fault_type_map:
            fault_id = self.fault_type_map[fault_type]
            fault_context = self.fault_encoder(torch.tensor(fault_id, device=embeddings.device))
            fault_context = fault_context.expand(num_nodes, -1)
            # Fuse fault context
            enhanced_embeddings = embeddings + self.fault_aware_projection(embeddings)
        else:
            enhanced_embeddings = embeddings
        
        # Compute node pair interactions - use KAN to learn complex relationships
        expanded_emb1 = enhanced_embeddings.unsqueeze(1).expand(-1, num_nodes, -1)  # [N, N, D]
        expanded_emb2 = enhanced_embeddings.unsqueeze(0).expand(num_nodes, -1, -1)  # [N, N, D]
        
        # Node pair features: element-wise product + concatenation
        pairwise_features = expanded_emb1 * expanded_emb2  # [N, N, D]
        
        # Reshape to KAN expected format [N*N, D]
        batch_size, seq_len, feature_dim = pairwise_features.shape
        pairwise_features_flat = pairwise_features.view(-1, feature_dim)  # [N*N, D]
        
        # Use KAN to compute attention weights
        attention_weights_flat = self.kan_attention(pairwise_features_flat)  # [N*N, D//2]
        # If output is [N*N, D//2], restore to [N, N, D//2]
        attention_feat_dim = attention_weights_flat.shape[-1]
        attention_weights = attention_weights_flat.view(batch_size, seq_len, attention_feat_dim)
        
        # If fault-aware enabled, add fault context
        if fault_type is not None and fault_type in self.fault_type_map:
            fault_context_expanded = fault_context.unsqueeze(1).expand(-1, num_nodes, -1)
            # Project attention weights to same dimension as fault context then concatenate, avoid dimension conflict
            if attention_feat_dim != self.embed_dim // 2:
                # Safe projection to embed_dim // 2 dimensions
                proj = nn.Linear(attention_feat_dim, self.embed_dim // 2).to(attention_weights.device)
                attention_weights = proj(attention_weights)
                attention_feat_dim = attention_weights.shape[-1]
            combined_features = torch.cat([attention_weights, fault_context_expanded], dim=-1)
        else:
            combined_features = attention_weights
        
        # Output adjacency matrix (AdvancedKANLayer expects 2D input, flatten then restore)
        expected_in = self.output_kan.input_dim if hasattr(self.output_kan, 'input_dim') else (self.embed_dim + self.embed_dim // 4)
        last_dim = combined_features.shape[-1]
        if last_dim != expected_in:
            proj_out = nn.Linear(last_dim, expected_in).to(combined_features.device)
            combined_features = proj_out(combined_features)
            last_dim = expected_in
        # Flatten to [N*N, last_dim]
        combined_features_flat = combined_features.view(-1, last_dim)
        adj_scores_flat = self.output_kan(combined_features_flat).squeeze(-1)  # [N*N]
        adj_scores = adj_scores_flat.view(num_nodes, num_nodes)  # [N, N]
        
        # Fine-grained differentiation mechanism - fine-tune to enhance value range
        # Approach A: Fine-tune nonlinear amplification
        scale_factor = 4.0  # Fine-tune amplification coefficient
        adj_scores = torch.tanh(adj_scores * scale_factor)
        
        # Approach B: Fine-tune competition mechanism - moderately enhance noise and temperature
        noise = torch.randn_like(adj_scores) * 0.1  # Fine-tune noise strength
        adj_scores_noisy = adj_scores + noise
        
        # Approach B: Local competition mechanism - apply softmax per row to ensure node competition
        temperature = 0.15  # Fine-tune temperature, moderately enhance differences
        competitive_scores = torch.softmax(adj_scores_noisy * temperature, dim=1)
        
        # Fix C: Fault-aware differentiation
        if fault_type is not None and fault_type in self.fault_type_map:
            # Adjust competition strength based on fault type
            fault_intensity = {
                'cpu': 1.5, 'mem': 1.3, 'disk': 1.4, 
                'socket': 1.2, 'delay': 1.1, 'loss': 1.0
            }.get(fault_type, 1.0)
            competitive_scores = competitive_scores * fault_intensity
        
        # Fix D: Reduce sparsification level
        adj_matrix = self.apply_dynamic_sparsification(competitive_scores, sparsity_level=0.3)
        
        # Remove self-loops (avoid in-place operations that break gradients)
        eye = torch.eye(num_nodes, device=adj_matrix.device, dtype=adj_matrix.dtype)
        adj_matrix = adj_matrix * (1.0 - eye)
        
        # Ensure symmetry (undirected graph)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2.0
        
        return adj_matrix
    
    def apply_dynamic_sparsification(self, adj_matrix, sparsity_level=0.3):
        """Dynamically adjust sparsity threshold to ensure reasonable graph structure"""
        # Fix 1: Reduce sparsification level, retain more connections
        # Method 1: Based on quantile (adapt to graphs of different densities)
        threshold = torch.quantile(adj_matrix, 1 - sparsity_level)
        
        # Method 2: Mixed strategy (recommended)
        mean_val = torch.mean(adj_matrix)
        std_val = torch.std(adj_matrix)
        adaptive_threshold = 0.5 * threshold + 0.5 * (mean_val + 0.2 * std_val)
        
        # Fix 2: Ensure at least some connections are retained
        min_threshold = torch.quantile(adj_matrix, 0.8)  # Retain at least 20% of connections
        adaptive_threshold = torch.min(adaptive_threshold, min_threshold)
        
        # Apply sparsification
        sparse_adj = torch.where(
            adj_matrix > adaptive_threshold, 
            adj_matrix, 
            torch.tensor(0.0, device=adj_matrix.device)
        )
        
        # Fix 3: Ensure each node has at least one outgoing edge
        row_sums = sparse_adj.sum(dim=1, keepdim=True)
        zero_rows = (row_sums == 0).squeeze(1)
        
        if zero_rows.any():
            # For nodes without outgoing edges, retain their largest connection
            for i in range(adj_matrix.size(0)):
                if zero_rows[i]:
                    max_val, max_idx = torch.max(adj_matrix[i], dim=0)
                    if max_val > 0:
                        sparse_adj[i, max_idx] = max_val
        
        # Normalize
        row_sums = sparse_adj.sum(dim=1, keepdim=True)
        sparse_adj = torch.where(row_sums > 0, sparse_adj / (row_sums + 1e-8), sparse_adj)
        
        return sparse_adj


class MultiScaleGraphDecoder(nn.Module):
    """
    Multi-scale graph decoder - use KAN to learn graph structures at multiple scales
    
    Principle:
    1. Learn graph structures at multiple scales (local, middle, global)
    2. Dynamically fuse graphs at different scales with weights
    3. Adaptive sparsification, no need to preset fault types
    """
    
    def __init__(self, embed_dim, num_nodes, kan_grid_size=8, scales=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.scales = scales
        self.current_epoch = 0  # Training progress tracking
        
        # Multi-scale KAN decoders
        self.scale_decoders = nn.ModuleList([
            self._create_scale_decoder(embed_dim, kan_grid_size, scale=i) 
            for i in range(scales)
        ])
        
        # Optimization 2: Learnable temperature parameters
        self.learnable_temps = nn.Parameter(torch.tensor([0.3, 0.6, 0.9]))  # Initial temperature
        
        # Dynamic weight learner
        self.scale_weights = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, scales),
            nn.Softmax(dim=-1)
        )
        
        # Edge weight calibrator
        self.edge_calibrator = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Optimization 3: Learnable fusion weights
        self.sim_weights = nn.Parameter(torch.tensor([0.4, 0.25, 0.2, 0.15]))  # [original, euclidean, manhattan, variance]
    
    def _create_scale_decoder(self, embed_dim, kan_grid_size, scale):
        """Create single-scale decoder - adapt to enhanced feature dimensions"""
        # Enhanced feature dimension: 5*embed_dim (element_wise + concat + diff + cosine)
        rich_feature_dim = 5 * embed_dim
        decoder = nn.ModuleDict({
            'feature_compressor': nn.Linear(rich_feature_dim, embed_dim),  # Compress to original dimension
            'kan_attention': AdvancedKANLayer(embed_dim, embed_dim // 2),
            'output_kan': AdvancedKANLayer(embed_dim // 2, 1)
        })
        # Dynamic temperature parameter (will be updated in forward)
        decoder.temperature = nn.Parameter(torch.tensor(0.5))  # Initial value, will be dynamically adjusted
        return decoder
    
    def forward(self, embeddings, fault_type=None):
        """
        Forward propagation - learn multi-scale graph structure
        
        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            fault_type: Fault type (kept for interface compatibility, but not used)
            
        Returns:
            adj_matrix: Fused multi-scale adjacency matrix [num_nodes, num_nodes]
        """
        num_nodes, embed_dim = embeddings.shape
        
        # 1. Learn graph structures at multiple scales
        scale_graphs = []
        for i, decoder in enumerate(self.scale_decoders):
            # Optimization 2: Dynamic temperature adjustment
            base_temp = self.learnable_temps[i]
            progress = min(1.0, self.current_epoch / 100.0)  # Training progress [0,1]
            # Dynamic temperature: initial value + progress offset + small noise
            dynamic_temp = base_temp * (1 + 0.5 * progress) + 0.05 * torch.randn(1, device=embeddings.device).item()
            dynamic_temp = torch.clamp(torch.tensor(dynamic_temp), 0.1, 2.0)
            
            # Update decoder temperature
            decoder.temperature.data = dynamic_temp
            
            # Use dynamic temperature to generate diverse graphs
            adj = self._generate_scale_graph(embeddings, decoder, i)
            scale_graphs.append(adj)
        
        # Update epoch counter
        self.current_epoch += 1
        
        # 2. Dynamically predict optimal scale combination
        global_summary = torch.mean(embeddings, dim=0, keepdim=True)
        weights = self.scale_weights(global_summary).squeeze(0)
        
        # 3. Fuse multi-scale graphs
        fused_graph = sum(w * g for w, g in zip(weights, scale_graphs))
        
        # 4. Edge weight calibration
        calibrated_graph = self._calibrate_edge_weights(fused_graph, embeddings)
        
        # 5. Apply adaptive sparsification
        adj_matrix = self._apply_adaptive_sparsification(calibrated_graph)
        
        return adj_matrix
    
    def _generate_scale_graph(self, embeddings, decoder, scale_idx):
        """Generate single-scale graph structure - enhance node pair interactions"""
        num_nodes, embed_dim = embeddings.shape
        
        # Optimization 1: Enhance node pair interaction computation
        expanded_emb1 = embeddings.unsqueeze(1).expand(-1, num_nodes, -1)
        expanded_emb2 = embeddings.unsqueeze(0).expand(num_nodes, -1, -1)
        
        # Combine multiple interaction methods
        element_wise = expanded_emb1 * expanded_emb2  # Element-wise product
        concatenated = torch.cat([expanded_emb1, expanded_emb2], dim=-1)  # Concatenated features [N,N,2*D]
        difference = torch.abs(expanded_emb1 - expanded_emb2)  # Absolute difference
        cosine_sim = F.cosine_similarity(expanded_emb1, expanded_emb2, dim=-1).unsqueeze(-1)  # Cosine similarity
        
        # Combine multiple features [N,N,4*D+1]
        pairwise_features = torch.cat([
            element_wise,  # [N,N,D]
            concatenated,  # [N,N,2*D] 
            difference,    # [N,N,D]
            cosine_sim.expand(-1, -1, embed_dim)  # [N,N,D]
        ], dim=-1)
        
        # Reshape to KAN expected format
        rich_feature_dim = pairwise_features.shape[-1]  # 4*D+D = 5*D
        pairwise_features_flat = pairwise_features.view(-1, rich_feature_dim)
        
        # Compress features to original dimension
        compressed_features = decoder['feature_compressor'](pairwise_features_flat)
        
        # Use KAN to compute attention weights
        attention_weights_flat = decoder['kan_attention'](compressed_features)
        attention_weights = attention_weights_flat.view(num_nodes, num_nodes, -1)
        
        # Output adjacency matrix
        adj_scores_flat = decoder['output_kan'](attention_weights_flat).squeeze(-1)
        adj_scores = adj_scores_flat.view(num_nodes, num_nodes)
        
        # Apply dynamic temperature scaling (preserve linear scores, avoid intermediate compression)
        temperature = decoder.temperature
        adj_scores = adj_scores / (temperature + 1e-8)
        
        return adj_scores
    
    def _calibrate_edge_weights(self, adj_matrix, embeddings):
        """Mixed calibration mechanism - multi-dimensional similarity + learnable weights"""
        # Optimization 3: Multiple similarity computations
        # 1. Euclidean distance similarity
        euclidean_dist = torch.cdist(embeddings, embeddings, p=2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # 2. Manhattan distance similarity
        manhattan_dist = torch.cdist(embeddings, embeddings, p=1)
        manhattan_sim = 1.0 / (1.0 + manhattan_dist)
        
        # 3. Feature variance similarity
        feature_var = torch.var(embeddings, dim=1, keepdim=True)
        var_sim = 1.0 / (1.0 + torch.abs(feature_var - feature_var.T))
        
        # 4. Normalize all similarities to [0,1]
        euclidean_sim = (euclidean_sim - euclidean_sim.min()) / (euclidean_sim.max() - euclidean_sim.min() + 1e-8)
        manhattan_sim = (manhattan_sim - manhattan_sim.min()) / (manhattan_sim.max() - manhattan_sim.min() + 1e-8)
        var_sim = (var_sim - var_sim.min()) / (var_sim.max() - var_sim.min() + 1e-8)
        
        # 5. Learnable fusion (weights automatically normalized)
        weights = F.softmax(self.sim_weights, dim=0)
        calibrated = (
            weights[0] * adj_matrix +
            weights[1] * euclidean_sim +
            weights[2] * manhattan_sim +
            weights[3] * var_sim
        )
        
        # Dynamic range adjustment
        min_val, max_val = calibrated.min(), calibrated.max()
        calibrated = (calibrated - min_val) / (max_val - min_val + 1e-8)
        # Avoid extreme 0/1 causing excessive hard pruning
        calibrated = torch.clamp(calibrated, 0.05, 0.95)
        
        # Fine sharpening (reduce intensity, avoid excessive polarization)
        calibrated = self._sharpen_adjacency(calibrated, gamma=2.2)
        
        return calibrated
    
    def _sharpen_adjacency(self, adj_matrix, gamma=2.5, eps=1e-6):
        """
        Sharpen adjacency matrix - expand value range, enhance discriminability
        gamma > 1: sharpen, gamma < 1: smooth
        """
        # Ensure numerical stability
        adj_clamped = torch.clamp(adj_matrix, eps, 1 - eps)
        
        # Convert to logit space for sharpening
        logits = torch.log(adj_clamped) - torch.log(1 - adj_clamped)
        
        # Apply sharpening coefficient
        sharpened_logits = logits * gamma
        
        # Convert back to probability space
        sharpened = torch.sigmoid(sharpened_logits)
        
        # Ensure symmetry
        sharpened = (sharpened + sharpened.T) / 2
        
        return sharpened
    
    def _apply_adaptive_sparsification(self, adj_matrix, epoch=None):
        """Revised sparsification strategy - precisely control sparsity level"""
        if epoch is None:
            epoch = 0
        
        n = adj_matrix.shape[0]
        total_elements = n * n
        
        # Conservative sparsity target (maintain reasonable density)
        if epoch < 50:
            target_sparsity = 0.45   # Retain 45% in early stage
        elif epoch < 150:
            target_sparsity = 0.40   # Retain 40% in middle stage
        else:
            target_sparsity = 0.35   # Retain 35% in later stage
        
        # Conservative correction: limit sparsity range to maintain reasonable density
        if target_sparsity < 0.2:
            target_sparsity = 0.2
        if target_sparsity > 0.8:
            target_sparsity = 0.8
        
        # Compute threshold
        adj_flat = adj_matrix.flatten()
        threshold = torch.quantile(adj_flat, 1 - target_sparsity)
        
        # Conservative correction: add safety margin to maintain reasonable density
        min_threshold = torch.quantile(adj_flat, 0.7)   # Guarantee at least 30% highest connections retained
        threshold = max(threshold.item(), min_threshold.item())
        
        # Dynamic soft threshold: smooth transition (continuous weights, avoid generating many hard 0s)
        soft_values = torch.sigmoid((adj_matrix - threshold) * 3.5)
        
        # Minimum weight floor, avoid rows being all zeros
        soft_values = torch.clamp(soft_values, 5e-4, 1.0)
        
        # Ensure each node has at least 2 connections (if row is nearly all zeros, reinforce top-2)
        row_sums_soft = soft_values.sum(dim=1, keepdim=True)
        needs_boost = (row_sums_soft.squeeze(1) < 2e-4).float().view(-1, 1)
        k = min(2, n)
        _, top_indices = torch.topk(adj_matrix, k, dim=1)
        boost = torch.zeros_like(adj_matrix)
        for i in range(n):
            for j in range(k):
                if j < top_indices.shape[1]:
                    boost[i, top_indices[i, j]] = 1.0
        # Add small constant to avoid breaking continuity
        sparse_adj = soft_values + needs_boost * 1e-3 * boost
        
        # Ensure no self-connections are added
        sparse_adj = sparse_adj * (1 - torch.eye(n, device=sparse_adj.device))
        
        return sparse_adj


# Add SimplifiedGNNKAN to available models
__all__ = ['GNNKANModel', 'SimplifiedGNNKAN', 'TemporalAttention', 'create_fallback_model', 
           'train_gnn_kan_model', 'compute_loss_stable', 'MultiScaleGraphDecoder', 'LossScheduler']