"""
GAT Training Module
Uses the same training strategy as GNN-KAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


def train_gat_model(model, node_features, edge_index, config, **kwargs):
    """
    Train GAT model - Uses the same training strategy as GNN-KAN
    
    Args:
        model: GAT model
        node_features: Node features
        edge_index: Edge indices
        config: Configuration object (same as GNN-KAN)
        **kwargs: Other parameters
        
    Returns:
        trained_model: Trained model
        training_history: Training history
    """
    print("Starting GAT model training...")
    
    # Set device
    device = node_features.device
    model = model.to(device)
    
    # Optimizer (same settings as GNN-KAN)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler (same as GNN-KAN, but without verbose)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    training_history = {
        'loss': [],
        'adj_min': [],
        'adj_max': [],
        'adj_mean': [],
        'sparsity_01': [],
        'sparsity_03': [],
        'sparsity_05': []
    }
    
    # Training loop (same number of epochs as GNN-KAN)
    num_epochs = config.num_epochs
    best_loss = float('inf')
    patience_counter = 0
    patience = kwargs.get('patience', config.patience)
    
    print(f"  Training parameters: epochs={num_epochs}, lr={config.learning_rate}")
    print(f"  Fair comparison settings: layers={config.num_gnn_layers}, hidden_dims={config.hidden_dims}, output_dim={config.output_dim}")
    
    for epoch in range(num_epochs):
        model.train()
        
        # Forward propagation
        embeddings, adj_scores = model(node_features, edge_index)
        
        # Calculate loss
        if adj_scores is not None:
            # Build target adjacency matrix (based on edge indices)
            num_nodes = node_features.size(0)
            target_adj = torch.zeros(num_nodes, num_nodes, device=device)
            
            if edge_index.shape[1] > 0:
                row, col = edge_index
                target_adj[row, col] = 1.0
                target_adj[col, row] = 1.0  # Symmetric
            
            # Reconstruction loss
            recon_loss = criterion(adj_scores, target_adj)
            
            # Sparsity regularization (same as GNN-KAN)
            adj_probs = torch.sigmoid(adj_scores)
            sparsity_loss = torch.mean(adj_probs)
            
            # Total loss
            total_loss = recon_loss + 0.01 * sparsity_loss
        else:
            # If no adjacency matrix, use contrastive loss on embeddings
            total_loss = torch.var(embeddings)
        
        # Backward propagation
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (same as GNN-KAN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        
        optimizer.step()
        
        # Record training history
        training_history['loss'].append(total_loss.item())
        
        if adj_scores is not None:
            adj_probs = torch.sigmoid(adj_scores)
            training_history['adj_min'].append(adj_probs.min().item())
            training_history['adj_max'].append(adj_probs.max().item())
            training_history['adj_mean'].append(adj_probs.mean().item())
            training_history['sparsity_01'].append((adj_probs > 0.1).float().mean().item())
            training_history['sparsity_03'].append((adj_probs > 0.3).float().mean().item())
            training_history['sparsity_05'].append((adj_probs > 0.5).float().mean().item())
        
        # Learning rate scheduling
        scheduler.step(total_loss)
        
        # Early stopping (same as GNN-KAN)
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        min_epochs = kwargs.get('min_epochs', getattr(config, 'min_epochs', 30))
        if patience_counter >= patience and epoch >= min_epochs:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        # Periodic output
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6e}")
    
    print(f"GAT training completed, final loss: {training_history['loss'][-1]:.6f}")
    
    return model, training_history
