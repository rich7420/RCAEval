"""
Pure GNN Training Module
Pure GNN training module
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


def train_gnn_model(model, node_features, edge_index, config, **kwargs):
    """
    Train pure GNN model
    
    Args:
        model: GNN model
        node_features: Node features
        edge_index: Edge indices
        config: Configuration object
        **kwargs: Other parameters
        
    Returns:
        trained_model: Trained model
        training_history: Training history
    """
    
    # Set device
    device = node_features.device
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    # Note: Some older torch versions don't support verbose parameter, so don't pass it to maintain compatibility
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
    
    # Training loop
    num_epochs = config.num_epochs
    best_loss = float('inf')
    patience_counter = 0
    patience = kwargs.get('patience', 20)
    
    
    for epoch in range(num_epochs):
        model.train()
        
        # Forward propagation
        embeddings, adj_scores = model(node_features, edge_index)
        
        # Compute loss
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
            
            # Sparsity regularization
            adj_probs = torch.sigmoid(adj_scores)
            sparsity_loss = torch.mean(adj_probs)
            
            # Total loss
            total_loss = recon_loss + 0.01 * sparsity_loss
        else:
            # If no adjacency matrix, use contrastive loss on embeddings
            # Simplified version: minimize embedding variance
            total_loss = torch.var(embeddings)
        
        # Backward propagation
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
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
        
        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience and epoch >= kwargs.get('min_epochs', 30):
            break
        
        # Periodic output
        if (epoch + 1) % 20 == 0:
            pass
    
    
    return model, training_history

