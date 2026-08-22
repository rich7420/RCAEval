"""
Training Module for GNN-KAN
Training module - handles training and optimization of GNN-KAN models
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import math
try:  # torch_geometric pulls torch_sparse (compiled ext); negative_sampling is unused here
    from torch_geometric.utils import negative_sampling
except Exception:  # pragma: no cover - keep GNN_KAN importable without torch_geometric
    negative_sampling = None

# Use the centralized, full implementation of GradientStabilizer
# GradientStabilizer removed, use simplified version
from .config import SimplifiedGNNKANConfig


def learning_rate_scheduler(optimizer, current_epoch, total_epochs):
    """Cosine annealing learning rate scheduler - smoothly reduce LR"""
    lr_max = 5e-4  # Increase initial LR to 5e-4, make early training faster
    min_lr = 1e-6
    # Cosine curve for smooth decay
    cos_anneal = 0.5 * (1 + math.cos(math.pi * current_epoch / total_epochs))
    current_lr = min_lr + (lr_max - min_lr) * cos_anneal
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr
# Import the model from the models module, not a local copy
from .models import SimplifiedGNNKAN, GNNKANModel, TemporalAttention

# Fix model import
try:
    from .models import SimplifiedGNNKAN
except ImportError:
    pass


def create_model_with_config(config):
    """
    Create model based on configuration - redirect to unified implementation in models.py
    
    Args:
        config: SimplifiedGNNKANConfig configuration object
        
    Returns:
        model: Created model instance
    """
    from .models import create_model_with_config as _create_model
    return _create_model(config)


def create_model_from_checkpoint(checkpoint_path, config=None):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Checkpoint file path
        config: Configuration object (optional)
        
    Returns:
        model: Loaded model
        metadata: Checkpoint metadata
    """
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get configuration from checkpoint
        if 'config' in checkpoint and config is None:
            config = checkpoint['config']
        
        # Create model
        if config:
            model = create_model_with_config(config)
        else:
            # Infer model structure from state_dict
            model = _infer_model_from_state_dict(checkpoint['model_state_dict'])
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', None),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', None)
        }
        
        return model, metadata
        
    except Exception as e:
        if config:
            return create_model_with_config(config), {}
        else:
            raise e


def _infer_model_from_state_dict(state_dict):
    """Infer model structure from state_dict"""
    # Analyze state_dict keys to infer model parameters
    keys = list(state_dict.keys())
    
    # Infer number of layers
    conv_layers = [k for k in keys if 'conv_layers' in k and 'weight' in k]
    num_layers = len(set(k.split('.')[1] for k in conv_layers))
    
    # Infer dimensions
    first_conv = [k for k in keys if 'conv_layers.0' in k and 'weight' in k]
    if first_conv:
        first_weight = state_dict[first_conv[0]]
        if len(first_weight.shape) >= 2:
            input_dim = first_weight.shape[1]
            hidden_dim = first_weight.shape[0]
        else:
            input_dim = hidden_dim = 64
    else:
        input_dim = hidden_dim = 64
    
    # Create model
    model = SimplifiedGNNKAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_layers=max(num_layers, 2)
    )
    
    return model


class ModelManager:
    """Model manager - unified management of model creation, training, saving"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def create_model(self):
        """Create model"""
        self.model = create_model_with_config(self.config)
        return self.model
    
    def setup_training(self):
        """Setup training components"""
        if self.model is None:
            self.create_model()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.config, 'learning_rate', 0.001),
            weight_decay=getattr(self.config, 'weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10
        )
        
        return self.optimizer, self.scheduler
    
    def train_model(self, node_features, edge_index):
        """Train model"""
        if self.model is None or self.optimizer is None:
            self.setup_training()
        
        return train_gnn_kan_model(
            self.model, node_features, edge_index, self.config
        )
    
    def save_checkpoint(self, epoch, loss, metrics, save_path):
        """Save checkpoint"""
        # This function is being consolidated into utils.py
        # save_model_checkpoint(
        #     self.model, self.optimizer, epoch, loss, metrics, save_path, self.config
        # )
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        self.model, metadata = create_model_from_checkpoint(checkpoint_path, self.config)
        return metadata


class GNNKANLoss(nn.Module):
    """GNN-KAN specific loss function - Stage 1 improvement: solve over-condensation/overfitting"""
    
    def __init__(self, config):
        super(GNNKANLoss, self).__init__()
        self.config = config
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # Stage 1: Dynamic loss weights, prevent condensation
        self.polar_weight = getattr(config, 'polar_weight', 0.8)  # Reduce Polar weight
        self.contrast_weight = getattr(config, 'contrast_weight', 1.5)  # Increase Contrast weight
        self.sparsity_target = getattr(config, 'sparsity_target', 0.3)  # Target sparsity
        
    def forward(self, pred_adj, true_adj, node_embeddings=None):
        """
        Compute combined loss - Stage 1 improvement: solve over-condensation/overfitting
        
        Args:
            pred_adj: Predicted adjacency matrix
            true_adj: True adjacency matrix  
            node_embeddings: Node embeddings (optional)
        """
        # Fix CUDA assertion error: ensure pred_adj is in [0,1] range
        pred_adj_safe = torch.clamp(pred_adj, min=1e-7, max=1.0-1e-7)
        true_adj_safe = torch.clamp(true_adj, min=0.0, max=1.0)
        
        # Numerical stability check
        has_nan = torch.isnan(pred_adj_safe).any()
        has_inf = torch.isinf(pred_adj_safe).any()
        
        if has_nan or has_inf:
            pred_adj_clean = torch.nan_to_num(pred_adj, nan=0.0, posinf=1.0, neginf=0.0)
            recon_loss = self.mse_loss(pred_adj_clean, true_adj_safe)
        else:
            # Reconstruction loss
            recon_loss = self.bce_loss(pred_adj_safe, true_adj_safe)
        
        total_loss = recon_loss
        
        # Dynamic weight adjustment strategy - adjust loss weights based on training progress
        epoch = getattr(self, 'current_epoch', 0)
        total_epochs = getattr(self, 'total_epochs', 100)
        
        # Corrected dynamic weight computation - balance loss terms
        w_recon = 0.5  # Further reduce reconstruction loss weight
        w_polar = 0.2 + 0.5 * (epoch / max(total_epochs, 1))  # Polarization loss enhancement
        w_contrast = min(8.0, 2.0 + epoch * 0.1 )  # More aggressive contrast loss enhancement
        w_sparsity = min(0.8, 0.1 + epoch * 0.01)  # Reduce sparsity constraint pressure
        
        # Improved sparsity penalty
        current_sparsity = (pred_adj_safe > 0.1).float().mean().item()
        sparsity_penalty = torch.mean(torch.abs(pred_adj_safe)) * (1 - current_sparsity)
        total_loss += sparsity_penalty * w_sparsity
        
        # Enhanced contrastive loss
        if node_embeddings is not None:
            contrast_loss = self.enhanced_contrastive_loss(node_embeddings, pred_adj_safe)
            total_loss += contrast_loss * w_contrast
            
            # L2 regularization
            l2_reg = torch.norm(node_embeddings, p=2)
            total_loss += self.config.l2_lambda * l2_reg
            
            # New: Variance regularization - encourage node representations to have sufficient discriminability
            std_per_dim = torch.clamp(node_embeddings.std(dim=0), 1e-3, None)
            var_loss = torch.relu(1.0 - std_per_dim).mean()
            total_loss += 0.05 * var_loss
            
            # Smoothness regularization
            if node_embeddings.size(0) > 1:
                diff = torch.diff(node_embeddings, dim=0)
                smoothness_reg = torch.norm(diff, p=2)
                total_loss += self.config.smoothness_lambda * smoothness_reg
        
        # Stabilized polarization loss
        polar_loss = self.stabilized_polar_loss(pred_adj_safe, alpha=0.5)
        total_loss += polar_loss * w_polar
        
        # Fine Margin Loss - fine-tune to ensure top/bottom gap
        if node_embeddings is not None:
            margin = 0.08  # Fine-tune gap requirement
            margin_loss = self.discriminative_loss(pred_adj_safe.flatten(), margin)
            total_loss += 0.15 * margin_loss
        
        # Direction 3: Ranking Loss - directly optimize ranking (enabled later)
        if epoch > 50 and node_embeddings is not None:
            try:
                # Generate pseudo ranks
                ground_truth_ranks = self.generate_pseudo_ranks(pred_adj_safe)
                
                # Compute PageRank scores as pr_scores
                pr_scores = pred_adj_safe.sum(dim=1)  # Simplified PageRank scores
                
                # Apply ListNet loss
                rank_loss = self.listnet_loss(pr_scores, ground_truth_ranks)
                total_loss += 0.2 * rank_loss
                
                if epoch % 10 == 0:  # Print every 10 epochs
                    pass
            except Exception as e:
                pass
        
        return total_loss
    
    def enhanced_contrastive_loss(self, embeddings, adj_matrix, 
                             base_temp=0.2, min_temp=0.1, max_temp=0.5):
        """
        Revised contrastive learning - ensure consistent magnitude with other loss terms
        * No fault type information needed *
        """
        batch_size = embeddings.shape[0]
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        logits = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # Key fix: use adjacency matrix as positive sample strength, but strengthen differences
        # Map adjacency weights from [0.5, 0.73] to [-1.0, 1.0] to expand differences
        weights = (adj_matrix - 0.62) * 10.0  # Center point set to range midpoint
        
        # Limit to reasonable range [-1.0, 1.0]
        weights = torch.clamp(weights, -1.0, 1.0)
        
        # Key fix: enhance high-value connections
        strong_connections = (adj_matrix > 0.68).float()
        weights = weights + strong_connections * 0.3
        
        # Remove self-connections
        eye = torch.eye(batch_size, device=embeddings.device)
        weights = weights * (1 - eye)
        
        # Fix: use weighted similarity to compute loss
        positive_logits = logits * weights * (1 - eye)
        negative_logits = logits * (1 - weights) * (1 - eye)
        
        # Key fix: increase contrastive loss magnitude, ensure numerical stability
        pos_exp = torch.exp(torch.clamp(positive_logits.sum(dim=1) / 10.0, -10, 10))
        neg_exp = torch.exp(torch.clamp(negative_logits.sum(dim=1) / 10.0, -10, 10))
        
        # Avoid division by zero and log(0)
        ratio = pos_exp / (neg_exp + 1e-8)
        ratio = torch.clamp(ratio, min=1e-8, max=1e8)
        
        contrastive_loss = -torch.log(ratio).mean()
        
        # Check NaN and handle
        if torch.isnan(contrastive_loss):
            contrastive_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Association constraint: ensure node representations are consistent with adjacency matrix
        recon_loss = torch.nn.functional.mse_loss(
            torch.mm(norm_embeddings, norm_embeddings.t()),
            adj_matrix.detach()
        ) * 0.1
        
        return contrastive_loss * 5.0 + recon_loss  # Explicitly increase contrastive loss weight
    
    def discriminative_loss(self, pr_scores, margin=0.1):
        """
        Discriminative loss - directly optimize top-1 separation
        pr_scores: PageRank score list (sorted)
        margin: Minimum gap requirement
        """
        if len(pr_scores) < 2:
            return torch.tensor(0.0, device=pr_scores[0].device)
        
        # Ensure scores are tensor
        if not isinstance(pr_scores, torch.Tensor):
            pr_scores = torch.tensor(pr_scores, dtype=torch.float32)
        
        # Sort scores
        sorted_scores, _ = torch.sort(pr_scores, descending=True)
        
        # Calculate gap between top-1 and top-2
        score_diff = sorted_scores[0] - sorted_scores[1]
        
        # Loss: generate loss when difference is less than margin
        loss = torch.relu(margin - score_diff)
        
        return loss
    
    def listnet_loss(self, pr_scores, ground_truth_ranks, temperature=1.0):
        """ListNet ranking loss - optimize PageRank ranking, no fault type needed"""
        # Convert to probability distribution
        predicted_probs = torch.softmax(pr_scores / temperature, dim=0)
        
        # Generate ideal distribution based on ground_truth_ranks (data-driven: high rank should have high prob)
        ideal_probs = torch.zeros_like(predicted_probs)
        for idx, rank in ground_truth_ranks.items():
            ideal_probs[idx] = rank / sum(ground_truth_ranks.values())
        
        # KL divergence as loss
        kl_div = F.kl_div(
            predicted_probs.log(),
            ideal_probs, 
            reduction='batchmean'
        )
        
        return kl_div
    
    def generate_pseudo_ranks(self, adj_matrix):
        """Generate pseudo label (data-driven)"""
        row_sums = adj_matrix.sum(dim=1).detach()
        pseudo_ranks = row_sums / row_sums.max()
        return {i: pseudo_ranks[i].item() for i in range(adj_matrix.shape[0])}
    
    def stabilized_polar_loss(self, adj_matrix, alpha=0.5):
        """
        Stabilized polarization loss, avoid gradient explosion
        alpha: Control balance between sparse and dense (0=pure sparse, 1=pure dense)
        """
        # First apply sigmoid to ensure value range is [0,1]
        adj_probs = torch.sigmoid(adj_matrix)
        
        # Combine two objectives: sparse and structured
        sparse_loss = torch.mean(adj_probs) * (1 - alpha)
        structured_loss = torch.var(adj_probs) * alpha
        
        # Add gradient stabilization term
        stable_term = 0.001 * torch.log(torch.var(adj_probs) + 1e-8)
        
        return sparse_loss + structured_loss - stable_term


class LossScheduler:
    """Loss weight adaptive controller without fault type"""
    
    def __init__(self, initial_weights, window_size=10):
        self.weights = initial_weights.copy()
        self.window_size = window_size
        self.history = []
    
    def update_weights(self, metrics):
        """Automatically adjust weights based on training dynamics"""
        self.history.append(metrics)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # 1. Dynamically adjust based on contrast loss performance
        if len(self.history) >= 2:
            contrast_trend = (self.history[-1]['contrast'] - 
                             self.history[-2]['contrast'])
            
            # Contrast loss decreasing too fast? May indicate insufficient learning
            if contrast_trend < -0.05:
                self.weights['contrast'] = min(3.0, 
                                             self.weights['contrast'] * 1.1)
            # Contrast loss stagnant? Reduce attention
            elif abs(contrast_trend) < 0.01:
                self.weights['contrast'] = max(0.5, 
                                             self.weights['contrast'] * 0.95)
        
        # 2. Automatically adjust polarization loss based on graph sparsity
        sparsity = 1.0 - metrics['density']
        if sparsity < 0.3:  # Graph too dense
            self.weights['polar'] = min(2.0, self.weights['polar'] * 1.05)
        elif sparsity > 0.7:  # Graph too sparse
            self.weights['polar'] = max(0.1, self.weights['polar'] * 0.95)
        
        # 3. Automatically balance reconstruction and structure loss
        recon_ratio = metrics['recon'] / (metrics['sparsity'] + 1e-5)
        if recon_ratio > 0.5:  # High reconstruction demand
            self.weights['recon'] = min(2.0, self.weights['recon'] * 1.02)
        
        return self.weights.copy()


def train_gnn_kan_model(model, node_features, edge_index, config, sparsity_lambda=None, fault_type=None, **kwargs):
    """
    🚨 ISSUE 6: 實時性能力不足分析
    
    Real-time issues:
    1. **Training time too long**:
       - Currently requires 200 epochs, average 60-180 seconds
       - Actual fault response requirement: <10 seconds for preliminary analysis
       - On-the-fly training not feasible in production environment
    
    2. **Cold start problem**:
       - New systems or new fault types require retraining
       - Lack of pretrained models or transfer learning mechanisms
       - Cannot leverage experience from historically similar faults
    
    3. **Missing incremental learning**:
       - Complete retraining every time
       - Cannot continuously learn from new faults
       - Model cannot evolve and improve over time
    
    4. **Inference complexity**:
       - O(V²) adjacency matrix computation still needed during inference
       - KAN layer B-spline computation more time-consuming than MLP matrix multiplication
       - Multi-stage post-processing increases response latency
    
    Real-time improvement suggestions:
    1. Pretraining + fine-tuning strategy:
       # pretrained_model = load_universal_pretrained_model()
       # quick_adapted = few_shot_adaptation(pretrained_model, current_data)
    
    2. Model compression:
       # compressed_model = knowledge_distillation(full_model, student_model)
       # quantized_model = dynamic_quantization(compressed_model)
    
    3. Approximate inference:
       # sparse_adj = approximate_adjacency(embeddings, top_k=20)
       # fast_pagerank = power_iteration_early_stop(sparse_adj, max_iter=10)
    
    4. Tiered response:
       # immediate_response = fast_heuristic_ranking(raw_features)  # <1s
       # refined_response = simplified_kan_inference(processed_data)  # <5s  
       # detailed_analysis = full_gnn_kan_pipeline(all_data)  # <30s
    
    Current implementation - NOT suitable for real-time production
    """
    # Fix 6: Real-time improvement - add fast mode
    fast_mode = getattr(config, 'fast_mode', False) or sparsity_lambda is None
    if fast_mode:
        # Fast mode: reduce training epochs, increase convergence speed
        config.num_epochs = min(50, config.num_epochs)
        config.learning_rate = config.learning_rate * 1.5  # More aggressive learning rate
    
    # Set sparsity parameters
    if sparsity_lambda is None:
        sparsity_lambda = 1e-5 if fast_mode else 1e-4
    
    device = 'cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu'
    model.to(device)
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.patience, factor=0.5)
    
    # Stage 1: Improved early stopping mechanism
    from .models import EarlyStopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    # Stage 1: Validation data preparation
    val_data = kwargs.get('val_data', None)
    
    # Initialize loss function
    loss_fn = GNNKANLoss(config)
    val_ground_truth = kwargs.get('val_ground_truth', None)
    monitor_metric = kwargs.get('monitor_metric', 'val_precision')
    
    best_val_metric = 0.0
    patience_counter = 0
    
    
    training_history = {'loss': [], 'adj_min': [], 'adj_max': [], 'adj_mean': [], 'sparsity_01': [], 'sparsity_03': [], 'sparsity_05': []}
    
    # Create weighted true adjacency matrix - based on feature similarity rather than simple binarization
    with torch.no_grad():
        num_nodes = node_features.size(0)
        true_adj = torch.zeros(num_nodes, num_nodes, device=device)
        
        # Compute similarity based on node features as true weights
        if num_nodes > 1:
            # Normalize features
            norm_features = F.normalize(node_features, p=2, dim=1)
            # Compute cosine similarity
            similarity_matrix = torch.mm(norm_features, norm_features.t())
            # Convert similarity to [0,1] range weights
            true_adj = (similarity_matrix + 1) / 2
            # Remove self-loop influence
            true_adj.fill_diagonal_(0)
            
            # Enhancement: strengthen important connections based on original edges (from improvement scheme)
            if edge_index.size(1) > 0:
                # Give additional weight to original edges - increase to 0.3
                edge_boost = 0.3
                true_adj[edge_index[0], edge_index[1]] += edge_boost
                true_adj[edge_index[1], edge_index[0]] += edge_boost
                # Ensure weights are in [0,1] range
                true_adj = torch.clamp(true_adj, 0, 1)
        else:
            # Fallback for single node case
            if edge_index.size(1) > 0:
                true_adj[edge_index[0], edge_index[1]] = 1
                true_adj[edge_index[1], edge_index[0]] = 1
    
    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Direction 2: Learning rate scheduling - smoothly reduce LR
        current_lr = learning_rate_scheduler(optimizer, epoch, config.num_epochs)
        
        # Set current epoch information for loss function use
        if hasattr(loss_fn, 'current_epoch'):
            loss_fn.current_epoch = epoch
        if hasattr(loss_fn, 'total_epochs'):
            loss_fn.total_epochs = config.num_epochs
            
        # Forward propagation - use fault type awareness
        node_embedding, pred_adj = model(node_features, edge_index, fault_type)
            
        # Loss computation - handle KNN Baseline case
        if pred_adj is None:
            # Use KNN Baseline, skip reconstruction loss computation
            recon_loss = torch.tensor(0.0, device=node_embedding.device, requires_grad=True)
            pred_adj_sigmoid = torch.eye(node_embedding.size(0), device=node_embedding.device)
        else:
            # Graph Decoder case
            # If pred_adj is already in [0,1], treat as probability and use BCE directly; otherwise treat as logits and use BCEWithLogits
            try:
                pred_min = float(pred_adj.detach().min())
                pred_max = float(pred_adj.detach().max())
                is_prob = (pred_min >= 0.0) and (pred_max <= 1.0)
            except Exception:
                is_prob = False

            if is_prob:
                recon_loss = F.binary_cross_entropy(pred_adj, true_adj, reduction='mean')
                pred_adj_sigmoid = pred_adj
            else:
                recon_loss = F.binary_cross_entropy_with_logits(pred_adj, true_adj, reduction='mean')
                pred_adj_sigmoid = torch.sigmoid(pred_adj)
        
        # Strengthen polarization loss - use more aggressive strategy
        # Method 1: Reward values close to 0 and 1, penalize middle values
        distance_from_center = torch.abs(pred_adj_sigmoid - 0.5)
        polarization_loss = -torch.mean(distance_from_center ** 2) * 6  # Reduce intensity to avoid extremization
        
        # Method 2: Add top-k contrast loss - strengthen most important edges
        if pred_adj_sigmoid.numel() > 4:  # Ensure enough elements
            flat_probs = pred_adj_sigmoid.flatten()
            k = max(2, len(flat_probs) // 4)  # Select top 25%
            top_k_vals, _ = torch.topk(flat_probs, k)
            bottom_k_vals, _ = torch.topk(flat_probs, k, largest=False)
            
            # Encourage top-k close to 1, bottom-k close to 0
            top_contrast = -torch.mean((1 - top_k_vals) ** 2)
            bottom_contrast = -torch.mean(bottom_k_vals ** 2)
            contrast_enhance = (top_contrast + bottom_contrast) * 0.5
            polarization_loss += contrast_enhance
        
        # New: top-K/bottom-K contrast to enhance discriminability (from improvement scheme)
        if pred_adj_sigmoid.numel() > 8:  # Ensure enough elements for contrast
            flat_adj = pred_adj_sigmoid.flatten()
            top_k = min(5, len(flat_adj) // 4)  # Select top 25% or at most 5
            bottom_k = min(5, len(flat_adj) // 4)
            
            top_k_vals = torch.topk(flat_adj, top_k)[0]
            bottom_k_vals = torch.topk(flat_adj, bottom_k, largest=False)[0]
            
            # Moderately strengthen discriminability: top-K close to 1, bottom-K close to 0
            top_contrast_loss = torch.mean((1 - top_k_vals) ** 2) * 5
            bottom_contrast_loss = torch.mean(bottom_k_vals ** 2) * 5
            
            # New: moderate score difference loss
            if len(top_k_vals) > 0 and len(bottom_k_vals) > 0:
                score_diff_loss = torch.mean((top_k_vals[0] - bottom_k_vals[0]) ** 2) * 6
                polarization_loss += score_diff_loss
            
            # Add to polarization loss
            polarization_loss += top_contrast_loss + bottom_contrast_loss
        
        # Improvement 2: Add contrastive loss, encourage learning meaningful representations
        contrast_loss = 0
        if node_embedding.size(0) > 1:
            # Compute node embedding similarity
            embedding_sim = torch.mm(F.normalize(node_embedding, dim=1), F.normalize(node_embedding, dim=1).t())
            # Contrast with true adjacency matrix
            contrast_loss = F.mse_loss(embedding_sim, true_adj) * 0.1
        
        # 2. KAN regularization loss (should be retained)
        kan_reg_loss = 0
        if hasattr(model, 'get_reg_loss'):
            # Ensure kan_reg_loss is a scalar
            reg_loss = model.get_reg_loss()
            if isinstance(reg_loss, torch.Tensor):
                kan_reg_loss = reg_loss
            else: # Assume it's a list or tuple
                kan_reg_loss = sum(reg_loss)

        # Improvement 3: More effective sparsity loss
        if sparsity_lambda > 0:
            adj_probs_for_loss = pred_adj_sigmoid
            # Use L1 regularization instead of simple mean
            sparsity_loss = torch.mean(torch.abs(adj_probs_for_loss))
        else:
            sparsity_loss = torch.tensor(0.0, device=device)
        
        # New: Score differentiation margin-based loss, force top and bottom to differ by at least margin
        margin_loss = 0
        flat_probs_for_margin = pred_adj_sigmoid.flatten()
        if flat_probs_for_margin.numel() >= 2:
            max_val = torch.max(flat_probs_for_margin)
            min_val = torch.min(flat_probs_for_margin)
            # Dynamic margin: use higher margin for small graphs
            margin = 0.5 if flat_probs_for_margin.numel() < 50 else 0.3
            # Use hinge-style: max(0, margin - (max - min))
            # Increased from 4.0 to 8.0 to enhance score differentiation
            margin_loss = torch.relu(margin - (max_val - min_val)) * 8.0

        # New: Training period feature decorrelation regularization
        decor_loss = 0
        if node_embedding.size(0) > 1 and node_embedding.size(1) > 1:
            # Compute embedding covariance matrix
            embeddings_mean = node_embedding.mean(dim=0)
            centered = node_embedding - embeddings_mean
            cov = (centered.T @ centered) / (centered.shape[0] - 1)
            # Penalize off-diagonal elements (decorrelation)
            off_diag_cov = cov - torch.diag(cov.diag())
            decor_loss = torch.mean(off_diag_cov ** 2) * 0.1  # Small weight to avoid over-penalization
        
        # New: Graph sparsity loss - encourage balanced density
        graph_sparsity_loss = 0
        if pred_adj_sigmoid.numel() > 0:
            # Compute graph density
            graph_density = (pred_adj_sigmoid > 0.1).float().mean().item()
            # Target density 0.3-0.5, penalize too dense or too sparse
            target_density = 0.4
            density_penalty = torch.abs(torch.tensor(graph_density - target_density)) * 2.0
            graph_sparsity_loss = density_penalty
        
        # Improvement 4: Adjust loss weights to ensure effective learning (reconstruction primary, contrast/polarization secondary), support config override
        w_recon = getattr(config, 'w_recon', 1.2)
        w_contrast = getattr(config, 'w_contrast', 0.3)
        w_polar = getattr(config, 'w_polar', 0.2)
        w_margin = getattr(config, 'w_margin', 0.1)
        w_kan_reg = getattr(config, 'w_kan_reg', 0.005)
        w_decor = getattr(config, 'w_decor', 0.1)
        w_graph = getattr(config, 'w_graph', 0.2)

        total_loss = (
            w_recon * recon_loss
            + w_contrast * contrast_loss
            + w_polar * polarization_loss
            + w_margin * margin_loss
            + w_kan_reg * kan_reg_loss
            + (sparsity_lambda * sparsity_loss)
            + w_decor * decor_loss
            + w_graph * graph_sparsity_loss
        )
        
        # Improvement 5: Ensure loss is within reasonable range (only adjust when non-finite or clearly degraded)
        if not torch.isfinite(total_loss):
            total_loss = recon_loss + kan_reg_loss
        elif total_loss.item() < 0 and recon_loss.item() < 0.01:
            # Only consider degraded when total loss is negative and reconstruction term is very small, avoid noise-like warnings
            total_loss = recon_loss + contrast_loss + (sparsity_lambda * sparsity_loss)
        
        # Critical fix: NaN value detection and handling
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            
            # Use safe loss values
            safe_recon = torch.tensor(0.0, device=recon_loss.device) if torch.isnan(recon_loss) else recon_loss
            safe_contrast = torch.tensor(0.0, device=node_features.device) if contrast_loss == 0 else torch.tensor(contrast_loss, device=node_features.device)
            safe_polar = torch.tensor(0.0, device=node_features.device) if torch.isnan(polarization_loss) else polarization_loss
            safe_margin = torch.tensor(0.0, device=node_features.device) if margin_loss == 0 else torch.tensor(margin_loss, device=node_features.device)
            
            total_loss = w_recon * safe_recon + w_contrast * safe_contrast + w_polar * safe_polar + w_margin * safe_margin
            
            # If still problematic, use minimum loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(0.1, device=total_loss.device, requires_grad=True)
        
        # Backward propagation
        total_loss.backward()
        
        # New: Global gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # New: KAN coefficient regularization
        kan_reg = 0.0
        for m in model.modules():
            if hasattr(m, 'spline_coeffs'):
                kan_reg = kan_reg + m.spline_coeffs.pow(2).sum()
            if hasattr(m, 'activation_weights'):
                kan_reg = kan_reg + m.activation_weights.pow(2).sum()
        
        if kan_reg > 0:
            kan_reg_loss = 1e-4 * kan_reg
            total_loss = total_loss + kan_reg_loss
        
        optimizer.step()
        
        # Record and print
        training_history['loss'].append(total_loss.item())
        
        # Compute adjacency matrix statistics
        with torch.no_grad():
            if pred_adj is not None:
                # Use probability matrix already determined by value range, avoid sigmoid compression again
                adj_probs = pred_adj_sigmoid
                adj_min = adj_probs.min().item()
                adj_max = adj_probs.max().item()
                adj_mean = adj_probs.mean().item()
            else:
                # KNN Baseline case
                adj_min = 0.0
                adj_max = 1.0
                adj_mean = 0.5
            
            if pred_adj is not None:
                sparsity_01 = (adj_probs < 0.1).float().mean().item()
                sparsity_03 = (adj_probs < 0.3).float().mean().item()
                sparsity_05 = (adj_probs < 0.5).float().mean().item()
            else:
                # KNN Baseline case
                sparsity_01 = 0.5
                sparsity_03 = 0.3
                sparsity_05 = 0.1
        
        training_history['adj_min'].append(adj_min)
        training_history['adj_max'].append(adj_max)
        training_history['adj_mean'].append(adj_mean)
        training_history['sparsity_01'].append(sparsity_01)
        training_history['sparsity_03'].append(sparsity_03)
        training_history['sparsity_05'].append(sparsity_05)

        # Stage 1: Validation and early stopping check
        if val_data is not None and (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            model.eval()
            with torch.no_grad():
                # Extract val_edge_index, filter before sending to model
                val_node_features = val_data['node_features'].to(device)
                val_edge_index = val_data['edge_index'].to(device)

                val_used_edge_index = val_edge_index
                if val_edge_index.size(1) > 0:
                    valid_mask = (val_edge_index[0] < val_node_features.size(0)) & (val_edge_index[1] < val_node_features.size(0))
                    val_valid_edge_index = val_edge_index[:, valid_mask]
                    val_used_edge_index = val_valid_edge_index if val_valid_edge_index.size(1) > 0 else torch.empty((2,0), dtype=torch.long, device=device)

                val_embeddings, val_pred_adj = model(val_node_features, val_used_edge_index)
                
                # Compute validation accuracy
                if val_pred_adj is not None:
                    val_adj_sigmoid = torch.sigmoid(val_pred_adj)
                    val_density = (val_adj_sigmoid > 0.1).float().mean().item()
                else:
                    # KNN Baseline case
                    val_density = 0.3
                
                # Compute PageRank ranking
                from ..graph_heads.page_rank import page_rank
                if val_pred_adj is not None:
                    val_ranks = page_rank(val_adj_sigmoid.cpu().numpy())
                else:
                    # KNN Baseline case, use identity matrix
                    val_ranks = page_rank(torch.eye(val_embeddings.size(0)).numpy())
                
                # Compute precision@1
                if val_ground_truth and val_ranks:
                    val_precision = 1.0 if val_ground_truth in val_ranks[:1] else 0.0
                else:
                    val_precision = 0.0
                
                # Update best validation metric
                if val_precision > best_val_metric:
                    best_val_metric = val_precision
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
            
            model.train()
        
        # Add minimum training epochs check
        if epoch + 1 < config.min_epochs:
            continue

        # Improvement: Intelligent early stopping and sharpening coordination mechanism
        early_stop_triggered = early_stopping(total_loss.item()) or patience_counter >= config.patience
        
        if early_stop_triggered:
            # Analyze early stopping reason and decide sharpening strategy
            convergence_state = _analyze_convergence_state(
                training_history, epoch, patience_counter, config.patience
            )
            
            if patience_counter >= config.patience:
                # Restore best model
                if 'best_model_state' in locals():
                    model.load_state_dict(best_model_state)
                
                # Decide whether to sharpen based on convergence state
                if convergence_state['needs_sharpening']:
                    _apply_adaptive_sharpening(model, convergence_state)
            else:
                # Loss plateau early stopping usually doesn't need sharpening, as model may be overfitting
                if convergence_state['needs_sharpening'] and convergence_state['is_healthy_convergence']:
                    _apply_adaptive_sharpening(model, convergence_state, intensity='light')
            break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Improvement 6: More detailed training information - add polarization loss tracking and density monitoring
            adj_density = pred_adj_sigmoid.mean().item()

    return model, training_history


def _analyze_convergence_state(training_history, epoch, patience_counter, max_patience):
    """
    Analyze convergence state, decide if sharpening is needed
    
    Args:
        training_history: Training history
        epoch: Current epoch
        patience_counter: Patience counter
        max_patience: Maximum patience value
        
    Returns:
        dict: Convergence state analysis results
    """
    # Analyze loss trend
    recent_losses = training_history['loss'][-10:] if len(training_history['loss']) >= 10 else training_history['loss']
    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0] if len(recent_losses) > 1 else 0
    
    # Analyze adjacency matrix statistics
    recent_adj_max = training_history['adj_max'][-5:] if len(training_history['adj_max']) >= 5 else training_history['adj_max']
    recent_adj_min = training_history['adj_min'][-5:] if len(training_history['adj_min']) >= 5 else training_history['adj_min']
    
    # Compute value range gap
    avg_max = np.mean(recent_adj_max) if recent_adj_max else 0.5
    avg_min = np.mean(recent_adj_min) if recent_adj_min else 0.0
    score_gap = avg_max - avg_min
    
    # Judge convergence type
    is_healthy_convergence = (
        loss_trend < 0.001 and  # Loss is basically stable
        patience_counter >= max_patience * 0.7 and  # Patience value is high
        score_gap > 0.1  # Has some discriminability
    )
    
    # Judge if sharpening is needed
    needs_sharpening = (
        score_gap < 0.15 and  # Gap is insufficient
        avg_max < 0.8 and  # Maximum value is not high enough
        avg_min > 0.2  # Minimum value is not low enough
    )
    
    return {
        'loss_trend': loss_trend,
        'score_gap': score_gap,
        'avg_max': avg_max,
        'avg_min': avg_min,
        'is_healthy_convergence': is_healthy_convergence,
        'needs_sharpening': needs_sharpening,
        'convergence_type': 'healthy' if is_healthy_convergence else 'plateau'
    }


def _apply_adaptive_sharpening(model, convergence_state, intensity='adaptive'):
    """
    Apply adaptive sharpening strategy
    
    Args:
        model: Trained model
        convergence_state: Convergence state analysis results
        intensity: Sharpening intensity ('light', 'adaptive', 'strong')
    """
    # Decide sharpening parameters based on convergence state and intensity
    score_gap = convergence_state['score_gap']
    
    if intensity == 'light':
        gamma = 1.5
        min_gap = 0.08
    elif intensity == 'strong':
        gamma = 3.0
        min_gap = 0.12
    else:  # adaptive
        # Dynamically adjust based on gap
        if score_gap < 0.05:
            gamma = 2.5
            min_gap = 0.10
        elif score_gap < 0.10:
            gamma = 2.0
            min_gap = 0.08
        else:
            gamma = 1.8
            min_gap = 0.06
    
    
    # Set sharpening parameters on model (if model supports it)
    if hasattr(model, 'set_sharpening_params'):
        model.set_sharpening_params(gamma=gamma, min_gap=min_gap)
    
    # If model has sharpening method, apply directly
    if hasattr(model, 'apply_sharpening'):
        model.apply_sharpening(gamma=gamma, min_gap=min_gap)


class AdvancedGNNKANTrainer:
    """Advanced GNN-KAN trainer - includes more training strategies"""
    
    def __init__(self, config):
        self.config = config
        
    def train_with_curriculum(self, model, node_features, edge_index):
        """
        Curriculum learning training strategy
        
        Args:
            model: GNN-KAN model
            node_features: Node features
            edge_index: Edge indices
            
        Returns:
            model: Trained model
            training_history: Training history
        """
        
        device = next(model.parameters()).device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        
        training_history = {
            'losses': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Phase 1: Simple task (few nodes)
        subset_size = min(10, node_features.size(0))
        subset_features = node_features[:subset_size]
        subset_edge_index = self._filter_edge_index(edge_index, subset_size)
        
        model = self._train_phase(
            model, subset_features, subset_edge_index, 
            epochs=self.config.num_epochs // 3,
            phase_name="Phase 1"
        )
        
        # Phase 2: Medium complexity
        if node_features.size(0) > subset_size:
            medium_size = min(20, node_features.size(0))
            medium_features = node_features[:medium_size]
            medium_edge_index = self._filter_edge_index(edge_index, medium_size)
            
            model = self._train_phase(
                model, medium_features, medium_edge_index,
                epochs=self.config.num_epochs // 3,
                phase_name="Phase 2"
            )
        
        # Phase 3: Full training
        model = self._train_phase(
            model, node_features, edge_index,
            epochs=self.config.num_epochs // 3,
            phase_name="Phase 3"
        )
        
        return model, training_history
    
    def _filter_edge_index(self, edge_index, max_nodes):
        """Filter edge indices, only keep edges involving first max_nodes nodes"""
        mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
        return edge_index[:, mask]
    
    def _train_phase(self, model, features, edge_index, epochs, phase_name):
        """Train one phase"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        criterion = GNNKANLoss(self.config)
        
        # Create target adjacency matrix - fix edge index out-of-bounds issue
        num_nodes = features.size(0)
        device = features.device
        target_adj = torch.zeros(num_nodes, num_nodes, device=device)

        used_edge_index = edge_index  # Default
        if edge_index.size(1) > 0:
            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            valid_edge_index = edge_index[:, valid_edges_mask]

            if valid_edge_index.size(1) > 0:
                src = valid_edge_index[0]; dst = valid_edge_index[1]
                ones = torch.ones_like(src, dtype=target_adj.dtype)
                target_adj.index_put_((src, dst), ones, accumulate=True)
                target_adj.index_put_((dst, src), ones, accumulate=True)
                used_edge_index = valid_edge_index  # Key: use filtered edges afterwards
            else:
                target_adj = torch.eye(num_nodes, device=device) * 0.1
                used_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            try:
                node_embeddings, pred_adj = model(features, used_edge_index)
                loss = criterion(pred_adj, target_adj, node_embeddings)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    optimizer.step()
                    
            except Exception as e:
                break
        
        return model


def create_adaptive_targets(node_features, edge_index, method='similarity'):
    """
    創建自適應訓練目標
    
    Args:
        node_features: 節點特徵
        edge_index: 邊索引
        method: 目標創建方法
        
    Returns:
        target_adj: 目標鄰接矩陣
    """
    num_nodes = node_features.size(0)
    device = node_features.device
    
    if method == 'similarity':
        # Create target based on feature similarity
        features_np = node_features.detach().cpu().numpy()
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity = cosine_similarity(features_np)
        # Thresholding
        threshold = np.percentile(similarity, 80)
        target_adj = torch.tensor(
            (similarity > threshold).astype(float),
            device=device,
            dtype=torch.float
        )
    
    elif method == 'knn':
        # Create target based on k-nearest neighbors
        from sklearn.neighbors import kneighbors_graph
        features_np = node_features.detach().cpu().numpy()
        
        k = min(5, num_nodes - 1)
        knn_graph = kneighbors_graph(
            features_np, n_neighbors=k, mode='connectivity'
        )
        target_adj = torch.tensor(
            knn_graph.toarray().astype(float),
            device=device,
            dtype=torch.float
        )
    
    else:
        # Default: based on edge indices
        target_adj = torch.zeros(num_nodes, num_nodes, device=device)
        if edge_index.size(1) > 0:
            src = edge_index[0]
            dst = edge_index[1]
            ones = torch.ones_like(src, dtype=target_adj.dtype)
            target_adj.index_put_((src, dst), ones, accumulate=True)
            target_adj.index_put_((dst, src), ones, accumulate=True)
    
    return target_adj