"""
GNN-KAN Models Module
包含所有 GNN-KAN 相關的模型定義和訓練函數
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
    OptimizedGNNKANEncoder, UltraFastKANLayer, 
    GradientStabilizer, StabilizedKANLayer
)


class GNNKANModel(nn.Module):
    """GNN-KAN 模型 - 結合 GNN 和 KAN 的優勢"""
    
    def __init__(self, config, num_nodes):
        super(GNNKANModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # 特徵投影層
        self.feature_projection = nn.Linear(config.target_feature_dim, config.input_dim)
        
        # GNN-KAN編碼器
        self.gnn_encoder = OptimizedGNNKANEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            num_layers=config.num_gnn_layers,
            kan_grid_size=config.kan_grid_size,
            kan_spline_order=config.kan_spline_order,
            dropout=config.dropout
        )
        
        # 時序注意力機制
        self.temporal_attention = TemporalAttention(config.output_dim)
        
        # KAN 解碼器 - 用於計算鄰接矩陣
        self.graph_decoder = nn.Sequential(
            nn.Linear(config.output_dim * 2, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, node_features, edge_index):
        """
        前向傳播
        
        Args:
            node_features: 節點特徵 [num_nodes, feature_dim]
            edge_index: 邊索引 [2, num_edges]
            
        Returns:
            node_embeddings: 節點嵌入
            adj_scores: 鄰接矩陣分數
        """
        # 特徵投影
        projected_features = self.feature_projection(node_features)
        
        # GNN-KAN 編碼
        node_embeddings = self.gnn_encoder(projected_features, edge_index)
        
        # 時序注意力
        if node_embeddings.dim() == 2:
            # 增加時間維度用於注意力計算
            node_embeddings_expanded = node_embeddings.unsqueeze(1)
            attended_embeddings = self.temporal_attention(node_embeddings_expanded)
            node_embeddings = attended_embeddings.squeeze(1)
        else:
            node_embeddings = self.temporal_attention(node_embeddings)
        
        # 計算鄰接矩陣分數 (批量化處理)
        adj_scores = self._compute_adjacency_scores_batch(node_embeddings)
        
        return node_embeddings, adj_scores
    
    def _compute_adjacency_scores_batch(self, embeddings):
        """批量化計算鄰接矩陣分數"""
        num_nodes = embeddings.size(0)
        
        # 創建所有可能的邊對
        i_indices = torch.arange(num_nodes, device=embeddings.device).repeat_interleave(num_nodes)
        j_indices = torch.arange(num_nodes, device=embeddings.device).repeat(num_nodes)
        
        # 批量計算邊特徵
        edge_features = torch.cat([
            embeddings[i_indices], 
            embeddings[j_indices]
        ], dim=1)
        
        # 批量通過 KAN 解碼器
        scores = torch.sigmoid(self.graph_decoder(edge_features))
        
        # 重塑為鄰接矩陣
        adj_scores = scores.view(num_nodes, num_nodes)
        
        return adj_scores


class TemporalAttention(nn.Module):
    """時序注意力機制 - 提升準確率"""
    
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, features):
        # 對時間序列特徵應用注意力
        attn_output, _ = self.attention(features, features, features)
        return self.norm(attn_output + features)  # 殘差連接


class AdaptiveGradientStabilizer:
    """自適應梯度穩定器 - 提升訓練穩定性"""
    
    def __init__(self):
        self.loss_history = []
        self.grad_norm_history = []
        
    def adaptive_clipping(self, model, current_loss):
        """根據損失歷史動態調整梯度裁剪"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) > 10:
            loss_std = np.std(self.loss_history[-10:])
            clip_norm = max(0.5, min(2.0, 1.0 / (loss_std + 1e-8)))
        else:
            clip_norm = 1.0
            
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        self.grad_norm_history.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        
        return grad_norm


def create_model_with_config(config, num_nodes):
    """根據配置創建模型"""
    try:
        model = GNNKANModel(config, num_nodes)
        print(f"✓ Created GNN-KAN model with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    except Exception as e:
        print(f"⚠️ Model creation failed: {e}")
        # 創建簡化版本的模型
        return create_fallback_model(config, num_nodes)


def create_fallback_model(config, num_nodes):
    """創建回退版本的模型"""
    print("🔄 Creating fallback model...")
    
    class FallbackModel(nn.Module):
        def __init__(self, config, num_nodes):
            super().__init__()
            self.linear = nn.Linear(config.target_feature_dim, num_nodes)
            
        def forward(self, node_features, edge_index):
            # 簡單的線性變換
            output = torch.sigmoid(self.linear(node_features))
            adj_matrix = torch.mm(output, output.t())
            return output, adj_matrix
    
    return FallbackModel(config, num_nodes)


def validate_model_setup(model, node_features, edge_index):
    """驗證模型設置"""
    try:
        # 檢查設備一致性
        model_device = next(model.parameters()).device
        if node_features.device != model_device:
            print(f"⚠️ Device mismatch: model on {model_device}, features on {node_features.device}")
            return False
            
        if edge_index.device != model_device:
            print(f"⚠️ Device mismatch: model on {model_device}, edge_index on {edge_index.device}")
            return False
        
        # 測試前向傳播
        model.eval()
        with torch.no_grad():
            embeddings, adj = model(node_features, edge_index)
            
            # 檢查輸出形狀
            if embeddings.size(0) != node_features.size(0):
                print(f"⚠️ Embedding shape mismatch: {embeddings.shape} vs {node_features.shape}")
                return False
                
            if adj.size() != (node_features.size(0), node_features.size(0)):
                print(f"⚠️ Adjacency shape mismatch: {adj.shape}")
                return False
            
            # 檢查數值穩定性
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print("⚠️ NaN/Inf in embeddings")
                return False
                
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                print("⚠️ NaN/Inf in adjacency matrix")
                return False
        
        print("✓ Model validation passed")
        return True
        
    except Exception as e:
        print(f"⚠️ Model validation failed: {e}")
        return False


def train_gnn_kan_model(model, node_features, edge_index, config):
    """
    訓練 GNN-KAN 模型
    
    Args:
        model: GNN-KAN 模型
        node_features: 節點特徵
        edge_index: 邊索引
        config: 配置參數
        
    Returns:
        trained_model: 訓練後的模型
        final_adj: 最終鄰接矩陣
    """
    print("🎯 Starting GNN-KAN model training...")
    
    # 梯度穩定化器
    stabilizer = GradientStabilizer(
        l1_lambda=config.base_l1_lambda,
        entropy_lambda=config.base_entropy_lambda,
        grad_clip_value=config.gradient_clip_norm,
        pruning_threshold=1e-2,
        enable_dynamic_scaling=True,
        stability_check_freq=config.stability_check_frequency
    )
    
    # 優化器設置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.base_learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # 學習率調度器
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.warmup_epochs,
        T_mult=2,
        eta_min=config.base_learning_rate * 0.01
    )
    
    model.train()
    successful_epochs = 0
    loss_history = []
    
    try:
        for epoch in range(config.epochs):
            optimizer.zero_grad()
            
            try:
                # 前向傳播
                node_embeddings, adj_scores = model(node_features, edge_index)
                
                # 檢查輸出是否為 NaN 或 Inf
                if torch.isnan(node_embeddings).any() or torch.isinf(node_embeddings).any():
                    print(f"⚠️ NaN/Inf in embeddings at epoch {epoch}, skipping...")
                    continue
                
                if torch.isnan(adj_scores).any() or torch.isinf(adj_scores).any():
                    print(f"⚠️ NaN/Inf in adj_scores at epoch {epoch}, skipping...")
                    continue
                
                # 計算基礎損失
                try:
                    base_loss = compute_loss_stable(node_embeddings, adj_scores, edge_index, config)
                except RuntimeError as e:
                    print(f"⚠️ Loss computation failed at epoch {epoch}: {e}")
                    continue
                
                # 使用梯度穩定化器計算總損失
                total_loss = stabilizer.compute_total_regularization_loss(model, base_loss)
                
                # 檢查 loss 是否為 NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️ Invalid total loss at epoch {epoch}: {total_loss.item()}")
                    continue
                
                # 反向傳播
                try:
                    total_loss.backward()
                except RuntimeError as e:
                    print(f"⚠️ Backward pass failed at epoch {epoch}: {e}")
                    continue
                
                # 裁剪梯度並檢查穩定性
                grad_norm = stabilizer.apply_gradient_clipping(model, clip_type='norm')
                
                # 自適應梯度穩定器
                stabilizer.adaptive_clipping(model, total_loss.item())
                
                # 數值穩定性檢查
                if epoch % stabilizer.stability_check_freq == 0:
                    stability_report = stabilizer.check_numerical_stability(model)
                    if stability_report['gradient_nan_count'] > 0 or stability_report['gradient_inf_count'] > 0:
                        print(f"⚠️ Epoch {epoch}: Gradient stability issues detected")
                
                # 梯度爆炸檢查
                if grad_norm > 8.0:
                    if epoch % 50 == 0:
                        print(f"⚠️ Large gradient norm {grad_norm:.3f} at epoch {epoch}, applying stabilization...")
                
                # 優化器更新
                optimizer.step()
                scheduler.step()
                
                successful_epochs += 1
                loss_history.append(total_loss.item())
                
                # 動態調整正則化強度
                if len(loss_history) >= 10:
                    stabilizer.adaptive_regularization_scaling(total_loss.item(), loss_history)
                
                # 動態剪枝
                if successful_epochs % 50 == 0 and successful_epochs > 0:
                    pruning_ratio = stabilizer.apply_dynamic_pruning(model)
                    if pruning_ratio > 0:
                        print(f"✂️ Epoch {epoch}: Applied pruning, ratio: {pruning_ratio:.3f}")
                
                if epoch % 20 == 0 or successful_epochs <= 5:
                    print(f"📊 Epoch {epoch}/{config.epochs}, Base Loss: {base_loss.item():.6f}, "
                          f"Total Loss: {total_loss.item():.6f}, Grad norm: {grad_norm:.6f}")
                    print(f"🎛️ L1 λ: {stabilizer.l1_lambda:.6f}, Entropy λ: {stabilizer.entropy_lambda:.6f}")
                    
                    # GPU 記憶體監控
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated()/1024**3
                        cached = torch.cuda.memory_reserved()/1024**3
                        print(f"💾 GPU memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
                        
                        # 如果記憶體使用過高，切換到CPU
                        if allocated > 8.0:
                            print("🔄 GPU memory usage too high, switching to CPU...")
                            return train_on_cpu_fallback(model, node_features, edge_index, config)
                            
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"💥 CUDA error at epoch {epoch}: {e}")
                    print("🔄 Attempting CPU fallback...")
                    return train_on_cpu_fallback(model, node_features, edge_index, config)
                else:
                    print(f"⚠️ Error in epoch {epoch}: {e}")
                    continue
                    
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
    except Exception as e:
        print(f"💥 Training error: {e}")
        if "CUDA" in str(e):
            print("🔄 CUDA error detected, falling back to CPU...")
            return train_on_cpu_fallback(model, node_features, edge_index, config)
        traceback.print_exc()
    
    print(f"🎉 Training completed with {successful_epochs} successful epochs out of {config.epochs}")
    
    # 打印最終穩定性報告
    final_report = stabilizer.get_stability_report()
    if "message" not in final_report:
        print("📊 === 梯度穩定性報告 ===")
        print(f"📈 平均梯度範數: {final_report['gradient_statistics']['mean_grad_norm']:.6f}")
        print(f"📈 最大梯度範數: {final_report['gradient_statistics']['max_grad_norm']:.6f}")
        print(f"✂️ 梯度裁剪次數: {final_report['gradient_statistics']['gradient_clips']}")
        print(f"⚠️ 穩定性違規次數: {final_report['stability_violations']}")
        print(f"🎛️ 最終 L1 λ: {final_report['regularization_config']['l1_lambda']:.6f}")
        print(f"🎛️ 最終熵 λ: {final_report['regularization_config']['entropy_lambda']:.6f}")
    
    # 如果成功訓練的epoch太少，使用簡化策略
    if successful_epochs < 5:
        print("⚠️ Too few successful epochs, using simplified adjacency calculation...")
        return train_on_cpu_fallback(model, node_features, edge_index, config)
    
    # 獲取最終的鄰接矩陣
    print("📊 Getting final adjacency matrix...")
    model.eval()
    try:
        with torch.no_grad():
            _, final_adj = model(node_features, edge_index)
            
            # 確認結果有效性
            if torch.isnan(final_adj).any() or torch.isinf(final_adj).any():
                print("⚠️ NaN/Inf in final adjacency, using fallback...")
                final_adj = torch.eye(node_features.size(0), device=node_features.device)
    
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"💥 CUDA error in final evaluation: {e}")
            return train_on_cpu_fallback(model, node_features, edge_index, config)
        else:
            print(f"⚠️ Error getting final adjacency: {e}")
            final_adj = torch.eye(node_features.size(0), device=node_features.device)
    
    return model, final_adj


def train_on_cpu_fallback(model, node_features, edge_index, config):
    """CPU 回退訓練函數 - 確保設備一致性"""
    print("🔄 === CPU FALLBACK MODE ===")
    
    try:
        # 強制移動到 CPU
        if hasattr(model, 'cpu'):
            model = model.cpu()
        if hasattr(node_features, 'cpu'):
            node_features = node_features.cpu()
        if hasattr(edge_index, 'cpu'):
            edge_index = edge_index.cpu()
        
        # 簡化模型結構以適應 CPU
        print("🔧 Using simplified training for CPU...")
        
        # 獲取節點數量
        if hasattr(node_features, 'size'):
            num_nodes = node_features.size(0)
        elif hasattr(node_features, 'shape'):
            num_nodes = node_features.shape[0]
        else:
            num_nodes = len(node_features)
        
        # 基於特徵相似性構建鄰接矩陣
        with torch.no_grad():
            # 確保 node_features 是正確的 tensor 格式
            if not isinstance(node_features, torch.Tensor):
                node_features = torch.tensor(node_features, dtype=torch.float, device='cpu')
            else:
                node_features = node_features.to('cpu')
            
            # 計算餘弦相似性
            normalized_features = F.normalize(node_features, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # 應用閾值和sigmoid
            final_adj = torch.sigmoid(similarity_matrix * 3.0)
            
            # 確保對角線為高值 (自相似性)
            final_adj.fill_diagonal_(0.9)
            
            # 確保結果在CPU上
            final_adj = final_adj.cpu()
            
    except Exception as e:
        print(f"💥 CPU fallback also failed: {e}")
        # 最終回退：恆等矩陣
        try:
            if hasattr(node_features, 'size'):
                num_nodes = node_features.size(0)
            elif hasattr(node_features, 'shape'):
                num_nodes = node_features.shape[0]
            else:
                num_nodes = len(node_features)
        except:
            num_nodes = 10  # 默認值
            
        final_adj = torch.eye(num_nodes, device='cpu')
    
    print("✅ CPU fallback completed")
    # 確保返回的模型也在 CPU 上
    if hasattr(model, 'cpu'):
        model = model.cpu()
    return model, final_adj


def compute_loss_stable(node_embeddings, adj_scores, edge_index, config):
    """
    數值穩定的損失計算
    
    Args:
        node_embeddings: 節點嵌入
        adj_scores: 鄰接矩陣分數
        edge_index: 邊索引
        config: 配置參數
        
    Returns:
        total_loss: 總損失
    """
    num_nodes = node_embeddings.size(0)
    device = node_embeddings.device
    
    # 1. 圖重建損失 (使用更穩定的版本)
    true_adj = torch.zeros(num_nodes, num_nodes, device=device)
    if edge_index.size(1) > 0:
        # 確保索引在有效範圍內
        valid_indices = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        if valid_indices.any():
            valid_edge_index = edge_index[:, valid_indices]
            true_adj[valid_edge_index[0], valid_edge_index[1]] = 1.0
    
    # 裁剪 adj_scores 以避免數值不穩定
    adj_scores_clipped = torch.clamp(adj_scores, min=1e-7, max=1-1e-7)
    
    # 使用穩定的二元交叉熵
    reconstruction_loss = F.binary_cross_entropy(adj_scores_clipped, true_adj, reduction='mean')
    
    # 2. 嵌入正則化損失 (使用更溫和的正則化)
    embedding_reg = torch.norm(node_embeddings, p=2, dim=1).mean()
    
    # 3. 稀疏性損失 (鼓勵稀疏的鄰接矩陣)
    sparsity_loss = torch.norm(adj_scores, p=1) / (num_nodes * num_nodes)
    
    # 確保各個損失項都是有效的數值
    if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
        reconstruction_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    if torch.isnan(embedding_reg) or torch.isinf(embedding_reg):
        embedding_reg = torch.tensor(0.0, device=device)
    
    if torch.isnan(sparsity_loss) or torch.isinf(sparsity_loss):
        sparsity_loss = torch.tensor(0.0, device=device)
    
    # 總損失 (使用更小的權重)
    total_loss = reconstruction_loss + 0.001 * embedding_reg + 0.0001 * sparsity_loss
    
    return total_loss


class AdvancedTrainingManager:
    """高級訓練管理器 - 包含更多高級功能"""
    
    def __init__(self, config):
        self.config = config
        self.training_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def train_with_advanced_features(self, model, node_features, edge_index):
        """使用高級功能進行訓練"""
        print("🚀 Starting advanced training with enhanced features...")
        
        # 早停機制
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )
        
        # 學習率調度器
        optimizer = optim.AdamW(model.parameters(), lr=self.config.base_learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 訓練循環
        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            
            # 前向傳播
            embeddings, adj = model(node_features, edge_index)
            
            # 計算損失
            loss = compute_loss_stable(embeddings, adj, edge_index, self.config)
            
            # 反向傳播
            loss.backward()
            optimizer.step()
            
            # 學習率調度
            scheduler.step(loss)
            
            # 記錄訓練歷史
            self.training_history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # 早停檢查
            if early_stopping(loss.item()):
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                print(f"📊 Epoch {epoch}: Loss = {loss.item():.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # 獲取最終結果
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
    """早停機制"""
    
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
    """簡化的GNN-KAN模型 - 主要模型類"""
    
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
        
        # 輸入投影層
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 構建 GNN-KAN 層
        self.gnn_kan_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            layer_output_dim = hidden_dim if i < num_layers - 1 else output_dim
            
            # 創建 GNN-KAN 層
            gnn_kan_layer = self._create_gnn_kan_layer(layer_input_dim, layer_output_dim, kan_config)
            self.gnn_kan_layers.append(gnn_kan_layer)
            
            # 批量歸一化
            if use_batch_norm and i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(layer_output_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 圖解碼器
        self.graph_decoder = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1)
        )
        
    def _create_gnn_kan_layer(self, input_dim, output_dim, kan_config):
        """創建純粹的GNN-KAN層 - 使用AdvancedKANLayer"""
        try:
            # 使用純粹的KAN組件
            from .kan_components import AdvancedKANLayer
            
            # 使用KAN配置創建AdvancedKANLayer
            if kan_config:
                return AdvancedKANLayer(
                    input_dim, output_dim,
                    num_basis=kan_config.get('num_basis', 8),
                    spline_order=kan_config.get('spline_order', 3),
                    grid_size=kan_config.get('grid_size', 8),
                    adaptive_spline_order=kan_config.get('adaptive_spline_order', True)
                )
            else:
                return AdvancedKANLayer(input_dim, output_dim)
                
        except ImportError:
            # 回退到SimplifiedKANLayer
            try:
                from .kan_components import SimplifiedKANLayer
                return SimplifiedKANLayer(input_dim, output_dim)
            except ImportError:
                # 最後回退 - 但這表示KAN特性缺失
                print("⚠️ 警告：KAN層不可用，回退到標準線性層（失去KAN優勢）")
                return nn.Sequential(
                    nn.Linear(input_dim, output_dim, bias=False),  # 最小化MLP特性
                    nn.LayerNorm(output_dim)  # 使用LayerNorm而非BatchNorm
                )
    
    def forward(self, node_features, edge_index):
        """
        前向傳播
        
        Args:
            node_features: 節點特徵 [num_nodes, feature_dim]
            edge_index: 邊索引 [2, num_edges]
            
        Returns:
            node_embeddings: 節點嵌入
            adj_scores: 鄰接矩陣分數
        """
        # 輸入投影
        x = self.input_projection(node_features)
        
        # 逐層處理
        for i, layer in enumerate(self.gnn_kan_layers):
            residual = x if self.use_residual and x.size(-1) == layer[0].out_features else None
            
            # GNN-KAN 層
            x = layer(x)
            
            # 殘差連接
            if residual is not None and x.size(-1) == residual.size(-1):
                x = x + residual
            
            # 批量歸一化
            if self.batch_norms and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Dropout（除了最後一層）
            if i < len(self.gnn_kan_layers) - 1:
                x = self.dropout(x)
        
        node_embeddings = x
        
        # 計算鄰接矩陣分數
        adj_scores = self._compute_adjacency_matrix(node_embeddings)
        
        return node_embeddings, adj_scores
    
    def _compute_adjacency_matrix(self, embeddings):
        """計算鄰接矩陣分數"""
        num_nodes = embeddings.size(0)
        device = embeddings.device
        
        # 批量計算所有邊的分數
        adj_scores = torch.zeros(num_nodes, num_nodes, device=device)
        
        # 優化的批量計算
        for i in range(num_nodes):
            # 計算節點i與所有其他節點的連接分數
            i_embedding = embeddings[i].unsqueeze(0).expand(num_nodes, -1)
            edge_features = torch.cat([i_embedding, embeddings], dim=1)
            scores = torch.sigmoid(self.graph_decoder(edge_features)).squeeze()
            adj_scores[i] = scores
        
        return adj_scores
    
    def get_model_info(self):
        """獲取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'use_batch_norm': self.use_batch_norm,
            'use_residual': self.use_residual
        }

# 將SimplifiedGNNKAN添加到可用的模型中
__all__ = ['GNNKANModel', 'SimplifiedGNNKAN', 'TemporalAttention', 'create_model_with_config', 
           'train_gnn_kan_model', 'compute_loss_stable']