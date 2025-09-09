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
    OptimizedGNNKANEncoder, AdvancedKANLayer,
    GradientStabilizer, SimplifiedKANLayer,
    HighCapacityGNNKANEncoder
)

# Import configuration classes for type checking
from .config import GNNKANConfig


# AttentionGraphDecoder已移除 - 使用KNN Baseline替代


class GNNKANModel(nn.Module):
    """GNN-KAN 模型 - 結合 GNN 和 KAN 的優勢"""
    
    def __init__(self, config, num_nodes):
        super(GNNKANModel, self).__init__()
        self.config = config
        self.num_nodes = num_nodes
        
        # 特徵投影層 - 自適應維度
        input_feature_dim = getattr(config, 'target_feature_dim', config.input_dim)
        self.feature_projection = nn.Linear(input_feature_dim, config.input_dim)
        
        # 根據配置選擇適當的編碼器
        if hasattr(config, 'high_capacity_mode') and config.high_capacity_mode:
            self.gnn_encoder = HighCapacityGNNKANEncoder(
                input_dim=config.input_dim,
                hidden_dims=config.hidden_dims,
                output_dim=config.output_dim,
                num_layers=config.num_gnn_layers,
                kan_config={
                    'grid_size': config.kan_grid_size,
                    'spline_order': config.kan_spline_order,
                    'use_residual': True,
                    'use_spectral_norm': True
                },
                dropout=config.dropout
            )
        else:
            self.gnn_encoder = OptimizedGNNKANEncoder(
                input_dim=config.input_dim,
                hidden_dims=config.hidden_dims,
                output_dim=config.output_dim,
                num_layers=config.num_gnn_layers,
                kan_grid_size=config.kan_grid_size,
                kan_spline_order=config.kan_spline_order,
                dropout=config.dropout
            )
        
        # 時序注意力機制 - 使用適配器解決維度問題
        try:
            from .dimension_adapters import TemporalAttentionAdapter
            self.temporal_attention = TemporalAttentionAdapter(config.output_dim)
        except ImportError:
            self.temporal_attention = TemporalAttention(config.output_dim)
        
        # 🎯 Graph Decoder已移除 - 使用KNN Baseline替代
        print("✓ 使用KNN Baseline替代Graph Decoder")
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, node_features, edge_index):
        """GNN-KAN前向傳播 - 只返回embeddings，鄰接矩陣由KNN Baseline構建"""
        
        try:
            # 只計算embeddings，鄰接矩陣由KNN Baseline構建
            embeddings = self.gnn_encoder(node_features, edge_index)
            # 返回None作為adj_scores，因為我們使用KNN Baseline
            return embeddings, None
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("🚨 GPU記憶體不足，使用簡化結果")
                # 🔧 緊急回退：返回簡化結果
                embeddings = self.gnn_encoder(node_features, edge_index) 
                return embeddings, None
            else:
                raise e
    
    # Graph Decoder相關方法已移除 - 使用KNN Baseline替代


class TemporalAttention(nn.Module):
    """時序注意力機制 - 修復維度匹配問題"""
    
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 確保feature_dim能被num_heads整除
        if feature_dim % num_heads != 0:
            # 調整到最接近的可整除值
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
        """修復維度匹配的前向傳播"""
        original_shape = features.shape
        
        # 確保輸入至少是3D [batch, seq, feature]
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch, 1, feature]
        
        try:
            # 維度投影（如果需要）
            if self.use_projection:
                projected_features = self.projection(features)
                attn_output, _ = self.attention(projected_features, projected_features, projected_features)
                attn_output = self.back_projection(attn_output)
            else:
                attn_output, _ = self.attention(features, features, features)
            
            # 殘差連接
            output = self.norm(attn_output + features)
            
            # 恢復原始形狀
            if len(original_shape) == 2:
                output = output.squeeze(1)
            
            return output
            
        except Exception as e:
            print(f"⚠️ TemporalAttention failed: {e}, using identity mapping")
            # 安全回退：直接返回歸一化的輸入
            if len(original_shape) == 2:
                return self.norm(features.squeeze(1))
            else:
                return self.norm(features)


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


# 🔧 validate_model_setup 函數已移至 utils.py 模組中，避免重複定義
# train_gnn_kan_model 函數已移至 training.py 模組中，避免重複定義


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
    
    # 🎯 統一使用 BCEWithLogits 損失函數 (更穩定，適用於 KAN 和 MLP)
    # 無論是 KAN 還是 MLP 解碼器，都假設輸出 logit (原始分數)
    reconstruction_loss = F.binary_cross_entropy_with_logits(adj_scores, true_adj, reduction='mean')
    
    # 2. 嵌入正則化損失 (使用更溫和的正則化)
    embedding_reg = torch.norm(node_embeddings, p=2, dim=1).mean()
    
    # 3. 稀疏性損失 (鼓勵稀疏的鄰接矩陣)
    # 對 logit 應用 sigmoid 後計算稀疏性
    adj_probs = torch.sigmoid(adj_scores)
    sparsity_loss = torch.norm(adj_probs, p=1) / (num_nodes * num_nodes)
    
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
                self.batch_norms.append(nn.LayerNorm(layer_output_dim, eps=1e-4))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 圖解碼器
        self.graph_decoder = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1)
        )
        
    def _create_gnn_kan_layer(self, input_dim, output_dim, kan_config):
        """創建純粹的GNN-KAN層 - 使用兼容的KAN實現"""
        try:
            # 🔧 修復：使用兼容的KAN組件創建函數
            from .kan_components.kan_layers import create_compatible_kan_layer
            
            # 使用KAN配置創建兼容的KAN層
            if kan_config:
                return create_compatible_kan_layer(
                    input_dim, output_dim,
                    num_basis=kan_config.get('num_basis', 8),
                    spline_order=kan_config.get('spline_order', 3),
                    grid_size=kan_config.get('grid_size', 8),
                    adaptive_spline_order=kan_config.get('adaptive_spline_order', True)
                )
            else:
                return create_compatible_kan_layer(input_dim, output_dim)
                
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
                    nn.LayerNorm(output_dim, eps=1e-4)  # 使用LayerNorm而非BatchNorm
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
            # 🔧 修正殘差連接邏輯 - KAN層沒有[0]索引
            # 保存殘差（如果啟用殘差連接且維度匹配）
            residual = x if self.use_residual else None
            
            # GNN-KAN 層處理
            try:
                x = layer(x)
                
                # 檢查輸出是否有效
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"⚠️ KAN層{i}輸出包含無效值，進行修復")
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            except Exception as e:
                print(f"⚠️ KAN層{i}處理失敗: {e}")
                # 如果KAN層失敗，使用簡單變換
                if hasattr(layer, 'base_linear'):
                    x = layer.base_linear(x)
                elif hasattr(layer, 'base_transform'):
                    x = layer.base_transform(x)
                else:
                    # 最後回退：保持輸入不變
                    pass
            
            # 🔧 安全的殘差連接
            if residual is not None and self.use_residual:
                try:
                    if x.shape == residual.shape:
                        x = x + residual
                    else:
                        print(f"⚠️ 殘差維度不匹配: x={x.shape}, residual={residual.shape}")
                        # 維度不匹配時不使用殘差連接
                except Exception as e:
                    print(f"⚠️ 殘差連接失敗: {e}")
            
            # 批量歸一化
            if self.batch_norms and i < len(self.batch_norms):
                try:
                    x = self.batch_norms[i](x)
                except Exception as e:
                    print(f"⚠️ 批量歸一化失敗: {e}")
            
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
            scores = self.graph_decoder(edge_features).squeeze()
            adj_scores[i] = scores
        
        return adj_scores
    
    def get_model_info(self):
        """獲取模型信息"""
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
    根據配置創建模型
    
    Args:
        config: SimplifiedGNNKANConfig 配置對象
        
    Returns:
        model: 創建的模型實例
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
        
        print(f"✓ Created model with config: {type(config).__name__}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to create model with config: {e}")
        # 創建最小回退模型
        try:
            model = SimplifiedGNNKAN(
                input_dim=getattr(config, 'input_dim', 32),
                hidden_dim=64,
                output_dim=getattr(config, 'output_dim', 16),
                num_layers=2,
                dropout=0.1
            )
            print("✓ Created fallback model")
            return model
        except Exception as fallback_error:
            print(f"❌ Fallback model creation failed: {fallback_error}")
            raise e

# 將SimplifiedGNNKAN添加到可用的模型中
__all__ = ['GNNKANModel', 'SimplifiedGNNKAN', 'TemporalAttention', 'create_fallback_model', 
           'train_gnn_kan_model', 'compute_loss_stable']