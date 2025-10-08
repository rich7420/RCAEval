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
    SimplifiedKANLayer
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
        
        # 使用優化的GNN-KAN編碼器
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
        
        # 時序注意力機制 - 使用適配器解決維度問題
        try:
            from .dimension_adapters import TemporalAttentionAdapter
            self.temporal_attention = TemporalAttentionAdapter(config.output_dim)
        except ImportError:
            self.temporal_attention = TemporalAttention(config.output_dim)
        
        # 🎯 恢復Graph Decoder - 使用多尺度KAN進行圖結構學習
        self.graph_decoder = MultiScaleGraphDecoder(
            embed_dim=config.output_dim,
            num_nodes=num_nodes,
            kan_grid_size=config.kan_grid_size,
            scales=3  # 3個尺度：局部、中層、全局
        )
        print("✓ 恢復KAN-based Graph Decoder")
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, node_features, edge_index, fault_type=None):
        """GNN-KAN前向傳播 - 使用KAN學習圖結構，優化版本"""
        
        try:
            # 🔧 優化：緩存中間結果，減少重複計算
            if not hasattr(self, '_cache') or self._cache is None:
                self._cache = {}
            
            # 檢查是否可以使用緩存
            cache_key = f"{node_features.shape}_{edge_index.shape}"
            if cache_key in self._cache and not self.training:
                return self._cache[cache_key]
            
            # GNN特徵提取
            embeddings = self.gnn_encoder(node_features, edge_index)
            
            # 使用KAN-based Graph Decoder學習圖結構
            adj_scores = self.graph_decoder(embeddings, fault_type)
            
            # 緩存結果（僅在推理時）
            if not self.training:
                self._cache[cache_key] = (embeddings, adj_scores)
            
            return embeddings, adj_scores
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("🚨 GPU記憶體不足，使用簡化結果")
                # 🔧 緊急回退：返回簡化結果
                embeddings = self.gnn_encoder(node_features, edge_index) 
                return embeddings, None
            else:
                raise e


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
            loss.backward(retain_graph=False)
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
                    adaptive_spline_order=kan_config.get('adaptive_spline_order', True),
                    basis_function=kan_config.get('basis_function', 'chebyshev'),
                    basis_kwargs=kan_config.get('basis_kwargs', {})
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


class FaultAwareGraphDecoder(nn.Module):
    """
    故障感知圖解碼器 - 使用KAN學習故障相關的圖結構
    
    原理：
    1. 使用KAN層進行節點對交互學習
    2. 故障類型感知的權重調整
    3. 端到端學習圖結構而非使用KNN
    """
    
    def __init__(self, embed_dim, num_nodes, kan_grid_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        
        # KAN-based attention for graph construction
        self.kan_attention = AdvancedKANLayer(embed_dim, embed_dim // 2)
        self.fault_aware_projection = nn.Linear(embed_dim, embed_dim)
        
        # 故障類型編碼器
        self.fault_encoder = nn.Embedding(6, embed_dim // 4)  # 6種故障類型
        
        # KAN-based圖結構輸出
        self.output_kan = AdvancedKANLayer(embed_dim + embed_dim // 4, 1)
        
        # 故障類型映射
        self.fault_type_map = {
            'cpu': 0, 'mem': 1, 'disk': 2, 
            'socket': 3, 'delay': 4, 'loss': 5
        }
        
    def forward(self, embeddings, fault_type=None):
        """
        前向傳播 - 學習故障相關的圖結構
        
        Args:
            embeddings: 節點嵌入 [num_nodes, embed_dim]
            fault_type: 故障類型字符串
            
        Returns:
            adj_matrix: 鄰接矩陣 [num_nodes, num_nodes]
        """
        num_nodes, embed_dim = embeddings.shape
        
        # 故障類型感知投影
        if fault_type is not None and fault_type in self.fault_type_map:
            fault_id = self.fault_type_map[fault_type]
            fault_context = self.fault_encoder(torch.tensor(fault_id, device=embeddings.device))
            fault_context = fault_context.expand(num_nodes, -1)
            # 融合故障上下文
            enhanced_embeddings = embeddings + self.fault_aware_projection(embeddings)
        else:
            enhanced_embeddings = embeddings
        
        # 計算節點對交互 - 使用KAN學習複雜關係
        expanded_emb1 = enhanced_embeddings.unsqueeze(1).expand(-1, num_nodes, -1)  # [N, N, D]
        expanded_emb2 = enhanced_embeddings.unsqueeze(0).expand(num_nodes, -1, -1)  # [N, N, D]
        
        # 節點對特徵：元素級乘積 + 拼接
        pairwise_features = expanded_emb1 * expanded_emb2  # [N, N, D]
        
        # 重塑為KAN期望的格式 [N*N, D]
        batch_size, seq_len, feature_dim = pairwise_features.shape
        pairwise_features_flat = pairwise_features.view(-1, feature_dim)  # [N*N, D]
        
        # 使用KAN計算注意力權重
        attention_weights_flat = self.kan_attention(pairwise_features_flat)  # [N*N, D//2]
        # 如果輸出是 [N*N, D//2]，則恢復為 [N, N, D//2]
        attention_feat_dim = attention_weights_flat.shape[-1]
        attention_weights = attention_weights_flat.view(batch_size, seq_len, attention_feat_dim)
        
        # 如果啟用故障感知，添加故障上下文
        if fault_type is not None and fault_type in self.fault_type_map:
            fault_context_expanded = fault_context.unsqueeze(1).expand(-1, num_nodes, -1)
            # 將注意力權重投影到與fault上下文相同的維度再拼接，避免維度衝突
            if attention_feat_dim != self.embed_dim // 2:
                # 安全投影到 embed_dim // 2 維
                proj = nn.Linear(attention_feat_dim, self.embed_dim // 2).to(attention_weights.device)
                attention_weights = proj(attention_weights)
                attention_feat_dim = attention_weights.shape[-1]
            combined_features = torch.cat([attention_weights, fault_context_expanded], dim=-1)
        else:
            combined_features = attention_weights
        
        # 輸出鄰接矩陣（AdvancedKANLayer 期望 2D 輸入，先展平再還原）
        expected_in = self.output_kan.input_dim if hasattr(self.output_kan, 'input_dim') else (self.embed_dim + self.embed_dim // 4)
        last_dim = combined_features.shape[-1]
        if last_dim != expected_in:
            proj_out = nn.Linear(last_dim, expected_in).to(combined_features.device)
            combined_features = proj_out(combined_features)
            last_dim = expected_in
        # 展平為 [N*N, last_dim]
        combined_features_flat = combined_features.view(-1, last_dim)
        adj_scores_flat = self.output_kan(combined_features_flat).squeeze(-1)  # [N*N]
        adj_scores = adj_scores_flat.view(num_nodes, num_nodes)  # [N, N]
        
        # 🎯 細微差異化機制 - 微調增強值域
        # 方案A: 微調非線性放大
        scale_factor = 4.0  # 微調放大係數
        adj_scores = torch.tanh(adj_scores * scale_factor)
        
        # 🎯 細微B: 微調競爭機制 - 適度增強噪聲和溫度
        noise = torch.randn_like(adj_scores) * 0.1  # 微調噪聲強度
        adj_scores_noisy = adj_scores + noise
        
        # 方案B: 局部競爭機制 - 每行進行softmax確保節點間競爭
        temperature = 0.15  # 微調溫度，適度增強差異
        competitive_scores = torch.softmax(adj_scores_noisy * temperature, dim=1)
        
        # 🎯 修復C: 故障感知差異化
        if fault_type is not None and fault_type in self.fault_type_map:
            # 根據故障類型調整競爭強度
            fault_intensity = {
                'cpu': 1.5, 'mem': 1.3, 'disk': 1.4, 
                'socket': 1.2, 'delay': 1.1, 'loss': 1.0
            }.get(fault_type, 1.0)
            competitive_scores = competitive_scores * fault_intensity
        
        # 🎯 修復D: 降低稀疏化程度
        adj_matrix = self.apply_dynamic_sparsification(competitive_scores, sparsity_level=0.3)
        
        # 移除自環（避免就地操作破壞梯度）
        eye = torch.eye(num_nodes, device=adj_matrix.device, dtype=adj_matrix.dtype)
        adj_matrix = adj_matrix * (1.0 - eye)
        
        # 確保對稱性（無向圖）
        adj_matrix = (adj_matrix + adj_matrix.T) / 2.0
        
        return adj_matrix
    
    def apply_dynamic_sparsification(self, adj_matrix, sparsity_level=0.3):
        """動態調整稀疏閾值，確保圖結構合理"""
        # 🎯 修復1: 降低稀疏化程度，保留更多連接
        # 方法1: 基於分位數（適應不同密度的圖）
        threshold = torch.quantile(adj_matrix, 1 - sparsity_level)
        
        # 方法2: 混合策略（推薦）
        mean_val = torch.mean(adj_matrix)
        std_val = torch.std(adj_matrix)
        adaptive_threshold = 0.5 * threshold + 0.5 * (mean_val + 0.2 * std_val)
        
        # 🎯 修復2: 確保至少保留一些連接
        min_threshold = torch.quantile(adj_matrix, 0.8)  # 至少保留20%的連接
        adaptive_threshold = torch.min(adaptive_threshold, min_threshold)
        
        # 應用稀疏化
        sparse_adj = torch.where(
            adj_matrix > adaptive_threshold, 
            adj_matrix, 
            torch.tensor(0.0, device=adj_matrix.device)
        )
        
        # 🎯 修復3: 確保每個節點至少有一個出邊
        row_sums = sparse_adj.sum(dim=1, keepdim=True)
        zero_rows = (row_sums == 0).squeeze(1)
        
        if zero_rows.any():
            # 對沒有出邊的節點，保留其最大的連接
            for i in range(adj_matrix.size(0)):
                if zero_rows[i]:
                    max_val, max_idx = torch.max(adj_matrix[i], dim=0)
                    if max_val > 0:
                        sparse_adj[i, max_idx] = max_val
        
        # 標準化
        row_sums = sparse_adj.sum(dim=1, keepdim=True)
        sparse_adj = torch.where(row_sums > 0, sparse_adj / (row_sums + 1e-8), sparse_adj)
        
        return sparse_adj


class MultiScaleGraphDecoder(nn.Module):
    """
    多尺度圖解碼器 - 使用KAN學習多種尺度的圖結構
    
    原理：
    1. 學習多種尺度的圖結構（局部、中層、全局）
    2. 動態權重融合不同尺度的圖
    3. 自適應稀疏化，無需預設故障類型
    """
    
    def __init__(self, embed_dim, num_nodes, kan_grid_size=8, scales=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.scales = scales
        self.current_epoch = 0  # 訓練進度追蹤
        
        # 多尺度KAN解碼器
        self.scale_decoders = nn.ModuleList([
            self._create_scale_decoder(embed_dim, kan_grid_size, scale=i) 
            for i in range(scales)
        ])
        
        # 🎯 優化2: 可學習溫度參數
        self.learnable_temps = nn.Parameter(torch.tensor([0.3, 0.6, 0.9]))  # 初始溫度
        
        # 動態權重學習器
        self.scale_weights = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, scales),
            nn.Softmax(dim=-1)
        )
        
        # 邊權重校準器
        self.edge_calibrator = nn.Sequential(
            nn.Linear(embed_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 🎯 優化3: 可學習融合權重
        self.sim_weights = nn.Parameter(torch.tensor([0.4, 0.25, 0.2, 0.15]))  # [original, euclidean, manhattan, variance]
    
    def _create_scale_decoder(self, embed_dim, kan_grid_size, scale):
        """創建單一尺度的解碼器 - 適應增強特徵維度"""
        # 增強特徵維度: 5*embed_dim (element_wise + concat + diff + cosine)
        rich_feature_dim = 5 * embed_dim
        decoder = nn.ModuleDict({
            'feature_compressor': nn.Linear(rich_feature_dim, embed_dim),  # 壓縮到原始維度
            'kan_attention': AdvancedKANLayer(embed_dim, embed_dim // 2),
            'output_kan': AdvancedKANLayer(embed_dim // 2, 1)
        })
        # 動態溫度參數（會在forward中更新）
        decoder.temperature = nn.Parameter(torch.tensor(0.5))  # 初始值，會被動態調整
        return decoder
    
    def forward(self, embeddings, fault_type=None):
        """
        前向傳播 - 學習多尺度圖結構
        
        Args:
            embeddings: 節點嵌入 [num_nodes, embed_dim]
            fault_type: 故障類型（保留接口兼容性，但不使用）
            
        Returns:
            adj_matrix: 融合的多尺度鄰接矩陣 [num_nodes, num_nodes]
        """
        num_nodes, embed_dim = embeddings.shape
        
        # 1. 學習多種尺度的圖結構
        scale_graphs = []
        for i, decoder in enumerate(self.scale_decoders):
            # 🎯 優化2: 動態溫度調整
            base_temp = self.learnable_temps[i]
            progress = min(1.0, self.current_epoch / 100.0)  # 訓練進度 [0,1]
            # 動態溫度: 初始值 + 進度偏移 + 小噪聲
            dynamic_temp = base_temp * (1 + 0.5 * progress) + 0.05 * torch.randn(1, device=embeddings.device).item()
            dynamic_temp = torch.clamp(torch.tensor(dynamic_temp), 0.1, 2.0)
            
            # 更新decoder溫度
            decoder.temperature.data = dynamic_temp
            
            # 使用動態溫度生成多樣化圖
            adj = self._generate_scale_graph(embeddings, decoder, i)
            scale_graphs.append(adj)
        
        # 更新epoch計數器
        self.current_epoch += 1
        
        # 2. 動態預測最佳尺度組合
        global_summary = torch.mean(embeddings, dim=0, keepdim=True)
        weights = self.scale_weights(global_summary).squeeze(0)
        
        # 3. 融合多尺度圖
        fused_graph = sum(w * g for w, g in zip(weights, scale_graphs))
        
        # 4. 邊權重校準
        calibrated_graph = self._calibrate_edge_weights(fused_graph, embeddings)
        
        # 5. 應用自適應稀疏化
        adj_matrix = self._apply_adaptive_sparsification(calibrated_graph)
        
        return adj_matrix
    
    def _generate_scale_graph(self, embeddings, decoder, scale_idx):
        """生成單一尺度的圖結構 - 增強節點對交互"""
        num_nodes, embed_dim = embeddings.shape
        
        # 🎯 優化1: 增強節點對交互計算
        expanded_emb1 = embeddings.unsqueeze(1).expand(-1, num_nodes, -1)
        expanded_emb2 = embeddings.unsqueeze(0).expand(num_nodes, -1, -1)
        
        # 多種交互方式組合
        element_wise = expanded_emb1 * expanded_emb2  # 元素級乘積
        concatenated = torch.cat([expanded_emb1, expanded_emb2], dim=-1)  # 拼接特徵 [N,N,2*D]
        difference = torch.abs(expanded_emb1 - expanded_emb2)  # 絕對差異
        cosine_sim = F.cosine_similarity(expanded_emb1, expanded_emb2, dim=-1).unsqueeze(-1)  # 餘弦相似度
        
        # 組合多種特徵 [N,N,4*D+1]
        pairwise_features = torch.cat([
            element_wise,  # [N,N,D]
            concatenated,  # [N,N,2*D] 
            difference,    # [N,N,D]
            cosine_sim.expand(-1, -1, embed_dim)  # [N,N,D]
        ], dim=-1)
        
        # 重塑為KAN期望的格式
        rich_feature_dim = pairwise_features.shape[-1]  # 4*D+D = 5*D
        pairwise_features_flat = pairwise_features.view(-1, rich_feature_dim)
        
        # 特徵壓縮到原始維度
        compressed_features = decoder['feature_compressor'](pairwise_features_flat)
        
        # 使用KAN計算注意力權重
        attention_weights_flat = decoder['kan_attention'](compressed_features)
        attention_weights = attention_weights_flat.view(num_nodes, num_nodes, -1)
        
        # 輸出鄰接矩陣
        adj_scores_flat = decoder['output_kan'](attention_weights_flat).squeeze(-1)
        adj_scores = adj_scores_flat.view(num_nodes, num_nodes)
        
        # 應用動態溫度縮放（保留線性分數，避免中途壓縮）
        temperature = decoder.temperature
        adj_scores = adj_scores / (temperature + 1e-8)
        
        return adj_scores
    
    def _calibrate_edge_weights(self, adj_matrix, embeddings):
        """混合校準機制 - 多維相似度 + 可學習權重"""
        # 🎯 優化3: 多種相似度計算
        # 1. 歐幾里得距離相似度
        euclidean_dist = torch.cdist(embeddings, embeddings, p=2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # 2. 曼哈頓距離相似度
        manhattan_dist = torch.cdist(embeddings, embeddings, p=1)
        manhattan_sim = 1.0 / (1.0 + manhattan_dist)
        
        # 3. 特徵方差相似度
        feature_var = torch.var(embeddings, dim=1, keepdim=True)
        var_sim = 1.0 / (1.0 + torch.abs(feature_var - feature_var.T))
        
        # 4. 正規化所有相似度到[0,1]
        euclidean_sim = (euclidean_sim - euclidean_sim.min()) / (euclidean_sim.max() - euclidean_sim.min() + 1e-8)
        manhattan_sim = (manhattan_sim - manhattan_sim.min()) / (manhattan_sim.max() - manhattan_sim.min() + 1e-8)
        var_sim = (var_sim - var_sim.min()) / (var_sim.max() - var_sim.min() + 1e-8)
        
        # 5. 可學習融合（權重會自動正規化）
        weights = F.softmax(self.sim_weights, dim=0)
        calibrated = (
            weights[0] * adj_matrix +
            weights[1] * euclidean_sim +
            weights[2] * manhattan_sim +
            weights[3] * var_sim
        )
        
        # 動態範圍調整
        min_val, max_val = calibrated.min(), calibrated.max()
        calibrated = (calibrated - min_val) / (max_val - min_val + 1e-8)
        # 避免極端0/1導致後續硬剪枝過度
        calibrated = torch.clamp(calibrated, 0.05, 0.95)
        
        # 細微銳化（降低強度，避免過度兩極）
        calibrated = self._sharpen_adjacency(calibrated, gamma=2.2)
        
        return calibrated
    
    def _sharpen_adjacency(self, adj_matrix, gamma=2.5, eps=1e-6):
        """
        銳化鄰接矩陣 - 擴大值域範圍，增強判別性
        gamma > 1: 銳化，gamma < 1: 平滑
        """
        # 確保數值穩定性
        adj_clamped = torch.clamp(adj_matrix, eps, 1 - eps)
        
        # 轉換為logit空間進行銳化
        logits = torch.log(adj_clamped) - torch.log(1 - adj_clamped)
        
        # 應用銳化係數
        sharpened_logits = logits * gamma
        
        # 轉回概率空間
        sharpened = torch.sigmoid(sharpened_logits)
        
        # 確保對稱性
        sharpened = (sharpened + sharpened.T) / 2
        
        return sharpened
    
    def _apply_adaptive_sparsification(self, adj_matrix, epoch=None):
        """修正版稀疏化策略 - 精確控制稀疏程度"""
        if epoch is None:
            epoch = 0
        
        n = adj_matrix.shape[0]
        total_elements = n * n
        
        # 保守稀疏目標 (保持合理密度)
        if epoch < 50:
            target_sparsity = 0.45   # 初期保留45%
        elif epoch < 150:
            target_sparsity = 0.40   # 中期保留40%
        else:
            target_sparsity = 0.35   # 後期保留35%
        
        # 保守修正: 限制稀疏範圍保持合理密度
        if target_sparsity < 0.2:
            target_sparsity = 0.2
        if target_sparsity > 0.8:
            target_sparsity = 0.8
        
        # 計算閾值
        adj_flat = adj_matrix.flatten()
        threshold = torch.quantile(adj_flat, 1 - target_sparsity)
        
        # 保守修正: 添加安全邊界保持合理密度
        min_threshold = torch.quantile(adj_flat, 0.7)   # 保證至少保留30%最高連接
        threshold = max(threshold.item(), min_threshold.item())
        
        # 動態軟閾值: 平滑過渡（連續權重，避免產生大量硬0）
        soft_values = torch.sigmoid((adj_matrix - threshold) * 3.5)
        
        # 最小權重地板，避免行全0
        soft_values = torch.clamp(soft_values, 5e-4, 1.0)
        
        # 確保每個節點至少有2個連接（若行近乎全0，補強 top-2）
        row_sums_soft = soft_values.sum(dim=1, keepdim=True)
        needs_boost = (row_sums_soft.squeeze(1) < 2e-4).float().view(-1, 1)
        k = min(2, n)
        _, top_indices = torch.topk(adj_matrix, k, dim=1)
        boost = torch.zeros_like(adj_matrix)
        for i in range(n):
            for j in range(k):
                if j < top_indices.shape[1]:
                    boost[i, top_indices[i, j]] = 1.0
        # 以小常數加入，避免破壞連續性
        sparse_adj = soft_values + needs_boost * 1e-3 * boost
        
        # 確保不添加自連接
        sparse_adj = sparse_adj * (1 - torch.eye(n, device=sparse_adj.device))
        
        return sparse_adj


class DualGraphGNNKANModel(nn.Module):
    """
    雙圖融合 GNN-KAN 模型
    
    核心理念：
    - 相似度圖：捕捉服務間的靜態相似性
    - 傳播圖：捕捉異常傳播的動態關係
    - 自適應融合：根據故障類型動態調整兩個圖的權重
    
    Args:
        config: GNNKANConfig 配置對象
        num_nodes: 節點數量
        similarity_threshold: 相似度圖閾值（默認 0.5）
        propagation_threshold: 傳播圖閾值（默認 0.3）
        max_lag: 最大時滯（默認 10）
    
    Example:
        >>> config = GNNKANConfig()
        >>> model = DualGraphGNNKANModel(config, num_nodes=10)
        >>> node_features = torch.randn(10, 128)
        >>> sim_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        >>> prop_edge_index = torch.tensor([[0, 1]], dtype=torch.long)
        >>> output = model(node_features, sim_edge_index, prop_edge_index)
    """
    
    def __init__(self, config, num_nodes, similarity_threshold=0.5, 
                 propagation_threshold=0.3, max_lag=10):
        super(DualGraphGNNKANModel, self).__init__()
        
        self.config = config
        self.num_nodes = num_nodes
        self.similarity_threshold = similarity_threshold
        self.propagation_threshold = propagation_threshold
        self.max_lag = max_lag
        
        # 特徵維度
        self.input_dim = config.hidden_dims[0] if hasattr(config, 'hidden_dims') else 128
        self.hidden_dim = config.hidden_dims[0] if hasattr(config, 'hidden_dims') else 128
        self.output_dim = config.hidden_dims[0] if hasattr(config, 'hidden_dims') else 128
        
        print(f"🧠 初始化 DualGraphGNNKANModel:")
        print(f"   - 節點數: {num_nodes}")
        print(f"   - 輸入維度: {self.input_dim}")
        print(f"   - 隱藏維度: {self.hidden_dim}")
        print(f"   - 相似度閾值: {similarity_threshold}")
        print(f"   - 傳播閾值: {propagation_threshold}")
        
        # 1. 相似度圖分支 (GNN + KAN)
        self.similarity_gnn = OptimizedGNNKANEncoder(
            input_dim=self.input_dim,
            hidden_dims=config.hidden_dims if hasattr(config, 'hidden_dims') else [128, 96, 64],
            output_dim=self.hidden_dim,
            num_layers=getattr(config, 'num_layers', 3),
            dropout=getattr(config, 'dropout', 0.1),
            kan_grid_size=getattr(config, 'kan_grid_size', 8),
            kan_spline_order=getattr(config, 'kan_spline_order', 3),
            basis_function=getattr(config, 'basis_function', 'chebyshev')
        )
        
        # 2. 傳播圖分支 (GNN + KAN)
        self.propagation_gnn = OptimizedGNNKANEncoder(
            input_dim=self.input_dim,
            hidden_dims=config.hidden_dims if hasattr(config, 'hidden_dims') else [128, 96, 64],
            output_dim=self.hidden_dim,
            num_layers=getattr(config, 'num_layers', 3),
            dropout=getattr(config, 'dropout', 0.1),
            kan_grid_size=getattr(config, 'kan_grid_size', 8),
            kan_spline_order=getattr(config, 'kan_spline_order', 3),
            basis_function=getattr(config, 'basis_function', 'chebyshev')
        )
        
        # 3. 自適應融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(getattr(config, 'dropout', 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 4. 可學習的分支權重（用於自適應融合）
        self.branch_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # 5. 圖解碼器（用於重建鄰接矩陣）
        self.graph_decoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(getattr(config, 'dropout', 0.1)),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 6. 注意力機制（用於特徵融合）
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=getattr(config, 'dropout', 0.1),
            batch_first=True
        )
        
        print(f"✅ DualGraphGNNKANModel 初始化完成")
        print(f"   - 相似度分支參數: {sum(p.numel() for p in self.similarity_gnn.parameters())}")
        print(f"   - 傳播分支參數: {sum(p.numel() for p in self.propagation_gnn.parameters())}")
        print(f"   - 總參數: {sum(p.numel() for p in self.parameters())}")
    
    def forward(self, node_features, sim_edge_index, prop_edge_index=None, 
                sim_edge_weights=None, prop_edge_weights=None, fault_type=None):
        """
        前向傳播
        
        Args:
            node_features: 節點特徵 [N, D]
            sim_edge_index: 相似度圖邊索引 [2, E_sim]
            prop_edge_index: 傳播圖邊索引 [2, E_prop] (可選)
            sim_edge_weights: 相似度圖邊權重 [E_sim] (可選)
            prop_edge_weights: 傳播圖邊權重 [E_prop] (可選)
            fault_type: 故障類型 (可選，用於自適應融合)
            
        Returns:
            node_embedding: 融合後的節點嵌入 [N, D]
            pred_adj: 預測的鄰接矩陣 [N, N] (可選)
        """
        batch_size, num_nodes = node_features.shape[0], node_features.shape[1]
        
        # 🔧 優化：緩存機制
        if not hasattr(self, '_cache') or self._cache is None:
            self._cache = {}
        
        # 檢查是否可以使用緩存
        cache_key = f"{node_features.shape}_{sim_edge_index.shape}_{prop_edge_index.shape if prop_edge_index is not None else 'None'}"
        if cache_key in self._cache and not self.training:
            return self._cache[cache_key]
        
        # 1. 相似度圖分支前向傳播
        sim_embedding = self.similarity_gnn(node_features, sim_edge_index)
        
        # 2. 傳播圖分支前向傳播（如果提供）
        if prop_edge_index is not None and prop_edge_index.size(1) > 0:
            prop_embedding = self.propagation_gnn(node_features, prop_edge_index)
            
            # 3. 自適應融合
            # 根據故障類型調整分支權重
            if fault_type is not None:
                # 對於 DELAY 和 LOSS 故障，增加傳播圖權重
                if fault_type in ['delay', 'loss']:
                    weights = F.softmax(torch.tensor([0.3, 0.7]), dim=0)
                else:
                    weights = F.softmax(torch.tensor([0.7, 0.3]), dim=0)
            else:
                # 使用可學習權重
                weights = F.softmax(self.branch_weights, dim=0)
            
            # 加權融合
            fused_embedding = weights[0] * sim_embedding + weights[1] * prop_embedding
            
            # 4. 注意力機制增強融合
            # 將兩個嵌入拼接進行自注意力
            combined_embedding = torch.stack([sim_embedding, prop_embedding], dim=1)  # [N, 2, D]
            attended_embedding, _ = self.attention(
                combined_embedding, combined_embedding, combined_embedding
            )
            attended_embedding = attended_embedding.mean(dim=1)  # [N, D]
            
            # 5. 最終融合
            final_embedding = self.fusion_layer(
                torch.cat([fused_embedding, attended_embedding], dim=-1)
            )
            
        else:
            # 只有相似度圖的情況
            final_embedding = sim_embedding
        
        # 6. 圖解碼器（重建鄰接矩陣）
        pred_adj = None
        if self.training or hasattr(self, 'enable_graph_decoder'):
            # 計算節點間的相似度
            node_sim = torch.mm(final_embedding, final_embedding.t())
            # 直接使用相似度矩陣作為鄰接矩陣
            pred_adj = torch.sigmoid(node_sim)
        
        # 緩存結果（僅在推理時）
        if not self.training:
            self._cache[cache_key] = (final_embedding, pred_adj)
        
        return final_embedding, pred_adj
    
    def get_reg_loss(self):
        """獲取正則化損失（用於 KAN 層）"""
        reg_loss = 0.0
        
        # 相似度分支正則化
        if hasattr(self.similarity_gnn, 'get_reg_loss'):
            reg_loss += self.similarity_gnn.get_reg_loss()
        
        # 傳播分支正則化
        if hasattr(self.propagation_gnn, 'get_reg_loss'):
            reg_loss += self.propagation_gnn.get_reg_loss()
        
        return reg_loss
    
    def enable_graph_decoder(self):
        """啟用圖解碼器"""
        self.enable_graph_decoder = True
    
    def disable_graph_decoder(self):
        """禁用圖解碼器"""
        if hasattr(self, 'enable_graph_decoder'):
            delattr(self, 'enable_graph_decoder')


# 將SimplifiedGNNKAN添加到可用的模型中
__all__ = ['GNNKANModel', 'SimplifiedGNNKAN', 'DualGraphGNNKANModel', 'TemporalAttention', 'create_fallback_model', 
           'train_gnn_kan_model', 'compute_loss_stable', 'MultiScaleGraphDecoder', 'LossScheduler']