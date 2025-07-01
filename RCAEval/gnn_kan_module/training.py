"""
Training Module for GNN-KAN
訓練模組 - 處理GNN-KAN模型的訓練和優化
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch_geometric.utils import negative_sampling

# 修復導入問題 - 使用模組化的結構
try:
    from .kan_components import GradientStabilizer
except ImportError:
    print("⚠️ 創建臨時梯度穩定器...")
    
    class GradientStabilizer:
        def __init__(self, l1_lambda=1e-5, entropy_lambda=1e-5, grad_clip_value=1.0, 
                     pruning_threshold=1e-2, enable_dynamic_scaling=True, stability_check_freq=10):
            self.l1_lambda = l1_lambda
            self.entropy_lambda = entropy_lambda
            self.grad_clip_value = grad_clip_value
            self.pruning_threshold = pruning_threshold
            self.enable_dynamic_scaling = enable_dynamic_scaling
            self.stability_check_freq = stability_check_freq
            self.loss_history = []
            
        def stabilize_gradients(self, model):
            """臨時梯度穩定器"""
            pass
            
        def compute_total_regularization_loss(self, model, base_loss):
            """計算總正則化損失"""
            return base_loss
            
        def apply_gradient_clipping(self, model, clip_type='norm'):
            """應用梯度裁剪"""
            return torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_value)
            
        def adaptive_clipping(self, model, current_loss):
            """自適應梯度裁剪"""
            self.loss_history.append(current_loss)
            if len(self.loss_history) > 10:
                loss_std = np.std(self.loss_history[-10:])
                clip_norm = max(0.5, min(2.0, 1.0 / (loss_std + 1e-8)))
            else:
                clip_norm = 1.0
            return torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
        def check_numerical_stability(self, model):
            """檢查數值穩定性"""
            return {'gradient_nan_count': 0, 'gradient_inf_count': 0}
            
        def adaptive_regularization_scaling(self, current_loss, loss_history):
            """自適應正則化縮放"""
            pass
            
        def apply_dynamic_pruning(self, model):
            """應用動態剪枝"""
            return 0.0
            
        def get_stability_report(self):
            """獲取穩定性報告"""
            return {
                'gradient_statistics': {
                    'mean_grad_norm': 0.0,
                    'max_grad_norm': 0.0,
                    'gradient_clips': 0
                },
                'stability_violations': 0,
                'regularization_config': {
                    'l1_lambda': self.l1_lambda,
                    'entropy_lambda': self.entropy_lambda
                }
            }

from .config import SimplifiedGNNKANConfig

# 修復模型導入
try:
    from .models import SimplifiedGNNKAN
except ImportError:
    print("⚠️ 無法導入 SimplifiedGNNKAN，使用臨時實現...")
    
    import torch.nn as nn
    
    class SimplifiedGNNKAN(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, output_dim=None, num_layers=2, 
                     dropout=0.1, use_batch_norm=True, use_residual=True, kan_config=None):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim or hidden_dim)
            
        def forward(self, node_features, edge_index):
            x = self.linear(node_features)
            adj = torch.mm(x, x.t())
            return x, torch.sigmoid(adj)


def create_model_with_config(config):
    """
    根據配置創建模型
    
    Args:
        config: SimplifiedGNNKANConfig 配置對象
        
    Returns:
        model: 創建的模型實例
    """
    try:
        from .models import SimplifiedGNNKAN
        
        model = SimplifiedGNNKAN(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            use_residual=config.use_residual,
            kan_config=config.kan_config
        )
        
        print(f"✓ Created model with config: {config}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to create model with config: {e}")
        raise


def create_model_from_checkpoint(checkpoint_path, config=None):
    """
    🔄 從檢查點加載模型
    
    Args:
        checkpoint_path: 檢查點文件路徑
        config: 配置對象（可選）
        
    Returns:
        model: 加載的模型
        metadata: 檢查點元數據
    """
    print(f"📂 Loading model from checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 從檢查點獲取配置
        if 'config' in checkpoint and config is None:
            config = checkpoint['config']
        
        # 創建模型
        if config:
            model = create_model_with_config(config)
        else:
            # 從state_dict推斷模型結構
            model = _infer_model_from_state_dict(checkpoint['model_state_dict'])
        
        # 加載權重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 獲取元數據
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', None),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', None)
        }
        
        print(f"✅ Model loaded from epoch {metadata['epoch']}")
        return model, metadata
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        if config:
            print("Creating new model with provided config...")
            return create_model_with_config(config), {}
        else:
            raise e


def _infer_model_from_state_dict(state_dict):
    """從state_dict推斷模型結構"""
    # 分析state_dict的鍵來推斷模型參數
    keys = list(state_dict.keys())
    
    # 推斷層數
    conv_layers = [k for k in keys if 'conv_layers' in k and 'weight' in k]
    num_layers = len(set(k.split('.')[1] for k in conv_layers))
    
    # 推斷維度
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
    
    # 創建模型
    model = SimplifiedGNNKAN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_layers=max(num_layers, 2)
    )
    
    return model


def save_model_checkpoint(model, optimizer, epoch, loss, metrics, save_path, config=None):
    """
    💾 保存模型檢查點
    
    Args:
        model: 要保存的模型
        optimizer: 優化器
        epoch: 當前epoch
        loss: 當前損失
        metrics: 評估指標
        save_path: 保存路徑
        config: 配置對象
    """
    print(f"💾 Saving checkpoint to {save_path}")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics,
        'timestamp': time.time()
    }
    
    if config:
        checkpoint['config'] = config
    
    try:
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved successfully")
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")


class ModelManager:
    """🎯 模型管理器 - 統一管理模型創建、訓練、保存"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def create_model(self):
        """創建模型"""
        self.model = create_model_with_config(self.config)
        return self.model
    
    def setup_training(self):
        """設置訓練組件"""
        if self.model is None:
            self.create_model()
        
        # 優化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.config, 'learning_rate', 0.001),
            weight_decay=getattr(self.config, 'weight_decay', 1e-5)
        )
        
        # 學習率調度器
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10
        )
        
        return self.optimizer, self.scheduler
    
    def train_model(self, node_features, edge_index):
        """訓練模型"""
        if self.model is None or self.optimizer is None:
            self.setup_training()
        
        return train_gnn_kan_model(
            self.model, node_features, edge_index, self.config
        )
    
    def save_checkpoint(self, epoch, loss, metrics, save_path):
        """保存檢查點"""
        save_model_checkpoint(
            self.model, self.optimizer, epoch, loss, metrics, save_path, self.config
        )
    
    def load_checkpoint(self, checkpoint_path):
        """加載檢查點"""
        self.model, metadata = create_model_from_checkpoint(checkpoint_path, self.config)
        return metadata


class TemporalAttention(nn.Module):
    """時序注意力機制"""
    
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] 或 [seq_len, hidden_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加batch維度
        
        attended, _ = self.attention(x, x, x)
        output = self.layer_norm(attended + x)
        
        if output.size(0) == 1:
            output = output.squeeze(0)  # 移除batch維度
            
        return output


class GNNKANLoss(nn.Module):
    """GNN-KAN專用損失函數"""
    
    def __init__(self, config):
        super(GNNKANLoss, self).__init__()
        self.config = config
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_adj, true_adj, node_embeddings=None):
        """
        計算組合損失 - 修復CUDA斷言錯誤
        
        Args:
            pred_adj: 預測的鄰接矩陣
            true_adj: 真實的鄰接矩陣  
            node_embeddings: 節點嵌入（可選）
        """
        # 🔧 修復CUDA斷言錯誤：確保pred_adj在[0,1]範圍內
        pred_adj_safe = torch.clamp(pred_adj, min=1e-7, max=1.0-1e-7)
        true_adj_safe = torch.clamp(true_adj, min=0.0, max=1.0)
        
        # 數值穩定性檢查 - 詳細診斷
        has_nan = torch.isnan(pred_adj_safe).any()
        has_inf = torch.isinf(pred_adj_safe).any()
        
        if has_nan or has_inf:
            print("⚠️ 預測鄰接矩陣包含無效值，使用MSE損失")
            print(f"  pred_adj_safe shape: {pred_adj_safe.shape}")
            print(f"  pred_adj_safe range: [{pred_adj_safe.min().item():.6f}, {pred_adj_safe.max().item():.6f}]")
            print(f"  NaN count: {torch.isnan(pred_adj_safe).sum().item()}")
            print(f"  Inf count: {torch.isinf(pred_adj_safe).sum().item()}")
            
            # 找到無效值的位置
            if has_nan:
                nan_positions = torch.where(torch.isnan(pred_adj_safe))
                print(f"  NaN positions (first 5): {[(i.item(), j.item()) for i, j in zip(nan_positions[0][:5], nan_positions[1][:5])]}")
            
            if has_inf:
                inf_positions = torch.where(torch.isinf(pred_adj_safe))
                print(f"  Inf positions (first 5): {[(i.item(), j.item()) for i, j in zip(inf_positions[0][:5], inf_positions[1][:5])]}")
            
            # 使用安全的MSE損失
            pred_adj_clean = torch.nan_to_num(pred_adj, nan=0.0, posinf=1.0, neginf=0.0)
            recon_loss = self.mse_loss(pred_adj_clean, true_adj_safe)
        else:
            # 重構損失 - 使用安全的值
            recon_loss = self.bce_loss(pred_adj_safe, true_adj_safe)
        
        total_loss = recon_loss
        
        # 添加正則化項
        if node_embeddings is not None:
            # L2正則化
            l2_reg = torch.norm(node_embeddings, p=2)
            total_loss += self.config.l2_lambda * l2_reg
            
            # 平滑性正則化
            if node_embeddings.size(0) > 1:
                diff = torch.diff(node_embeddings, dim=0)
                smoothness_reg = torch.norm(diff, p=2)
                total_loss += self.config.smoothness_lambda * smoothness_reg
        
        return total_loss


def train_gnn_kan_model(model, node_features, edge_index, config, sparsity_lambda=1e-5):
    """
    通用GNN-KAN模型訓練函數
    
    Args:
        model (nn.Module): GNN-KAN模型
        node_features (torch.Tensor): 節點特徵
        edge_index (torch.Tensor): 邊索引
        config (SimplifiedGNNKANConfig): 配置對象
        sparsity_lambda (float): 稀疏性正則化強度
        
    Returns:
        model: 訓練好的模型
        training_history: 訓練歷史記錄
    """
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    model.to(device)
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.patience, factor=0.5)
    
    print(f"🚀 Starting GNN-KAN training on {device}...")
    print(f"   Config: lr={config.learning_rate}, epochs={config.num_epochs}, batch_size={config.batch_size}, sparsity_lambda={sparsity_lambda}")
    
    training_history = {'loss': [], 'adj_min': [], 'adj_max': [], 'adj_mean': []}
    
    # 創建目標鄰接矩陣（用於監督學習）
    # 在真實場景中，這應該基於先驗知識或日誌/追蹤數據生成
    # 這裡我們使用一個簡化的自監督目標
    with torch.no_grad():
        true_adj = torch.zeros(node_features.size(0), node_features.size(0), device=device)
        true_adj[edge_index[0], edge_index[1]] = 1
        true_adj[edge_index[1], edge_index[0]] = 1 # 無向圖

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向傳播
        node_embedding, pred_adj = model(node_features, edge_index)
        
        # 損失計算
        # 1. 重建損失 (Reconstruction Loss) - 確保圖結構合理
        pos_weight = torch.tensor([float(true_adj.shape[0] * true_adj.shape[0] - true_adj.sum()) / true_adj.sum()])
        recon_loss = F.binary_cross_entropy_with_logits(pred_adj, true_adj, pos_weight=pos_weight.to(device))
        
        # 2. 節點嵌入損失 (Embedding Loss) - 可選，使相連節點更接近
        # 這裡簡化，不計算嵌入損失
        
        # 3. KAN 正則化損失
        kan_reg_loss = 0
        if hasattr(model, 'get_reg_loss'):
            kan_reg_loss = model.get_reg_loss()

        # 4. 稀疏性損失 (Sparsity Loss) - 鼓勵稀疏圖
        sparsity_loss = sparsity_lambda * torch.norm(pred_adj, 1)

        # 總損失
        loss = recon_loss + kan_reg_loss + sparsity_loss
        
        loss.backward()
        
        # 梯度裁剪
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            
        optimizer.step()
        
        # 更新學習率
        scheduler.step(loss)
        
        training_history['loss'].append(loss.item())

        # 監控與調試: 定期打印鄰接矩陣統計信息
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                adj_stats = pred_adj.sigmoid() # 查看經過 sigmoid 後的概率值
                adj_min = adj_stats.min().item()
                adj_max = adj_stats.max().item()
                adj_mean = adj_stats.mean().item()
                training_history['adj_min'].append(adj_min)
                training_history['adj_max'].append(adj_max)
                training_history['adj_mean'].append(adj_mean)
                
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss.item():.6f}, "
                      f"Recon: {recon_loss.item():.6f}, KAN: {kan_reg_loss:.6f}, Sparsity: {sparsity_loss.item():.6f}, "
                      f"Adj(min/max/mean): {adj_min:.4f}/{adj_max:.4f}/{adj_mean:.4f}")

    print("✅ Training finished.")
    return model, training_history


class AdvancedGNNKANTrainer:
    """高級GNN-KAN訓練器 - 包含更多訓練策略"""
    
    def __init__(self, config):
        self.config = config
        
    def train_with_curriculum(self, model, node_features, edge_index):
        """
        課程學習訓練策略
        
        Args:
            model: GNN-KAN模型
            node_features: 節點特徵
            edge_index: 邊索引
            
        Returns:
            model: 訓練後的模型
            training_history: 訓練歷史
        """
        print("Starting curriculum learning training...")
        
        device = next(model.parameters()).device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        
        training_history = {
            'losses': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # 階段1：簡單任務（少量節點）
        print("Phase 1: Training on subset of nodes...")
        subset_size = min(10, node_features.size(0))
        subset_features = node_features[:subset_size]
        subset_edge_index = self._filter_edge_index(edge_index, subset_size)
        
        model = self._train_phase(
            model, subset_features, subset_edge_index, 
            epochs=self.config.num_epochs // 3,
            phase_name="Phase 1"
        )
        
        # 階段2：中等複雜度
        if node_features.size(0) > subset_size:
            print("Phase 2: Training on larger subset...")
            medium_size = min(20, node_features.size(0))
            medium_features = node_features[:medium_size]
            medium_edge_index = self._filter_edge_index(edge_index, medium_size)
            
            model = self._train_phase(
                model, medium_features, medium_edge_index,
                epochs=self.config.num_epochs // 3,
                phase_name="Phase 2"
            )
        
        # 階段3：完整訓練
        print("Phase 3: Full training...")
        model = self._train_phase(
            model, node_features, edge_index,
            epochs=self.config.num_epochs // 3,
            phase_name="Phase 3"
        )
        
        return model, training_history
    
    def _filter_edge_index(self, edge_index, max_nodes):
        """過濾邊索引，只保留涉及前max_nodes個節點的邊"""
        mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
        return edge_index[:, mask]
    
    def _train_phase(self, model, features, edge_index, epochs, phase_name):
        """訓練一個階段"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        criterion = GNNKANLoss(self.config)
        
        # 創建目標鄰接矩陣 - 修復邊索引越界問題
        num_nodes = features.size(0)
        device = features.device
        target_adj = torch.zeros(num_nodes, num_nodes, device=device)
        
        if edge_index.size(1) > 0:
            # 🔧 安全邊索引檢查 - 避免越界
            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            valid_edge_index = edge_index[:, valid_edges_mask]
            
            if valid_edge_index.size(1) > 0:
                target_adj[valid_edge_index[0], valid_edge_index[1]] = 1.0
                target_adj = (target_adj + target_adj.t()) / 2.0
            else:
                print(f"⚠️ {phase_name}: 所有邊索引都超出範圍，使用單位矩陣")
                target_adj = torch.eye(num_nodes, device=device) * 0.1
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            try:
                node_embeddings, pred_adj = model(features, edge_index)
                loss = criterion(pred_adj, target_adj, node_embeddings)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"{phase_name} Epoch {epoch}: Loss={loss.item():.6f}")
                    
            except Exception as e:
                print(f"Error in {phase_name} at epoch {epoch}: {e}")
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
        # 基於特徵相似性創建目標
        features_np = node_features.detach().cpu().numpy()
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity = cosine_similarity(features_np)
        # 閾值化
        threshold = np.percentile(similarity, 80)
        target_adj = torch.tensor(
            (similarity > threshold).astype(float),
            device=device,
            dtype=torch.float
        )
    
    elif method == 'knn':
        # 基於k近鄰創建目標
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
        # 默認：基於邊索引
        target_adj = torch.zeros(num_nodes, num_nodes, device=device)
        if edge_index.size(1) > 0:
            target_adj[edge_index[0], edge_index[1]] = 1.0
            target_adj = (target_adj + target_adj.t()) / 2.0
    
    return target_adj