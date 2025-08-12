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

# Use the centralized, full implementation of GradientStabilizer
from .kan_components.gradient_stabilizer import GradientStabilizer
from .config import SimplifiedGNNKANConfig
# Import the model from the models module, not a local copy
from .models import SimplifiedGNNKAN, GNNKANModel, TemporalAttention

# 修復模型導入
try:
    from .models import SimplifiedGNNKAN
except ImportError:
    print("⚠️ 使用models.py中的統一實現...")


def create_model_with_config(config):
    """
    根據配置創建模型 - 重定向到models.py的統一實現
    
    Args:
        config: SimplifiedGNNKANConfig 配置對象
        
    Returns:
        model: 創建的模型實例
    """
    from .models import create_model_with_config as _create_model
    return _create_model(config)


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
        # This function is being consolidated into utils.py
        # save_model_checkpoint(
        #     self.model, self.optimizer, epoch, loss, metrics, save_path, self.config
        # )
    
    def load_checkpoint(self, checkpoint_path):
        """加載檢查點"""
        self.model, metadata = create_model_from_checkpoint(checkpoint_path, self.config)
        return metadata


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


def train_gnn_kan_model(model, node_features, edge_index, config, sparsity_lambda=None):
    """
    🚨 ISSUE 6: 實時性能力不足分析
    
    實時性問題：
    1. **訓練時間過長**：
       - 當前需要200輪訓練，平均60-180秒
       - 實際故障響應要求：<10秒內給出初步分析
       - On-the-fly訓練在生產環境不可行
    
    2. **冷啟動問題**：
       - 新系統或新故障類型需要重新訓練
       - 缺乏預訓練模型或遷移學習機制
       - 無法利用歷史相似故障的經驗
    
    3. **增量學習缺失**：
       - 每次都是完全重新訓練
       - 無法持續從新故障中學習
       - 模型不能隨時間演化改進
    
    4. **推理複雜度**：
       - O(V²)的鄰接矩陣計算在推理時仍然需要
       - KAN層的B-spline計算比MLP的矩陣乘法更耗時
       - 多階段後處理增加了響應延遲
    
    🔧 實時化改進建議：
    1. 預訓練+微調策略：
       # pretrained_model = load_universal_pretrained_model()
       # quick_adapted = few_shot_adaptation(pretrained_model, current_data)
    
    2. 模型壓縮：
       # compressed_model = knowledge_distillation(full_model, student_model)
       # quantized_model = dynamic_quantization(compressed_model)
    
    3. 近似推理：
       # sparse_adj = approximate_adjacency(embeddings, top_k=20)
       # fast_pagerank = power_iteration_early_stop(sparse_adj, max_iter=10)
    
    4. 分級響應：
       # immediate_response = fast_heuristic_ranking(raw_features)  # <1s
       # refined_response = simplified_kan_inference(processed_data)  # <5s  
       # detailed_analysis = full_gnn_kan_pipeline(all_data)  # <30s
    
    Current implementation - NOT suitable for real-time production
    """
    # 🔧 修正6：實時性改進 - 添加快速模式
    fast_mode = getattr(config, 'fast_mode', False) or sparsity_lambda is None
    if fast_mode:
        print("🚀 啟用快速訓練模式")
        # 快速模式：減少訓練輪數，提高收斂速度
        config.num_epochs = min(50, config.num_epochs)
        config.learning_rate = config.learning_rate * 1.5  # 更激進的學習率
        print(f"   快速模式配置：{config.num_epochs}輪訓練，學習率×1.5")
    
    # 設置稀疏性參數
    if sparsity_lambda is None:
        sparsity_lambda = 1e-5 if fast_mode else 1e-4
    
    device = 'cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu'
    model.to(device)
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.patience, factor=0.5)
    
    print(f"🚀 Starting GNN-KAN training on {device}...")
    print(f"   Config: lr={config.learning_rate}, epochs={config.num_epochs}, batch_size={config.batch_size}, sparsity_lambda={sparsity_lambda}")
    
    training_history = {'loss': [], 'adj_min': [], 'adj_max': [], 'adj_mean': [], 'sparsity_01': [], 'sparsity_03': [], 'sparsity_05': []}
    
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
        
        # 2. KAN 正則化損失 (應保留)
        kan_reg_loss = 0
        if hasattr(model, 'get_reg_loss'):
            # 確保 kan_reg_loss 是一個純量
            reg_loss = model.get_reg_loss()
            if isinstance(reg_loss, torch.Tensor):
                kan_reg_loss = reg_loss
            else: # 假設它是一個列表或元組
                kan_reg_loss = sum(reg_loss)

        # 3. 稀疏性損失 (Sparsity Loss)
        if sparsity_lambda > 0:
            adj_probs_for_loss = torch.sigmoid(pred_adj)
            sparsity_loss = torch.mean(adj_probs_for_loss)
        else:
            sparsity_loss = torch.tensor(0.0, device=device) # 如果 lambda 為 0，則不計算
        
        # 總損失
        loss = recon_loss + kan_reg_loss + (sparsity_lambda * sparsity_loss)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        # 記錄和打印
        training_history['loss'].append(loss.item())
        
        # 計算鄰接矩陣統計
        with torch.no_grad():
            adj_probs = torch.sigmoid(pred_adj)
            adj_min = adj_probs.min().item()
            adj_max = adj_probs.max().item()
            adj_mean = adj_probs.mean().item()
            
            sparsity_01 = (adj_probs < 0.1).float().mean().item()
            sparsity_03 = (adj_probs < 0.3).float().mean().item()
            sparsity_05 = (adj_probs < 0.5).float().mean().item()
        
        training_history['adj_min'].append(adj_min)
        training_history['adj_max'].append(adj_max)
        training_history['adj_mean'].append(adj_mean)
        training_history['sparsity_01'].append(sparsity_01)
        training_history['sparsity_03'].append(sparsity_03)
        training_history['sparsity_05'].append(sparsity_05)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # 打印時，顯示未加權的 sparsity_loss，更能反映真實的平均概率
            print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss.item():.6f}, Recon: {recon_loss.item():.4f}, Sparsity: {sparsity_loss.item():.6f} | "
                  f"Adj Probs(min/max/mean): {adj_min:.4f}/{adj_max:.4f}/{adj_mean:.4f} | "
                  f"Graph Sparsity: {sparsity_03:.4f} (0.1:{sparsity_01:.3f}, 0.3:{sparsity_03:.3f}, 0.5:{sparsity_05:.3f})")

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
                src = valid_edge_index[0]
                dst = valid_edge_index[1]
                ones = torch.ones_like(src, dtype=target_adj.dtype)
                target_adj.index_put_((src, dst), ones, accumulate=True)
                target_adj.index_put_((dst, src), ones, accumulate=True)
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
            src = edge_index[0]
            dst = edge_index[1]
            ones = torch.ones_like(src, dtype=target_adj.dtype)
            target_adj.index_put_((src, dst), ones, accumulate=True)
            target_adj.index_put_((dst, src), ones, accumulate=True)
    
    return target_adj