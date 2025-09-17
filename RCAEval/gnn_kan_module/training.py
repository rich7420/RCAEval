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
import math
from torch_geometric.utils import negative_sampling

# Use the centralized, full implementation of GradientStabilizer
# GradientStabilizer 已移除，使用簡化版本
from .config import SimplifiedGNNKANConfig


def learning_rate_scheduler(optimizer, current_epoch, total_epochs):
    """Cosine annealing learning rate scheduler - 平滑減少LR"""
    lr_max = 5e-4  # 初始LR提高到5e-4, 讓早期訓練更快
    min_lr = 1e-6
    # Cosine curve for smooth decay
    cos_anneal = 0.5 * (1 + math.cos(math.pi * current_epoch / total_epochs))
    current_lr = min_lr + (lr_max - min_lr) * cos_anneal
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr
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
    """GNN-KAN專用損失函數 - 階段1改進：解決過度凝聚/擬合"""
    
    def __init__(self, config):
        super(GNNKANLoss, self).__init__()
        self.config = config
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # 🔧 階段1：動態損失權重，防止凝聚
        self.polar_weight = getattr(config, 'polar_weight', 0.8)  # 降低Polar權重
        self.contrast_weight = getattr(config, 'contrast_weight', 1.5)  # 提升Contrast權重
        self.sparsity_target = getattr(config, 'sparsity_target', 0.3)  # 目標稀疏性
        
    def forward(self, pred_adj, true_adj, node_embeddings=None):
        """
        計算組合損失 - 階段1改進：解決過度凝聚/擬合
        
        Args:
            pred_adj: 預測的鄰接矩陣
            true_adj: 真實的鄰接矩陣  
            node_embeddings: 節點嵌入（可選）
        """
        # 🔧 修復CUDA斷言錯誤：確保pred_adj在[0,1]範圍內
        pred_adj_safe = torch.clamp(pred_adj, min=1e-7, max=1.0-1e-7)
        true_adj_safe = torch.clamp(true_adj, min=0.0, max=1.0)
        
        # 數值穩定性檢查
        has_nan = torch.isnan(pred_adj_safe).any()
        has_inf = torch.isinf(pred_adj_safe).any()
        
        if has_nan or has_inf:
            print("⚠️ 預測鄰接矩陣包含無效值，使用MSE損失")
            pred_adj_clean = torch.nan_to_num(pred_adj, nan=0.0, posinf=1.0, neginf=0.0)
            recon_loss = self.mse_loss(pred_adj_clean, true_adj_safe)
        else:
            # 重構損失
            recon_loss = self.bce_loss(pred_adj_safe, true_adj_safe)
        
        total_loss = recon_loss
        
        # 🎯 動態權重調整策略 - 根據訓練進度調整損失權重
        epoch = getattr(self, 'current_epoch', 0)
        total_epochs = getattr(self, 'total_epochs', 100)
        
        # 修正動態權重計算 - 平衡各損失項
        w_recon = 0.5  # 進一步降低重構損失權重
        w_polar = 0.2 + 0.5 * (epoch / max(total_epochs, 1))  # 極化損失增強
        w_contrast = min(8.0, 2.0 + epoch * 0.1)  # 對比損失更激進增強
        w_sparsity = min(0.8, 0.1 + epoch * 0.01)  # 降低稀疏性約束壓力
        
        # 🔧 改進的稀疏性懲罰
        current_sparsity = (pred_adj_safe > 0.1).float().mean().item()
        sparsity_penalty = torch.mean(torch.abs(pred_adj_safe)) * (1 - current_sparsity)
        total_loss += sparsity_penalty * w_sparsity
        
        # 🔧 增強的對比損失
        if node_embeddings is not None:
            contrast_loss = self.enhanced_contrastive_loss(node_embeddings, pred_adj_safe)
            total_loss += contrast_loss * w_contrast
            
            # L2正則化
            l2_reg = torch.norm(node_embeddings, p=2)
            total_loss += self.config.l2_lambda * l2_reg
            
            # 🎯 新增: 方差正則化 - 鼓勵節點表示有足夠判別性
            std_per_dim = torch.clamp(node_embeddings.std(dim=0), 1e-3, None)
            var_loss = torch.relu(1.0 - std_per_dim).mean()
            total_loss += 0.05 * var_loss
            
            # 平滑性正則化
            if node_embeddings.size(0) > 1:
                diff = torch.diff(node_embeddings, dim=0)
                smoothness_reg = torch.norm(diff, p=2)
                total_loss += self.config.smoothness_lambda * smoothness_reg
        
        # 🔧 穩定化的極化損失
        polar_loss = self.stabilized_polar_loss(pred_adj_safe, alpha=0.5)
        total_loss += polar_loss * w_polar
        
        # 🎯 細微Margin Loss - 微調確保top/bottom差距
        if node_embeddings is not None:
            margin = 0.08  # 微調差距要求
            margin_loss = self.discriminative_loss(pred_adj_safe.flatten(), margin)
            total_loss += 0.15 * margin_loss
        
        # 🎯 方向3: Ranking Loss - 直接優化排名 (後期啟用)
        if epoch > 50 and node_embeddings is not None:
            try:
                # 生成pseudo ranks
                ground_truth_ranks = self.generate_pseudo_ranks(pred_adj_safe)
                
                # 計算PageRank分數作為pr_scores
                pr_scores = pred_adj_safe.sum(dim=1)  # 簡化版PageRank分數
                
                # 應用ListNet loss
                rank_loss = self.listnet_loss(pr_scores, ground_truth_ranks)
                total_loss += 0.2 * rank_loss
                
                if epoch % 10 == 0:  # 每10個epoch打印一次
                    print(f"  Ranking Loss: {rank_loss.item():.4f}")
            except Exception as e:
                print(f"⚠️ Ranking Loss計算失敗: {e}")
        
        return total_loss
    
    def enhanced_contrastive_loss(self, embeddings, adj_matrix, 
                             base_temp=0.2, min_temp=0.1, max_temp=0.5):
        """
        修正版對比學習 - 確保與其他損失項量級一致
        * 無需故障類型信息 *
        """
        batch_size = embeddings.shape[0]
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 計算余弦相似度
        logits = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # 關鍵修正: 使用鄰接矩陣作為正樣本強度，但強化差異
        # 將鄰接權重從[0.5, 0.73]映射到[-1.0, 1.0]以擴大差異
        weights = (adj_matrix - 0.62) * 10.0  # 中心點設為值域中點
        
        # 限制在合理範圍 [-1.0, 1.0]
        weights = torch.clamp(weights, -1.0, 1.0)
        
        # 關鍵修正: 增強高價值連接
        strong_connections = (adj_matrix > 0.68).float()
        weights = weights + strong_connections * 0.3
        
        # 移除自連接
        eye = torch.eye(batch_size, device=embeddings.device)
        weights = weights * (1 - eye)
        
        # 修正: 使用加權相似度計算損失
        positive_logits = logits * weights * (1 - eye)
        negative_logits = logits * (1 - weights) * (1 - eye)
        
        # 關鍵修正: 提升對比損失量級，確保數值穩定性
        pos_exp = torch.exp(torch.clamp(positive_logits.sum(dim=1) / 10.0, -10, 10))
        neg_exp = torch.exp(torch.clamp(negative_logits.sum(dim=1) / 10.0, -10, 10))
        
        # 避免除零和log(0)
        ratio = pos_exp / (neg_exp + 1e-8)
        ratio = torch.clamp(ratio, min=1e-8, max=1e8)
        
        contrastive_loss = -torch.log(ratio).mean()
        
        # 檢查NaN並處理
        if torch.isnan(contrastive_loss):
            contrastive_loss = torch.tensor(0.0, device=embeddings.device)
        
        # 關聯約束: 確保節點表示與鄰接矩陣一致
        recon_loss = torch.nn.functional.mse_loss(
            torch.mm(norm_embeddings, norm_embeddings.t()),
            adj_matrix.detach()
        ) * 0.1
        
        return contrastive_loss * 5.0 + recon_loss  # 明確提升對比損失權重
    
    def discriminative_loss(self, pr_scores, margin=0.1):
        """
        辨識度損失 - 直接優化top-1分離度
        pr_scores: PageRank分數列表 (已排序)
        margin: 最小間距要求
        """
        if len(pr_scores) < 2:
            return torch.tensor(0.0, device=pr_scores[0].device)
        
        # 確保分數是tensor
        if not isinstance(pr_scores, torch.Tensor):
            pr_scores = torch.tensor(pr_scores, dtype=torch.float32)
        
        # 排序分數
        sorted_scores, _ = torch.sort(pr_scores, descending=True)
        
        # 計算top-1與top-2的差距
        score_diff = sorted_scores[0] - sorted_scores[1]
        
        # 損失: 當差異小於margin時產生損失
        loss = torch.relu(margin - score_diff)
        
        return loss
    
    def listnet_loss(self, pr_scores, ground_truth_ranks, temperature=1.0):
        """ListNet ranking loss - 優化PageRank排名, 無需故障類型"""
        # 轉為概率distribution
        predicted_probs = torch.softmax(pr_scores / temperature, dim=0)
        
        # 基於ground_truth_ranks生成理想分布 (數據驅動: 高rank應該高prob)
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
        """生成pseudo label (數據驅動)"""
        row_sums = adj_matrix.sum(dim=1).detach()
        pseudo_ranks = row_sums / row_sums.max()
        return {i: pseudo_ranks[i].item() for i in range(adj_matrix.shape[0])}
    
    def stabilized_polar_loss(self, adj_matrix, alpha=0.5):
        """
        穩定化極化損失，避免梯度爆炸
        alpha: 控制稀疏和密集的平衡(0=純稀疏, 1=純密集)
        """
        # 先應用sigmoid確保值域在[0,1]
        adj_probs = torch.sigmoid(adj_matrix)
        
        # 結合兩種目標：稀疏與有結構
        sparse_loss = torch.mean(adj_probs) * (1 - alpha)
        structured_loss = torch.var(adj_probs) * alpha
        
        # 添加梯度穩定項
        stable_term = 0.001 * torch.log(torch.var(adj_probs) + 1e-8)
        
        return sparse_loss + structured_loss - stable_term


class LossScheduler:
    """無需故障類型的損失權重自適應調控器"""
    
    def __init__(self, initial_weights, window_size=10):
        self.weights = initial_weights.copy()
        self.window_size = window_size
        self.history = []
    
    def update_weights(self, metrics):
        """基於訓練動態自動調整權重"""
        self.history.append(metrics)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # 1. 根據對比損失表現動態調整
        if len(self.history) >= 2:
            contrast_trend = (self.history[-1]['contrast'] - 
                             self.history[-2]['contrast'])
            
            # 對比損失下降過快？可能學習不足
            if contrast_trend < -0.05:
                self.weights['contrast'] = min(3.0, 
                                             self.weights['contrast'] * 1.1)
            # 對比損失停滯？減少關注
            elif abs(contrast_trend) < 0.01:
                self.weights['contrast'] = max(0.5, 
                                             self.weights['contrast'] * 0.95)
        
        # 2. 基於圖稀疏度自動調整極化損失
        sparsity = 1.0 - metrics['density']
        if sparsity < 0.3:  # 圖太密
            self.weights['polar'] = min(2.0, self.weights['polar'] * 1.05)
        elif sparsity > 0.7:  # 圖太稀疏
            self.weights['polar'] = max(0.1, self.weights['polar'] * 0.95)
        
        # 3. 自動平衡重構與結構損失
        recon_ratio = metrics['recon'] / (metrics['sparsity'] + 1e-5)
        if recon_ratio > 0.5:  # 重構需求高
            self.weights['recon'] = min(2.0, self.weights['recon'] * 1.02)
        
        return self.weights.copy()


def train_gnn_kan_model(model, node_features, edge_index, config, sparsity_lambda=None, fault_type=None, **kwargs):
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
    
    # 🔧 階段1：改進的早停機制 / Improved early stopping mechanism
    from .models import EarlyStopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    
    # 🔧 階段1：驗證數據準備 / Validation data preparation
    val_data = kwargs.get('val_data', None)
    
    # 初始化損失函數
    loss_fn = GNNKANLoss(config)
    val_ground_truth = kwargs.get('val_ground_truth', None)
    monitor_metric = kwargs.get('monitor_metric', 'val_precision')
    
    best_val_metric = 0.0
    patience_counter = 0
    
    print(f"🚀 Starting GNN-KAN training on {device}...")
    print(f"   Config: lr={config.learning_rate}, epochs={config.num_epochs}, batch_size={config.batch_size}, sparsity_lambda={sparsity_lambda}")
    
    training_history = {'loss': [], 'adj_min': [], 'adj_max': [], 'adj_mean': [], 'sparsity_01': [], 'sparsity_03': [], 'sparsity_05': []}
    
    # 🔥 創建加權的真實鄰接矩陣 - 基於特徵相似性而非簡單二值化
    with torch.no_grad():
        num_nodes = node_features.size(0)
        true_adj = torch.zeros(num_nodes, num_nodes, device=device)
        
        # 基於節點特徵計算相似性作為真實權重
        if num_nodes > 1:
            # 歸一化特徵
            norm_features = F.normalize(node_features, p=2, dim=1)
            # 計算餘弦相似性
            similarity_matrix = torch.mm(norm_features, norm_features.t())
            # 將相似性轉換為 [0,1] 範圍的權重
            true_adj = (similarity_matrix + 1) / 2
            # 移除自環的影響
            true_adj.fill_diagonal_(0)
            
            # 🎯 增強: 根據原始邊強化重要連接 (來自改進方案)
            if edge_index.size(1) > 0:
                # 對原始邊給予額外權重 - 提升到0.3
                edge_boost = 0.3
                true_adj[edge_index[0], edge_index[1]] += edge_boost
                true_adj[edge_index[1], edge_index[0]] += edge_boost
                # 確保權重在[0,1]範圍內
                true_adj = torch.clamp(true_adj, 0, 1)
        else:
            # 單節點情況的回退
            if edge_index.size(1) > 0:
                true_adj[edge_index[0], edge_index[1]] = 1
                true_adj[edge_index[1], edge_index[0]] = 1
    
    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 🎯 方向2: 學習率調度 - 平滑減少LR
        current_lr = learning_rate_scheduler(optimizer, epoch, config.num_epochs)
        
        # 設置當前epoch信息供損失函數使用
        if hasattr(loss_fn, 'current_epoch'):
            loss_fn.current_epoch = epoch
        if hasattr(loss_fn, 'total_epochs'):
            loss_fn.total_epochs = config.num_epochs
            
        # 前向傳播 - 使用故障類型感知
        node_embedding, pred_adj = model(node_features, edge_index, fault_type)
            
        # 損失計算 - 處理KNN Baseline情況
        if pred_adj is None:
            # 使用KNN Baseline，跳過重建損失計算
            recon_loss = torch.tensor(0.0, device=node_embedding.device, requires_grad=True)
            pred_adj_sigmoid = torch.eye(node_embedding.size(0), device=node_embedding.device)
        else:
            # 使用Graph Decoder的情況
            # 若pred_adj已在[0,1]，視為概率直接用 BCE；否則視為logits用 BCEWithLogits
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
        
        # 🎯 強化極化損失 - 使用更激進的策略
        # 方法1: 獎勵接近0和1的值，懲罰中間值
        distance_from_center = torch.abs(pred_adj_sigmoid - 0.5)
        polarization_loss = -torch.mean(distance_from_center ** 2) * 6  # 降低力度避免極端化
        
        # 🎯 方法2: 添加 top-k 對比損失 - 強化最重要的邊
        if pred_adj_sigmoid.numel() > 4:  # 確保有足夠的元素
            flat_probs = pred_adj_sigmoid.flatten()
            k = max(2, len(flat_probs) // 4)  # 選擇前25%
            top_k_vals, _ = torch.topk(flat_probs, k)
            bottom_k_vals, _ = torch.topk(flat_probs, k, largest=False)
            
            # 鼓勵top-k接近1，bottom-k接近0
            top_contrast = -torch.mean((1 - top_k_vals) ** 2)
            bottom_contrast = -torch.mean(bottom_k_vals ** 2)
            contrast_enhance = (top_contrast + bottom_contrast) * 0.5
            polarization_loss += contrast_enhance
        
        # 🎯 新增: top-K/bottom-K 對比增強判別力 (來自改進方案)
        if pred_adj_sigmoid.numel() > 8:  # 確保有足夠的元素進行對比
            flat_adj = pred_adj_sigmoid.flatten()
            top_k = min(5, len(flat_adj) // 4)  # 選擇前25%或最多5個
            bottom_k = min(5, len(flat_adj) // 4)
            
            top_k_vals = torch.topk(flat_adj, top_k)[0]
            bottom_k_vals = torch.topk(flat_adj, bottom_k, largest=False)[0]
            
            # 適度強化判別力：top-K接近1，bottom-K接近0
            top_contrast_loss = torch.mean((1 - top_k_vals) ** 2) * 5
            bottom_contrast_loss = torch.mean(bottom_k_vals ** 2) * 5
            
            # 新增：適度分數差異損失
            if len(top_k_vals) > 0 and len(bottom_k_vals) > 0:
                score_diff_loss = torch.mean((top_k_vals[0] - bottom_k_vals[0]) ** 2) * 6
                polarization_loss += score_diff_loss
            
            # 添加到極化損失中
            polarization_loss += top_contrast_loss + bottom_contrast_loss
        
        # 🎯 改進2: 添加對比損失，鼓勵學習有意義的表示
        contrast_loss = 0
        if node_embedding.size(0) > 1:
            # 計算節點嵌入的相似性
            embedding_sim = torch.mm(F.normalize(node_embedding, dim=1), F.normalize(node_embedding, dim=1).t())
            # 與真實鄰接矩陣對比
            contrast_loss = F.mse_loss(embedding_sim, true_adj) * 0.1
        
        # 2. KAN 正則化損失 (應保留)
        kan_reg_loss = 0
        if hasattr(model, 'get_reg_loss'):
            # 確保 kan_reg_loss 是一個純量
            reg_loss = model.get_reg_loss()
            if isinstance(reg_loss, torch.Tensor):
                kan_reg_loss = reg_loss
            else: # 假設它是一個列表或元組
                kan_reg_loss = sum(reg_loss)

        # 🎯 改進3: 更有效的稀疏性損失
        if sparsity_lambda > 0:
            adj_probs_for_loss = pred_adj_sigmoid
            # 使用 L1 正則化而非簡單平均
            sparsity_loss = torch.mean(torch.abs(adj_probs_for_loss))
        else:
            sparsity_loss = torch.tensor(0.0, device=device)
        
        # 🎯 新增: 分數差異化的 margin-based 損失，強制 top 與 bottom 至少相差 margin
        margin_loss = 0
        flat_probs_for_margin = pred_adj_sigmoid.flatten()
        if flat_probs_for_margin.numel() >= 2:
            max_val = torch.max(flat_probs_for_margin)
            min_val = torch.min(flat_probs_for_margin)
            # 動態margin：小圖使用更高margin
            margin = 0.5 if flat_probs_for_margin.numel() < 50 else 0.3
            # 使用 hinge-style：max(0, margin - (max - min))
            # 從4.0提升到8.0以增強分數差異化
            margin_loss = torch.relu(margin - (max_val - min_val)) * 8.0

        # 🎯 新增: 訓練期特徵去相關正則化
        decor_loss = 0
        if node_embedding.size(0) > 1 and node_embedding.size(1) > 1:
            # 計算嵌入的協方差矩陣
            embeddings_mean = node_embedding.mean(dim=0)
            centered = node_embedding - embeddings_mean
            cov = (centered.T @ centered) / (centered.shape[0] - 1)
            # 懲罰非對角線元素（去相關）
            off_diag_cov = cov - torch.diag(cov.diag())
            decor_loss = torch.mean(off_diag_cov ** 2) * 0.1  # 小權重避免過度懲罰
        
        # 🎯 新增: 圖稀疏性損失 - 鼓勵平衡密度
        graph_sparsity_loss = 0
        if pred_adj_sigmoid.numel() > 0:
            # 計算圖密度
            graph_density = (pred_adj_sigmoid > 0.1).float().mean().item()
            # 目標密度 0.3-0.5，懲罰過密或過疏
            target_density = 0.4
            density_penalty = torch.abs(torch.tensor(graph_density - target_density)) * 2.0
            graph_sparsity_loss = density_penalty
        
        # 🎯 改進4: 調整損失權重確保有效學習（以重建為主，對比/極化為輔），並支持配置覆寫
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
        
        # 🎯 改進5: 確保損失在合理範圍內（僅在非有限或明顯退化時調整）
        if not torch.isfinite(total_loss):
            print(f"⚠️ 損失非有限 ({total_loss.item() if total_loss.numel()==1 else 'tensor'})，重設為重建導向")
            total_loss = recon_loss + kan_reg_loss
        elif total_loss.item() < 0 and recon_loss.item() < 0.01:
            # 僅在總損失為負且重建項極小時視為退化，避免噪音式警告
            print(f"⚠️ 損失退化 (total={total_loss.item():.4f}, recon={recon_loss.item():.4f})，切換為重建優先")
            total_loss = recon_loss + contrast_loss + (sparsity_lambda * sparsity_loss)
        
        # 反向傳播
        total_loss.backward()
        
        # 🔧 新增：全局梯度裁剪 - 基于您的建议
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 🔧 新增：KAN系数正则化
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
        
        # 記錄和打印
        training_history['loss'].append(total_loss.item())
        
        # 計算鄰接矩陣統計
        with torch.no_grad():
            if pred_adj is not None:
                # 使用前面已根據數值域決定的機率矩陣，避免再次sigmoid壓縮
                adj_probs = pred_adj_sigmoid
                adj_min = adj_probs.min().item()
                adj_max = adj_probs.max().item()
                adj_mean = adj_probs.mean().item()
            else:
                # KNN Baseline情況
                adj_min = 0.0
                adj_max = 1.0
                adj_mean = 0.5
            
            if pred_adj is not None:
                sparsity_01 = (adj_probs < 0.1).float().mean().item()
                sparsity_03 = (adj_probs < 0.3).float().mean().item()
                sparsity_05 = (adj_probs < 0.5).float().mean().item()
            else:
                # KNN Baseline情況
                sparsity_01 = 0.5
                sparsity_03 = 0.3
                sparsity_05 = 0.1
        
        training_history['adj_min'].append(adj_min)
        training_history['adj_max'].append(adj_max)
        training_history['adj_mean'].append(adj_mean)
        training_history['sparsity_01'].append(sparsity_01)
        training_history['sparsity_03'].append(sparsity_03)
        training_history['sparsity_05'].append(sparsity_05)

        # 🔧 階段1：驗證和早停檢查 / Validation and early stopping check
        if val_data is not None and (epoch + 1) % 5 == 0:  # 每5個epoch驗證一次
            model.eval()
            with torch.no_grad():
                # 取出 val_edge_index 後，先過濾再送模型
                val_node_features = val_data['node_features'].to(device)
                val_edge_index = val_data['edge_index'].to(device)

                val_used_edge_index = val_edge_index
                if val_edge_index.size(1) > 0:
                    valid_mask = (val_edge_index[0] < val_node_features.size(0)) & (val_edge_index[1] < val_node_features.size(0))
                    val_valid_edge_index = val_edge_index[:, valid_mask]
                    val_used_edge_index = val_valid_edge_index if val_valid_edge_index.size(1) > 0 else torch.empty((2,0), dtype=torch.long, device=device)

                val_embeddings, val_pred_adj = model(val_node_features, val_used_edge_index)
                
                # 計算驗證精度
                if val_pred_adj is not None:
                    val_adj_sigmoid = torch.sigmoid(val_pred_adj)
                    val_density = (val_adj_sigmoid > 0.1).float().mean().item()
                else:
                    # KNN Baseline情況
                    val_density = 0.3
                
                # 計算PageRank排序
                from ..graph_heads.page_rank import page_rank
                if val_pred_adj is not None:
                    val_ranks = page_rank(val_adj_sigmoid.cpu().numpy())
                else:
                    # KNN Baseline情況，使用單位矩陣
                    val_ranks = page_rank(torch.eye(val_embeddings.size(0)).numpy())
                
                # 計算precision@1
                if val_ground_truth and val_ranks:
                    val_precision = 1.0 if val_ground_truth in val_ranks[:1] else 0.0
                else:
                    val_precision = 0.0
                
                # 更新最佳驗證指標
                if val_precision > best_val_metric:
                    best_val_metric = val_precision
                    patience_counter = 0
                    # 保存最佳模型
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                print(f"  Val: Precision@1={val_precision:.3f}, Density={val_density:.3f}, Best={best_val_metric:.3f}, Patience={patience_counter}/{config.patience}")
            
            model.train()
        
        # 早停檢查
        if early_stopping(total_loss.item()) or patience_counter >= config.patience:
            if patience_counter >= config.patience:
                print(f"🛑 Early stopping at epoch {epoch+1} (validation patience exceeded)")
                # 恢復最佳模型
                if 'best_model_state' in locals():
                    model.load_state_dict(best_model_state)
            else:
                print(f"🛑 Early stopping at epoch {epoch+1} (loss plateau)")
            break
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # 🎯 改進6: 更詳細的訓練信息 - 添加極化損失跟蹤和密度監控
            adj_density = pred_adj_sigmoid.mean().item()
            print(f"Epoch [{epoch+1}/{config.num_epochs}], Total: {total_loss.item():.6f}, Recon: {recon_loss.item():.4f}, "
                  f"Polar: {polarization_loss.item():.4f}, Contrast: {contrast_loss:.4f}, KAN: {kan_reg_loss:.4f}, Sparsity: {sparsity_loss.item():.6f} | "
                  f"Adj Probs(min/max/mean): {adj_min:.4f}/{adj_max:.4f}/{adj_mean:.4f} | "
                  f"Graph Sparsity: {sparsity_03:.4f} (0.1:{sparsity_01:.3f}, 0.3:{sparsity_03:.3f}, 0.5:{sparsity_05:.3f}) | "
                  f"Adj Density: {adj_density:.3f}")

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

        used_edge_index = edge_index  # 預設
        if edge_index.size(1) > 0:
            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            valid_edge_index = edge_index[:, valid_edges_mask]

            if valid_edge_index.size(1) > 0:
                src = valid_edge_index[0]; dst = valid_edge_index[1]
                ones = torch.ones_like(src, dtype=target_adj.dtype)
                target_adj.index_put_((src, dst), ones, accumulate=True)
                target_adj.index_put_((dst, src), ones, accumulate=True)
                used_edge_index = valid_edge_index  # 關鍵：之後都用過濾後的邊
            else:
                print(f"⚠️ {phase_name}: 所有邊索引都超出範圍，使用單位矩陣")
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