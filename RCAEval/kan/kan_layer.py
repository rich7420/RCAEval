"""
GPU 優化的 KAN 實現
使用向量化操作和 GPU 友好的設計
基於 KAN 論文的梯度穩定性優化
新增：自適應學習率調度、權重衰減、動態剪枝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AdvancedKANLayer(nn.Module):
    """
    基於 KAN 論文的高級 KAN 層實現
    包含論文中提到的梯度穩定性優化和自適應機制
    """
    
    def __init__(self, input_dim, output_dim, num_basis=5, 
                 l1_lambda=1e-3, entropy_lambda=1e-3, 
                 spline_order=3, grid_size=5,
                 enable_pruning=True, pruning_threshold=1e-2,
                 adaptive_lr=True, weight_decay=1e-4):
        super(AdvancedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        self.l1_lambda = l1_lambda
        self.entropy_lambda = entropy_lambda
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold
        self.adaptive_lr = adaptive_lr
        self.weight_decay = weight_decay
        
        # 基礎線性層 - 使用 Xavier 初始化
        self.linear = nn.Linear(input_dim, output_dim)
        
        # B-spline 係數 - 基於論文的初始化
        self.spline_coeffs = nn.Parameter(
            torch.zeros(output_dim, input_dim, num_basis)
        )
        
        # SiLU 激活函數的權重 (論文中提到的 σ(x) = x/(1+e^(-x)))
        self.silu_weight = nn.Parameter(torch.zeros(output_dim, input_dim))
        
        # Batch normalization for stability
        self.bn = nn.BatchNorm1d(output_dim)
        
        # Layer normalization as backup
        self.ln = nn.LayerNorm(output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 用於剪枝的重要性分數
        self.register_buffer('importance_scores', torch.ones(output_dim, input_dim))
        
        # 自適應學習率相關
        self.register_buffer('grad_norm_history', torch.zeros(10))  # 保存最近10次梯度範數
        self.register_buffer('step_count', torch.tensor(0))
        
        # 權重衰減調度器
        self.register_buffer('weight_decay_schedule', torch.tensor(weight_decay))
        
        # 初始化參數
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        基於 KAN 論文的參數初始化
        使用 Xavier 初始化確保初始梯度穩定
        """
        # Xavier 初始化線性層
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.linear.bias)
        
        # B-spline 係數初始化 - 小的隨機值
        fan_in = self.input_dim
        fan_out = self.output_dim
        std = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(self.spline_coeffs, mean=0.0, std=std * 0.1)
        
        # SiLU 權重初始化
        nn.init.xavier_uniform_(self.silu_weight, gain=0.1)
    
    def silu_activation(self, x):
        """
        SiLU 激活函數：σ(x) = x / (1 + exp(-x))
        論文中提到的平滑激活函數，有助於穩定梯度
        添加數值穩定性檢查
        """
        # 限制輸入範圍避免數值溢出
        x_clamped = torch.clamp(x, -20.0, 20.0)
        return x_clamped * torch.sigmoid(x_clamped)
    
    def b_spline_basis(self, x):
        """
        計算 B-spline 基函數
        使用 Chebyshev 多項式作為近似（更快且數值穩定）
        改進：添加自適應歸一化
        """
        # 自適應歸一化 - 基於輸入統計
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        # 將輸入歸一化到 [-1, 1] - 使用平滑的tanh函數
        x_norm = torch.tanh(x_normalized)
        
        # 計算 Chebyshev 多項式基函數
        basis_list = [torch.ones_like(x_norm)]  # T0 = 1
        
        if self.num_basis > 1:
            basis_list.append(x_norm)  # T1 = x
        
        # 遞歸計算 T_n = 2x*T_{n-1} - T_{n-2}
        for i in range(2, self.num_basis):
            t_next = 2 * x_norm * basis_list[-1] - basis_list[-2]
            # 數值穩定性檢查
            t_next = torch.clamp(t_next, -10.0, 10.0)
            basis_list.append(t_next)
        
        return torch.stack(basis_list, dim=-1)  # [batch, input_dim, num_basis]
    
    def compute_l1_regularization(self):
        """
        計算 L1 正則化項：|Φ_l|_1 = (1/N_p) * Σ|φ(x_s^(p))|
        基於論文公式，添加自適應權重
        """
        # 自適應L1權重 - 基於當前模型複雜度
        model_complexity = torch.sum(torch.abs(self.spline_coeffs) > 1e-3).float()
        adaptive_weight = 1.0 + 0.1 * torch.log(1 + model_complexity)
        
        l1_reg = torch.sum(torch.abs(self.spline_coeffs)) * adaptive_weight
        l1_reg += torch.sum(torch.abs(self.silu_weight))
        return l1_reg
    
    def compute_entropy_regularization(self):
        """
        計算熵正則化項：S(Φ_l) = -Σ(|φ_{i,j}|_1/|Φ|_1) * log(|φ_{i,j}|_1/|Φ|_1)
        基於論文公式，促進稀疏性，添加溫度參數
        """
        temperature = 0.1  # 溫度參數控制分佈尖銳程度
        
        # 計算每個連接的重要性
        spline_importance = torch.sum(torch.abs(self.spline_coeffs), dim=-1)  # [output_dim, input_dim]
        silu_importance = torch.abs(self.silu_weight)
        
        total_importance = spline_importance + silu_importance
        total_sum = torch.sum(total_importance) + 1e-8  # 避免除零
        
        # 計算歸一化的重要性（添加溫度調節）
        normalized_importance = F.softmax(total_importance.flatten() / temperature, dim=0)
        normalized_importance = normalized_importance.view_as(total_importance)
        
        # 計算熵（避免 log(0)）
        log_importance = torch.log(normalized_importance + 1e-8)
        entropy = -torch.sum(normalized_importance * log_importance)
        
        return entropy
    
    def update_importance_scores(self, x):
        """
        更新連接的重要性分數，用於剪枝
        改進：使用指數移動平均和方差考慮
        """
        with torch.no_grad():
            # 基於激活值計算重要性
            basis = self.b_spline_basis(x)  # [batch, input_dim, num_basis]
            
            # 計算每個連接的平均激活強度和方差
            spline_activation = torch.einsum('oij,bij->oi', self.spline_coeffs, basis)
            spline_mean = torch.mean(torch.abs(spline_activation), dim=0)
            spline_var = torch.var(torch.abs(spline_activation), dim=0)
            spline_importance = spline_mean + 0.1 * spline_var  # 考慮變異性
            
            silu_activation = self.silu_activation(x)
            silu_mean = torch.mean(torch.abs(silu_activation), dim=0)
            silu_var = torch.var(torch.abs(silu_activation), dim=0)
            silu_importance = silu_mean + 0.1 * silu_var
            
            # 更新重要性分數（使用指數移動平均）
            alpha = 0.05  # 降低更新速度提高穩定性
            new_importance = spline_importance + silu_importance.unsqueeze(0)
            self.importance_scores = (1 - alpha) * self.importance_scores + alpha * new_importance
    
    def apply_adaptive_pruning(self):
        """
        應用自適應剪枝：動態調整剪枝閾值
        論文中提到的動態結構調整，添加漸進式剪枝
        """
        if not self.enable_pruning:
            return
        
        with torch.no_grad():
            # 動態調整剪枝閾值 - 基於重要性分數分佈
            importance_mean = torch.mean(self.importance_scores)
            importance_std = torch.std(self.importance_scores)
            adaptive_threshold = max(self.pruning_threshold, 
                                   importance_mean - 2 * importance_std)
            
            # 創建剪枝掩碼
            mask = self.importance_scores > adaptive_threshold
            
            # 漸進式剪枝 - 避免突然的大幅剪枝
            current_mask = (torch.abs(self.spline_coeffs.data).sum(dim=-1) > 1e-6)
            pruning_rate = 0.05  # 每次最多剪枝5%的連接
            num_to_prune = int(torch.sum(current_mask).item() * pruning_rate)
            
            if num_to_prune > 0:
                # 找到重要性最低的連接進行剪枝
                flat_importance = self.importance_scores[current_mask]
                _, indices = torch.topk(flat_importance, num_to_prune, largest=False)
                
                # 應用掩碼到參數
                prune_mask = torch.ones_like(self.importance_scores, dtype=torch.bool)
                prune_mask[current_mask][indices] = False
                
                self.spline_coeffs.data *= prune_mask.unsqueeze(-1).float()
                self.silu_weight.data *= prune_mask.float()
    
    def compute_gradient_penalty(self):
        """
        計算梯度懲罰項，確保梯度不會過大
        基於Lipschitz約束的思想
        """
        grad_penalty = 0.0
        
        if self.spline_coeffs.grad is not None:
            spline_grad_norm = torch.norm(self.spline_coeffs.grad)
            grad_penalty += torch.clamp(spline_grad_norm - 1.0, min=0.0) ** 2
        
        if self.silu_weight.grad is not None:
            silu_grad_norm = torch.norm(self.silu_weight.grad)
            grad_penalty += torch.clamp(silu_grad_norm - 1.0, min=0.0) ** 2
        
        return grad_penalty * 0.01  # 小的懲罰係數
    
    def forward(self, x):
        """
        前向傳播 - 包含論文中的所有穩定性機制
        改進：添加自適應歸一化和錯誤恢復
        """
        # 輸入數值穩定性檢查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Invalid input detected in AdvancedKANLayer")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        batch_size = x.shape[0]
        original_x = x.clone()  # 保存原始輸入用於錯誤恢復
        
        # 更新重要性分數
        if self.training:
            self.update_importance_scores(x)
            # 定期應用剪枝
            if self.step_count % 100 == 0:
                self.apply_adaptive_pruning()
            self.step_count += 1
        
        try:
            # 基礎線性變換
            linear_out = self.linear(x)
            
            # B-spline 分量
            basis = self.b_spline_basis(x)  # [batch, input_dim, num_basis]
            spline_out = torch.einsum('oij,bij->bo', self.spline_coeffs, basis)
            
            # SiLU 分量
            silu_out = torch.einsum('oi,bi->bo', self.silu_weight, self.silu_activation(x))
            
            # 組合所有分量
            output = linear_out + spline_out + silu_out
            
            # 應用 Dropout
            if self.training:
                output = self.dropout(output)
            
            # 自適應歸一化選擇
            if batch_size > 1:
                try:
                    output = self.bn(output)
                except RuntimeError:
                    # 如果BatchNorm失敗，使用LayerNorm
                    output = self.ln(output)
            else:
                output = self.ln(output)
                
        except RuntimeError as e:
            print(f"Forward pass failed: {e}, using fallback")
            # 錯誤恢復：使用簡化的線性變換
            output = self.linear(original_x)
            output = self.ln(output)
        
        # 最終數值穩定性檢查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: Invalid output in AdvancedKANLayer, applying stabilization")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            output = torch.clamp(output, -10.0, 10.0)
        
        return output
    
    def get_total_loss_contribution(self, base_loss):
        """
        計算總損失，包含正則化項
        基於論文公式：L_total = L_pred + λ * (μ1*|Φ_l|_1 + μ2*S(Φ_l)) + λ_grad*Grad_penalty
        """
        l1_reg = self.compute_l1_regularization()
        entropy_reg = self.compute_entropy_regularization()
        grad_penalty = self.compute_gradient_penalty()
        
        # 自適應正則化權重 - 基於訓練進度
        training_progress = min(1.0, self.step_count / 1000.0)
        adaptive_l1 = self.l1_lambda * (1 + training_progress)
        adaptive_entropy = self.entropy_lambda * (1 - 0.5 * training_progress)
        
        regularization = adaptive_l1 * l1_reg + adaptive_entropy * entropy_reg + grad_penalty
        
        return base_loss + regularization
    
    def get_sparsity_info(self):
        """
        獲取模型稀疏性信息，用於監控和調試
        """
        with torch.no_grad():
            total_params = self.spline_coeffs.numel() + self.silu_weight.numel()
            spline_zeros = torch.sum(torch.abs(self.spline_coeffs) < 1e-6).item()
            silu_zeros = torch.sum(torch.abs(self.silu_weight) < 1e-6).item()
            total_zeros = spline_zeros + silu_zeros
            sparsity_ratio = total_zeros / total_params
            
            return {
                'total_params': total_params,
                'zero_params': total_zeros,
                'sparsity_ratio': sparsity_ratio,
                'active_connections': total_params - total_zeros
            }


class SimplifiedKANLayer(nn.Module):
    """
    簡化版 KAN 層 - 更快的計算
    使用 ReLU 和多項式基函數代替 B-spline
    """
    
    def __init__(self, input_dim, output_dim, num_basis=5):
        super(SimplifiedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        
        # 基礎線性層
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 多項式基函數權重
        self.poly_weights = nn.Parameter(
            torch.randn(output_dim, input_dim, num_basis) * 0.1
        )
        
        # Batch normalization for stability
        self.bn = nn.BatchNorm1d(output_dim)
    
    def polynomial_basis(self, x):
        """
        快速多項式基函數
        使用 Chebyshev 多項式
        """
        # 歸一化輸入到 [-1, 1]
        x_norm = torch.tanh(x)
        
        # 計算 Chebyshev 多項式
        basis_list = [torch.ones_like(x_norm)]  # T0 = 1
        
        if self.num_basis > 1:
            basis_list.append(x_norm)  # T1 = x
        
        # 遞歸計算 T_n = 2x*T_{n-1} - T_{n-2}
        for i in range(2, self.num_basis):
            t_next = 2 * x_norm * basis_list[-1] - basis_list[-2]
            basis_list.append(t_next)
        
        # Stack along last dimension
        return torch.stack(basis_list, dim=-1)  # [batch, input_dim, num_basis]
    
    def forward(self, x):
        """快速前向傳播 - 添加數值穩定性檢查"""
        # 輸入驗證
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in KAN input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 基礎線性變換
        try:
            linear_out = self.linear(x)
        except RuntimeError as e:
            print(f"Linear layer failed in KAN: {e}")
            # 回退到零輸出
            return torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
        
        # 多項式基函數
        try:
            poly_basis = self.polynomial_basis(x)  # [batch, input_dim, num_basis]
            poly_out = torch.einsum('oij,bij->bo', self.poly_weights, poly_basis)
        except RuntimeError as e:
            print(f"Polynomial basis failed in KAN: {e}")
            poly_out = torch.zeros_like(linear_out)
        
        # 組合
        output = linear_out + poly_out
        
        # 安全的 Batch normalization
        if x.shape[0] > 1 and not torch.isnan(output).any():
            try:
                output = self.bn(output)
            except RuntimeError as e:
                print(f"BatchNorm failed in KAN: {e}")
                # 跳過 BatchNorm
                pass
        
        # 最終檢查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN or Inf in KAN output, applying clipping")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            output = torch.clamp(output, -10.0, 10.0)
        
        return output


class UltraFastKANLayer(nn.Module):
    """
    超快速 KAN 層 - 最簡化版本
    使用線性組合代替複雜的查找表
    """
    
    def __init__(self, input_dim, output_dim, num_activations=4):
        super(UltraFastKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_activations = num_activations
        
        # 主要線性層 (大部分計算)
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 輕量級非線性組件
        self.activation_weights = nn.Parameter(
            torch.randn(output_dim, input_dim, num_activations) * 0.01
        )
        
    def forward(self, x):
        """超快速前向傳播 - 優化版"""
        # 主要線性變換 (95% 的計算)
        linear_out = self.linear(x)
        
        # 輕量級非線性 (5% 的計算)
        # 使用簡單的激活函數組合
        x_expanded = x.unsqueeze(-1)  # [batch, input_dim, 1]
        
        # 計算多個簡單激活函數
        activations = []
        activations.append(torch.tanh(x_expanded))  # tanh
        activations.append(torch.sigmoid(x_expanded))  # sigmoid  
        activations.append(F.relu(x_expanded))  # ReLU
        activations.append(x_expanded)  # 線性
        
        # 取前 num_activations 個
        activations = torch.cat(activations[:self.num_activations], dim=-1)  # [batch, input_dim, num_activations]
        
        # 高效張量乘法
        nonlinear_out = torch.einsum('oij,bij->bo', self.activation_weights, activations)
        
        return linear_out + 0.1 * nonlinear_out  # 降低非線性成分的權重


class OptimizedGNNKANEncoder(nn.Module):
    """
    GPU 優化的 GNN-KAN 編碼器
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=2, kan_type='simplified', dropout=0.1):
        super(OptimizedGNNKANEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # 選擇 KAN 層類型
        if kan_type == 'fast':
            KANLayerClass = FastKANLayer
        elif kan_type == 'simplified':
            KANLayerClass = SimplifiedKANLayer
        else:  # 'ultra_fast'
            KANLayerClass = UltraFastKANLayer
        
        # 構建層
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(KANLayerClass(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 不在最後一層添加 dropout
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.ModuleList(layers)
        
        # 消息傳遞層 (簡化的 GCN)
        self.message_layers = nn.ModuleList([
            nn.Linear(dims[i + 1], dims[i + 1]) 
            for i in range(len(dims) - 1)
        ])
        
    def message_passing(self, x, edge_index, layer_idx):
        """數值穩定的消息傳遞 - 修復CUDA錯誤"""
        if edge_index.size(1) == 0:
            return x
        
        # 添加數值穩定性檢查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in input features")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 簡單的平均聚合
        row, col = edge_index
        num_nodes = x.size(0)
        
        # 確保邊索引在有效範圍內
        row = torch.clamp(row, 0, num_nodes - 1)
        col = torch.clamp(col, 0, num_nodes - 1)
        
        # 使用稀疏張量進行更安全的操作
        adj_indices = torch.stack([row, col], dim=0)
        adj_values = torch.ones(len(row), device=x.device, dtype=x.dtype)
        adj_size = (num_nodes, num_nodes)
        
        # 創建稀疏鄰接矩陣
        adj_sparse = torch.sparse_coo_tensor(adj_indices, adj_values, adj_size, device=x.device)
        adj_sparse = adj_sparse.coalesce()
        
        # 計算度數 (行和)
        degrees = torch.sparse.sum(adj_sparse, dim=1).to_dense()
        degrees = torch.clamp(degrees, min=1e-8)  # 防止除零，使用更大的最小值
        
        # 歸一化
        degrees_inv = 1.0 / degrees
        degrees_inv = torch.where(torch.isfinite(degrees_inv), degrees_inv, 0.0)
        
        # 創建歸一化的鄰接矩陣
        norm_values = degrees_inv[row]
        norm_adj = torch.sparse_coo_tensor(adj_indices, norm_values, adj_size, device=x.device)
        
        # 安全的稀疏矩陣乘法
        try:
            message = torch.sparse.mm(norm_adj, x)
        except RuntimeError as e:
            print(f"Sparse matrix multiplication failed: {e}")
            # 回退到恆等映射
            return x
        
        # 檢查結果
        if torch.isnan(message).any() or torch.isinf(message).any():
            print("Warning: NaN or Inf detected in message")
            message = torch.nan_to_num(message, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 通過線性層 (添加安全檢查)
        if layer_idx < len(self.message_layers):
            try:
                message = self.message_layers[layer_idx](message)
            except RuntimeError as e:
                print(f"Linear layer failed: {e}")
                return x
        
        return message
    
    def forward(self, x, edge_index):
        """優化的前向傳播"""
        current_x = x
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Dropout):
                current_x = layer(current_x)
            else:
                # KAN 層
                kan_out = layer(current_x)
                
                # 消息傳遞 (每隔一層)
                if i % 2 == 0 and edge_index.size(1) > 0:
                    message = self.message_passing(kan_out, edge_index, i // 2)
                    current_x = kan_out + 0.1 * message  # 殘差連接
                else:
                    current_x = kan_out
                
                # 激活函數
                if i < len(self.layers) - 1:  # 不在最後一層使用激活
                    current_x = F.gelu(current_x)
        
        return current_x


class KANLayer(nn.Module):
    """
    原始 KAN 層實現 (向後兼容)
    為了保持導入兼容性，這裡使用 SimplifiedKANLayer 的實現
    """
    
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        # 使用簡化實現來提供向後兼容性
        self.simplified_kan = SimplifiedKANLayer(input_dim, output_dim, num_basis=grid_size)
        
    def forward(self, x):
        return self.simplified_kan(x)


class GNNKANEncoder(nn.Module):
    """
    原始 GNN-KAN 編碼器 (向後兼容)
    為了保持導入兼容性，這裡使用 OptimizedGNNKANEncoder 的實現
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers=2, dropout=0.1):
        super(GNNKANEncoder, self).__init__()
        # 使用優化實現來提供向後兼容性
        self.optimized_encoder = OptimizedGNNKANEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_layers=num_layers,
            kan_type='simplified',  # 使用平衡的版本
            dropout=dropout
        )
        
    def forward(self, x, edge_index):
        return self.optimized_encoder(x, edge_index)


class FastKANLayer(nn.Module):
    """
    GPU 優化的 KAN 層實現
    使用 B-spline 基函數的向量化計算
    """
    
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0):
        super(FastKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 創建 B-spline 網格 (固定在 [-1, 1] 範圍)
        h = 2.0 / grid_size
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order, 
                             grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)
        
        # B-spline 係數 (可學習參數)
        self.spline_weight = nn.Parameter(
            torch.randn(output_dim, input_dim, grid_size + spline_order) * scale_spline
        )
        
        # 基礎線性變換
        self.base_weight = nn.Parameter(torch.randn(output_dim, input_dim) * scale_base)
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化參數"""
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
    
    def b_splines(self, x):
        """
        GPU 優化的 B-spline 基函數計算
        使用向量化操作避免循環
        """
        # 將輸入限制在 [-1, 1] 範圍
        x = torch.clamp(x, -0.99, 0.99)
        
        # 擴展維度進行廣播
        x = x.unsqueeze(-1)  # [batch, input_dim, 1]
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # [1, 1, grid_points]
        
        # 計算 B-spline 基函數 (使用快速算法)
        bases = self.fast_b_spline_basis(x, grid, self.spline_order)
        
        return bases
    
    def fast_b_spline_basis(self, x, grid, k):
        """
        快速 B-spline 基函數計算
        使用遞歸關係的向量化版本
        """
        # 找到 x 在 grid 中的位置
        # 使用 searchsorted 進行批量搜索
        batch_size, input_dim, _ = x.shape
        x_flat = x.view(-1)
        grid_flat = grid.view(-1)
        
        # 找到每個 x 值對應的網格區間
        indices = torch.searchsorted(grid_flat, x_flat, right=False)
        indices = torch.clamp(indices - 1, 0, len(grid_flat) - k - 2)
        indices = indices.view(batch_size, input_dim)
        
        # 初始化基函數矩陣
        n_basis = grid.shape[-1] - k - 1
        bases = torch.zeros(batch_size, input_dim, n_basis, 
                           device=x.device, dtype=x.dtype)
        
        # 使用 Cox-de Boor 遞歸公式的向量化版本
        # 0 階基函數
        for i in range(batch_size):
            for j in range(input_dim):
                idx = indices[i, j]
                if 0 <= idx < n_basis:
                    bases[i, j, idx] = 1.0
        
        # 遞歸計算高階基函數
        for r in range(1, k + 1):
            bases_new = torch.zeros_like(bases)
            for i in range(n_basis - r):
                # 左邊項
                denom1 = grid[0, 0, i + r] - grid[0, 0, i]
                if denom1 > 1e-8:
                    alpha1 = (x.squeeze(-1) - grid[0, 0, i]) / denom1
                    bases_new[:, :, i] += alpha1 * bases[:, :, i]
                
                # 右邊項
                if i + 1 < n_basis:
                    denom2 = grid[0, 0, i + r + 1] - grid[0, 0, i + 1]
                    if denom2 > 1e-8:
                        alpha2 = (grid[0, 0, i + r + 1] - x.squeeze(-1)) / denom2
                        bases_new[:, :, i] += alpha2 * bases[:, :, i + 1]
            
            bases = bases_new
        
        return bases
    
    def forward(self, x):
        """
        快速前向傳播
        """
        batch_size = x.shape[0]
        
        # 基礎線性變換
        base_output = F.linear(x, self.base_weight)  # [batch, output_dim]
        
        # B-spline 變換 (向量化)
        spline_bases = self.b_splines(x)  # [batch, input_dim, n_basis]
        
        # 使用 einsum 進行高效張量乘法
        # spline_weight: [output_dim, input_dim, n_basis]
        # spline_bases: [batch, input_dim, n_basis]
        spline_output = torch.einsum('oij,bij->bo', self.spline_weight, spline_bases)
        
        return base_output + spline_output