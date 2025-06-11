"""
KAN Components: Core KAN Layer Implementations
核心 KAN 層實現 - 從原始 kan 模組整合而來
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
        self.register_buffer('grad_norm_history', torch.zeros(10))
        self.register_buffer('step_count', torch.tensor(0))
        
        # 權重衰減調度器
        self.register_buffer('weight_decay_schedule', torch.tensor(weight_decay))
        
        # 初始化參數
        self.reset_parameters()
    
    def reset_parameters(self):
        """基於 KAN 論文的參數初始化"""
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
        """SiLU 激活函數：σ(x) = x / (1 + exp(-x))"""
        x_clamped = torch.clamp(x, -20.0, 20.0)
        return x_clamped * torch.sigmoid(x_clamped)
    
    def b_spline_basis(self, x):
        """計算 B-spline 基函數"""
        # 自適應歸一化 - 基於輸入統計
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        # 將輸入歸一化到 [-1, 1] - 使用平滑的tanh函數
        x_norm = torch.tanh(x_normalized)
        
        # 計算 Chebyshev 多項式基函數
        basis_list = [torch.ones_like(x_norm)]
        
        if self.num_basis > 1:
            basis_list.append(x_norm)
        
        for i in range(2, self.num_basis):
            t_next = 2 * x_norm * basis_list[-1] - basis_list[-2]
            t_next = torch.clamp(t_next, -10.0, 10.0)
            basis_list.append(t_next)
        
        return torch.stack(basis_list, dim=-1)
    
    def forward(self, x):
        """前向傳播"""
        # 輸入數值穩定性檢查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Invalid input detected in AdvancedKANLayer")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        batch_size = x.shape[0]
        original_x = x.clone()
        
        try:
            # 基礎線性變換
            linear_out = self.linear(x)
            
            # B-spline 分量
            basis = self.b_spline_basis(x)
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
                    output = self.ln(output)
            else:
                output = self.ln(output)
                
        except RuntimeError as e:
            print(f"Forward pass failed: {e}, using fallback")
            output = self.linear(original_x)
            output = self.ln(output)
        
        # 最終數值穩定性檢查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: Invalid output in AdvancedKANLayer, applying stabilization")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            output = torch.clamp(output, -10.0, 10.0)
        
        return output


class SimplifiedKANLayer(nn.Module):
    """簡化版 KAN 層 - 更快的計算"""
    
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
        """快速多項式基函數"""
        x_norm = torch.tanh(x)
        
        basis_list = [torch.ones_like(x_norm)]
        
        if self.num_basis > 1:
            basis_list.append(x_norm)
        
        for i in range(2, self.num_basis):
            t_next = 2 * x_norm * basis_list[-1] - basis_list[-2]
            basis_list.append(t_next)
        
        return torch.stack(basis_list, dim=-1)
    
    def forward(self, x):
        """快速前向傳播"""
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in KAN input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            linear_out = self.linear(x)
        except RuntimeError as e:
            print(f"Linear layer failed in KAN: {e}")
            return torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
        
        try:
            poly_basis = self.polynomial_basis(x)
            poly_out = torch.einsum('oij,bij->bo', self.poly_weights, poly_basis)
        except RuntimeError as e:
            print(f"Polynomial basis failed in KAN: {e}")
            poly_out = torch.zeros_like(linear_out)
        
        output = linear_out + poly_out
        
        if x.shape[0] > 1 and not torch.isnan(output).any():
            try:
                output = self.bn(output)
            except RuntimeError as e:
                print(f"BatchNorm failed in KAN: {e}")
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN or Inf in KAN output, applying clipping")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            output = torch.clamp(output, -10.0, 10.0)
        
        return output


class UltraFastKANLayer(nn.Module):
    """超快速 KAN 層 - 最簡化版本"""
    
    def __init__(self, input_dim, output_dim, num_activations=4):
        super(UltraFastKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_activations = num_activations
        
        # 主要線性層
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 輕量級非線性組件
        self.activation_weights = nn.Parameter(
            torch.randn(output_dim, input_dim, num_activations) * 0.01
        )
        
    def forward(self, x):
        """超快速前向傳播"""
        linear_out = self.linear(x)
        
        x_expanded = x.unsqueeze(-1)
        
        activations = []
        activations.append(torch.tanh(x_expanded))
        activations.append(torch.sigmoid(x_expanded))
        activations.append(F.relu(x_expanded))
        activations.append(x_expanded)
        
        activations = torch.cat(activations[:self.num_activations], dim=-1)
        
        nonlinear_out = torch.einsum('oij,bij->bo', self.activation_weights, activations)
        
        return linear_out + 0.1 * nonlinear_out


class FastKANLayer(nn.Module):
    """GPU 優化的 KAN 層實現"""
    
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3, 
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0):
        super(FastKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 創建 B-spline 網格
        h = 2.0 / grid_size
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order, 
                             grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)
        
        # B-spline 係數
        self.spline_weight = nn.Parameter(
            torch.randn(output_dim, input_dim, grid_size + spline_order) * scale_spline
        )
        
        # 基礎線性變換
        self.base_weight = nn.Parameter(torch.randn(output_dim, input_dim) * scale_base)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化參數"""
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
    
    def forward(self, x):
        """前向傳播"""
        batch_size = x.shape[0]
        
        # 基礎線性變換
        base_output = F.linear(x, self.base_weight)
        
        # 簡化的 B-spline 計算
        x_clamped = torch.clamp(x, -0.99, 0.99)
        
        # 使用簡化的多項式基函數代替復雜的 B-spline
        x_norm = torch.tanh(x_clamped)
        basis_list = [torch.ones_like(x_norm)]
        
        grid_size_actual = self.spline_weight.size(-1)
        for i in range(1, min(5, grid_size_actual)):  # 限制基函數數量
            if i == 1:
                basis_list.append(x_norm)
            else:
                t_next = 2 * x_norm * basis_list[-1] - basis_list[-2]
                basis_list.append(t_next)
        
        # 處理維度不匹配
        while len(basis_list) < grid_size_actual:
            basis_list.append(torch.zeros_like(x_norm))
        
        basis_tensor = torch.stack(basis_list[:grid_size_actual], dim=-1)
        
        # 計算 spline 輸出
        spline_output = torch.einsum('oij,bij->bo', self.spline_weight, basis_tensor)
        
        return base_output + spline_output


class OptimizedGNNKANEncoder(nn.Module):
    """GPU 優化的 GNN-KAN 編碼器"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=2, kan_grid_size=5, kan_spline_order=3, dropout=0.1):
        super(OptimizedGNNKANEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # 構建層
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(SimplifiedKANLayer(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.ModuleList(layers)
        
        # 消息傳遞層
        self.message_layers = nn.ModuleList([
            nn.Linear(dims[i + 1], dims[i + 1]) 
            for i in range(len(dims) - 1)
        ])
        
    def message_passing(self, x, edge_index, layer_idx):
        """數值穩定的消息傳遞"""
        if edge_index.size(1) == 0:
            return x
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in input features")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        row, col = edge_index
        num_nodes = x.size(0)
        
        row = torch.clamp(row, 0, num_nodes - 1)
        col = torch.clamp(col, 0, num_nodes - 1)
        
        adj_indices = torch.stack([row, col], dim=0)
        adj_values = torch.ones(len(row), device=x.device, dtype=x.dtype)
        adj_size = (num_nodes, num_nodes)
        
        adj_sparse = torch.sparse_coo_tensor(adj_indices, adj_values, adj_size, device=x.device)
        adj_sparse = adj_sparse.coalesce()
        
        degrees = torch.sparse.sum(adj_sparse, dim=1).to_dense()
        degrees = torch.clamp(degrees, min=1e-8)
        
        degrees_inv = 1.0 / degrees
        degrees_inv = torch.where(torch.isfinite(degrees_inv), degrees_inv, 0.0)
        
        norm_values = degrees_inv[row]
        norm_adj = torch.sparse_coo_tensor(adj_indices, norm_values, adj_size, device=x.device)
        
        try:
            message = torch.sparse.mm(norm_adj, x)
        except RuntimeError as e:
            print(f"Sparse matrix multiplication failed: {e}")
            return x
        
        if torch.isnan(message).any() or torch.isinf(message).any():
            print("Warning: NaN or Inf detected in message")
            message = torch.nan_to_num(message, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
                kan_out = layer(current_x)
                
                if i % 2 == 0 and edge_index.size(1) > 0:
                    message = self.message_passing(kan_out, edge_index, i // 2)
                    current_x = kan_out + 0.1 * message
                else:
                    current_x = kan_out
                
                if i < len(self.layers) - 1:
                    current_x = F.gelu(current_x)
        
        return current_x


# 向後兼容的別名
class KANLayer(SimplifiedKANLayer):
    """原始 KAN 層實現 (向後兼容)"""
    
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__(input_dim, output_dim, num_basis=grid_size)


class GNNKANEncoder(OptimizedGNNKANEncoder):
    """原始 GNN-KAN 編碼器 (向後兼容)"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers=2, dropout=0.1):
        super(GNNKANEncoder, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )


# 注意：compute_service_criticality_weights 函數已移動到 utils.py 中統一管理
# 如需使用，請從 ..utils 導入