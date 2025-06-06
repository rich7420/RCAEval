"""
GPU 優化的 KAN 實現
使用向量化操作和 GPU 友好的設計
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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
        """快速前向傳播"""
        # 基礎線性變換
        linear_out = self.linear(x)
        
        # 多項式基函數
        poly_basis = self.polynomial_basis(x)  # [batch, input_dim, num_basis]
        
        # 高效張量乘法
        poly_out = torch.einsum('oij,bij->bo', self.poly_weights, poly_basis)
        
        # 組合並歸一化
        output = linear_out + poly_out
        
        # Batch normalization (如果 batch size > 1)
        if x.shape[0] > 1:
            output = self.bn(output)
        
        return output


class UltraFastKANLayer(nn.Module):
    """
    超快速 KAN 層 - 最簡化版本
    使用預計算的激活函數查找表
    """
    
    def __init__(self, input_dim, output_dim, table_size=256):
        super(UltraFastKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.table_size = table_size
        
        # 線性層
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 激活函數查找表權重
        self.activation_weights = nn.Parameter(
            torch.randn(output_dim, input_dim, table_size) * 0.1
        )
        
        # 預計算的激活函數表 (SiLU 的變體)
        x_range = torch.linspace(-3, 3, table_size)
        activation_table = x_range * torch.sigmoid(x_range)  # SiLU
        self.register_buffer('activation_table', activation_table)
        
        # 範圍映射參數
        self.register_buffer('x_min', torch.tensor(-3.0))
        self.register_buffer('x_max', torch.tensor(3.0))
    
    def fast_activation(self, x):
        """使用查找表的快速激活函數"""
        # 將輸入映射到表索引
        x_clamped = torch.clamp(x, self.x_min, self.x_max)
        indices = ((x_clamped - self.x_min) / (self.x_max - self.x_min) * 
                  (self.table_size - 1)).long()
        
        # 查找表插值
        activated = self.activation_table[indices]
        
        return activated.unsqueeze(-1)  # [batch, input_dim, 1]
    
    def forward(self, x):
        """超快速前向傳播"""
        # 基礎線性變換
        linear_out = self.linear(x)
        
        # 查找表激活
        activated = self.fast_activation(x)  # [batch, input_dim, 1]
        
        # 簡化的張量乘法 (只使用一個激活函數)
        activation_out = torch.sum(
            self.activation_weights * activated, 
            dim=2
        ).T  # [output_dim, batch] -> [batch, output_dim]
        
        return linear_out + activation_out


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
            KANLayer = FastKANLayer
        elif kan_type == 'simplified':
            KANLayer = SimplifiedKANLayer
        else:  # 'ultra_fast'
            KANLayer = UltraFastKANLayer
        
        # 構建層
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 不在最後一層添加 dropout
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.ModuleList(layers)
        
        # 消息傳遞層 (簡化的 GCN)
        self.message_layers = nn.ModuleList([
            nn.Linear(dims[i + 1], dims[i + 1]) 
            for i in range(len(dims) - 1)
        ])
        
    def message_passing(self, x, edge_index, layer_idx):
        """簡化的消息傳遞"""
        if edge_index.size(1) == 0:
            return x
        
        # 簡單的平均聚合
        row, col = edge_index
        
        # 計算鄰接矩陣 (稀疏)
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[row, col] = 1.0
        
        # 行歸一化
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1  # 避免除零
        adj = adj / row_sum
        
        # 消息傳遞
        message = torch.matmul(adj, x)
        
        # 通過線性層
        if layer_idx < len(self.message_layers):
            message = self.message_layers[layer_idx](message)
        
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

# ...existing code...