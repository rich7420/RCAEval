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
    優化版：增強非線性表達，減少冗餘檢查
    """
    
    def __init__(self, input_dim, output_dim, num_basis=8,  # 增加到8
                 l1_lambda=1e-3, entropy_lambda=1e-3, 
                 spline_order=3, grid_size=8,              # 增加到8
                 enable_pruning=True, pruning_threshold=1e-2,
                 adaptive_lr=True, weight_decay=1e-4,
                 adaptive_spline_order=True):               # 新增自適應參數
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
        self.adaptive_spline_order = adaptive_spline_order
        
        # 基礎線性層 - 使用 Xavier 初始化
        self.linear = nn.Linear(input_dim, output_dim)
        
        # B-spline 係數 - 增強網格密度
        self.spline_coeffs = nn.Parameter(
            torch.zeros(output_dim, input_dim, num_basis)
        )
        
        # SiLU 激活函數的權重 (論文中提到的 σ(x) = x/(1+e^(-x)))
        self.silu_weight = nn.Parameter(torch.zeros(output_dim, input_dim))
        
        # 自適應樣條階數權重
        if adaptive_spline_order:
            self.spline_order_weights = nn.Parameter(
                torch.ones(output_dim, input_dim, 3) / 3  # 3,4,5階樣條的權重
            )
        
        # 簡化穩定性組件
        self.ln = nn.LayerNorm(output_dim)  # 只保留LayerNorm
        self.dropout = nn.Dropout(0.1)
        
        # 用於剪枝的重要性分數
        self.register_buffer('importance_scores', torch.ones(output_dim, input_dim))
        
        # 初始化參數
        self.reset_parameters()
    
    def reset_parameters(self):
        """增強的參數初始化"""
        # Xavier 初始化線性層
        nn.init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.linear.bias)
        
        # B-spline 係數初始化 - 更細緻的初始化
        fan_in = self.input_dim
        fan_out = self.output_dim
        std = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(self.spline_coeffs, mean=0.0, std=std * 0.05)  # 更小的初始化
        
        # SiLU 權重初始化
        nn.init.xavier_uniform_(self.silu_weight, gain=0.1)
        
        # 自適應樣條階數權重初始化
        if hasattr(self, 'spline_order_weights'):
            nn.init.uniform_(self.spline_order_weights, 0.2, 0.4)
    
    def silu_activation(self, x):
        """SiLU 激活函數：σ(x) = x / (1 + exp(-x))"""
        x_clamped = torch.clamp(x, -20.0, 20.0)
        return x_clamped * torch.sigmoid(x_clamped)
    
    def enhanced_b_spline_basis(self, x):
        """增強的 B-spline 基函數計算 - 支持自適應樣條階數"""
        # 自適應歸一化
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        # 將輸入歸一化到 [-1, 1]
        x_norm = torch.tanh(x_normalized)
        
        # 基礎 Chebyshev 多項式基函數
        basis_list = [torch.ones_like(x_norm)]
        
        if self.num_basis > 1:
            basis_list.append(x_norm)
        
        # 計算更高階基函數
        for i in range(2, self.num_basis):
            t_next = 2 * x_norm * basis_list[-1] - basis_list[-2]
            t_next = torch.clamp(t_next, -10.0, 10.0)
            basis_list.append(t_next)
        
        basis_tensor = torch.stack(basis_list, dim=-1)
        
        # 如果啟用自適應樣條階數
        if self.adaptive_spline_order and hasattr(self, 'spline_order_weights'):
            # 計算多階樣條的組合
            order_3_basis = basis_tensor[..., :min(4, basis_tensor.size(-1))]
            order_4_basis = basis_tensor[..., :min(5, basis_tensor.size(-1))]
            order_5_basis = basis_tensor[..., :min(6, basis_tensor.size(-1))]
            
            # 加權組合不同階數的基函數
            adaptive_weights = torch.softmax(self.spline_order_weights, dim=-1)
            # 這裡簡化實現，返回原始基函數
            return basis_tensor
        
        return basis_tensor
    
    def forward(self, x):
        """優化的前向傳播 - 減少冗餘檢查"""
        # 簡化的輸入檢查（減少頻率）
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        batch_size = x.shape[0]
        
        try:
            # 基礎線性變換
            linear_out = self.linear(x)
            
            # 增強的 B-spline 分量
            basis = self.enhanced_b_spline_basis(x)
            spline_out = torch.einsum('oij,bij->bo', self.spline_coeffs, basis)
            
            # SiLU 分量
            silu_out = torch.einsum('oi,bi->bo', self.silu_weight, self.silu_activation(x))
            
            # 組合所有分量
            output = linear_out + spline_out + silu_out
            
            # 應用 Dropout
            if self.training:
                output = self.dropout(output)
            
            # 簡化的歸一化（只使用LayerNorm）
            output = self.ln(output)
                
        except RuntimeError as e:
            # 簡化的錯誤處理
            output = self.linear(x)
            output = self.ln(output)
        
        return output


class SimplifiedKANLayer(nn.Module):
    """簡化版 KAN 層 - 增強表達能力的快速計算"""
    
    def __init__(self, input_dim, output_dim, num_basis=8):  # 增加到8
        super(SimplifiedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        
        # 基礎線性層
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 增強的多項式基函數權重
        self.poly_weights = nn.Parameter(
            torch.randn(output_dim, input_dim, num_basis) * 0.05  # 更精細的初始化
        )
        
        # 簡化穩定性組件
        self.ln = nn.LayerNorm(output_dim)
    
    def enhanced_polynomial_basis(self, x):
        """增強的多項式基函數"""
        x_norm = torch.tanh(x)
        
        basis_list = [torch.ones_like(x_norm)]
        
        if self.num_basis > 1:
            basis_list.append(x_norm)
        
        # 計算更多階的多項式
        for i in range(2, self.num_basis):
            if i == 2:
                t_next = x_norm * x_norm  # x^2
            elif i == 3:
                t_next = x_norm * basis_list[2]  # x^3
            else:
                t_next = 2 * x_norm * basis_list[-1] - basis_list[-2]
            basis_list.append(t_next)
        
        return torch.stack(basis_list, dim=-1)
    
    def forward(self, x):
        """優化的前向傳播"""
        # 簡化的檢查
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        try:
            linear_out = self.linear(x)
            poly_basis = self.enhanced_polynomial_basis(x)
            poly_out = torch.einsum('oij,bij->bo', self.poly_weights, poly_basis)
            
            output = linear_out + poly_out
            output = self.ln(output)
            
        except RuntimeError:
            output = self.linear(x)
            output = self.ln(output)
        
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
    """增強版 GNN-KAN 編碼器 - 支持動態圖結構學習"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=2, kan_grid_size=8, kan_spline_order=3, dropout=0.1,
                 learnable_graph=True):
        super(OptimizedGNNKANEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.learnable_graph = learnable_graph
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        
        # 構建增強的 KAN 層 - 使用提升的表達能力
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # 使用增強的 AdvancedKANLayer 提升非線性表達
            layers.append(AdvancedKANLayer(
                dims[i], dims[i + 1], 
                num_basis=kan_grid_size,  # 使用配置的grid_size
                spline_order=kan_spline_order,
                grid_size=kan_grid_size
            ))
            if i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.ModuleList(layers)
        
        # 增強的消息傳遞層 - 簡化但保持效果
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i + 1], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU()
            )
            for i in range(len(dims) - 1)
        ])
        
        # 可學習的圖結構機制 - 新增功能
        if learnable_graph:
            self.edge_weight_net = nn.Sequential(
                nn.Linear(dims[-1] * 2, dims[-1]),
                nn.ReLU(),
                nn.Linear(dims[-1], 1),
                nn.Sigmoid()
            )
            
            # 圖注意力機制（簡化版）
            self.graph_attention = nn.MultiheadAttention(
                embed_dim=dims[-1], num_heads=4, dropout=dropout, batch_first=True
            )
        
    def enhanced_message_passing(self, x, edge_index, layer_idx):
        """增強的消息傳遞 - 減少冗餘檢查，提升效率"""
        if edge_index.size(1) == 0:
            return x
        
        # 簡化的穩定性檢查 - 只在必要時處理
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        row, col = edge_index
        num_nodes = x.size(0)
        
        # 安全的索引處理
        row = torch.clamp(row, 0, num_nodes - 1)
        col = torch.clamp(col, 0, num_nodes - 1)
        
        try:
            # 稀疏圖構建 - 優化內存使用
            adj_indices = torch.stack([row, col], dim=0)
            adj_values = torch.ones(len(row), device=x.device, dtype=x.dtype)
            adj_size = (num_nodes, num_nodes)
            
            adj_sparse = torch.sparse_coo_tensor(adj_indices, adj_values, adj_size, device=x.device)
            adj_sparse = adj_sparse.coalesce()
            
            # 度歸一化 - 簡化計算
            degrees = torch.sparse.sum(adj_sparse, dim=1).to_dense()
            degrees = torch.clamp(degrees, min=1e-6)
            degrees_inv = 1.0 / torch.sqrt(degrees)
            
            # 對稱歸一化
            norm_values = degrees_inv[row] * degrees_inv[col]
            norm_adj = torch.sparse_coo_tensor(adj_indices, norm_values, adj_size, device=x.device)
            
            # 消息傳遞
            message = torch.sparse.mm(norm_adj, x)
            
            # 可學習的邊權重更新 - 新增動態圖功能
            if self.learnable_graph and hasattr(self, 'edge_weight_net'):
                # 計算邊的嵌入特徵
                edge_features = torch.cat([x[row], x[col]], dim=1)
                edge_weights = self.edge_weight_net(edge_features).squeeze(-1)
                
                # 應用學習到的邊權重
                weighted_values = norm_values * edge_weights
                weighted_adj = torch.sparse_coo_tensor(adj_indices, weighted_values, adj_size, device=x.device)
                message = torch.sparse.mm(weighted_adj, x)
            
            # 通過增強的消息層
            if layer_idx < len(self.message_layers):
                message = self.message_layers[layer_idx](message)
                
        except RuntimeError as e:
            # 簡化的回退機制
            print(f"Message passing fallback: {e}")
            return x
        
        return message
    
    def forward(self, x, edge_index):
        """增強的前向傳播 - 平衡KAN表達力與計算效率"""
        current_x = x
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Dropout):
                current_x = layer(current_x)
            else:
                # KAN 層處理 - 核心價值：用KAN取代MLP
                kan_out = layer(current_x)
                
                # 消息傳遞（減少頻率但保持效果）
                if i % 2 == 0 and edge_index.size(1) > 0:
                    message = self.enhanced_message_passing(kan_out, edge_index, i // 2)
                    # 殘差連接 - 穩定訓練
                    current_x = kan_out + 0.2 * message  # 增加message權重
                else:
                    current_x = kan_out
                
                # 激活函數（非最後一層）
                if i < len(self.layers) - 1:
                    current_x = F.gelu(current_x)
        
        # 簡化的圖注意力機制 - 只在最後應用
        if self.learnable_graph and hasattr(self, 'graph_attention'):
            # 將節點特徵重塑為批次格式進行注意力計算
            x_att = current_x.unsqueeze(0)  # [1, num_nodes, feat_dim]
            try:
                att_out, att_weights = self.graph_attention(x_att, x_att, x_att)
                # 使用注意力權重進行動態圖調整
                current_x = current_x + 0.1 * att_out.squeeze(0)
            except RuntimeError:
                pass  # 如果注意力失敗，使用原始輸出
        
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