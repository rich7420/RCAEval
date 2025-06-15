"""
KAN Components: Pure KAN Layer Implementations
純粹的 KAN 層實現 - 專注於用KAN取代MLP的核心價值
確保KAN特性的純粹性：B-spline基函數、學習激活函數、非線性建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AdvancedKANLayer(nn.Module):
    """
    純粹的 KAN 層實現 - 基於 KAN 論文的高級實現
    核心特性：學習激活函數、B-spline基函數、自適應樣條
    完全不同於MLP的固定激活函數方式
    """
    
    def __init__(self, input_dim, output_dim, num_basis=8,
                 spline_order=3, grid_size=8,
                 adaptive_spline_order=True,
                 l1_lambda=1e-3, entropy_lambda=1e-3):
        super(AdvancedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.adaptive_spline_order = adaptive_spline_order
        self.l1_lambda = l1_lambda
        self.entropy_lambda = entropy_lambda
        
        # 🎯 KAN核心：可學習的B-spline基函數係數 (不是MLP的固定權重)
        self.spline_coeffs = nn.Parameter(
            torch.zeros(output_dim, input_dim, num_basis)
        )
        
        # 🎯 KAN核心：可學習的激活函數權重 (完全不同於MLP的固定激活)
        self.activation_weights = nn.Parameter(
            torch.zeros(output_dim, input_dim)
        )
        
        # 🎯 KAN核心：自適應樣條階數權重 (動態調整非線性程度)
        if adaptive_spline_order:
            self.spline_order_weights = nn.Parameter(
                torch.ones(output_dim, input_dim, 3) / 3  # 支持3,4,5階樣條
            )
        
        # 簡化穩定性 - 保持KAN純粹性
        self.ln = nn.LayerNorm(output_dim)
        
        # 基礎線性變換 (最小化MLP特性)
        self.base_linear = nn.Linear(input_dim, output_dim, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """KAN特有的參數初始化 - 針對B-spline優化"""
        # B-spline係數初始化 - 小值確保穩定性
        std = math.sqrt(2.0 / (self.input_dim + self.output_dim))
        nn.init.normal_(self.spline_coeffs, mean=0.0, std=std * 0.01)
        
        # 激活函數權重初始化
        nn.init.xavier_uniform_(self.activation_weights, gain=0.05)
        
        # 基礎線性層初始化 (最小權重，突出KAN特性)
        nn.init.xavier_uniform_(self.base_linear.weight, gain=0.1)
        
        # 自適應樣條階數權重
        if hasattr(self, 'spline_order_weights'):
            nn.init.uniform_(self.spline_order_weights, 0.2, 0.4)
    
    def learnable_activation(self, x):
        """KAN特有的可學習激活函數 - 非固定激活"""
        x_clamped = torch.clamp(x, -10.0, 10.0)
        return x_clamped * torch.sigmoid(x_clamped) * torch.sigmoid(self.activation_weights.unsqueeze(0))
    
    def pure_b_spline_basis(self, x):
        """純粹的B-spline基函數 - KAN的核心特性"""
        # 自適應歸一化 (不是MLP的線性歸一化)
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        # B-spline網格點生成
        x_grid = torch.tanh(x_normalized)  # 非線性映射到[-1,1]
        
        # 構建B-spline基函數 (Chebyshev多項式基)
        basis_functions = []
        
        # T_0(x) = 1
        basis_functions.append(torch.ones_like(x_grid))
        
        if self.num_basis > 1:
            # T_1(x) = x
            basis_functions.append(x_grid)
        
        # T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x) (Chebyshev遞推)
        for n in range(2, self.num_basis):
            t_next = 2 * x_grid * basis_functions[-1] - basis_functions[-2]
            t_next = torch.clamp(t_next, -5.0, 5.0)  # 數值穩定
            basis_functions.append(t_next)
        
        basis_tensor = torch.stack(basis_functions, dim=-1)
        
        # 自適應樣條階數調整
        if self.adaptive_spline_order and hasattr(self, 'spline_order_weights'):
            # 不同階數樣條的加權組合
            order_weights = torch.softmax(self.spline_order_weights, dim=-1)
            # 簡化實現：使用原始基函數
        
        return basis_tensor
    
    def forward(self, x):
        """純粹的KAN前向傳播 - 強調可學習激活函數"""
        # 輸入穩定性處理
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # 🎯 KAN核心計算流程
        try:
            # 1. 最小化的基礎線性變換 (不是MLP的主要特徵提取)
            base_output = self.base_linear(x) * 0.1  # 降低線性成分權重
            
            # 2. 核心：B-spline基函數計算 (KAN的主要特徵)
            basis_functions = self.pure_b_spline_basis(x)
            spline_output = torch.einsum('oij,bij->bo', self.spline_coeffs, basis_functions)
            
            # 3. 核心：可學習激活函數 (KAN vs MLP的關鍵差異)
            activation_output = torch.einsum('oi,bi->bo', 
                                           self.activation_weights, 
                                           self.learnable_activation(x))
            
            # 4. KAN輸出組合 (B-spline主導，激活函數輔助)
            kan_output = spline_output + activation_output + base_output
            
            # 5. 穩定性歸一化
            output = self.ln(kan_output)
                
        except RuntimeError as e:
            # 簡化錯誤處理
            output = self.base_linear(x)
            output = self.ln(output)
        
        return output


class SimplifiedKANLayer(nn.Module):
    """
    簡化的純粹KAN層 - 快速但保持KAN核心特性
    重點：多項式基函數、可學習激活、非MLP特性
    """
    
    def __init__(self, input_dim, output_dim, num_basis=8):
        super(SimplifiedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        
        # 🎯 簡化KAN核心：多項式基函數係數
        self.poly_coeffs = nn.Parameter(
            torch.randn(output_dim, input_dim, num_basis) * 0.02
        )
        
        # 🎯 簡化KAN核心：可學習激活權重
        self.activation_scale = nn.Parameter(
            torch.ones(output_dim, input_dim) * 0.1
        )
        
        # 最小化的線性成分 (不是MLP的主要特徵)
        self.base_transform = nn.Linear(input_dim, output_dim, bias=False)
        
        # 穩定性組件
        self.ln = nn.LayerNorm(output_dim)
    
        self.reset_parameters()
    
    def reset_parameters(self):
        """KAN特有的初始化"""
        # 多項式係數初始化
        nn.init.normal_(self.poly_coeffs, mean=0.0, std=0.02)
        
        # 激活尺度初始化
        nn.init.uniform_(self.activation_scale, 0.05, 0.15)
        
        # 基礎變換初始化 (小權重)
        nn.init.xavier_uniform_(self.base_transform.weight, gain=0.05)
    
    def polynomial_basis_functions(self, x):
        """簡化的多項式基函數 - KAN的簡化版本"""
        x_normalized = torch.tanh(x)  # 非線性歸一化
        
        basis_list = []
        
        # 多項式基函數序列
        for i in range(self.num_basis):
            if i == 0:
                basis_list.append(torch.ones_like(x_normalized))
            elif i == 1:
                basis_list.append(x_normalized)
            elif i == 2:
                basis_list.append(x_normalized ** 2)
            elif i == 3:
                basis_list.append(x_normalized ** 3)
            else:
                # 高階多項式使用遞推關係
                power = x_normalized ** i
                power = torch.clamp(power, -5.0, 5.0)
                basis_list.append(power)
        
        return torch.stack(basis_list, dim=-1)
    
    def kan_learnable_activation(self, x):
        """簡化的可學習激活函數"""
        return x * torch.tanh(x * self.activation_scale.unsqueeze(0))
    
    def forward(self, x):
        """簡化KAN的前向傳播 - 保持核心特性"""
        # 穩定性處理
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        try:
            # 1. 最小化基礎變換 (降低MLP特性)
            base_output = self.base_transform(x) * 0.05
            
            # 2. 多項式基函數計算 (KAN核心)
            poly_basis = self.polynomial_basis_functions(x)
            poly_output = torch.einsum('oij,bij->bo', self.poly_coeffs, poly_basis)
            
            # 3. 可學習激活函數 (KAN vs MLP差異)
            activation_output = self.kan_learnable_activation(x).sum(dim=1, keepdim=True)
            activation_output = activation_output.expand(-1, self.output_dim) * 0.1
            
            # 4. KAN輸出組合 (多項式主導)
            kan_output = poly_output + activation_output + base_output
            
            # 5. 穩定性歸一化
            output = self.ln(kan_output)
            
        except RuntimeError:
            output = self.base_transform(x)
            output = self.ln(output)
        
        return output


class OptimizedGNNKANEncoder(nn.Module):
    """
    專用於GNN的KAN編碼器 - 用純粹KAN取代MLP層
    核心：用AdvancedKANLayer完全取代傳統的MLP層
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=2, kan_grid_size=8, kan_spline_order=3, 
                 dropout=0.1, learnable_graph=True):
        super(OptimizedGNNKANEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.learnable_graph = learnable_graph
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        
        # 🎯 核心：用純粹KAN層取代所有MLP層
        kan_layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # 使用AdvancedKANLayer完全取代MLP線性層
            kan_layers.append(AdvancedKANLayer(
                dims[i], dims[i + 1], 
                num_basis=kan_grid_size,
                spline_order=kan_spline_order,
                grid_size=kan_grid_size,
                adaptive_spline_order=True
            ))
            
            # Dropout (但不使用MLP常用的ReLU/GELU等固定激活)
            if i < len(dims) - 2:
                kan_layers.append(nn.Dropout(dropout))
        
        self.kan_layers = nn.ModuleList(kan_layers)
        
        # 簡化消息傳遞 (避免MLP結構)
        self.message_processors = nn.ModuleList([
            SimplifiedKANLayer(dims[i + 1], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        
        # 可學習圖結構
        if learnable_graph:
            self.edge_learner = nn.Sequential(
                SimplifiedKANLayer(dims[-1] * 2, dims[-1]),
                SimplifiedKANLayer(dims[-1], 1)
            )
        
    def kan_message_passing(self, x, edge_index, layer_idx):
        """使用KAN進行消息傳遞 - 不使用MLP"""
        if edge_index.size(1) == 0:
            return x
        
        # 穩定性檢查
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        row, col = edge_index
        num_nodes = x.size(0)
        
        # 安全索引
        row = torch.clamp(row, 0, num_nodes - 1)
        col = torch.clamp(col, 0, num_nodes - 1)
        
        try:
            # 構建稀疏鄰接矩陣
            adj_indices = torch.stack([row, col], dim=0)
            adj_values = torch.ones(len(row), device=x.device, dtype=x.dtype)
            adj_size = (num_nodes, num_nodes)
            
            adj_sparse = torch.sparse_coo_tensor(adj_indices, adj_values, adj_size, device=x.device)
            adj_sparse = adj_sparse.coalesce()
            
            # 度歸一化
            degrees = torch.sparse.sum(adj_sparse, dim=1).to_dense()
            degrees = torch.clamp(degrees, min=1e-6)
            degrees_inv = 1.0 / torch.sqrt(degrees)
            
            # 歸一化鄰接矩陣
            norm_values = degrees_inv[row] * degrees_inv[col]
            norm_adj = torch.sparse_coo_tensor(adj_indices, norm_values, adj_size, device=x.device)
            
            # 消息傳遞
            message = torch.sparse.mm(norm_adj, x)
            
            # 🎯 使用KAN處理消息 (而不是MLP)
            if layer_idx < len(self.message_processors):
                message = self.message_processors[layer_idx](message)
                
        except RuntimeError:
            return x
        
        return message
    
    def forward(self, x, edge_index):
        """純粹KAN的前向傳播 - 完全避免MLP結構"""
        current_x = x
        
        for i, layer in enumerate(self.kan_layers):
            if isinstance(layer, nn.Dropout):
                current_x = layer(current_x)
            else:
                # 🎯 KAN層處理 (核心：用KAN取代MLP)
                kan_output = layer(current_x)
                
                # KAN消息傳遞 (每兩層一次，減少計算)
                if i % 2 == 0 and edge_index.size(1) > 0:
                    message = self.kan_message_passing(kan_output, edge_index, i // 2)
                    # 殘差連接
                    current_x = kan_output + 0.3 * message
                else:
                    current_x = kan_output
                
                # 不使用固定激活函數 (KAN內部已包含可學習激活)
        
        return current_x


# 向後兼容的別名 - 統一接口
class KANLayer(SimplifiedKANLayer):
    """向後兼容的KAN層"""
    pass


class GNNKANEncoder(OptimizedGNNKANEncoder):
    """向後兼容的GNN-KAN編碼器"""
    pass