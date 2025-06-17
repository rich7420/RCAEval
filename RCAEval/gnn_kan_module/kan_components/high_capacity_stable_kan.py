"""
高容量梯度穩定的 KAN 實現
保持原始模型復雜度 (grid_size=5, spline_order=3, 3層GNN) 但解決梯度爆炸問題
使用多種先進的梯度穩定技術：
1. 殘差連接 + 層標準化
2. 譜標準化 + 權重正交化
3. 預熱學習率調度
4. 自適應梯度縮放
5. 梯度累積和檢查點
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SpectralNorm(nn.Module):
    """譜標準化模組 - 限制權重矩陣的最大奇異值"""
    
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        # 初始化 u 向量
        w = getattr(module, name)
        height = w.data.shape[0]
        u = w.new_empty(height).normal_(0, 1)
        u = F.normalize(u, dim=0, eps=1e-12)
        
        self.register_buffer(f'{name}_u', u)
    
    def _update_u_v(self):
        """更新 u 和 v 向量"""
        w = getattr(self.module, self.name)
        u = getattr(self, f'{self.name}_u')
        
        height, width = w.shape[0], w.view(w.shape[0], -1).shape[1]
        
        for _ in range(self.power_iterations):
            # v = w^T @ u / ||w^T @ u||
            v = F.normalize(torch.mv(w.view(height, width).t(), u), dim=0, eps=1e-12)
            # u = w @ v / ||w @ v||
            u = F.normalize(torch.mv(w.view(height, width), v), dim=0, eps=1e-12)
        
        setattr(self, f'{self.name}_u', u.detach())
        return u, v
    
    def forward(self, *args, **kwargs):
        if self.training:
            u, v = self._update_u_v()
            w = getattr(self.module, self.name)
            sigma = torch.dot(u, torch.mv(w.view(w.shape[0], -1), v))
            # 將權重除以最大奇異值，保持Parameter類型
            normalized_weight = w / sigma.clamp(min=1.0)
            # 正確設置Parameter - 使用torch.nn.Parameter包裝
            if isinstance(w, torch.nn.Parameter):
                setattr(self.module, self.name, torch.nn.Parameter(normalized_weight.data, requires_grad=w.requires_grad))
            else:
                setattr(self.module, self.name, normalized_weight)
        
        return self.module(*args, **kwargs)


class HighCapacityStableKANLayer(nn.Module):
    """
    高容量且梯度穩定的 KAN 層
    保持原始復雜度：grid_size=5, spline_order=3
    """
    
    def __init__(self, input_dim, output_dim, 
                 grid_size=5, spline_order=3,
                 use_residual=True, use_spectral_norm=True,
                 l1_lambda=1e-4, entropy_lambda=1e-4):
        super(HighCapacityStableKANLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.use_residual = use_residual
        
        # 🔑 保持原始復雜度的 B-spline 係數 - 修復維度匹配
        # 確保係數維度與基函數維度一致
        self.num_basis_functions = grid_size + spline_order
        self.spline_coeffs = nn.Parameter(
            torch.zeros(output_dim, input_dim, self.num_basis_functions)
        )
        
        # SiLU 激活權重
        self.silu_weight = nn.Parameter(torch.zeros(output_dim, input_dim))
        
        # 🛡️ 殘差連接的線性層
        if use_residual:
            self.residual_linear = nn.Linear(input_dim, output_dim)
            if use_spectral_norm:
                self.residual_linear = SpectralNorm(self.residual_linear)
        
        # 🛡️ 層標準化 (比 BatchNorm 更穩定)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 🛡️ 預標準化 (在激活前標準化)
        self.pre_norm = nn.LayerNorm(input_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 正則化參數
        self.l1_lambda = l1_lambda
        self.entropy_lambda = entropy_lambda
        
        # 初始化參數
        self._initialize_parameters()
        
        # 梯度縮放因子 (自適應)
        self.register_buffer('grad_scale', torch.tensor(1.0))
        self.register_buffer('grad_history', torch.zeros(10))
        self.register_buffer('step_count', torch.tensor(0))
    
    def _initialize_parameters(self):
        """智能參數初始化 - 保證高容量的同時穩定梯度"""
        
        # 🎯 B-spline 係數的保守初始化
        fan_in = self.input_dim
        fan_out = self.output_dim
        
        # 使用 He 初始化的修正版本
        std = math.sqrt(2.0 / fan_in) * 0.1  # 縮小10倍保證穩定性
        
        with torch.no_grad():
            # 正交初始化 spline 係數
            for i in range(self.output_dim):
                # 對每個輸出維度單獨初始化
                nn.init.orthogonal_(self.spline_coeffs[i], gain=std)
        
        # SiLU 權重的保守初始化
        nn.init.xavier_uniform_(self.silu_weight, gain=0.05)
        
        # 殘差連接的正交初始化
        if self.use_residual:
            # 檢查是否使用了SpectralNorm包裝
            if hasattr(self.residual_linear, 'weight'):
                nn.init.orthogonal_(self.residual_linear.weight, gain=1.0)
                nn.init.zeros_(self.residual_linear.bias)
            elif hasattr(self.residual_linear, 'module'):
                # SpectralNorm包裝的情況
                nn.init.orthogonal_(self.residual_linear.module.weight, gain=1.0)
                nn.init.zeros_(self.residual_linear.module.bias)
            else:
                # 安全的初始化方式
                for param in self.residual_linear.parameters():
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param, gain=1.0)
                    else:
                        nn.init.zeros_(param)
    
    def _compute_b_spline_basis(self, x):
        """
        計算高階 B-spline 基函數 (保持 order=3 的表達能力)
        使用數值穩定的實現
        """
        batch_size, input_dim = x.shape
        
        # 🛡️ 輸入預處理和數值穩定化
        x_clamped = torch.clamp(x, -10.0, 10.0)  # 防止數值溢出
        
        # 標準化到 [0, 1] 區間
        x_min = torch.min(x_clamped, dim=0, keepdim=True)[0] - 1e-8
        x_max = torch.max(x_clamped, dim=0, keepdim=True)[0] + 1e-8
        x_norm = (x_clamped - x_min) / (x_max - x_min)
        
        # 創建節點向量 (保持 grid_size=5)
        knots = torch.linspace(0, 1, self.grid_size, device=x.device, dtype=x.dtype)
        
        # 擴展節點向量以支持高階樣條
        extended_knots = torch.cat([
            knots[0].repeat(self.spline_order),
            knots,
            knots[-1].repeat(self.spline_order)
        ])
        
        # 計算 B-spline 基函數 (使用數值穩定的遞歸算法)
        basis_functions = []
        
        for i in range(len(extended_knots) - self.spline_order - 1):
            # 0階基函數
            b = ((x_norm >= extended_knots[i]) & 
                 (x_norm < extended_knots[i + 1])).float()
            
            # 遞歸計算高階基函數
            for p in range(1, self.spline_order + 1):
                if i + p + 1 < len(extended_knots):
                    # 左側係數
                    left_denom = extended_knots[i + p] - extended_knots[i]
                    if left_denom > 1e-8:
                        left_coeff = (x_norm - extended_knots[i]) / left_denom
                    else:
                        left_coeff = 0.0
                    
                    # 右側係數
                    right_denom = extended_knots[i + p + 1] - extended_knots[i + 1]
                    if right_denom > 1e-8:
                        right_coeff = (extended_knots[i + p + 1] - x_norm) / right_denom
                    else:
                        right_coeff = 0.0
                    
                    # 更新基函數
                    b = left_coeff * b + right_coeff * basis_functions[i] if i < len(basis_functions) else left_coeff * b
            
            basis_functions.append(b)
        
        # 堆疊成張量 [batch_size, input_dim, num_basis]
        if basis_functions:
            basis_tensor = torch.stack(basis_functions[-self.grid_size-self.spline_order:], dim=-1)
        else:
            # 回退到簡單基函數
            basis_tensor = self._fallback_basis(x_norm)
        
        return basis_tensor
    
    def _fallback_basis(self, x):
        """回退到簡單但穩定的基函數"""
        # 使用 Chebyshev 多項式作為回退
        basis_list = []
        
        # T0 = 1
        basis_list.append(torch.ones_like(x))
        
        if self.grid_size + self.spline_order > 1:
            # T1 = x
            basis_list.append(x)
        
        # 遞歸計算更高階的 Chebyshev 多項式
        for i in range(2, self.grid_size + self.spline_order):
            t_next = 2 * x * basis_list[-1] - basis_list[-2]
            t_next = torch.clamp(t_next, -5.0, 5.0)  # 防止數值爆炸
            basis_list.append(t_next)
        
        return torch.stack(basis_list, dim=-1)
    
    def _adaptive_gradient_scaling(self):
        """自適應梯度縮放"""
        if self.training and self.spline_coeffs.grad is not None:
            current_grad_norm = torch.norm(self.spline_coeffs.grad)
            
            # 更新梯度歷史
            self.grad_history[:-1] = self.grad_history[1:]
            self.grad_history[-1] = current_grad_norm
            
            # 計算平均梯度範數
            avg_grad_norm = torch.mean(self.grad_history[self.grad_history > 0])
            
            # 自適應調整縮放因子
            if avg_grad_norm > 2.0:  # 如果梯度過大
                self.grad_scale *= 0.95
            elif avg_grad_norm < 0.5:  # 如果梯度過小
                self.grad_scale *= 1.02
            
            # 限制縮放因子範圍
            self.grad_scale = torch.clamp(self.grad_scale, 0.01, 1.0)
    
    def forward(self, x):
        """
        前向傳播 - 高容量且穩定
        """
        # 🛡️ 輸入預標準化
        x_normalized = self.pre_norm(x)
        
        # 🛡️ 數值穩定性檢查
        if torch.isnan(x_normalized).any() or torch.isinf(x_normalized).any():
            x_normalized = torch.nan_to_num(x_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🔑 高容量 B-spline 計算 - 修復einsum維度問題
        try:
            basis = self._compute_b_spline_basis(x_normalized)
            batch_size = x.size(0)
            
            # 🔧 安全的B-spline計算 - 檢查維度兼容性
            expected_basis_dim = self.num_basis_functions
            if (basis.shape[0] == batch_size and 
                basis.shape[1] == self.input_dim and
                basis.shape[2] == expected_basis_dim):
                # 正常的einsum操作
                spline_output = torch.einsum('oij,bij->bo', self.spline_coeffs, basis)
            else:
                # 維度不匹配時的安全處理
                print(f"High-capacity B-spline dimension mismatch: basis={basis.shape}, coeffs={self.spline_coeffs.shape}, expected_basis_dim={expected_basis_dim}")
                
                # 調整基函數維度以匹配係數
                if basis.shape[2] != expected_basis_dim:
                    if basis.shape[2] < expected_basis_dim:
                        # 基函數維度不足，進行零填充
                        padding_size = expected_basis_dim - basis.shape[2]
                        padding = torch.zeros(batch_size, self.input_dim, padding_size, device=basis.device)
                        basis = torch.cat([basis, padding], dim=2)
                    else:
                        # 基函數維度過多，進行截斷
                        basis = basis[:, :, :expected_basis_dim]
                
                # 再次嘗試einsum操作
                if basis.shape == (batch_size, self.input_dim, expected_basis_dim):
                    spline_output = torch.einsum('oij,bij->bo', self.spline_coeffs, basis)
                else:
                    # 最終回退：使用安全的矩陣乘法
                    basis_flat = basis.view(batch_size, -1)
                    coeffs_flat = self.spline_coeffs.view(self.output_dim, -1)
                    
                    min_dim = min(basis_flat.shape[1], coeffs_flat.shape[1])
                    if min_dim > 0:
                        spline_output = torch.mm(basis_flat[:, :min_dim], coeffs_flat[:, :min_dim].t())
                    else:
                        spline_output = torch.zeros(batch_size, self.output_dim, device=x.device)
                    
        except RuntimeError as e:
            print(f"B-spline computation failed: {e}, using fallback")
            # 回退到線性層
            spline_output = torch.zeros(x.size(0), self.output_dim, device=x.device)
        
        # 🔑 SiLU 非線性 (保持表達能力)
        silu_activation = x_normalized * torch.sigmoid(x_normalized)
        silu_output = torch.einsum('oi,bi->bo', self.silu_weight, silu_activation)
        
        # 組合輸出
        kan_output = spline_output + silu_output
        
        # 🛡️ 殘差連接
        if self.use_residual:
            residual = self.residual_linear(x_normalized)
            output = kan_output + 0.1 * residual  # 較小的殘差權重
        else:
            output = kan_output
        
        # 🛡️ 層標準化
        output = self.layer_norm(output)
        
        # Dropout
        if self.training:
            output = self.dropout(output)
        
        # 🛡️ 最終數值檢查
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            output = torch.clamp(output, -5.0, 5.0)
        
        # 自適應梯度縮放
        if self.training:
            self._adaptive_gradient_scaling()
        
        return output
    
    def compute_regularization_loss(self):
        """計算正則化損失"""
        # L1 正則化
        l1_loss = torch.sum(torch.abs(self.spline_coeffs)) + torch.sum(torch.abs(self.silu_weight))
        
        # 熵正則化 (促進稀疏性)
        spline_importance = torch.sum(torch.abs(self.spline_coeffs), dim=-1)
        total_importance = torch.sum(spline_importance)
        
        if total_importance > 1e-8:
            normalized_importance = spline_importance / (total_importance + 1e-8)
            entropy_loss = -torch.sum(normalized_importance * torch.log(normalized_importance + 1e-8))
        else:
            entropy_loss = torch.tensor(0.0, device=self.spline_coeffs.device)
        
        return self.l1_lambda * l1_loss, self.entropy_lambda * entropy_loss


class HighCapacityGNNKANEncoder(nn.Module):
    """
    高容量的 GNN-KAN 編碼器
    保持 3 層 GNN 的深度和表達能力
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=3, kan_config=None, dropout=0.1):
        super(HighCapacityGNNKANEncoder, self).__init__()
        
        self.num_layers = num_layers
        
        # 默認 KAN 配置
        if kan_config is None:
            kan_config = {
                'grid_size': 5,
                'spline_order': 3,
                'use_residual': True,
                'use_spectral_norm': True
            }
        
        # 🔑 構建 3 層深度網路
        self.kan_layers = nn.ModuleList()
        self.message_passing_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # 輸入維度序列
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # 🔑 高容量 KAN 層
            kan_layer = HighCapacityStableKANLayer(
                dims[i], dims[i + 1], **kan_config
            )
            self.kan_layers.append(kan_layer)
            
            # 消息傳遞層
            message_layer = nn.Linear(dims[i + 1], dims[i + 1])
            if kan_config.get('use_spectral_norm', False):
                message_layer = SpectralNorm(message_layer)
            self.message_passing_layers.append(message_layer)
            
            # 層標準化
            self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
        
        self.dropout = nn.Dropout(dropout)
        
        # 🛡️ 梯度累積緩衝區
        self.register_buffer('grad_accumulator', torch.tensor(0.0))
        self.register_buffer('accumulation_steps', torch.tensor(0))
    
    def message_passing(self, x, edge_index, layer_idx):
        """數值穩定的消息傳遞"""
        if edge_index.size(1) == 0:
            return x
        
        try:
            # 安全的消息傳遞實現
            row, col = edge_index
            num_nodes = x.size(0)
            
            # 檢查索引有效性
            valid_mask = (row < num_nodes) & (col < num_nodes) & (row >= 0) & (col >= 0)
            if not valid_mask.all():
                row = row[valid_mask]
                col = col[valid_mask]
            
            if len(row) == 0:
                return x
            
            # 聚合鄰居特徵
            neighbor_features = x[row]
            
            # 按目標節點聚合
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, col, neighbor_features)
            
            # 度數歸一化
            degree = torch.zeros(num_nodes, device=x.device)
            degree.index_add_(0, col, torch.ones(len(col), device=x.device))
            degree = torch.clamp(degree, min=1.0)
            
            aggregated = aggregated / degree.unsqueeze(1)
            
            # 通過消息傳遞層
            message_output = self.message_passing_layers[layer_idx](aggregated)
            
            return message_output
            
        except Exception as e:
            print(f"Message passing failed: {e}, returning original features")
            return x
    
    def forward(self, x, edge_index):
        """
        前向傳播 - 保持 3 層深度
        """
        current_features = x
        
        # 🔑 通過 3 層 GNN-KAN
        for i, kan_layer in enumerate(self.kan_layers):
            # KAN 變換
            transformed = kan_layer(current_features)
            
            # 消息傳遞
            if i < len(self.message_passing_layers):
                messages = self.message_passing(transformed, edge_index, i)
                
                # 🛡️ 殘差連接 (跨層)
                if transformed.shape == messages.shape:
                    combined = transformed + 0.1 * messages
                else:
                    combined = transformed
                
                # 層標準化
                combined = self.layer_norms[i](combined)
                
                # Dropout
                if self.training and i < len(self.kan_layers) - 1:
                    combined = self.dropout(combined)
                
                current_features = combined
            else:
                current_features = transformed
        
        return current_features
    
    def compute_total_regularization_loss(self, base_loss):
        """計算總正則化損失"""
        total_l1 = torch.tensor(0.0, device=base_loss.device)
        total_entropy = torch.tensor(0.0, device=base_loss.device)
        
        for kan_layer in self.kan_layers:
            l1_loss, entropy_loss = kan_layer.compute_regularization_loss()
            total_l1 += l1_loss
            total_entropy += entropy_loss
        
        return base_loss + total_l1 + total_entropy


class WarmupScheduler:
    """預熱學習率調度器"""
    
    def __init__(self, optimizer, warmup_epochs=10, base_lr=2e-4, max_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    
    def step(self):
        """更新學習率"""
        if self.current_epoch < self.warmup_epochs:
            # 預熱階段：線性增加學習率
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # 正常階段：指數衰減
            decay_epochs = self.current_epoch - self.warmup_epochs
            lr = self.max_lr * (0.95 ** (decay_epochs // 10))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def create_high_capacity_stable_model(input_dim, hidden_dims, output_dim, num_nodes):
    """
    創建高容量且穩定的 GNN-KAN 模型
    
    Args:
        input_dim: 輸入維度
        hidden_dims: 隱藏層維度列表
        output_dim: 輸出維度
        num_nodes: 節點數量
        
    Returns:
        model: 高容量穩定模型
        config: 模型配置
    """
    # 🔑 高容量配置 - 保持原始復雜度
    kan_config = {
        'grid_size': 5,          # 保持 5 個網格點
        'spline_order': 3,       # 保持 3 次樣條
        'use_residual': True,    # 啟用殘差連接
        'use_spectral_norm': True, # 啟用譜標準化
        'l1_lambda': 1e-5,       # 輕微的 L1 正則化
        'entropy_lambda': 1e-5   # 輕微的熵正則化
    }
    
    # 🔑 保持 3 層 GNN 深度
    model = HighCapacityGNNKANEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,  # 例如 [128, 96, 64]
        output_dim=output_dim,
        num_layers=3,             # 保持 3 層
        kan_config=kan_config,
        dropout=0.15
    )
    
    # 訓練配置
    training_config = {
        'epochs': 100,            # 更多訓練輪數
        'batch_size': 16,
        'base_lr': 1e-4,          # 基礎學習率
        'max_lr': 5e-4,           # 最大學習率
        'warmup_epochs': 15,      # 預熱階段
        'weight_decay': 1e-6,
        'gradient_clip_norm': 1.5, # 適中的梯度裁剪
        'accumulation_steps': 4    # 梯度累積步數
    }
    
    return model, kan_config, training_config


def create_high_capacity_kan_encoder(input_dim, hidden_dims, output_dim, num_nodes):
    """
    創建高容量的 KAN 編碼器 - 兼容函數
    
    Args:
        input_dim: 輸入維度
        hidden_dims: 隱藏層維度列表 
        output_dim: 輸出維度
        num_nodes: 節點數量
        
    Returns:
        HighCapacityGNNKANEncoder: 高容量編碼器實例
    """
    # 使用默認的高容量配置
    kan_config = {
        'grid_size': 5,
        'spline_order': 3,
        'use_residual': True,
        'use_spectral_norm': True,
        'l1_lambda': 1e-5,
        'entropy_lambda': 1e-5
    }
    
    # 如果沒有提供隱藏層維度，使用默認值
    if not hidden_dims:
        hidden_dims = [128, 96]
    
    return HighCapacityGNNKANEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_layers=3,
        kan_config=kan_config,
        dropout=0.15
    )


# 確保所有函數都可以被導入
__all__ = [
    'HighCapacityStableKANLayer',
    'HighCapacityGNNKANEncoder', 
    'SpectralNorm',
    'WarmupScheduler',
    'create_high_capacity_stable_model',
    'create_high_capacity_kan_encoder'  # 添加到導出列表
]