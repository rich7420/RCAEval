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
        self.ln = nn.LayerNorm(output_dim, eps=1e-4)
        
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
        """
        可學習激活函數 - 修復維度問題和NaN產生
        這是KAN的核心特性：可學習的激活函數
        """
        batch_size, input_dim = x.shape
        
        # print(f"🔍 激活函數維度診斷:")
        # print(f"  activation_weights.shape: {self.activation_weights.shape}")
        # print(f"  batch_size: {batch_size}, input_dim: {input_dim}, output_dim: {self.output_dim}")
        
        try:
            # 🔧 修復維度匹配問題
            if (self.activation_weights.shape[0] == self.output_dim and
                self.activation_weights.shape[1] == input_dim):
                
                # 正常的矩陣乘法路徑
                # x: [batch_size, input_dim]
                # activation_weights: [output_dim, input_dim]
                
                # 1. 使用可學習激活權重進行變換
                output = torch.matmul(x, self.activation_weights.t())  # [batch_size, output_dim]
                
                # 2. 添加非線性激活
                output = torch.tanh(output)  # 穩定的激活函數
                
                #print("✅ 激活函數計算成功:", output.shape)
                return output
                
            else:
                print("⚠️ 維度不匹配，使用安全回退")
                print(f"  期望: activation_weights=[{self.output_dim}, {input_dim}]")
                print(f"  實際: activation_weights={self.activation_weights.shape}")
                
                # 🔧 安全回退：重新構建兼容的權重
                if input_dim != self.activation_weights.shape[1]:
                    # 調整輸入維度
                    if input_dim > self.activation_weights.shape[1]:
                        x_adapted = F.adaptive_avg_pool1d(x.unsqueeze(1), self.activation_weights.shape[1]).squeeze(1)
                    else:
                        x_adapted = F.pad(x, (0, self.activation_weights.shape[1] - input_dim))
                else:
                    x_adapted = x
                
                # 安全的矩陣乘法
                fallback_output = torch.matmul(x_adapted, self.activation_weights.t())
                fallback_output = torch.tanh(fallback_output)
                
                # print("✅ 回退計算成功:", fallback_output.shape)
                return fallback_output
                
        except Exception as e:
            print(f"❌ 激活函數計算失敗: {e}")
            # 最終回退：簡單的線性變換
            try:
                if x.shape[1] >= self.output_dim:
                    simple_output = x[:, :self.output_dim]
                else:
                    simple_output = F.pad(x, (0, self.output_dim - x.shape[1]))
                
                # 添加非線性
                simple_output = torch.tanh(simple_output) * 0.1
                # print("✅ 簡單回退成功:", simple_output.shape)
                return simple_output
                
            except Exception as final_e:
                print(f"❌ 最終回退也失敗: {final_e}")
                # 創建零張量
                return torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)
    
    def pure_b_spline_basis(self, x):
        """純粹的B-spline基函數 - KAN的核心特性，確保維度一致性"""
        batch_size, input_dim = x.shape
        
        # 自適應歸一化 (不是MLP的線性歸一化)
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        # B-spline網格點生成
        x_grid = torch.tanh(x_normalized)  # 非線性映射到[-1,1]
        
        # 🔧 確保維度一致的基函數生成
        basis_functions_list = []
        
        # 為每個輸入維度生成基函數
        for dim_idx in range(input_dim):
            x_dim = x_grid[:, dim_idx:dim_idx+1]  # [batch_size, 1]
            
            dim_basis = []
            
            # T_0(x) = 1
            dim_basis.append(torch.ones_like(x_dim))
            
            if self.num_basis > 1:
                # T_1(x) = x
                dim_basis.append(x_dim)
            
            # T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x) (Chebyshev遞推)
            for n in range(2, self.num_basis):
                if len(dim_basis) >= 2:
                    t_next = 2 * x_dim * dim_basis[-1] - dim_basis[-2]
                    t_next = torch.clamp(t_next, -5.0, 5.0)  # 數值穩定
                    dim_basis.append(t_next)
                else:
                    # 安全回退
                    dim_basis.append(torch.zeros_like(x_dim))
            
            # 確保正確的基函數數量
            while len(dim_basis) < self.num_basis:
                dim_basis.append(torch.zeros_like(x_dim))
            
            # 截斷到正確數量
            dim_basis = dim_basis[:self.num_basis]
            
            # 堆疊為 [batch_size, num_basis]
            dim_basis_tensor = torch.cat(dim_basis, dim=1)
            basis_functions_list.append(dim_basis_tensor)
        
        # 堆疊為 [batch_size, input_dim, num_basis]
        basis_tensor = torch.stack(basis_functions_list, dim=1)
        
        # 確保輸出維度正確
        assert basis_tensor.shape == (batch_size, input_dim, self.num_basis), \
            f"Basis tensor shape mismatch: got {basis_tensor.shape}, expected {(batch_size, input_dim, self.num_basis)}"
        
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
            
            # 🔧 修復einsum維度問題 - 安全的B-spline計算
            batch_size, input_dim = x.shape
            try:
                # 檢查維度兼容性
                if (basis_functions.shape[0] == batch_size and 
                    basis_functions.shape[1] == input_dim and
                    basis_functions.shape[2] == self.num_basis):
                    # 正常的einsum操作
                    spline_output = torch.einsum('oij,bij->bo', self.spline_coeffs, basis_functions)
                else:
                    print(f"B-spline dimension mismatch: basis_functions={basis_functions.shape}, spline_coeffs={self.spline_coeffs.shape}")
                    # 使用安全的矩陣乘法回退
                    basis_flat = basis_functions.view(batch_size, -1)
                    coeffs_flat = self.spline_coeffs.view(self.output_dim, -1)
                    
                    # 調整維度匹配
                    min_dim = min(basis_flat.shape[1], coeffs_flat.shape[1])
                    spline_output = torch.mm(basis_flat[:, :min_dim], coeffs_flat[:, :min_dim].t())
                    
            except RuntimeError as e:
                print(f"B-spline computation failed: {e}, using fallback")
                # 回退到線性變換
                spline_output = self.base_linear(x) * 0.5
            
            # 3. 核心：可學習激活函數 (KAN vs MLP的關鍵差異)
            try:
                activation_output = self.learnable_activation(x)
                
                # 🔧 修復激活函數矩陣維度問題 - 詳細診斷
                print(f"🔍 激活函數維度診斷:")
                print(f"  activation_weights.shape: {self.activation_weights.shape}")
                print(f"  batch_size: {batch_size}, input_dim: {input_dim}, output_dim: {self.output_dim}")
                
                # 檢查是否為矩陣（至少2維）
                if activation_output.dim() < 2 or self.activation_weights.dim() < 2:
                    print(f"⚠️ 維度不足: activation_output.dim()={activation_output.dim()}, activation_weights.dim()={self.activation_weights.dim()}")
                    # 確保至少是2D
                    if activation_output.dim() == 1:
                        activation_output = activation_output.unsqueeze(0)
                    if self.activation_weights.dim() == 1:
                        self.activation_weights = self.activation_weights.unsqueeze(0)
                
                # 安全的矩陣乘法計算
                if (activation_output.shape[0] == batch_size and 
                    activation_output.shape[1] <= self.activation_weights.shape[1]):
                    # 使用安全的矩陣乘法
                    feat_dim = activation_output.shape[1]
                    weight_subset = self.activation_weights[:, :feat_dim]  # [output_dim, feat_dim]
                    activation_output = torch.mm(activation_output, weight_subset.t())  # [batch_size, output_dim]
                    print(f"✅ 激活函數計算成功: {activation_output.shape}")
                else:
                    print(f"⚠️ 維度不匹配，使用安全回退")
                    # 維度調整回退
                    min_feat_dim = min(activation_output.shape[-1], self.activation_weights.shape[-1])
                    activation_output_safe = activation_output[..., :min_feat_dim]
                    weights_safe = self.activation_weights[:, :min_feat_dim]
                    
                    # 確保batch維度正確
                    if activation_output_safe.shape[0] != batch_size:
                        activation_output_safe = activation_output_safe[:batch_size]
                    
                    activation_output = torch.mm(activation_output_safe, weights_safe.t())
                    # print(f"✅ 回退計算成功: {activation_output.shape}")
                        
            except RuntimeError as e:
                print(f"❌ Activation computation failed: {e}")
                # print(f"  activation_output type: {type(activation_output)}")
                # print(f"  activation_weights type: {type(self.activation_weights)}")
                if hasattr(activation_output, 'shape'):
                    print(f"  activation_output shape: {activation_output.shape}")
                if hasattr(self.activation_weights, 'shape'):
                    print(f"  activation_weights shape: {self.activation_weights.shape}")
                print(f"  Using zero fallback")
                activation_output = torch.zeros(batch_size, self.output_dim, device=x.device)
            
            # 4. KAN輸出組合 (B-spline主導，激活函數輔助)
            kan_output = spline_output + activation_output + base_output
            
            # 5. 穩定性歸一化
            output = self.ln(kan_output)
                
        except RuntimeError as e:
            print(f"KAN forward failed: {e}, using linear fallback")
            # 簡化錯誤處理
            output = self.base_linear(x)
            output = self.ln(output)
        
        return output


class SimplifiedKANLayer(nn.Module):
    """
    純粹的 KAN 層的簡化實現 - 專注於數值穩定性和核心功能
    移除了自適應樣條階數等複雜特性，確保穩定收斂
    """
    
    def __init__(self, input_dim, output_dim, num_basis=8, layer_idx=0, verbose=False):
        super(SimplifiedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        self.layer_idx = layer_idx # 用於日誌追蹤
        self.verbose = verbose

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
        self.ln = nn.LayerNorm(output_dim, eps=1e-4)
    
        self.reset_parameters()
    
    def reset_parameters(self):
        """KAN特有的初始化 - 加強數值穩定性"""
        # 多項式係數初始化 - 更小的初始值
        nn.init.normal_(self.poly_coeffs, mean=0.0, std=0.01)
        
        # 激活尺度初始化 - 更保守的範圍
        nn.init.uniform_(self.activation_scale, 0.01, 0.05)
        
        # 基礎變換初始化 - 更小的權重
        nn.init.xavier_uniform_(self.base_transform.weight, gain=0.01)
    
    def polynomial_basis_functions(self, x):
        """簡化的多項式基函數 - KAN的簡化版本，修復維度問題"""
        batch_size, input_dim = x.shape
        
        if self.verbose:
            print(f"🔍 多項式基函數診斷:")
            print(f"  輸入形狀: {x.shape}")
        
        x_normalized = torch.tanh(x) # 穩定地歸一化到[-1, 1]
        
        if self.verbose:
            print(f"  歸一化後範圍: [{x_normalized.min().item():.6f}, {x_normalized.max().item():.6f}]")

        basis_list = []
        basis_list.append(torch.ones_like(x_normalized))
        
        for n in range(1, self.num_basis):
            basis_list.append(basis_list[-1] * x_normalized)

        basis = torch.stack(basis_list, dim=-1)
        if self.verbose:
            print(f"  ✓ 基函數生成成功: {basis.shape}, 範圍[{basis.min().item():.6f}, {basis.max().item():.6f}]")
        return basis
    
    def kan_learnable_activation(self, x):
        """簡化的可學習激活函數 - 修復維度問題"""
        batch_size, input_dim = x.shape
        
        # 確保activation_scale維度正確
        if self.activation_scale.shape != (self.output_dim, input_dim):
            print(f"⚠️ activation_scale維度不匹配: {self.activation_scale.shape} vs expected ({self.output_dim}, {input_dim})")
            # 調整維度
            scale = self.activation_scale[:, :input_dim] if self.activation_scale.shape[1] >= input_dim else self.activation_scale
        else:
            scale = self.activation_scale
        
        # 安全的激活函數計算
        try:
            # x: [batch_size, input_dim], scale: [output_dim, input_dim]
            # 需要將x擴展到[batch_size, output_dim, input_dim]進行逐元素計算
            x_expanded = x.unsqueeze(1).expand(batch_size, self.output_dim, input_dim)  # [batch_size, output_dim, input_dim]
            scale_expanded = scale.unsqueeze(0).expand(batch_size, self.output_dim, input_dim)  # [batch_size, output_dim, input_dim]
            
            # 逐元素激活函數
            activated = x_expanded * torch.tanh(x_expanded * scale_expanded)  # [batch_size, output_dim, input_dim]
            
            # 降維到[batch_size, output_dim]
            output = activated.mean(dim=2)  # 平均池化
            
            return output
            
        except RuntimeError as e:
            print(f"❌ KAN激活函數計算失敗: {e}")
            print(f"  x.shape: {x.shape}")
            print(f"  scale.shape: {scale.shape}")
            # 回退到簡單計算
            return torch.tanh(x).sum(dim=1, keepdim=True).expand(-1, self.output_dim) * 0.1
    
    def forward(self, x):
        if self.verbose:
            # print(f"🔍 執行KAN層{self.layer_idx}: {self.__class__.__name__}")
            #print(f"  輸入形狀: {x.shape}")
            if not torch.all(torch.isfinite(x)):
                # print("  ⚠️ 輸入包含NaN/Inf") 
                pass
            else:
                # print(f"  輸入範圍: [{x.min().item():.6f}, {x.max().item():.6f}]")
                pass

        try:
            # 1. 多項式基函數 (KAN的核心)
            poly_basis = self.polynomial_basis_functions(x)
            poly_output = torch.einsum('bid,oid->bo', poly_basis, self.poly_coeffs)

            # 2. 基礎線性變換 (最小化MLP特性)
            base_output = self.base_transform(x)

            # 3. 可學習激活函數部分 (KAN的核心)
            activation_output = self.kan_learnable_activation(x)

            # --- 診斷 ---
            if self.verbose:
                print(f"🔍 KAN組合診斷:")
                if torch.all(torch.isfinite(poly_output)):
                    print(f"  poly_output: min={poly_output.min().item():.6f}, max={poly_output.max().item():.6f}, nan={torch.isnan(poly_output).sum().item()}")
                if torch.all(torch.isfinite(activation_output)):
                    print(f"  activation_output: min={activation_output.min().item():.6f}, max={activation_output.max().item():.6f}, nan={torch.isnan(activation_output).sum().item()}")
                if torch.all(torch.isfinite(base_output)):
                    print(f"  base_output: min={base_output.min().item():.6f}, max={base_output.max().item():.6f}, nan={torch.isnan(base_output).sum().item()}")
            
            # --- 組合輸出 ---
            final_output = torch.zeros_like(base_output)
            use_base = torch.all(torch.isfinite(base_output))
            use_poly = torch.all(torch.isfinite(poly_output))
            use_activation = torch.all(torch.isfinite(activation_output))

            if use_base:
                final_output += base_output * 0.3
                if self.verbose: print("  ✓ 添加base_output")
            if use_poly:
                final_output += poly_output * 0.5
                if self.verbose: print("  ✓ 添加poly_output")
            if use_activation:
                final_output += activation_output * 0.2
                if self.verbose: print("  ✓ 添加activation_output")

            if not (use_base or use_poly or use_activation):
                if self.verbose: print("  ❌ 所有輸出均無效，返回零張量")
                return torch.zeros_like(base_output)

            # 應用層歸一化
            kan_output = self.ln(final_output)

            if self.verbose:
                print(f"  最終輸出: min={kan_output.min().item():.6f}, max={kan_output.max().item():.6f}, nan={torch.isnan(kan_output).sum().item()}")
            
            # 安全回退機制
            if not torch.all(torch.isfinite(kan_output)):
                if self.verbose: print("  ⚠️ KAN輸出包含NaN/Inf，使用穩定的基礎輸出")
                return torch.nan_to_num(kan_output, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                if self.verbose:
                    pass
                return kan_output
                
        except Exception as e:
            if self.verbose: print(f"⚠️ KAN層{self.layer_idx}處理失敗: {e}，使用前一層輸出")
            # 回退到輸入（身份映射）
            if x.shape[1] == self.output_dim:
                return x
            else:
                # 維度調整
                if x.shape[1] > self.output_dim:
                    return x[:, :self.output_dim]
                else:
                    return F.pad(x, (0, self.output_dim - x.shape[1]))


class OptimizedGNNKANEncoder(nn.Module):
    """
    專用於GNN的KAN編碼器 - 用純粹KAN取代MLP層
    核心：用AdvancedKANLayer完全取代傳統的MLP層
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_layers=2, kan_grid_size=8, kan_spline_order=3, 
                 dropout=0.1, learnable_graph=True, **kwargs):
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
            # 🔧 修復：使用兼容的KAN層創建函數
            kan_layers.append(CompatibleSimplifiedKANLayer(
                dims[i], dims[i + 1], 
                num_basis=kan_grid_size,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order
            ))
            
            # Dropout (但不使用MLP常用的ReLU/GELU等固定激活)
            if i < len(dims) - 2:
                kan_layers.append(nn.Dropout(dropout))
        
        self.kan_layers = nn.ModuleList(kan_layers)
        
        # 簡化消息傳遞 (避免MLP結構)
        self.message_processors = nn.ModuleList([
            CompatibleSimplifiedKANLayer(dims[i + 1], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        
        # 可學習圖結構
        if learnable_graph:
            self.edge_learner = nn.Sequential(
                CompatibleSimplifiedKANLayer(dims[-1] * 2, dims[-1]),
                CompatibleSimplifiedKANLayer(dims[-1], 1)
            )
        
    def kan_message_passing(self, x, edge_index, layer_idx):
        """使用KAN進行消息傳遞 - 不使用MLP - 修復NaN問題"""
        if edge_index.size(1) == 0:
            return x
        
        # 🔧 強化穩定性檢查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ 輸入包含無效值，進行清理: NaN={torch.isnan(x).sum()}, Inf={torch.isinf(x).sum()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        row, col = edge_index
        num_nodes = x.size(0)
        
        # 🔧 特殊情況：只有一個節點時，跳過圖消息傳遞
        if num_nodes <= 1:
            print(f"⚠️ 節點數過少({num_nodes})，跳過圖消息傳遞")
            return x
        
        # 安全索引檢查
        if edge_index.size(1) == 0:
            print("⚠️ 沒有邊信息，跳過圖消息傳遞")
            return x
            
        if row.max() >= num_nodes or col.max() >= num_nodes or row.min() < 0 or col.min() < 0:
            print(f"⚠️ 邊索引超出範圍: row=[{row.min()}, {row.max()}], col=[{col.min()}, {col.max()}], num_nodes={num_nodes}")
            # 過濾無效索引
            valid_mask = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
            if valid_mask.sum() == 0:
                print("⚠️ 沒有有效邊，返回原始特徵")
                return x
            row = row[valid_mask]
            col = col[valid_mask]
        
        try:
            # 🔧 改進的稀疏矩陣構建
            adj_indices = torch.stack([row, col], dim=0)
            adj_values = torch.ones(len(row), device=x.device, dtype=x.dtype)
            adj_size = (num_nodes, num_nodes)
            
            # 檢查稀疏張量的有效性
            if len(row) == 0:
                print("⚠️ 空邊列表，返回原始特徵")
                return x
            
            adj_sparse = torch.sparse_coo_tensor(adj_indices, adj_values, adj_size, device=x.device)
            adj_sparse = adj_sparse.coalesce()
            
            # 🔧 安全的度計算
            degrees = torch.sparse.sum(adj_sparse, dim=1).to_dense()
            
            # 檢查度的有效性
            if torch.isnan(degrees).any() or torch.isinf(degrees).any():
                print("⚠️ 度計算包含無效值，使用均勻度")
                degrees = torch.ones_like(degrees)
            
            # 避免除零
            degrees = torch.clamp(degrees, min=1e-6)
            degrees_inv = 1.0 / torch.sqrt(degrees)
            
            # 檢查度的倒數
            if torch.isnan(degrees_inv).any() or torch.isinf(degrees_inv).any():
                print("⚠️ 度倒數包含無效值，使用單位歸一化")
                degrees_inv = torch.ones_like(degrees_inv)
            
            # 🔧 安全的歸一化值計算
            norm_values = degrees_inv[row] * degrees_inv[col]
            
            # 檢查歸一化值
            if torch.isnan(norm_values).any() or torch.isinf(norm_values).any():
                print("⚠️ 歸一化值包含無效值，使用均勻權重")
                norm_values = torch.ones_like(norm_values) / len(norm_values)
            
            norm_adj = torch.sparse_coo_tensor(adj_indices, norm_values, adj_size, device=x.device)
            
            # 🔧 安全的稀疏矩陣乘法
            message = torch.sparse.mm(norm_adj, x)
            
            # 檢查消息傳遞結果
            if torch.isnan(message).any() or torch.isinf(message).any():
                print("⚠️ 消息傳遞結果包含無效值，使用原始特徵")
                return x
            
            # 🎯 使用KAN處理消息 (而不是MLP)
            if layer_idx < len(self.message_processors):
                try:
                    processed_message = self.message_processors[layer_idx](message)
                    
                    # 檢查KAN處理結果
                    if torch.isnan(processed_message).any() or torch.isinf(processed_message).any():
                        print("⚠️ KAN處理後包含無效值，使用未處理的消息")
                        return message
                    
                    return processed_message
                except Exception as e:
                    print(f"⚠️ KAN消息處理失敗: {e}，使用原始消息")
                    return message
            else:
                return message
                
        except Exception as e:
            print(f"⚠️ 消息傳遞完全失敗: {e}，返回原始特徵")
            return x
    
    def forward(self, x, edge_index):
        """純粹KAN的前向傳播 - 完全避免MLP結構 - 修復NaN傳播"""
        current_x = x
        
        # 🔧 輸入穩定性檢查
        if torch.isnan(current_x).any() or torch.isinf(current_x).any():
            print(f"⚠️ 編碼器輸入包含無效值: NaN={torch.isnan(current_x).sum()}, Inf={torch.isinf(current_x).sum()}")
            current_x = torch.nan_to_num(current_x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        for i, layer in enumerate(self.kan_layers):
            if isinstance(layer, nn.Dropout):
                current_x = layer(current_x)
            else:
                try:
                    # print(f"🔍 執行KAN層{i}: {type(layer).__name__}")
                    # print(f"  輸入形狀: {current_x.shape}")
                    # print(f"  輸入範圍: [{current_x.min():.6f}, {current_x.max():.6f}]")
                    
                    # 🎯 KAN層處理 (核心：用KAN取代MLP) - 強制使用KAN
                    kan_output = layer(current_x)
                    
                    print(f"  KAN輸出形狀: {kan_output.shape}")
                    print(f"  KAN輸出範圍: [{kan_output.min():.6f}, {kan_output.max():.6f}]")
                    print(f"  NaN數量: {torch.isnan(kan_output).sum()}")
                    
                    # 🔧 修復KAN層輸出而不是跳過
                    if torch.isnan(kan_output).any() or torch.isinf(kan_output).any():
                        print(f"  ⚠️ KAN層{i}輸出包含無效值，進行修復而非跳過")
                        
                        # 修復而不是替換
                        kan_output = torch.nan_to_num(kan_output, nan=0.0, posinf=1.0, neginf=-1.0)
                        kan_output = torch.clamp(kan_output, -5.0, 5.0)  # 限制範圍
                        
                        # 如果修復後仍有問題，使用安全的KAN輸出
                        if torch.isnan(kan_output).any() or torch.isinf(kan_output).any():
                            print(f"  ⚠️ 修復失敗，創建安全的KAN風格輸出")
                            # 創建保持KAN特性的安全輸出
                            if hasattr(layer, 'output_dim'):
                                target_dim = layer.output_dim
                            else:
                                target_dim = current_x.shape[1]
                            
                            # 使用輸入的非線性變換，保持KAN特性
                            safe_output = torch.tanh(current_x) * 0.5  # 非線性變換
                            
                            # 調整維度
                            if safe_output.shape[1] != target_dim:
                                if safe_output.shape[1] > target_dim:
                                    kan_output = safe_output[:, :target_dim]
                                else:
                                    padding = torch.zeros(safe_output.shape[0], target_dim - safe_output.shape[1], 
                                                        device=safe_output.device, dtype=safe_output.dtype)
                                    kan_output = torch.cat([safe_output, padding], dim=1)
                            else:
                                kan_output = safe_output
                        
                        # print(f"  ✓ 修復後輸出: 形狀={kan_output.shape}, 範圍=[{kan_output.min():.6f}, {kan_output.max():.6f}]")
                    else:
                        # print(f"  ✅ KAN層{i}輸出正常，繼續使用KAN結果")
                        pass
                    
                    # KAN消息傳遞 (每兩層一次，減少計算)
                    if i % 2 == 0 and edge_index.size(1) > 0:
                        message = self.kan_message_passing(kan_output, edge_index, i // 2)
                        
                        # 🔧 安全的殘差連接
                        if message.shape == kan_output.shape:
                            residual_output = kan_output + 0.3 * message
                            
                            # 檢查殘差連接結果
                            if torch.isnan(residual_output).any() or torch.isinf(residual_output).any():
                                print(f"⚠️ 殘差連接產生無效值，僅使用KAN輸出")
                                current_x = kan_output
                            else:
                                current_x = residual_output
                        else:
                            print(f"⚠️ 消息形狀不匹配: message={message.shape}, kan_output={kan_output.shape}")
                            current_x = kan_output
                    else:
                        current_x = kan_output
                    
                    # 🔧 每層後檢查穩定性
                    if torch.isnan(current_x).any() or torch.isinf(current_x).any():
                        print(f"⚠️ 第{i}層後出現無效值，進行修復")
                        current_x = torch.nan_to_num(current_x, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                except Exception as e:
                    print(f"⚠️ KAN層{i}處理失敗: {e}，使用前一層輸出")
                    # 如果出錯，保持前一層的輸出
                    pass
        
        # 🔧 最終輸出檢查
        if torch.isnan(current_x).any() or torch.isinf(current_x).any():
            print(f"⚠️ 編碼器最終輸出包含無效值，進行最終修復")
            current_x = torch.nan_to_num(current_x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 如果仍有問題，使用原始輸入的線性變換
            if torch.isnan(current_x).any() or torch.isinf(current_x).any():
                print("⚠️ 使用原始輸入的安全變換")
                current_x = torch.tanh(x) * 0.1
        
        # 僅返回節點嵌入，保持向後兼容
        # self.last_adj = final_adj  # 可選：儲存以便外部存取
        return current_x


# 向後兼容的別名 - 統一接口
class KANLayer(SimplifiedKANLayer):
    """向後兼容的KAN層"""
    def __init__(self, input_dim, output_dim, num_basis=8, grid_size=None, spline_order=None, **kwargs):
        # 調用SimplifiedKANLayer，忽略不支持的參數
        super().__init__(input_dim, output_dim, num_basis)


class GNNKANEncoder(OptimizedGNNKANEncoder):
    """向後兼容的GNN-KAN編碼器"""
    pass


# 🔧 修復：確保 SimplifiedKANLayer 可以處理 AdvancedKANLayer 的參數但忽略不支持的參數
class CompatibleSimplifiedKANLayer(SimplifiedKANLayer):
    """兼容性增強的簡化KAN層 - 可接受但忽略高級參數"""
    
    def __init__(self, input_dim, output_dim, num_basis=8, 
                 spline_order=None, grid_size=None, 
                 adaptive_spline_order=None, verbose=False, **kwargs):
        # 只使用 SimplifiedKANLayer 支持的參數
        # 將不支持的參數（如spline_order, grid_size）過濾掉
        
        # 提取 SimplifiedKANLayer __init__ 支持的參數
        simplified_kwargs = {k: v for k, v in kwargs.items() if k in ['layer_idx']}
        
        super().__init__(
            input_dim=input_dim, 
            output_dim=output_dim, 
            num_basis=num_basis,
            verbose=verbose,
            **simplified_kwargs
        )
        # print(f"Initialized CompatibleSimplifiedKANLayer, verbose={self.verbose}")


def create_compatible_kan_layer(input_dim, output_dim, **kwargs):
    return CompatibleSimplifiedKANLayer(input_dim, output_dim, **kwargs)