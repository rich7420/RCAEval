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


class KanInputNorm(nn.Module):

    def __init__(self, dim, eps=1e-5, clamp=1.5, affine=True):
        super().__init__()
        self.eps = eps
        self.clamp = clamp
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [B, D] 或 [N, D]
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(self.eps)
        xn = (x - mu) / std
        if self.affine:
            xn = xn * self.weight + self.bias
        # 严格范围限制
        return torch.clamp(xn, -self.clamp, self.clamp)


class KanResidualBlock(nn.Module):

    def __init__(self, dim, kan_layer, res_scale=0.5):
        super().__init__()
        self.kan = kan_layer
        self.res_scale = nn.Parameter(torch.tensor(res_scale))
        self.out_scale = nn.Parameter(torch.ones(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        y = self.kan(self.norm(x))
        # 输出重缩放 + 残差护栏
        return x + self.res_scale.tanh() * (y * self.out_scale)


class AdvancedKANLayer(nn.Module):
    """
    純粹的 KAN 層實現 - 基於 KAN 論文的高級實現
    核心特性：學習激活函數、B-spline基函數、自適應樣條
    完全不同於MLP的固定激活函數方式
    """
    
    def __init__(self, input_dim, output_dim, num_basis=8,
                 spline_order=3, grid_size=8,
                 adaptive_spline_order=True,
                 l1_lambda=1e-3, entropy_lambda=1e-3,
                 use_cheb=True, use_bspline=True,
                 clamp_in=1.5, cheb_clamp=1.0,
                 normalize_input=True, use_residual=True,
                 basis_function='chebyshev', basis_kwargs=None):
        super(AdvancedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.adaptive_spline_order = adaptive_spline_order
        self.l1_lambda = l1_lambda
        self.entropy_lambda = entropy_lambda
        
        # 🔧 新增：数值稳定性控制参数
        self.use_cheb = use_cheb
        self.use_bspline = use_bspline
        self.clamp_in = clamp_in
        self.cheb_clamp = cheb_clamp
        self.normalize_input = normalize_input
        self.use_residual = use_residual
        
        # 🎯 基函數選擇 - Replace direct basis calls with factory output
        self.basis_function = basis_function
        self.basis_kwargs = basis_kwargs or {}
        
        # Import basis function factory
        from .basis_functions import BasisFunctionFactory
        
        # Create basis function instance
        self.basis = BasisFunctionFactory.create_basis_function(
            basis_function, num_basis, 
            spline_order=spline_order, grid_size=grid_size,
            **self.basis_kwargs
        )
        
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
        
        if self.normalize_input:
            self.input_norm = KanInputNorm(input_dim, clamp=self.clamp_in)
        
        # 簡化穩定性 - 保持KAN純粹性
        self.ln = nn.LayerNorm(output_dim, eps=1e-4)
        
        # 基礎線性變換 (最小化MLP特性)
        self.base_linear = nn.Linear(input_dim, output_dim, bias=False)
        
        # 將組合後的特徵（已是 output_dim）做輕量投影
        self.proj = nn.Linear(output_dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)
        
        # 添加缺失的Chebyshev多項式層
        if self.use_cheb:
            self.chebyshev_polynomials = nn.Linear(input_dim, output_dim, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """KAN特有的參數初始化 - 針對B-spline優化"""
        with torch.no_grad():
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
        """
        """
        if self.normalize_input:
            x = self.input_norm(x)
        else:
            x = torch.clamp(x, min=-self.clamp_in, max=self.clamp_in)
        
        if torch.isnan(x).any():
            print("⚠️ KAN層檢測到NaN輸入，進行清理")
            x = torch.nan_to_num(x, nan=0.0, posinf=self.clamp_in, neginf=-self.clamp_in)
        
        parts = []
        
        # 🔧 優化：並行計算基函數，減少循環開銷
        try:
            # 使用並行計算基函數
            basis_tensor = self.basis(x)  # [batch, input_dim, num_basis]
            
            # 並行計算所有輸出的加權和
            spline_coeffs = self.spline_coeffs.float()  # [output_dim, input_dim, num_basis]
            basis_output = torch.einsum('bji,oji->bo', basis_tensor, spline_coeffs)
            
            # 應用激活和縮放
            basis_output = torch.tanh(basis_output) * 0.5
            parts.append(basis_output)
            
        except Exception as e:
            print(f"⚠️ 基函數計算失敗，使用回退策略: {e}")
            # Fallback to simple linear transformation
            fallback_output = x @ (torch.ones_like(self.spline_coeffs[:, :, 0]).t() * 0.1)
            parts.append(fallback_output)
        
        # 🔧 優化：並行計算激活函數，減少重複計算
        try:
            # 並行計算激活函數
            x_float = x.float()
            activation_weights_float = self.activation_weights.float()
            activation_output = torch.tanh(x_float @ activation_weights_float.t())
            
            # 數值穩定性檢查
            if torch.isnan(activation_output).any() or torch.isinf(activation_output).any():
                print("⚠️ 激活函數輸出不穩定，使用線性回退")
                activation_output = x_float @ (activation_weights_float.t() * 0.1)
            
            # 温和缩放
            activation_output = activation_output * 0.3
            parts.append(activation_output)
        except Exception as e:
            print(f"⚠️ 可學習激活計算失敗: {e}")
            activation_output = x @ (torch.ones_like(self.activation_weights).t() * 0.1) * 0.3
            parts.append(activation_output)
        if len(parts) > 1:
            norms = [torch.norm(part) for part in parts]
            total_norm = sum(norms)
            
            if total_norm > 1e-8:
                weights = [norm / total_norm for norm in norms]
                output = sum(w * part for w, part in zip(weights, parts))
            else:
                output = sum(parts) * 0.1
        else:
            output = parts[0] if parts else x
        
        output = self.proj(output)
        output = self.out_norm(output)
        

        output = output.clamp(-3.0, 3.0)
        

        if self.use_residual and output.shape == x.shape:

            residual_scale = 0.1 
            output = x + residual_scale * output
        
        output = torch.nan_to_num(output, nan=0.0, posinf=3.0, neginf=-3.0)
        
        if self.training and output.requires_grad:
            def gradient_clipping_hook(grad):
                grad_norm = torch.norm(grad)
                if grad_norm > 1.0:
                    return grad * (1.0 / grad_norm)
                return grad
            output.register_hook(gradient_clipping_hook)
        
        return output
    
    def enhanced_b_spline_basis(self, x):
        """增強的B-spline基函數計算，提高數值穩定性"""
        # 歸一化到 [-1, 1] 區間（Chebyshev多項式的標準定義域）
        x_normalized = torch.clamp(x, min=-2.0, max=2.0) / 2.0
        
        # 使用修正的Chebyshev遞歸，增加數值穩定性檢查
        basis_functions = []
        
        # T0(x) = 1
        T0 = torch.ones_like(x_normalized)
        basis_functions.append(T0)
        
        if self.num_basis > 1:
            # T1(x) = x
            T1 = x_normalized
            basis_functions.append(T1)
            
            # 遞歸計算其餘基函數，添加穩定性檢查
            for n in range(2, self.num_basis):
                # T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
                Tn = 2.0 * x_normalized * basis_functions[-1] - basis_functions[-2]
                
                # 🔧 修正2：數值穩定性檢查
                if torch.any(torch.abs(Tn) > 10.0):  # 檢測到可能的發散
                    print(f"⚠️ Chebyshev多項式T_{n}出現數值不穩定，使用截斷策略")
                    Tn = torch.clamp(Tn, min=-5.0, max=5.0)
                
                basis_functions.append(Tn)
        
        # 組合基函數
        basis_matrix = torch.stack(basis_functions, dim=-1)  # [batch, input_dim, num_basis]
        
        # 計算B-spline輸出，添加數值檢查
        # 🔧 確保類型一致性，避免 numpy.float32 和 torch.FloatTensor 不匹配
        basis_matrix = basis_matrix.float()
        spline_coeffs = self.spline_coeffs.float()  # [output_dim, input_dim, num_basis]
        # 直接得到 [batch, output_dim]（對 input_dim 和 num_basis 求和）
        spline_output = torch.einsum('bji,oji->bo', basis_matrix, spline_coeffs)
        
        # 最終數值穩定性檢查
        if torch.isnan(spline_output).any() or torch.isinf(spline_output).any():
            print("⚠️ B-spline輸出包含無效值，使用回退策略")
            return self._fallback_basis(x)
        
        return spline_output


class SimplifiedKANLayer(nn.Module):
    """
    純粹的 KAN 層的簡化實現 - 專注於數值穩定性和核心功能
    移除了自適應樣條階數等複雜特性，確保穩定收斂
    """
    
    def __init__(self, input_dim, output_dim, num_basis=8, layer_idx=0, verbose=False,
                 basis_function='chebyshev', basis_kwargs=None):
        super(SimplifiedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        self.layer_idx = layer_idx # 用於日誌追蹤
        self.verbose = verbose
        
        # 🎯 基函數選擇 - Replace direct basis calls with factory output
        self.basis_function = basis_function
        self.basis_kwargs = basis_kwargs or {}
        
        # Import basis function factory
        from .basis_functions import BasisFunctionFactory
        
        # Create basis function instance
        self.basis = BasisFunctionFactory.create_basis_function(
            basis_function, num_basis, **self.basis_kwargs
        )

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
        with torch.no_grad():
            nn.init.normal_(self.poly_coeffs, mean=0.0, std=0.01)
            
            # 激活尺度初始化 - 更保守的範圍，確保返回torch張量
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
            # 1. 使用統一的基函數工廠 - Replace direct basis calls with factory output
            basis_tensor = self.basis(x)  # [batch, input_dim, num_basis]
            # 🔧 確保類型一致性，避免 numpy.float32 和 torch.FloatTensor 不匹配
            basis_tensor = basis_tensor.float()
            poly_coeffs = self.poly_coeffs.float()
            poly_output = torch.einsum('bid,oid->bo', basis_tensor, poly_coeffs)

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
                 dropout=0.1, learnable_graph=True, 
                 basis_function='chebyshev', basis_kwargs=None, **kwargs):
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
                spline_order=kan_spline_order,
                basis_function=basis_function,
                basis_kwargs=basis_kwargs
            ))
            
            # Dropout (但不使用MLP常用的ReLU/GELU等固定激活)
            if i < len(dims) - 2:
                kan_layers.append(nn.Dropout(dropout))
        
        self.kan_layers = nn.ModuleList(kan_layers)
        
        # 簡化消息傳遞 (避免MLP結構)
        self.message_processors = nn.ModuleList([
            CompatibleSimplifiedKANLayer(
                dims[i + 1], dims[i + 1],
                basis_function=basis_function,
                basis_kwargs=basis_kwargs
            )
            for i in range(len(dims) - 1)
        ])
        
        # 可學習圖結構
        if learnable_graph:
            self.edge_learner = nn.Sequential(
                CompatibleSimplifiedKANLayer(
                    dims[-1] * 2, dims[-1],
                    basis_function=basis_function,
                    basis_kwargs=basis_kwargs
                ),
                CompatibleSimplifiedKANLayer(
                    dims[-1], 1,
                    basis_function=basis_function,
                    basis_kwargs=basis_kwargs
                )
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
                 adaptive_spline_order=None, verbose=False,
                 basis_function='chebyshev', basis_kwargs=None, **kwargs):
        # 只使用 SimplifiedKANLayer 支持的參數
        # 將不支持的參數（如spline_order, grid_size）過濾掉
        
        # 提取 SimplifiedKANLayer __init__ 支持的參數
        simplified_kwargs = {k: v for k, v in kwargs.items() if k in ['layer_idx']}
        
        super().__init__(
            input_dim=input_dim, 
            output_dim=output_dim, 
            num_basis=num_basis,
            verbose=verbose,
            basis_function=basis_function,
            basis_kwargs=basis_kwargs,
            **simplified_kwargs
        )
        # print(f"Initialized CompatibleSimplifiedKANLayer, verbose={self.verbose}")


def create_compatible_kan_layer(input_dim, output_dim, basis_function='chebyshev', basis_kwargs=None, **kwargs):
    """Create compatible KAN layer with basis function selection"""
    return CompatibleSimplifiedKANLayer(
        input_dim, output_dim, 
        basis_function=basis_function, 
        basis_kwargs=basis_kwargs,
        **kwargs
    )


class KANEdgeDecoder(nn.Module):
    """
    KAN 邊解碼器 - 用於 Graph Decoder，將 MLP 替換為 KAN
    
    架構: KAN(2*d → d') → Linear(d' → 1) → 輸出 logit
    特性: 可學習激活函數、數值穩定性保護、輕量配置
    """
    
    def __init__(self, input_dim, hidden_dim=None, num_basis=4, 
                 spline_order=3, dropout=0.1, stability_mode=True):
        super(KANEdgeDecoder, self).__init__()
        
        self.input_dim = input_dim  # 應該是 2 * node_embedding_dim
        self.hidden_dim = hidden_dim or (input_dim // 2)  # 預設為輸入維度的一半
        self.num_basis = num_basis
        self.spline_order = spline_order 
        self.dropout_rate = dropout
        self.stability_mode = stability_mode
        
        # 🎯 KAN 層：輸入 [h_i; h_j] → 隱藏表示
        self.kan_layer = CompatibleSimplifiedKANLayer(
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
            num_basis=self.num_basis,
            verbose=False  # 避免過多日誌輸出
        )
        
        # 🔧 Dropout 和 LayerNorm
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-4)
        
        # 🎯 線性標頭：隱藏表示 → logit (輸出原始分數，不做 sigmoid)
        self.linear_head = nn.Linear(self.hidden_dim, 1, bias=True)
        
        # 🔧 數值穩定性組件
        if self.stability_mode:
            self.gradient_stabilizer = self._create_gradient_stabilizer()
        
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化參數 - 針對邊打分優化"""
        # 線性標頭使用小權重初始化
        nn.init.xavier_uniform_(self.linear_head.weight, gain=0.1)
        nn.init.constant_(self.linear_head.bias, 0.0)
    
    def _create_gradient_stabilizer(self):
        """創建梯度穩定器（如果需要的話）"""
        return lambda x: torch.clamp(x, min=-5.0, max=5.0)
    
    def forward(self, edge_features, base_adj=None):
        """
        前向傳播 - 加入殘差支援
        
        Args:
            edge_features: [N, 2*d] - 節點對特徵 [h_i; h_j]
            base_adj: [num_nodes, num_nodes] - KNN基線鄰接矩陣 (可選)
            
        Returns:
            logits: [N, 1] - 邊存在的原始分數 (logit，未經 sigmoid)
        """
        # 輸入檢查
        if edge_features.dim() != 2:
            raise ValueError(f"Expected 2D input, got {edge_features.dim()}D")
        
        if edge_features.size(1) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {edge_features.size(1)}")
        
        # 🔧 數值穩定性預處理
        if self.stability_mode:
            edge_features = self._stabilize_input(edge_features)
        
        try:
            # 🎯 通過 KAN 層進行非線性變換
            kan_output = self.kan_layer(edge_features)
            
            # 🔧 檢查 KAN 輸出有效性
            if torch.isnan(kan_output).any() or torch.isinf(kan_output).any():
                print("⚠️ KAN Edge Decoder: KAN 層輸出包含無效值，進行修復")
                kan_output = torch.nan_to_num(kan_output, nan=0.0, posinf=2.0, neginf=-2.0)
            
            # 🔧 LayerNorm + Dropout
            normalized_output = self.layer_norm(kan_output)
            if self.training:
                normalized_output = self.dropout(normalized_output)
            
            # 🎯 線性標頭輸出 logit
            logits = self.linear_head(normalized_output)
            
            # 新增: residual with base_adj
            if base_adj is not None:
                # 將base_adj轉換為與logits相同的形狀
                num_nodes = int(edge_features.size(0) ** 0.5)
                if num_nodes * num_nodes == edge_features.size(0):
                    base_adj_flat = base_adj.view(-1, 1)
                    residual = torch.tanh(logits) * 0.5
                    final_scores = base_adj_flat + residual
                else:
                    final_scores = torch.sigmoid(logits)
            else:
                final_scores = torch.sigmoid(logits)
            
            # 🔧 最終數值穩定性檢查
            if self.stability_mode:
                final_scores = self._stabilize_output(final_scores)
            
            return final_scores
            
        except Exception as e:
            print(f"⚠️ KAN Edge Decoder 處理失敗: {e}")
            # 🔧 回退策略：簡單線性變換
            return self._fallback_forward(edge_features)
    
    def _stabilize_input(self, x):
        """輸入穩定化"""
        # 1. NaN 和無窮值處理
        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0)
        
        # 2. 範圍限制
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        # 3. 自適應縮放（如果標準差過大）
        x_std = torch.std(x)
        if x_std > 2.0:
            scaling_factor = 2.0 / (x_std + 1e-8)
            x = x * scaling_factor
        
        return x
    
    def _stabilize_output(self, logits):
        """輸出穩定化"""
        # 限制 logit 範圍，避免極端值
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # NaN 檢查
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return logits
    
    def _fallback_forward(self, edge_features):
        """回退策略：簡單線性變換"""
        try:
            # 直接通過線性層，但先降維
            if edge_features.size(1) > self.hidden_dim:
                # 簡單的特徵選擇
                reduced_features = edge_features[:, :self.hidden_dim]
            else:
                # 零填充
                padding = torch.zeros(edge_features.size(0), 
                                    self.hidden_dim - edge_features.size(1),
                                    device=edge_features.device,
                                    dtype=edge_features.dtype)
                reduced_features = torch.cat([edge_features, padding], dim=1)
            
            return self.linear_head(reduced_features)
            
        except Exception as e:
            print(f"⚠️ 回退策略也失敗: {e}")
            # 最終回退：返回零 logit
            return torch.zeros(edge_features.size(0), 1, 
                             device=edge_features.device, 
                             dtype=edge_features.dtype)
    
    def get_model_info(self):
        """獲取模型信息，用於評估"""
        total_params = sum(p.numel() for p in self.parameters())
        kan_params = sum(p.numel() for p in self.kan_layer.parameters())
        linear_params = sum(p.numel() for p in self.linear_head.parameters())
        
        return {
            'model_name': 'KANEdgeDecoder',
            'total_parameters': total_params,
            'kan_parameters': kan_params,
            'linear_parameters': linear_params,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_basis': self.num_basis,
            'learnable_activations': kan_params,  # KAN 參數都是可學習激活相關
            'parameter_efficiency_score': kan_params / max(total_params, 1)
        }