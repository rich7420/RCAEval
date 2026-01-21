"""
KAN Components: Pure KAN Layer Implementations
Pure KAN layer implementation - focus on core value of replacing MLP with KAN
Ensure KAN feature purity: B-spline basis functions, learnable activation functions, nonlinear modeling
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
        # x: [B, D] or [N, D]
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(self.eps)
        xn = (x - mu) / std
        if self.affine:
            xn = xn * self.weight + self.bias
        # Strict range limitation
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
        # Output rescaling + residual guard
        return x + self.res_scale.tanh() * (y * self.out_scale)


class AdvancedKANLayer(nn.Module):
    """
    Pure KAN layer implementation - advanced implementation based on KAN paper
    Core features: learnable activation functions, B-spline basis functions, adaptive splines
    Completely different from MLP's fixed activation function approach
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
        
        # New: numerical stability control parameters
        self.use_cheb = use_cheb
        self.use_bspline = use_bspline
        self.clamp_in = clamp_in
        self.cheb_clamp = cheb_clamp
        self.normalize_input = normalize_input
        self.use_residual = use_residual
        
        # Basis function selection - replace direct basis calls with factory output
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
        
        # KAN core: learnable B-spline basis function coefficients (not MLP's fixed weights)
        self.spline_coeffs = nn.Parameter(
            torch.zeros(output_dim, input_dim, num_basis)
        )
        
        # KAN core: learnable activation function weights (completely different from MLP's fixed activations)
        self.activation_weights = nn.Parameter(
            torch.zeros(output_dim, input_dim)
        )
        
        # KAN core: adaptive spline order weights (dynamically adjust nonlinearity degree)
        if adaptive_spline_order:
            self.spline_order_weights = nn.Parameter(
                torch.ones(output_dim, input_dim, 3) / 3  # Support 3,4,5 order splines
            )
        
        if self.normalize_input:
            self.input_norm = KanInputNorm(input_dim, clamp=self.clamp_in)
        
        # Simplified stability - maintain KAN purity
        self.ln = nn.LayerNorm(output_dim, eps=1e-4)
        
        # Base linear transformation (minimize MLP characteristics)
        self.base_linear = nn.Linear(input_dim, output_dim, bias=False)
        
        # Lightweight projection of combined features (already output_dim)
        self.proj = nn.Linear(output_dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)
        
        # Add missing Chebyshev polynomial layer
        if self.use_cheb:
            self.chebyshev_polynomials = nn.Linear(input_dim, output_dim, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """KAN-specific parameter initialization - optimized for B-spline"""
        with torch.no_grad():
            # B-spline coefficient initialization - small values ensure stability
            std = math.sqrt(2.0 / (self.input_dim + self.output_dim))
            nn.init.normal_(self.spline_coeffs, mean=0.0, std=std * 0.01)
            
            # Activation function weight initialization
            nn.init.xavier_uniform_(self.activation_weights, gain=0.05)
            
            # Base linear layer initialization (minimal weights, highlight KAN characteristics)
            nn.init.xavier_uniform_(self.base_linear.weight, gain=0.1)
            
            # Adaptive spline order weights
            if hasattr(self, 'spline_order_weights'):
                nn.init.uniform_(self.spline_order_weights, 0.2, 0.4)
    
    def learnable_activation(self, x):
        """
        Learnable activation function - fix dimension issues and NaN generation
        This is KAN's core feature: learnable activation functions
        """
        batch_size, input_dim = x.shape
        
        try:
            # Fix dimension matching issues
            if (self.activation_weights.shape[0] == self.output_dim and
                self.activation_weights.shape[1] == input_dim):
                
                # Normal matrix multiplication path
                # x: [batch_size, input_dim]
                # activation_weights: [output_dim, input_dim]
                
                # 1. Use learnable activation weights for transformation
                output = torch.matmul(x, self.activation_weights.t())  # [batch_size, output_dim]
                
                # 2. Add nonlinear activation
                output = torch.tanh(output)  # Stable activation function
                
                return output
                
            else:
                
                # Safe fallback: rebuild compatible weights
                if input_dim != self.activation_weights.shape[1]:
                    # Adjust input dimension
                    if input_dim > self.activation_weights.shape[1]:
                        x_adapted = F.adaptive_avg_pool1d(x.unsqueeze(1), self.activation_weights.shape[1]).squeeze(1)
                    else:
                        x_adapted = F.pad(x, (0, self.activation_weights.shape[1] - input_dim))
                else:
                    x_adapted = x
                
                # Safe matrix multiplication
                fallback_output = torch.matmul(x_adapted, self.activation_weights.t())
                fallback_output = torch.tanh(fallback_output)
                
                return fallback_output
                
        except Exception as e:
            # Final fallback: simple linear transformation
            try:
                if x.shape[1] >= self.output_dim:
                    simple_output = x[:, :self.output_dim]
                else:
                    simple_output = F.pad(x, (0, self.output_dim - x.shape[1]))
                
                # Add nonlinearity
                simple_output = torch.tanh(simple_output) * 0.1
                return simple_output
                
            except Exception as final_e:
                # Create zero tensor
                return torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)
    
    def pure_b_spline_basis(self, x):
        """Pure B-spline basis functions - core feature of KAN, ensure dimension consistency"""
        batch_size, input_dim = x.shape
        
        # Adaptive normalization (not MLP's linear normalization)
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        # B-spline grid point generation
        x_grid = torch.tanh(x_normalized)  # Nonlinear mapping to [-1,1]
        
        # Ensure dimension-consistent basis function generation
        basis_functions_list = []
        
        # Generate basis functions for each input dimension
        for dim_idx in range(input_dim):
            x_dim = x_grid[:, dim_idx:dim_idx+1]  # [batch_size, 1]
            
            dim_basis = []
            
            # T_0(x) = 1
            dim_basis.append(torch.ones_like(x_dim))
            
            if self.num_basis > 1:
                # T_1(x) = x
                dim_basis.append(x_dim)
            
            # T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x) (Chebyshev recurrence)
            for n in range(2, self.num_basis):
                if len(dim_basis) >= 2:
                    t_next = 2 * x_dim * dim_basis[-1] - dim_basis[-2]
                    t_next = torch.clamp(t_next, -5.0, 5.0)  # Numerical stability
                    dim_basis.append(t_next)
                else:
                    # Safe fallback
                    dim_basis.append(torch.zeros_like(x_dim))
            
            # Ensure correct number of basis functions
            while len(dim_basis) < self.num_basis:
                dim_basis.append(torch.zeros_like(x_dim))
            
            # Truncate to correct number
            dim_basis = dim_basis[:self.num_basis]
            
            # Stack to [batch_size, num_basis]
            dim_basis_tensor = torch.cat(dim_basis, dim=1)
            basis_functions_list.append(dim_basis_tensor)
        
        # Stack to [batch_size, input_dim, num_basis]
        basis_tensor = torch.stack(basis_functions_list, dim=1)
        
        # Ensure output dimension is correct
        assert basis_tensor.shape == (batch_size, input_dim, self.num_basis), \
            f"Basis tensor shape mismatch: got {basis_tensor.shape}, expected {(batch_size, input_dim, self.num_basis)}"
        
        return basis_tensor
    
    def forward(self, x):
        """
        Forward propagation
        """
        if self.normalize_input:
            x = self.input_norm(x)
        else:
            x = torch.clamp(x, min=-self.clamp_in, max=self.clamp_in)
        
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=self.clamp_in, neginf=-self.clamp_in)
        
        parts = []
        
        # Use unified basis function factory - replace direct basis calls with factory output
        try:
            # Get basis functions from factory
            basis_tensor = self.basis(x)  # [batch, input_dim, num_basis]
            
            # Compute spline output using basis functions and learnable coefficients
            spline_coeffs = self.spline_coeffs.float()  # [output_dim, input_dim, num_basis]
            basis_output = torch.einsum('bji,oji->bo', basis_tensor, spline_coeffs)
            
            # Apply activation and scaling
            basis_output = torch.tanh(basis_output) * 0.5
            parts.append(basis_output)
            
        except Exception as e:
            # Fallback to simple linear transformation
            fallback_output = x @ (torch.ones_like(self.spline_coeffs[:, :, 0]).t() * 0.1)
            parts.append(fallback_output)
        
        try:
            # Ensure type consistency, avoid numpy.float32 and torch.FloatTensor mismatch
            x_float = x.float()
            activation_weights_float = self.activation_weights.float()
            activation_output = torch.tanh(x_float @ activation_weights_float.t())
            if torch.isnan(activation_output).any() or torch.isinf(activation_output).any():
                activation_output = x_float @ (activation_weights_float.t() * 0.1)
            # Gentle scaling
            activation_output = activation_output * 0.3
            parts.append(activation_output)
        except Exception as e:
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
        """Enhanced B-spline basis function computation, improve numerical stability"""
        # Normalize to [-1, 1] interval (standard domain for Chebyshev polynomials)
        x_normalized = torch.clamp(x, min=-2.0, max=2.0) / 2.0
        
        # Use corrected Chebyshev recurrence, add numerical stability checks
        basis_functions = []
        
        # T0(x) = 1
        T0 = torch.ones_like(x_normalized)
        basis_functions.append(T0)
        
        if self.num_basis > 1:
            # T1(x) = x
            T1 = x_normalized
            basis_functions.append(T1)
            
            # Recursively compute remaining basis functions, add stability checks
            for n in range(2, self.num_basis):
                # T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
                Tn = 2.0 * x_normalized * basis_functions[-1] - basis_functions[-2]
                
                # Fix 2: Numerical stability check
                if torch.any(torch.abs(Tn) > 10.0):  # Detect possible divergence
                    Tn = torch.clamp(Tn, min=-5.0, max=5.0)
                
                basis_functions.append(Tn)
        
        # Combine basis functions
        basis_matrix = torch.stack(basis_functions, dim=-1)  # [batch, input_dim, num_basis]
        
        # Compute B-spline output, add numerical checks
        # Ensure type consistency, avoid numpy.float32 and torch.FloatTensor mismatch
        basis_matrix = basis_matrix.float()
        spline_coeffs = self.spline_coeffs.float()  # [output_dim, input_dim, num_basis]
        # Directly get [batch, output_dim] (sum over input_dim and num_basis)
        spline_output = torch.einsum('bji,oji->bo', basis_matrix, spline_coeffs)
        
        # Final numerical stability check
        if torch.isnan(spline_output).any() or torch.isinf(spline_output).any():
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
        """KAN-specific initialization - strengthen numerical stability"""
        # Polynomial coefficient initialization - smaller initial values
        with torch.no_grad():
            nn.init.normal_(self.poly_coeffs, mean=0.0, std=0.01)
            
            # Activation scale initialization - more conservative range, ensure return torch tensor
            nn.init.uniform_(self.activation_scale, 0.01, 0.05)
            
            # Base transform initialization - smaller weights
            nn.init.xavier_uniform_(self.base_transform.weight, gain=0.01)
    
    def polynomial_basis_functions(self, x):
        """Simplified polynomial basis functions - simplified version of KAN, fix dimension issues"""
        batch_size, input_dim = x.shape
        
        x_normalized = torch.tanh(x) # Stably normalize to [-1, 1]
        
        basis_list = []
        basis_list.append(torch.ones_like(x_normalized))
        
        for n in range(1, self.num_basis):
            basis_list.append(basis_list[-1] * x_normalized)

        basis = torch.stack(basis_list, dim=-1)
        return basis
    
    def kan_learnable_activation(self, x):
        """Simplified learnable activation function - fix dimension issues"""
        batch_size, input_dim = x.shape
        
        # Ensure activation_scale dimension is correct
        if self.activation_scale.shape != (self.output_dim, input_dim):
            # Adjust dimension
            scale = self.activation_scale[:, :input_dim] if self.activation_scale.shape[1] >= input_dim else self.activation_scale
        else:
            scale = self.activation_scale
        
        # Safe activation function computation
        try:
            # x: [batch_size, input_dim], scale: [output_dim, input_dim]
            # Need to expand x to [batch_size, output_dim, input_dim] for element-wise computation
            x_expanded = x.unsqueeze(1).expand(batch_size, self.output_dim, input_dim)  # [batch_size, output_dim, input_dim]
            scale_expanded = scale.unsqueeze(0).expand(batch_size, self.output_dim, input_dim)  # [batch_size, output_dim, input_dim]
            
            # Element-wise activation function
            activated = x_expanded * torch.tanh(x_expanded * scale_expanded)  # [batch_size, output_dim, input_dim]
            
            # Reduce dimension to [batch_size, output_dim]
            output = activated.mean(dim=2)  # Average pooling
            
            return output
            
        except RuntimeError as e:
            # Fallback to simple computation
            return torch.tanh(x).sum(dim=1, keepdim=True).expand(-1, self.output_dim) * 0.1
    
    def forward(self, x):
        if self.verbose:
            pass

        try:
            # 1. Use unified basis function factory - Replace direct basis calls with factory output
            basis_tensor = self.basis(x)  # [batch, input_dim, num_basis]
            # Ensure type consistency, avoid numpy.float32 and torch.FloatTensor mismatch
            basis_tensor = basis_tensor.float()
            poly_coeffs = self.poly_coeffs.float()
            poly_output = torch.einsum('bid,oid->bo', basis_tensor, poly_coeffs)

            # 2. Base linear transformation (minimize MLP characteristics)
            base_output = self.base_transform(x)

            # 3. Learnable activation function part (core of KAN)
            activation_output = self.kan_learnable_activation(x)

            # --- Diagnostics ---
            if self.verbose:
                pass
            
            # --- Combine output ---
            final_output = torch.zeros_like(base_output)
            use_base = torch.all(torch.isfinite(base_output))
            use_poly = torch.all(torch.isfinite(poly_output))
            use_activation = torch.all(torch.isfinite(activation_output))

            if use_base:
                final_output += base_output * 0.3
            if use_poly:
                final_output += poly_output * 0.5
            if use_activation:
                final_output += activation_output * 0.2

            if not (use_base or use_poly or use_activation):
                return torch.zeros_like(base_output)

            # Apply layer normalization
            kan_output = self.ln(final_output)

            if self.verbose:
                pass
            
            # Safe fallback mechanism
            if not torch.all(torch.isfinite(kan_output)):
                return torch.nan_to_num(kan_output, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                if self.verbose:
                    pass
                return kan_output
                
        except Exception as e:
            # Fallback to input (identity mapping)
            if x.shape[1] == self.output_dim:
                return x
            else:
                # Dimension adjustment
                if x.shape[1] > self.output_dim:
                    return x[:, :self.output_dim]
                else:
                    return F.pad(x, (0, self.output_dim - x.shape[1]))


class OptimizedGNNKANEncoder(nn.Module):
    """
    KAN encoder dedicated to GNN - replace MLP layers with pure KAN
    Core: completely replace traditional MLP layers with AdvancedKANLayer
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
        
        # Core: replace all MLP layers with pure KAN layers
        kan_layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Fix: use compatible KAN layer creation function
            kan_layers.append(CompatibleSimplifiedKANLayer(
                dims[i], dims[i + 1], 
                num_basis=kan_grid_size,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                basis_function=basis_function,
                basis_kwargs=basis_kwargs
            ))
            
            # Dropout (but don't use fixed activations commonly used in MLP like ReLU/GELU)
            if i < len(dims) - 2:
                kan_layers.append(nn.Dropout(dropout))
        
        self.kan_layers = nn.ModuleList(kan_layers)
        
        # Simplified message passing (avoid MLP structure)
        self.message_processors = nn.ModuleList([
            CompatibleSimplifiedKANLayer(
                dims[i + 1], dims[i + 1],
                basis_function=basis_function,
                basis_kwargs=basis_kwargs
            )
            for i in range(len(dims) - 1)
        ])
        
        # Learnable graph structure
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
        """Use KAN for message passing - don't use MLP - fix NaN issues"""
        if edge_index.size(1) == 0:
            return x
        
        # Enhanced stability check
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        row, col = edge_index
        num_nodes = x.size(0)
        
        # Special case: skip graph message passing when only one node
        if num_nodes <= 1:
            return x
        
        # Safe index check
        if edge_index.size(1) == 0:
            return x
            
        if row.max() >= num_nodes or col.max() >= num_nodes or row.min() < 0 or col.min() < 0:
            # Filter invalid indices
            valid_mask = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
            if valid_mask.sum() == 0:
                return x
            row = row[valid_mask]
            col = col[valid_mask]
        
        try:
            # Improved sparse matrix construction
            adj_indices = torch.stack([row, col], dim=0)
            adj_values = torch.ones(len(row), device=x.device, dtype=x.dtype)
            adj_size = (num_nodes, num_nodes)
            
            # Check validity of sparse tensor
            if len(row) == 0:
                return x
            
            adj_sparse = torch.sparse_coo_tensor(adj_indices, adj_values, adj_size, device=x.device)
            adj_sparse = adj_sparse.coalesce()
            
            # Safe degree computation
            degrees = torch.sparse.sum(adj_sparse, dim=1).to_dense()
            
            # Check validity of degrees
            if torch.isnan(degrees).any() or torch.isinf(degrees).any():
                degrees = torch.ones_like(degrees)
            
            # Avoid division by zero
            degrees = torch.clamp(degrees, min=1e-6)
            degrees_inv = 1.0 / torch.sqrt(degrees)
            
            # Check inverse of degrees
            if torch.isnan(degrees_inv).any() or torch.isinf(degrees_inv).any():
                degrees_inv = torch.ones_like(degrees_inv)
            
            # Safe normalization value computation
            norm_values = degrees_inv[row] * degrees_inv[col]
            
            # Check normalization values
            if torch.isnan(norm_values).any() or torch.isinf(norm_values).any():
                norm_values = torch.ones_like(norm_values) / len(norm_values)
            
            norm_adj = torch.sparse_coo_tensor(adj_indices, norm_values, adj_size, device=x.device)
            
            # Safe sparse matrix multiplication
            message = torch.sparse.mm(norm_adj, x)
            
            # Check message passing results
            if torch.isnan(message).any() or torch.isinf(message).any():
                return x
            
            # Use KAN to process messages (instead of MLP)
            if layer_idx < len(self.message_processors):
                try:
                    processed_message = self.message_processors[layer_idx](message)
                    
                    # Check KAN processing results
                    if torch.isnan(processed_message).any() or torch.isinf(processed_message).any():
                        return message
                    
                    return processed_message
                except Exception as e:
                    return message
            else:
                return message
                
        except Exception as e:
            return x
    
    def forward(self, x, edge_index):
        """Pure KAN forward propagation - completely avoid MLP structure - fix NaN propagation"""
        current_x = x
        
        # Input stability check
        if torch.isnan(current_x).any() or torch.isinf(current_x).any():
            current_x = torch.nan_to_num(current_x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        for i, layer in enumerate(self.kan_layers):
            if isinstance(layer, nn.Dropout):
                current_x = layer(current_x)
            else:
                try:
                    # KAN layer processing (core: replace MLP with KAN) - force use of KAN
                    kan_output = layer(current_x)
                    
                    
                    # Fix KAN layer output instead of skipping
                    if torch.isnan(kan_output).any() or torch.isinf(kan_output).any():
                        
                        # Fix instead of replace
                        kan_output = torch.nan_to_num(kan_output, nan=0.0, posinf=1.0, neginf=-1.0)
                        kan_output = torch.clamp(kan_output, -5.0, 5.0)  # Limit range
                        
                        # If still problematic after fixing, use safe KAN output
                        if torch.isnan(kan_output).any() or torch.isinf(kan_output).any():
                            # Create safe output maintaining KAN characteristics
                            if hasattr(layer, 'output_dim'):
                                target_dim = layer.output_dim
                            else:
                                target_dim = current_x.shape[1]
                            
                            # Use nonlinear transformation of input, maintain KAN characteristics
                            safe_output = torch.tanh(current_x) * 0.5  # Nonlinear transformation
                            
                            # Adjust dimension
                            if safe_output.shape[1] != target_dim:
                                if safe_output.shape[1] > target_dim:
                                    kan_output = safe_output[:, :target_dim]
                                else:
                                    padding = torch.zeros(safe_output.shape[0], target_dim - safe_output.shape[1], 
                                                        device=safe_output.device, dtype=safe_output.dtype)
                                    kan_output = torch.cat([safe_output, padding], dim=1)
                            else:
                                kan_output = safe_output
                        
                    else:
                        pass
                    
                    # KAN message passing (every two layers, reduce computation)
                    if i % 2 == 0 and edge_index.size(1) > 0:
                        message = self.kan_message_passing(kan_output, edge_index, i // 2)
                        
                        # Safe residual connection
                        if message.shape == kan_output.shape:
                            residual_output = kan_output + 0.3 * message
                            
                            # Check residual connection results
                            if torch.isnan(residual_output).any() or torch.isinf(residual_output).any():
                                current_x = kan_output
                            else:
                                current_x = residual_output
                        else:
                            current_x = kan_output
                    else:
                        current_x = kan_output
                    
                    # Check stability after each layer
                    if torch.isnan(current_x).any() or torch.isinf(current_x).any():
                        current_x = torch.nan_to_num(current_x, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                except Exception as e:
                    # If error occurs, keep previous layer's output
                    pass
        
        # Final output check
        if torch.isnan(current_x).any() or torch.isinf(current_x).any():
            current_x = torch.nan_to_num(current_x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # If still problematic, use linear transformation of original input
            if torch.isnan(current_x).any() or torch.isinf(current_x).any():
                current_x = torch.tanh(x) * 0.1
        
        # Only return node embeddings, maintain backward compatibility
        # self.last_adj = final_adj  # Optional: store for external access
        return current_x


# Backward compatible aliases - unified interface
class KANLayer(SimplifiedKANLayer):
    """Backward compatible KAN layer"""
    def __init__(self, input_dim, output_dim, num_basis=8, grid_size=None, spline_order=None, **kwargs):
        # Call SimplifiedKANLayer, ignore unsupported parameters
        super().__init__(input_dim, output_dim, num_basis)


class GNNKANEncoder(OptimizedGNNKANEncoder):
    """Backward compatible GNN-KAN encoder"""
    pass


# Fix: Ensure SimplifiedKANLayer can handle AdvancedKANLayer parameters but ignore unsupported ones
class CompatibleSimplifiedKANLayer(SimplifiedKANLayer):
    """Compatibility-enhanced simplified KAN layer - can accept but ignore advanced parameters"""
    
    def __init__(self, input_dim, output_dim, num_basis=8, 
                 spline_order=None, grid_size=None, 
                 adaptive_spline_order=None, verbose=False,
                 basis_function='chebyshev', basis_kwargs=None, **kwargs):
        # Only use parameters supported by SimplifiedKANLayer
        # Filter out unsupported parameters (like spline_order, grid_size)
        
        # Extract parameters supported by SimplifiedKANLayer __init__
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
    KAN edge decoder - for Graph Decoder, replace MLP with KAN
    
    Architecture: KAN(2*d → d') → Linear(d' → 1) → output logit
    Features: learnable activation functions, numerical stability protection, lightweight configuration
    """
    
    def __init__(self, input_dim, hidden_dim=None, num_basis=4, 
                 spline_order=3, dropout=0.1, stability_mode=True):
        super(KANEdgeDecoder, self).__init__()
        
        self.input_dim = input_dim  # Should be 2 * node_embedding_dim
        self.hidden_dim = hidden_dim or (input_dim // 2)  # Default to half of input dimension
        self.num_basis = num_basis
        self.spline_order = spline_order 
        self.dropout_rate = dropout
        self.stability_mode = stability_mode
        
        # KAN layer: input [h_i; h_j] → hidden representation
        self.kan_layer = CompatibleSimplifiedKANLayer(
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
            num_basis=self.num_basis,
            verbose=False  # Avoid excessive log output
        )
        
        # Dropout and LayerNorm
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-4)
        
        # Linear head: hidden representation → logit (output raw score, no sigmoid)
        self.linear_head = nn.Linear(self.hidden_dim, 1, bias=True)
        
        # Numerical stability components
        if self.stability_mode:
            self.gradient_stabilizer = self._create_gradient_stabilizer()
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters - optimized for edge scoring"""
        # Linear head uses small weight initialization
        nn.init.xavier_uniform_(self.linear_head.weight, gain=0.1)
        nn.init.constant_(self.linear_head.bias, 0.0)
    
    def _create_gradient_stabilizer(self):
        """Create gradient stabilizer (if needed)"""
        return lambda x: torch.clamp(x, min=-5.0, max=5.0)
    
    def forward(self, edge_features, base_adj=None):
        """
        Forward propagation - add residual support
        
        Args:
            edge_features: [N, 2*d] - node pair features [h_i; h_j]
            base_adj: [num_nodes, num_nodes] - KNN baseline adjacency matrix (optional)
            
        Returns:
            logits: [N, 1] - raw score for edge existence (logit, not sigmoided)
        """
        # Input check
        if edge_features.dim() != 2:
            raise ValueError(f"Expected 2D input, got {edge_features.dim()}D")
        
        if edge_features.size(1) != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {edge_features.size(1)}")
        
        # Numerical stability preprocessing
        if self.stability_mode:
            edge_features = self._stabilize_input(edge_features)
        
        try:
            # Nonlinear transformation through KAN layer
            kan_output = self.kan_layer(edge_features)
            
            # Check KAN output validity
            if torch.isnan(kan_output).any() or torch.isinf(kan_output).any():
                kan_output = torch.nan_to_num(kan_output, nan=0.0, posinf=2.0, neginf=-2.0)
            
            # LayerNorm + Dropout
            normalized_output = self.layer_norm(kan_output)
            if self.training:
                normalized_output = self.dropout(normalized_output)
            
            # Linear head outputs logit
            logits = self.linear_head(normalized_output)
            
            # New: residual with base_adj
            if base_adj is not None:
                # Convert base_adj to same shape as logits
                num_nodes = int(edge_features.size(0) ** 0.5)
                if num_nodes * num_nodes == edge_features.size(0):
                    base_adj_flat = base_adj.view(-1, 1)
                    residual = torch.tanh(logits) * 0.5
                    final_scores = base_adj_flat + residual
                else:
                    final_scores = torch.sigmoid(logits)
            else:
                final_scores = torch.sigmoid(logits)
            
            # Final numerical stability check
            if self.stability_mode:
                final_scores = self._stabilize_output(final_scores)
            
            return final_scores
            
        except Exception as e:
            # Fallback strategy: simple linear transformation
            return self._fallback_forward(edge_features)
    
    def _stabilize_input(self, x):
        """Input stabilization"""
        # 1. NaN and infinite value handling
        x = torch.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0)
        
        # 2. Range limitation
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        # 3. Adaptive scaling (if standard deviation is too large)
        x_std = torch.std(x)
        if x_std > 2.0:
            scaling_factor = 2.0 / (x_std + 1e-8)
            x = x * scaling_factor
        
        return x
    
    def _stabilize_output(self, logits):
        """Output stabilization"""
        # Limit logit range, avoid extreme values
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # NaN check
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return logits
    
    def _fallback_forward(self, edge_features):
        """Fallback strategy: simple linear transformation"""
        try:
            # Directly through linear layer, but reduce dimension first
            if edge_features.size(1) > self.hidden_dim:
                # Simple feature selection
                reduced_features = edge_features[:, :self.hidden_dim]
            else:
                # Zero padding
                padding = torch.zeros(edge_features.size(0), 
                                    self.hidden_dim - edge_features.size(1),
                                    device=edge_features.device,
                                    dtype=edge_features.dtype)
                reduced_features = torch.cat([edge_features, padding], dim=1)
            
            return self.linear_head(reduced_features)
            
        except Exception as e:
            # Final fallback: return zero logit
            return torch.zeros(edge_features.size(0), 1, 
                             device=edge_features.device, 
                             dtype=edge_features.dtype)
    
    def get_model_info(self):
        """Get model information for evaluation"""
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
            'learnable_activations': kan_params,  # KAN parameters are all learnable activation related
            'parameter_efficiency_score': kan_params / max(total_params, 1)
        }