"""
基於 KAN 論文的梯度爆炸優化模組
實施論文中提到的所有梯度穩定性機制

參考論文中的關鍵技術：
- L1 正則化：|Φ_l|_1 = (1/N_p) * Σ|φ(x_s^(p))|
- 熵正則化：S(Φ_l) = -Σ(|φ_{i,j}|_1/|Φ|_1) * log(|φ_{i,j}|_1/|Φ|_1)
- Xavier 初始化：σ = √(2/(n_in + n_out))
- 動態剪枝：閾值 θ = 10^-2
- 平滑激活函數：SiLU σ(x) = x/(1+e^(-x))
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import warnings


class GradientStabilizer:
    """
    基於 KAN 論文的梯度穩定化器
    實施所有論文中提到的梯度爆炸緩解技術
    """
    
    def __init__(self, 
                 l1_lambda: float = 1e-2,      # 論文建議 10^-2 或 10^-3
                 entropy_lambda: float = 1e-2,  # 熵正則化強度
                 grad_clip_value: float = 1.0,  # 梯度裁剪閾值
                 pruning_threshold: float = 1e-2, # 剪枝閾值 θ
                 enable_dynamic_scaling: bool = True,
                 stability_check_freq: int = 100):
        
        self.l1_lambda = l1_lambda
        self.entropy_lambda = entropy_lambda
        self.grad_clip_value = grad_clip_value
        self.pruning_threshold = pruning_threshold
        self.enable_dynamic_scaling = enable_dynamic_scaling
        self.stability_check_freq = stability_check_freq
        
        # 監控統計
        self.gradient_norms = []
        self.parameter_norms = []
        self.regularization_history = []
        self.stability_violations = 0
        self.step_count = 0
        
    def xavier_init_kan_layer(self, layer):
        """
        基於論文的 KAN 層 Xavier 初始化
        σ = √(2/(n_in + n_out))
        """
        # 統一處理不同類型的KAN層
        if hasattr(layer, 'spline_coeffs'):
            fan_in = layer.input_dim
            fan_out = layer.output_dim
            std = math.sqrt(2.0 / (fan_in + fan_out))
            
            # B-spline 係數初始化 - 較小的方差
            nn.init.normal_(layer.spline_coeffs, mean=0.0, std=std * 0.1)
            
        elif hasattr(layer, 'poly_weights'):  # SimplifiedKANLayer
            fan_in = layer.input_dim
            fan_out = layer.output_dim
            std = math.sqrt(2.0 / (fan_in + fan_out))
            nn.init.normal_(layer.poly_weights, mean=0.0, std=std * 0.1)
            
        elif hasattr(layer, 'spline_weight'):  # FastKANLayer
            nn.init.kaiming_uniform_(layer.spline_weight, a=math.sqrt(5))
            
        elif hasattr(layer, 'activation_weights'):  # UltraFastKANLayer
            fan_in = layer.input_dim
            fan_out = layer.output_dim
            std = math.sqrt(2.0 / (fan_in + fan_out))
            nn.init.normal_(layer.activation_weights, mean=0.0, std=std * 0.01)
            
        if hasattr(layer, 'silu_weight'):
            # SiLU 權重使用更保守的初始化
            nn.init.xavier_uniform_(layer.silu_weight, gain=0.1)
            
        if hasattr(layer, 'linear'):
            # 線性層使用標準 Xavier 初始化
            nn.init.xavier_uniform_(layer.linear.weight, gain=math.sqrt(2.0))
            if layer.linear.bias is not None:
                nn.init.zeros_(layer.linear.bias)
    
    def compute_l1_regularization(self, model) -> torch.Tensor:
        """
        計算 L1 正則化項：|Φ_l|_1 = (1/N_p) * Σ|φ(x_s^(p))|
        """
        l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        param_count = 0
        
        for module in model.modules():
            # AdvancedKANLayer
            if hasattr(module, 'spline_coeffs'):
                l1_reg += torch.sum(torch.abs(module.spline_coeffs))
                param_count += module.spline_coeffs.numel()
                
            # SimplifiedKANLayer
            if hasattr(module, 'poly_weights'):
                l1_reg += torch.sum(torch.abs(module.poly_weights))
                param_count += module.poly_weights.numel()
                
            # FastKANLayer
            if hasattr(module, 'spline_weight'):
                l1_reg += torch.sum(torch.abs(module.spline_weight))
                param_count += module.spline_weight.numel()
                
            # UltraFastKANLayer
            if hasattr(module, 'activation_weights'):
                l1_reg += torch.sum(torch.abs(module.activation_weights))
                param_count += module.activation_weights.numel()
                
            if hasattr(module, 'silu_weight'):
                l1_reg += torch.sum(torch.abs(module.silu_weight))
                param_count += module.silu_weight.numel()
        
        # 歸一化
        if param_count > 0:
            l1_reg = l1_reg / param_count
            
        return l1_reg
    
    def compute_entropy_regularization(self, model) -> torch.Tensor:
        """
        計算熵正則化項：S(Φ_l) = -Σ(|φ_{i,j}|_1/|Φ|_1) * log(|φ_{i,j}|_1/|Φ|_1)
        促進稀疏性，間接穩定梯度
        """
        total_importance = torch.tensor(0.0, device=next(model.parameters()).device)
        importance_list = []
        
        for module in model.modules():
            # AdvancedKANLayer
            if hasattr(module, 'spline_coeffs'):
                spline_importance = torch.sum(torch.abs(module.spline_coeffs), dim=-1)
                importance_list.append(spline_importance.flatten())
                total_importance += torch.sum(spline_importance)
                
            # SimplifiedKANLayer
            if hasattr(module, 'poly_weights'):
                poly_importance = torch.sum(torch.abs(module.poly_weights), dim=-1)
                importance_list.append(poly_importance.flatten())
                total_importance += torch.sum(poly_importance)
                
            # FastKANLayer
            if hasattr(module, 'spline_weight'):
                spline_importance = torch.abs(module.spline_weight)
                importance_list.append(spline_importance.flatten())
                total_importance += torch.sum(spline_importance)
                
            # UltraFastKANLayer
            if hasattr(module, 'activation_weights'):
                activation_importance = torch.sum(torch.abs(module.activation_weights), dim=-1)
                importance_list.append(activation_importance.flatten())
                total_importance += torch.sum(activation_importance)
                
            if hasattr(module, 'silu_weight'):
                silu_importance = torch.abs(module.silu_weight)
                importance_list.append(silu_importance.flatten())
                total_importance += torch.sum(silu_importance)
        
        if len(importance_list) == 0 or total_importance == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        # 計算歸一化重要性
        all_importance = torch.cat(importance_list)
        normalized_importance = all_importance / (total_importance + 1e-8)
        
        # 計算熵（避免 log(0)）
        log_importance = torch.log(normalized_importance + 1e-8)
        entropy = -torch.sum(normalized_importance * log_importance)
        
        return entropy
    
    def compute_total_regularization_loss(self, model, pred_loss: torch.Tensor) -> torch.Tensor:
        """
        計算總損失：L_total = L_pred + λ * Σ(μ1|Φ_l|_1 + μ2*S(Φ_l))
        """
        l1_reg = self.compute_l1_regularization(model)
        entropy_reg = self.compute_entropy_regularization(model)
        
        # 論文中 μ1 = μ2 = 1
        reg_loss = self.l1_lambda * (l1_reg + entropy_reg)
        total_loss = pred_loss + reg_loss
        
        # 記錄正則化歷史
        self.regularization_history.append({
            'l1_reg': l1_reg.item(),
            'entropy_reg': entropy_reg.item(),
            'reg_loss': reg_loss.item(),
            'pred_loss': pred_loss.item()
        })
        
        return total_loss
    
    def apply_gradient_clipping(self, model, clip_type: str = 'norm') -> float:
        """
        應用梯度裁剪防止梯度爆炸
        """
        if clip_type == 'norm':
            # L2 範數裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=self.grad_clip_value
            )
        elif clip_type == 'value':
            # 數值裁剪
            torch.nn.utils.clip_grad_value_(
                model.parameters(), 
                clip_value=self.grad_clip_value
            )
            grad_norm = self._compute_grad_norm(model)
        else:
            grad_norm = self._compute_grad_norm(model)
        
        self.gradient_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        return grad_norm
    
    def _compute_grad_norm(self, model) -> float:
        """計算梯度的 L2 範數"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def apply_dynamic_pruning(self, model):
        """
        應用動態剪枝：移除重要性低於 θ = 10^-2 的連接
        論文中提到的結構調整技術
        """
        pruned_params = 0
        total_params = 0
        
        for module in model.modules():
            # AdvancedKANLayer with importance tracking
            if hasattr(module, 'spline_coeffs') and hasattr(module, 'importance_scores'):
                importance = module.importance_scores
                mask = importance > self.pruning_threshold
                
                total_params += mask.numel()
                pruned_params += (mask == 0).sum().item()
                
                with torch.no_grad():
                    module.spline_coeffs.data *= mask.unsqueeze(-1)
                    
                    if hasattr(module, 'silu_weight'):
                        module.silu_weight.data *= mask
                        
            # 其他KAN層的簡單剪枝
            elif hasattr(module, 'poly_weights'):  # SimplifiedKANLayer
                importance = torch.sum(torch.abs(module.poly_weights), dim=-1)
                mask = importance > self.pruning_threshold
                
                total_params += mask.numel()
                pruned_params += (mask == 0).sum().item()
                
                with torch.no_grad():
                    module.poly_weights.data *= mask.unsqueeze(-1)
        
        pruning_ratio = pruned_params / max(total_params, 1)
        return pruning_ratio
    
    def check_numerical_stability(self, model, inputs: torch.Tensor = None) -> Dict[str, float]:
        """
        檢查數值穩定性
        """
        stability_report = {
            'parameter_inf_count': 0,
            'parameter_nan_count': 0,
            'gradient_inf_count': 0,
            'gradient_nan_count': 0,
            'max_parameter_value': 0.0,
            'max_gradient_value': 0.0
        }
        
        # 檢查參數
        for p in model.parameters():
            if torch.isinf(p.data).any():
                stability_report['parameter_inf_count'] += torch.isinf(p.data).sum().item()
            if torch.isnan(p.data).any():
                stability_report['parameter_nan_count'] += torch.isnan(p.data).sum().item()
            
            max_val = torch.abs(p.data).max().item()
            stability_report['max_parameter_value'] = max(
                stability_report['max_parameter_value'], max_val
            )
            
            # 檢查梯度
            if p.grad is not None:
                if torch.isinf(p.grad).any():
                    stability_report['gradient_inf_count'] += torch.isinf(p.grad).sum().item()
                if torch.isnan(p.grad).any():
                    stability_report['gradient_nan_count'] += torch.isnan(p.grad).sum().item()
                
                max_grad = torch.abs(p.grad).max().item()
                stability_report['max_gradient_value'] = max(
                    stability_report['max_gradient_value'], max_grad
                )
        
        # 檢查輸入
        if inputs is not None:
            if torch.isinf(inputs).any() or torch.isnan(inputs).any():
                warnings.warn("輸入數據包含 inf 或 nan 值")
        
        # 記錄不穩定事件
        if (stability_report['parameter_inf_count'] > 0 or 
            stability_report['parameter_nan_count'] > 0 or
            stability_report['gradient_inf_count'] > 0 or
            stability_report['gradient_nan_count'] > 0):
            self.stability_violations += 1
        
        return stability_report
    
    def adaptive_regularization_scaling(self, current_loss: float, loss_history: List[float]):
        """
        基於損失歷史動態調整正則化強度
        """
        if not self.enable_dynamic_scaling or len(loss_history) < 10:
            return
        
        recent_losses = loss_history[-10:]
        loss_variance = np.var(recent_losses)
        loss_trend = recent_losses[-1] - recent_losses[0]
        
        # 如果損失方差過大或損失增長，增強正則化
        if loss_variance > 1.0 or loss_trend > 0.1:
            self.l1_lambda = min(self.l1_lambda * 1.1, 0.1)
            self.entropy_lambda = min(self.entropy_lambda * 1.1, 0.1)
        
        # 如果訓練穩定，逐漸減少正則化
        elif loss_variance < 0.01 and loss_trend < -0.01:
            self.l1_lambda = max(self.l1_lambda * 0.95, 1e-4)
            self.entropy_lambda = max(self.entropy_lambda * 0.95, 1e-4)
    
    def get_stability_report(self) -> Dict:
        """
        獲取完整的穩定性報告
        """
        if len(self.gradient_norms) == 0:
            return {"message": "尚未收集穩定性數據"}
        
        return {
            "regularization_config": {
                "l1_lambda": self.l1_lambda,
                "entropy_lambda": self.entropy_lambda,
                "grad_clip_value": self.grad_clip_value,
                "pruning_threshold": self.pruning_threshold
            },
            "gradient_statistics": {
                "mean_grad_norm": np.mean(self.gradient_norms),
                "max_grad_norm": np.max(self.gradient_norms),
                "std_grad_norm": np.std(self.gradient_norms),
                "gradient_clips": sum(1 for g in self.gradient_norms if g > self.grad_clip_value)
            },
            "regularization_statistics": {
                "total_regularization_steps": len(self.regularization_history),
                "avg_l1_reg": np.mean([r['l1_reg'] for r in self.regularization_history]) if self.regularization_history else 0,
                "avg_entropy_reg": np.mean([r['entropy_reg'] for r in self.regularization_history]) if self.regularization_history else 0
            },
            "stability_violations": self.stability_violations,
            "total_steps": self.step_count
        }


class StabilizedKANLayer(nn.Module):
    """
    集成梯度穩定化的 KAN 層
    完全基於論文的設計和優化技術
    與專案中其他KAN層保持一致的介面
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_basis: int = 5, spline_order: int = 3,
                 stabilizer: Optional[GradientStabilizer] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        self.stabilizer = stabilizer or GradientStabilizer()
        
        # 核心 KAN 參數 - 與AdvancedKANLayer一致
        self.spline_coeffs = nn.Parameter(torch.zeros(output_dim, input_dim, num_basis))
        self.silu_weight = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 用於動態剪枝的重要性分數 - 與現有實現一致
        self.register_buffer('importance_scores', torch.ones(output_dim, input_dim))
        
        # 數值穩定性組件
        self.input_normalizer = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
        # 應用 Xavier 初始化
        self.stabilizer.xavier_init_kan_layer(self)
    
    def silu_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU 激活函數：σ(x) = x / (1 + exp(-x))
        論文中提到的平滑激活函數，導數有界，有助於梯度穩定
        與AdvancedKANLayer保持一致
        """
        # 限制輸入範圍避免數值溢出
        x_clamped = torch.clamp(x, -20.0, 20.0)
        return x_clamped * torch.sigmoid(x_clamped)
    
    def b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        計算平滑的 B-spline 基函數
        使用數值穩定的實現，與其他KAN層一致
        """
        # 自適應歸一化 - 基於輸入統計
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        # 輸入歸一化和裁剪
        x_norm = torch.tanh(x_normalized)  # 平滑歸一化到 [-1, 1]
        
        # 使用 Chebyshev 多項式作為穩定的基函數
        basis_list = [torch.ones_like(x_norm)]
        
        if self.num_basis > 1:
            basis_list.append(x_norm)
        
        # 遞歸計算 Chebyshev 多項式 T_n = 2x*T_{n-1} - T_{n-2}
        for i in range(2, self.num_basis):
            t_next = 2 * x_norm * basis_list[-1] - basis_list[-2]
            # 數值穩定性檢查
            t_next = torch.clamp(t_next, -10.0, 10.0)
            basis_list.append(t_next)
        
        return torch.stack(basis_list, dim=-1)
    
    def update_importance_scores(self, x: torch.Tensor):
        """
        更新連接重要性分數，用於動態剪枝
        與AdvancedKANLayer的實現保持一致
        """
        if not self.training:
            return
        
        with torch.no_grad():
            # 基於激活強度計算重要性
            basis = self.b_spline_basis(x)
            spline_activation = torch.einsum('oij,bij->oi', self.spline_coeffs, basis)
            spline_importance = torch.mean(torch.abs(spline_activation), dim=0)
            
            silu_activation = self.silu_activation(x)
            silu_importance = torch.mean(torch.abs(silu_activation), dim=0)
            
            # 指數移動平均更新
            alpha = 0.1
            self.importance_scores = (1 - alpha) * self.importance_scores + \
                                   alpha * (spline_importance + silu_importance.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播，包含所有穩定性機制
        """
        # 輸入數值穩定性檢查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 輸入歸一化提高穩定性
        x_norm = self.input_normalizer(x)
        
        # 更新重要性分數
        self.update_importance_scores(x_norm)
        
        # 線性變換
        linear_out = self.linear(x_norm)
        
        # B-spline 分量
        basis = self.b_spline_basis(x_norm)
        spline_out = torch.einsum('oij,bij->bo', self.spline_coeffs, basis)
        
        # SiLU 分量
        silu_out = torch.einsum('oi,bi->bo', self.silu_weight, self.silu_activation(x_norm))
        
        # 組合輸出
        output = linear_out + spline_out + silu_out
        
        # Dropout 提高穩健性
        output = self.dropout(output)
        
        # 輸出數值檢查
        if torch.isnan(output).any() or torch.isinf(output).any():
            warnings.warn("KAN 層輸出包含 NaN 或 Inf")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output