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
    簡化的梯度穩定化器 - 減少冗餘檢查，專注核心穩定性
    """
    
    def __init__(self, 
                 l1_lambda: float = 1e-3,      # 減少正則化強度
                 entropy_lambda: float = 1e-3,  
                 grad_clip_value: float = 1.0,  
                 pruning_threshold: float = 1e-2,
                 stability_check_freq: int = 50):  # 大幅減少檢查頻率
        
        self.l1_lambda = l1_lambda
        self.entropy_lambda = entropy_lambda
        self.grad_clip_value = grad_clip_value
        self.pruning_threshold = pruning_threshold
        self.stability_check_freq = stability_check_freq
        
        # 精簡監控統計
        self.gradient_norms = []
        self.stability_violations = 0
        self.step_count = 0
        self.enable_dynamic_scaling = False  # 關閉動態調整
        
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
        """簡化的 L1 正則化計算"""
        l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for module in model.modules():
            # 統一處理所有KAN層的主要權重
            if hasattr(module, 'spline_coeffs'):
                l1_reg += torch.sum(torch.abs(module.spline_coeffs))
            elif hasattr(module, 'poly_weights'):
                l1_reg += torch.sum(torch.abs(module.poly_weights))
            elif hasattr(module, 'spline_weight'):
                l1_reg += torch.sum(torch.abs(module.spline_weight))
            elif hasattr(module, 'activation_weights'):
                l1_reg += torch.sum(torch.abs(module.activation_weights))
            
            if hasattr(module, 'silu_weight'):
                l1_reg += torch.sum(torch.abs(module.silu_weight))
        
        return l1_reg * self.l1_lambda
    
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
        精簡的總損失計算 - 減少正則化複雜度
        """
        # 步數檢查 - 只在必要時計算正則化
        self.step_count += 1
        if self.step_count % self.stability_check_freq != 0:
            return pred_loss  # 跳過正則化計算
        
        # 簡化的L1正則化
        l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        for module in model.modules():
            if hasattr(module, 'spline_weight'):
                l1_reg += torch.sum(torch.abs(module.spline_weight)) * 0.1  # 降低權重
            elif hasattr(module, 'poly_weights'):
                l1_reg += torch.sum(torch.abs(module.poly_weights)) * 0.1
        
        # 簡化的正則化損失
        reg_loss = self.l1_lambda * l1_reg
        
        # 限制正則化影響
        reg_loss = torch.clamp(reg_loss, max=pred_loss.item() * 0.1)  # 從0.3降到0.1
        
        return pred_loss + reg_loss
    
    def apply_gradient_clipping(self, model, clip_type: str = 'norm') -> float:
        """
        簡化的梯度裁剪 - 只在必要時執行
        """
        # 只在檢查頻率內執行
        if self.step_count % self.stability_check_freq == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=self.grad_clip_value
            )
            return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        return 0.0  # 跳過裁剪
    
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
        極簡的穩定性檢查
        """
        # 只在檢查頻率內執行
        if self.step_count % (self.stability_check_freq * 2) != 0:
            return {'gradient_nan_count': 0, 'gradient_inf_count': 0}
        
        nan_count = 0
        inf_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_count += 1
                if torch.isinf(param.grad).any():
                    inf_count += 1
        
        if nan_count > 0 or inf_count > 0:
            self.stability_violations += 1
        
        return {
            'gradient_nan_count': nan_count,
            'gradient_inf_count': inf_count
        }
    
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
            "stability_violations": self.stability_violations,
            "total_steps": self.step_count
        }


# StabilizedKANLayer 已移除 - 保持KAN純粹性
# 只保留 AdvancedKANLayer 和 SimplifiedKANLayer
# 梯度穩定功能通過 GradientStabilizer 類提供

def adaptive_gradient_clipping(model, y: torch.Tensor, optimizer, clip_value: float = 0.1, epsilon: float = 1e-3):
    """
    自適應梯度裁剪，基於論文的梯度穩定性方案
    C_k = clip_value * ||G_k||_F / ||g_k||_F
    其中 G_k 是所有梯度，g_k 是最後一層的梯度
    """
    # 對y進行clone()，避免inplace操作導致的執行時錯誤
    y_clone = y.clone()
    
    # 實際計算梯度
    grads = torch.autograd.grad(
        y_clone,
        model.parameters(),
        grad_outputs=torch.ones_like(y_clone),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )
    
    # 計算最後一層的梯度範數
    last_layer_grad_norm = torch.tensor(0.0, device=y.device)
    if grads and grads[-1] is not None:
        last_layer_grad_norm = torch.norm(grads[-1], p=2)
    
    # 計算所有梯度的Frobenius範數
    total_grad_norm = torch.tensor(0.0, device=y.device)
    for grad in grads:
        if grad is not None:
            total_grad_norm += torch.norm(grad, p=2)**2
    total_grad_norm = torch.sqrt(total_grad_norm)
    
    # 計算裁剪係數
    clip_coefficient = clip_value * total_grad_norm / (last_layer_grad_norm + epsilon)
    
    # 應用梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_coefficient.item())
    
    return clip_coefficient.item()