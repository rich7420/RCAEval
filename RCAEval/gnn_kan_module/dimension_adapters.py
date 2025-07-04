"""
Dimension Adapters Module
維度適配器模組 - 解決KAN模型中的維度不匹配問題
專注於確保KAN取代MLP的過程中維度正確匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DimensionAdapter(nn.Module):
    """
    智能維度適配器 - 確保KAN層之間的維度匹配
    專門解決KAN取代MLP過程中的維度問題
    """
    
    def __init__(self, input_dim, output_dim, adaptation_method='linear'):
        super(DimensionAdapter, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adaptation_method = adaptation_method
        
        if input_dim != output_dim:
            if adaptation_method == 'linear':
                self.adapter = nn.Linear(input_dim, output_dim, bias=False)
                # 小權重初始化，保持KAN特性主導
                nn.init.xavier_uniform_(self.adapter.weight, gain=0.1)
            elif adaptation_method == 'projection':
                if input_dim > output_dim:
                    # 降維：使用學習的投影矩陣
                    self.adapter = nn.Linear(input_dim, output_dim, bias=False)
                    nn.init.orthogonal_(self.adapter.weight)
                else:
                    # 升維：零填充 + 小線性變換
                    self.adapter = nn.Linear(input_dim, output_dim, bias=False)
                    nn.init.xavier_uniform_(self.adapter.weight, gain=0.05)
            else:
                # 默認線性適配
                self.adapter = nn.Linear(input_dim, output_dim, bias=False)
                nn.init.xavier_uniform_(self.adapter.weight, gain=0.1)
        else:
            self.adapter = nn.Identity()
    
    def forward(self, x):
        """維度適配前向傳播"""
        if self.input_dim == self.output_dim:
            return x
        
        try:
            adapted = self.adapter(x)
            
            # 檢查輸出有效性
            if torch.isnan(adapted).any() or torch.isinf(adapted).any():
                print(f"⚠️ 維度適配產生無效值，使用安全回退")
                if self.input_dim > self.output_dim:
                    # 降維：截斷
                    adapted = x[:, :self.output_dim]
                else:
                    # 升維：零填充
                    padding = torch.zeros(x.size(0), self.output_dim - self.input_dim, device=x.device)
                    adapted = torch.cat([x, padding], dim=1)
            
            return adapted
            
        except Exception as e:
            print(f"⚠️ 維度適配失敗: {e}，使用基礎適配")
            if self.input_dim > self.output_dim:
                return x[:, :self.output_dim]
            elif self.input_dim < self.output_dim:
                padding = torch.zeros(x.size(0), self.output_dim - self.input_dim, device=x.device)
                return torch.cat([x, padding], dim=1)
            else:
                return x


class TemporalAttentionAdapter(nn.Module):
    """
    時序注意力適配器 - 專門解決注意力機制的維度問題
    確保embedding dimension與num_heads的兼容性
    """
    
    def __init__(self, feature_dim, num_heads=4, min_head_dim=8):
        super(TemporalAttentionAdapter, self).__init__()
        self.feature_dim = feature_dim
        self.original_num_heads = num_heads
        self.min_head_dim = min_head_dim
        
        # 智能調整num_heads以確保兼容性
        if feature_dim % num_heads != 0:
            # 找到最大的可整除的頭數
            for h in range(num_heads, 0, -1):
                if feature_dim % h == 0 and feature_dim // h >= min_head_dim:
                    self.num_heads = h
                    break
            else:
                # 如果找不到合適的頭數，調整feature_dim
                self.num_heads = min(num_heads, feature_dim // min_head_dim)
                if self.num_heads == 0:
                    self.num_heads = 1
                    
                # 調整feature_dim到最接近的可整除值
                adjusted_dim = ((feature_dim // self.num_heads) + 1) * self.num_heads
                self.use_projection = True
                self.projection = nn.Linear(feature_dim, adjusted_dim, bias=False)
                self.back_projection = nn.Linear(adjusted_dim, feature_dim, bias=False)
                self.adjusted_dim = adjusted_dim
                
                # 小權重初始化
                nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
                nn.init.xavier_uniform_(self.back_projection.weight, gain=0.1)
        else:
            self.num_heads = num_heads
            self.use_projection = False
            self.adjusted_dim = feature_dim
        
        # 創建注意力機制
        self.attention = nn.MultiheadAttention(
            self.adjusted_dim, 
            num_heads=self.num_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(feature_dim, eps=1e-4)
        
        print(f"✓ 注意力適配器: {feature_dim}→{self.adjusted_dim}, heads={self.num_heads}")
    
    def forward(self, features):
        """智能注意力前向傳播"""
        original_shape = features.shape
        
        # 確保至少是3D
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        try:
            # 維度投影（如果需要）
            if self.use_projection:
                projected_features = self.projection(features)
                attn_output, _ = self.attention(projected_features, projected_features, projected_features)
                attn_output = self.back_projection(attn_output)
            else:
                attn_output, _ = self.attention(features, features, features)
            
            # 殘差連接
            output = self.norm(attn_output + features)
            
            # 恢復原始形狀
            if len(original_shape) == 2:
                output = output.squeeze(1)
            
            return output
            
        except Exception as e:
            print(f"⚠️ 注意力計算失敗: {e}，使用身份映射")
            if len(original_shape) == 2:
                return self.norm(features.squeeze(1))
            else:
                return self.norm(features)


class KANLayerAdapter(nn.Module):
    """
    KAN層適配器 - 確保KAN層輸出維度正確
    解決KAN取代MLP過程中的維度不匹配問題
    """
    
    def __init__(self, kan_layer, expected_output_dim):
        super(KANLayerAdapter, self).__init__()
        self.kan_layer = kan_layer
        self.expected_output_dim = expected_output_dim
        
        # 檢查KAN層的輸出維度
        if hasattr(kan_layer, 'output_dim'):
            actual_output_dim = kan_layer.output_dim
        else:
            # 嘗試推斷輸出維度
            actual_output_dim = expected_output_dim
        
        # 如果維度不匹配，添加適配器
        if actual_output_dim != expected_output_dim:
            self.dimension_adapter = DimensionAdapter(
                actual_output_dim, 
                expected_output_dim, 
                adaptation_method='projection'
            )
            self.needs_adaptation = True
        else:
            self.dimension_adapter = nn.Identity()
            self.needs_adaptation = False
    
    def forward(self, x):
        """KAN層適配前向傳播"""
        try:
            # 通過KAN層
            kan_output = self.kan_layer(x)
            
            # 檢查輸出有效性
            if torch.isnan(kan_output).any() or torch.isinf(kan_output).any():
                print(f"⚠️ KAN層輸出包含無效值")
                kan_output = torch.nan_to_num(kan_output, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 維度適配
            if self.needs_adaptation:
                adapted_output = self.dimension_adapter(kan_output)
            else:
                adapted_output = kan_output
            
            # 最終檢查
            if adapted_output.size(1) != self.expected_output_dim:
                print(f"⚠️ 適配後維度仍不匹配: {adapted_output.size(1)} vs {self.expected_output_dim}")
                # 強制調整
                if adapted_output.size(1) > self.expected_output_dim:
                    adapted_output = adapted_output[:, :self.expected_output_dim]
                else:
                    padding = torch.zeros(
                        adapted_output.size(0), 
                        self.expected_output_dim - adapted_output.size(1), 
                        device=adapted_output.device
                    )
                    adapted_output = torch.cat([adapted_output, padding], dim=1)
            
            return adapted_output
            
        except Exception as e:
            print(f"⚠️ KAN層適配失敗: {e}，創建零輸出")
            return torch.zeros(x.size(0), self.expected_output_dim, device=x.device)


class MessagePassingAdapter(nn.Module):
    """
    消息傳遞適配器 - 確保消息傳遞過程中的維度一致性
    專門處理GNN中KAN層之間的消息傳遞
    """
    
    def __init__(self, feature_dim):
        super(MessagePassingAdapter, self).__init__()
        self.feature_dim = feature_dim
        
        # 簡化的消息處理（避免複雜的MLP結構）
        self.message_processor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 小權重初始化
        nn.init.xavier_uniform_(self.message_processor[1].weight, gain=0.1)
    
    def forward(self, node_features, edge_index):
        """安全的消息傳遞"""
        if edge_index.size(1) == 0:
            return node_features
        
        try:
            row, col = edge_index
            num_nodes = node_features.size(0)
            
            # 安全性檢查
            valid_mask = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
            if not valid_mask.all():
                row = row[valid_mask]
                col = col[valid_mask]
            
            if len(row) == 0:
                return node_features
            
            # 簡化的消息聚合
            messages = node_features[row]
            
            # 按目標節點聚合
            aggregated = torch.zeros_like(node_features)
            aggregated.scatter_add_(0, col.unsqueeze(1).expand_as(messages), messages)
            
            # 度數歸一化
            degree = torch.zeros(num_nodes, device=node_features.device)
            degree.scatter_add_(0, col, torch.ones_like(col, dtype=node_features.dtype))
            degree = torch.clamp(degree, min=1.0)
            
            aggregated = aggregated / degree.unsqueeze(1)
            
            # 處理消息
            processed_messages = self.message_processor(aggregated)
            
            return processed_messages
            
        except Exception as e:
            print(f"⚠️ 消息傳遞適配失敗: {e}，返回原始特徵")
            return node_features


def create_adaptive_kan_encoder(input_dim, hidden_dims, output_dim, **kwargs):
    """
    創建自適應KAN編碼器 - 自動處理維度匹配
    確保KAN取代MLP的過程順利進行
    """
    # 🔧 修復：使用兼容的KAN層創建函數
    from .kan_components.kan_layers import create_compatible_kan_layer, AdvancedKANLayer, SimplifiedKANLayer
    
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(dims) - 1):
        current_input = dims[i]
        current_output = dims[i + 1]
        
        # 🔧 修復：創建兼容的KAN層
        try:
            kan_layer = create_compatible_kan_layer(current_input, current_output, **kwargs)
        except Exception as first_error:
            try:
                # 回退到原始 AdvancedKANLayer
                kan_layer = AdvancedKANLayer(current_input, current_output)
            except Exception as second_error:
                try:
                    # 最後回退到 SimplifiedKANLayer
                    kan_layer = SimplifiedKANLayer(current_input, current_output)
                except Exception as final_error:
                    print(f"⚠️ 所有KAN層創建失敗: {final_error}，使用線性層")
                    kan_layer = nn.Linear(current_input, current_output)
        
        # 包裝為適配器
        adapted_layer = KANLayerAdapter(kan_layer, current_output)
        layers.append(adapted_layer)
        
        # 添加dropout（除了最後一層）
        if i < len(dims) - 2:
            layers.append(nn.Dropout(0.1))
    
    return nn.Sequential(*layers)


# 導出所有適配器
__all__ = [
    'DimensionAdapter',
    'TemporalAttentionAdapter', 
    'KANLayerAdapter',
    'MessagePassingAdapter',
    'create_adaptive_kan_encoder'
] 