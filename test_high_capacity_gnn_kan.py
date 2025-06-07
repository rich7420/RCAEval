#!/usr/bin/env python3
"""
🚀 高容量且梯度穩定的 GNN-KAN 測試
保持原始模型復雜度，但使用先進的梯度穩定技術
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class HighCapacityGNNKANConfig:
    """高容量且梯度穩定的配置"""
    
    def __init__(self):
        # 🚀 高容量模型架構 - 保持或提升原始復雜度
        self.input_dim = 128
        self.hidden_dims = [256, 192, 128, 96]  # 4層深度網絡
        self.output_dim = 64
        
        # 🔑 保持高表達能力的KAN設置 - 不降低！
        self.kan_grid_size = 5        # 保持原始 B-spline 網格點數 
        self.kan_spline_order = 3     # 保持 3次樣條的表達能力
        self.num_gnn_layers = 3       # 保持 3層 GNN 的深度
        
        # 🛡️ 進階梯度穩定技術
        self.use_gradient_scaling = True         # 動態梯度縮放
        self.use_mixed_precision = True          # 混合精度訓練
        self.use_residual_connections = True     # 殘差連接
        self.use_layer_norm = True              # 層標準化
        self.use_spectral_norm = True           # 譜標準化
        self.use_ema_weights = True             # 指數移動平均權重
        
        # 🔬 智能梯度管理
        self.gradient_clip_norm = 1.0           # 嚴格梯度裁剪
        self.gradient_clip_adaptive = True      # 自適應梯度裁剪
        self.gradient_accumulation_steps = 4    # 梯度累積
        self.use_gradient_checkpointing = True  # 梯度檢查點
        
        # 🧠 智能學習率策略
        self.base_learning_rate = 1e-4          # 中等學習率
        self.max_learning_rate = 5e-4           # 允許更高的峰值學習率
        self.use_cosine_annealing = True        # 餘弦退火
        self.warmup_epochs = 30                 # 充分預熱
        self.weight_decay = 1e-5                # 適度權重衰減
        
        # 🎯 高精度正則化
        self.l1_lambda = 1e-5                   # 平衡的L1正則化
        self.entropy_lambda = 1e-5              # 平衡的熵正則化
        self.adaptive_regularization = True     # 自適應正則化強度
        
        # 🔧 訓練設置
        self.epochs = 150                       # 增加訓練時間
        self.batch_size = 8
        self.patience = 20                      # 早停耐心
        self.min_delta = 1e-6                   # 最小改進閾值
        
        # 📊 監控設置
        self.stability_check_freq = 5           # 頻繁穩定性檢查
        self.log_interval = 10                  # 日誌間隔
        self.save_best_model = True             # 保存最佳模型
        
        # 🔋 硬件優化
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compile_model = True               # PyTorch 2.0 編譯
        self.memory_efficient = True           # 記憶體效率模式

class AdvancedGradientStabilizer:
    """進階梯度穩定器"""
    
    def __init__(self, config):
        self.config = config
        self.gradient_history = []
        self.loss_history = []
        self.ema_decay = 0.999
        self.gradient_scale = 1.0
        self.stability_violations = 0
        
    def adaptive_gradient_clipping(self, model, loss):
        """自適應梯度裁剪"""
        # 計算當前梯度範數
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # 記錄梯度歷史
        self.gradient_history.append(total_norm)
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)
        
        # 自適應裁剪閾值
        if len(self.gradient_history) > 10:
            recent_gradients = self.gradient_history[-10:]
            avg_grad_norm = np.mean(recent_gradients)
            std_grad_norm = np.std(recent_gradients)
            
            # 動態調整裁剪閾值
            adaptive_clip = min(
                self.config.gradient_clip_norm,
                avg_grad_norm + 2 * std_grad_norm
            )
        else:
            adaptive_clip = self.config.gradient_clip_norm
        
        # 應用裁剪
        if total_norm > adaptive_clip:
            clip_coef = adaptive_clip / (total_norm + 1e-8)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
            
        return total_norm, adaptive_clip
    
    def detect_gradient_explosion(self, grad_norm):
        """檢測梯度爆炸"""
        if len(self.gradient_history) > 5:
            recent_avg = np.mean(self.gradient_history[-5:])
            if grad_norm > recent_avg * 10:  # 梯度突然增大10倍
                self.stability_violations += 1
                return True
        return False
    
    def stabilize_kan_parameters(self, model):
        """穩定KAN參數"""
        for name, module in model.named_modules():
            if hasattr(module, 'spline_weight'):
                # 檢查並修復異常值
                with torch.no_grad():
                    if torch.isnan(module.spline_weight).any():
                        print(f"⚠️ NaN detected in {name}, resetting...")
                        nn.init.xavier_uniform_(module.spline_weight, gain=0.01)
                    
                    if torch.isinf(module.spline_weight).any():
                        print(f"⚠️ Inf detected in {name}, clipping...")
                        module.spline_weight.clamp_(-1, 1)
                    
                    # 溫和的權重約束
                    if module.spline_weight.abs().max() > 5.0:
                        module.spline_weight.clamp_(-2, 2)

def create_high_capacity_model(config, num_nodes):
    """創建高容量且穩定的模型"""
    from RCAEval.e2e.gnn_kan import GNNKANModel
    
    # 創建模型
    model = GNNKANModel(config, num_nodes)
    
    # 應用譜標準化到關鍵層
    if config.use_spectral_norm:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                setattr(model, name, nn.utils.spectral_norm(module))
    
    # 智能初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if config.kan_init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            elif config.kan_init_method == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif hasattr(m, 'spline_weight'):
            # KAN層的保守初始化
            nn.init.xavier_uniform_(m.spline_weight, gain=0.01)
    
    model.apply(init_weights)
    
    return model

def advanced_training_loop(model, node_features, edge_index, config):
    """進階訓練循環"""
    stabilizer = AdvancedGradientStabilizer(config)
    
    # 優化器設置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # 學習率調度器
    if config.use_cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=25, gamma=0.8
        )
    
    # 混合精度訓練
    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
    
    # 指數移動平均
    if config.use_ema_weights:
        ema_model = torch.optim.swa_utils.AveragedModel(model)
    
    best_loss = float('inf')
    patience_counter = 0
    
    print("🚀 開始高容量且梯度穩定的訓練...")
    print(f"模型參數: KAN grid_size={config.kan_grid_size}, spline_order={config.kan_spline_order}, GNN layers={config.num_gnn_layers}")
    
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        try:
            # 混合精度前向傳播
            if config.use_mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    embeddings, adj_scores = model(node_features, edge_index)
                    
                    # 計算損失
                    base_loss = torch.nn.functional.mse_loss(
                        adj_scores, torch.eye(adj_scores.size(0), device=adj_scores.device)
                    )
                    
                    # 正則化損失
                    l1_loss = sum(p.abs().sum() for p in model.parameters()) * config.l1_lambda
                    total_loss = base_loss + l1_loss
                
                # 混合精度反向傳播
                scaler.scale(total_loss).backward()
                
                # 自適應梯度裁剪
                scaler.unscale_(optimizer)
                grad_norm, clip_threshold = stabilizer.adaptive_gradient_clipping(model, total_loss)
                
                # 穩定性檢查
                if stabilizer.detect_gradient_explosion(grad_norm):
                    print(f"⚠️ Epoch {epoch}: 檢測到梯度爆炸，應用穩定化...")
                    stabilizer.stabilize_kan_parameters(model)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # 標準訓練
                embeddings, adj_scores = model(node_features, edge_index)
                
                base_loss = torch.nn.functional.mse_loss(
                    adj_scores, torch.eye(adj_scores.size(0), device=adj_scores.device)
                )
                
                l1_loss = sum(p.abs().sum() for p in model.parameters()) * config.l1_lambda
                total_loss = base_loss + l1_loss
                
                total_loss.backward()
                
                grad_norm, clip_threshold = stabilizer.adaptive_gradient_clipping(model, total_loss)
                
                if stabilizer.detect_gradient_explosion(grad_norm):
                    print(f"⚠️ Epoch {epoch}: 檢測到梯度爆炸，應用穩定化...")
                    stabilizer.stabilize_kan_parameters(model)
                
                optimizer.step()
            
            scheduler.step()
            
            # 更新EMA權重
            if config.use_ema_weights:
                ema_model.update_parameters(model)
            
            # 記錄損失歷史
            stabilizer.loss_history.append(total_loss.item())
            
            # 早停檢查
            if total_loss.item() < best_loss - config.min_delta:
                best_loss = total_loss.item()
                patience_counter = 0
                if config.save_best_model:
                    torch.save(model.state_dict(), 'best_high_capacity_model.pth')
            else:
                patience_counter += 1
            
            # 日誌輸出
            if epoch % config.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}/{config.epochs} | "
                      f"Loss: {total_loss.item():.6f} | "
                      f"Base: {base_loss.item():.6f} | "
                      f"L1: {l1_loss.item():.6f} | "
                      f"Grad: {grad_norm:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Clip: {clip_threshold:.3f}")
            
            # 穩定性檢查
            if epoch % config.stability_check_freq == 0:
                stabilizer.stabilize_kan_parameters(model)
            
            # 早停
            if patience_counter >= config.patience:
                print(f"早停於 epoch {epoch}, 最佳損失: {best_loss:.6f}")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU記憶體不足於 epoch {epoch}, 清理緩存...")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"訓練錯誤於 epoch {epoch}: {e}")
                break
    
    # 載入最佳模型
    if config.save_best_model and patience_counter < config.patience:
        model.load_state_dict(torch.load('best_high_capacity_model.pth'))
    
    # 使用EMA權重
    if config.use_ema_weights:
        final_model = ema_model.module
    else:
        final_model = model
    
    # 訓練報告
    print("\n🎯 高容量訓練完成!")
    print(f"最終損失: {best_loss:.6f}")
    print(f"穩定性違規次數: {stabilizer.stability_violations}")
    print(f"平均梯度範數: {np.mean(stabilizer.gradient_history[-10:]):.6f}")
    
    return final_model

def test_high_capacity_gnn_kan():
    """測試高容量GNN-KAN"""
    print("🚀 測試高容量且梯度穩定的 GNN-KAN")
    
    # 創建配置
    config = HighCapacityGNNKANConfig()
    
    # 創建測試數據
    num_nodes = 20
    node_features = torch.randn(num_nodes, config.input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 40))
    
    if config.device == 'cuda':
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
    
    print(f"測試數據: {num_nodes} 節點, {config.input_dim} 特徵維度")
    print(f"模型設置: grid_size={config.kan_grid_size}, spline_order={config.kan_spline_order}, layers={config.num_gnn_layers}")
    
    # 創建高容量模型
    model = create_high_capacity_model(config, num_nodes)
    if config.device == 'cuda':
        model = model.cuda()
    
    # 編譯模型 (PyTorch 2.0)
    if config.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✅ 模型編譯成功")
        except:
            print("⚠️ 模型編譯失敗，使用標準模式")
    
    # 計算模型參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"總參數: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
    
    # 開始訓練
    trained_model = advanced_training_loop(model, node_features, edge_index, config)
    
    # 測試推理
    trained_model.eval()
    with torch.no_grad():
        test_embeddings, test_adj = trained_model(node_features, edge_index)
        
        print(f"\n🔍 推理結果:")
        print(f"嵌入形狀: {test_embeddings.shape}")
        print(f"鄰接矩陣形狀: {test_adj.shape}")
        print(f"嵌入範圍: [{test_embeddings.min():.3f}, {test_embeddings.max():.3f}]")
        print(f"鄰接矩陣範圍: [{test_adj.min():.3f}, {test_adj.max():.3f}]")
        
        # 檢查數值穩定性
        has_nan = torch.isnan(test_embeddings).any() or torch.isnan(test_adj).any()
        has_inf = torch.isinf(test_embeddings).any() or torch.isinf(test_adj).any()
        
        if not has_nan and not has_inf:
            print("✅ 數值穩定性檢查通過")
        else:
            print("❌ 檢測到數值不穩定性")
    
    print("\n🎉 高容量且梯度穩定的 GNN-KAN 測試完成!")
    return trained_model

if __name__ == "__main__":
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 運行測試
    model = test_high_capacity_gnn_kan()