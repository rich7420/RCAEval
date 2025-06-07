import torch
import torch.nn as nn

class GNNKANModel(nn.Module):
    def __init__(self, config):
        super(GNNKANModel, self).__init__()
        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        # ...initialize layers...

    def forward(self, x):
        # ...forward pass implementation...
        pass

    def _setup_optimizers(self):
        """設置優化器"""
        try:
            self.optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=30, 
                gamma=0.1
            )
            return True
        except Exception as e:
            print(f"設置優化器時發生錯誤: {e}")
            return False
    
    def _validate_config(self):
        """驗證配置參數"""
        try:
            required_attrs = ['hidden_dim', 'num_layers', 'dropout', 'learning_rate']
            for attr in required_attrs:
                if not hasattr(self.config, attr):
                    raise ValueError(f"配置缺少必需參數: {attr}")
            return True
        except Exception as e:
            print(f"配置驗證失敗: {e}")
            return False
    
    def save_model(self, path):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
            }, path)
            print(f"模型已保存到: {path}")
        except Exception as e:
            print(f"保存模型失敗: {e}")
    
    def load_model(self, path):
        """加載模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            if hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"模型已從 {path} 加載")
        except Exception as e:
            print(f"加載模型失敗: {e}")
    
    def get_model_info(self):
        """獲取模型信息"""
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),
                'device': str(self.device),
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
            }
            return info
        except Exception as e:
            print(f"獲取模型信息失敗: {e}")
            return {}