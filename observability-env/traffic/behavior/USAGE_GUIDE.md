# 流量突波模擬器使用指南

## 概述

流量突波模擬器可以產生不規則的流量模式，包括突然的流量突波、漸進式增長、閃電人群效應等，非常適合測試系統在真實流量變化下的表現。

## 快速開始

### 1. 基本突波測試

```bash
# 快速測試
python test_traffic_spikes.py

# 基本突波示範
python example_traffic_spikes.py spike
```

### 2. 不同類型的流量模式

```bash
# 突然突波 (適合測試系統突發負載能力)
python example_traffic_spikes.py spike

# 閃電人群 (模擬限時搶購)
python example_traffic_spikes.py flash

# 波浪模式 (模擬自然流量起伏)
python example_traffic_spikes.py wave

# 隨機爆發 (不可預測的流量)
python example_traffic_spikes.py burst

# 黑色星期五模式 (極端購物場景)
python example_traffic_spikes.py blackfriday
```

### 3. 互動式選擇

```bash
python example_traffic_spikes.py
```

## 流量模式說明

### 1. 突然突波 (SUDDEN_SPIKE)
- **特點**: 瞬間大量用戶湧入，然後逐漸減少
- **適用場景**: 新聞事件、產品發布、系統壓力測試
- **流量曲線**: 快速上升 → 緩慢下降

### 2. 閃電人群 (FLASH_CROWD)
- **特點**: 極短時間內達到峰值，然後快速下降
- **適用場景**: 限時搶購、秒殺活動、病毒式傳播
- **流量曲線**: 極速上升 → 短暫維持 → 快速下降

### 3. 漸進增長 (GRADUAL_INCREASE)
- **特點**: 流量平穩上升
- **適用場景**: 營業時間開始、廣告效應、口碑傳播
- **流量曲線**: 平滑上升

### 4. 波浪模式 (WAVE_PATTERN)
- **特點**: 週期性的流量起伏
- **適用場景**: 一天中的自然流量變化、週期性活動
- **流量曲線**: 正弦波形

### 5. 隨機爆發 (RANDOM_BURSTS)
- **特點**: 不可預測的隨機流量爆發
- **適用場景**: 社群媒體效應、不規律的用戶行為
- **流量曲線**: 隨機峰值

### 6. 黑色星期五 (BLACK_FRIDAY)
- **特點**: 複雜的多波段流量模式
- **適用場景**: 大型購物節、促銷活動
- **流量曲線**: 早晨衝刺 → 午餐低谷 → 下午高峰 → 晚間狂潮

## 程式化使用

### 基本用法

```python
from traffic_spike_simulator import TrafficSpikeSimulator, TrafficPatternType

# 創建模擬器
simulator = TrafficSpikeSimulator("default")

# 安排突波事件
simulator.schedule_traffic_spike(
    TrafficPatternType.SUDDEN_SPIKE,
    delay_seconds=10,      # 10秒後開始
    duration=60,           # 持續60秒
    peak_users=50,         # 最高50個並發用戶
    description="午餐時間突波"
)

# 執行模擬
simulator.run_traffic_simulation(total_duration=120)
```

### 多重突波組合

```python
simulator = TrafficSpikeSimulator("mobile")

# 早晨漸增
simulator.schedule_traffic_spike(
    TrafficPatternType.GRADUAL_INCREASE,
    delay_seconds=0,
    duration=60,
    peak_users=30
)

# 午餐突波
simulator.schedule_traffic_spike(
    TrafficPatternType.SUDDEN_SPIKE,
    delay_seconds=80,
    duration=45,
    peak_users=70
)

# 下午閃購
simulator.schedule_traffic_spike(
    TrafficPatternType.FLASH_CROWD,
    delay_seconds=140,
    duration=30,
    peak_users=100
)

simulator.run_traffic_simulation(total_duration=200)
```

### 預設場景

```python
from traffic_spike_simulator import PresetTrafficScenarios

simulator = TrafficSpikeSimulator("default")

# 使用預設的電商日場景
PresetTrafficScenarios.create_ecommerce_day(simulator)

# 或使用黑色星期五場景
PresetTrafficScenarios.create_black_friday(simulator)

# 或創建混亂測試場景
PresetTrafficScenarios.create_random_chaos(simulator)

simulator.run_traffic_simulation(total_duration=600)
```

## 監控和狀態

### 即時狀態監控

```python
# 獲取當前狀態
status = simulator.get_simulation_status()
print(f"活躍用戶: {status['active_users']}")
print(f"當前事件: {status['active_event_types']}")
```

### 日誌輸出

模擬器會輸出詳細的日誌信息：

```
2024-01-20 15:30:15 - INFO - Scheduled sudden_spike spike: starts in 10s, duration 60s, peak 50 users
2024-01-20 15:30:25 - INFO - Traffic status: 15 active users, target: 25, active events: ['sudden_spike']
2024-01-20 15:30:35 - INFO - User spike_user_00023 completed: 4 behaviors, 28.3s, checkout: True
```

## 配置選項

### 行為配置檔案

可以使用不同的用戶行為配置檔案：

- `default`: 標準電商行為 (70% 瀏覽, 20% 購物車, 8% 結帳, 2% 錯誤)
- `mobile`: 手機用戶行為 (更多瀏覽, 較少結帳)
- `high_value`: 高價值客戶 (更多結帳)
- `error_testing`: 錯誤測試 (高錯誤率)

```python
# 使用不同配置檔案
mobile_sim = TrafficSpikeSimulator("mobile")
high_value_sim = TrafficSpikeSimulator("high_value")
```

## 實際應用場景

### 1. 系統壓力測試

```python
# 測試系統在突發流量下的表現
simulator.schedule_traffic_spike(
    TrafficPatternType.SUDDEN_SPIKE,
    delay_seconds=5,
    duration=120,
    peak_users=200,
    description="壓力測試"
)
```

### 2. 容量規劃

```python
# 測試不同負載級別
for peak in [50, 100, 150, 200]:
    simulator.schedule_traffic_spike(
        TrafficPatternType.GRADUAL_INCREASE,
        delay_seconds=i * 60,
        duration=60,
        peak_users=peak,
        description=f"容量測試 - {peak} 用戶"
    )
```

### 3. 自動擴展測試

```python
# 測試自動擴展系統
PresetTrafficScenarios.create_ecommerce_day(simulator)
# 觀察系統如何應對一天中的流量變化
```

### 4. 監控系統驗證

```python
# 驗證監控告警
simulator.schedule_traffic_spike(
    TrafficPatternType.FLASH_CROWD,
    delay_seconds=10,
    duration=30,
    peak_users=300,
    description="監控告警測試"
)
```

## 注意事項

1. **資源消耗**: 每個模擬用戶會消耗一定的 CPU 和記憶體
2. **網路負載**: 大量並發請求可能對目標系統造成實際負載
3. **測試環境**: 建議在測試環境中使用，避免影響生產系統
4. **監控配合**: 配合系統監控工具使用效果更佳

## 故障排除

### 常見問題

1. **模擬器啟動失敗**
   ```bash
   # 檢查依賴
   python -c "import yaml; print('PyYAML OK')"
   ```

2. **用戶創建過慢**
   - 調整 `spawn_rate` 參數
   - 減少 `peak_users` 數量

3. **記憶體使用過高**
   - 減少並發用戶數
   - 縮短會話持續時間

### 除錯模式

```bash
# 啟用詳細日誌
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from example_traffic_spikes import demo_sudden_spike
demo_sudden_spike()
"
```

這個流量突波模擬器讓你可以創建各種不規則的流量模式，非常適合測試系統在真實世界流量變化下的表現！