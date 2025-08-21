#!/usr/bin/env python3
"""
Example usage of Traffic Spike Simulator

This script demonstrates how to create various traffic spike patterns
for realistic load testing and observability data collection.
"""

import time
import logging
from traffic_spike_simulator import TrafficSpikeSimulator, TrafficPatternType, PresetTrafficScenarios

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_sudden_spike():
    """Demo: 突然的流量突波 (Sudden Traffic Spike)"""
    print("=== 突然流量突波示範 ===")
    
    simulator = TrafficSpikeSimulator("default")
    
    # 5秒後突然來50個用戶，持續60秒
    simulator.schedule_traffic_spike(
        TrafficPatternType.SUDDEN_SPIKE,
        delay_seconds=5,
        duration=60,
        peak_users=50,
        description="突然的購物高峰"
    )
    
    print("5秒後將有突然的流量突波...")
    simulator.run_traffic_simulation(total_duration=120)

def demo_flash_crowd():
    """Demo: 閃電人群效應 (Flash Crowd)"""
    print("=== 閃電人群效應示範 ===")
    
    simulator = TrafficSpikeSimulator("mobile")  # 使用手機用戶行為
    
    # 模擬限時搶購活動
    simulator.schedule_traffic_spike(
        TrafficPatternType.FLASH_CROWD,
        delay_seconds=10,
        duration=90,
        peak_users=100,
        description="限時搶購活動"
    )
    
    print("10秒後開始限時搶購，準備閃電人群...")
    simulator.run_traffic_simulation(total_duration=150)

def demo_wave_pattern():
    """Demo: 波浪式流量模式 (Wave Pattern)"""
    print("=== 波浪式流量模式示範 ===")
    
    simulator = TrafficSpikeSimulator("high_value")
    
    # 波浪式流量，模擬一天中的自然起伏
    simulator.schedule_traffic_spike(
        TrafficPatternType.WAVE_PATTERN,
        delay_seconds=5,
        duration=120,
        peak_users=40,
        description="一天中的自然流量起伏"
    )
    
    print("5秒後開始波浪式流量模式...")
    simulator.run_traffic_simulation(total_duration=150)

def demo_random_bursts():
    """Demo: 隨機爆發流量 (Random Bursts)"""
    print("=== 隨機爆發流量示範 ===")
    
    simulator = TrafficSpikeSimulator("default")
    
    # 隨機爆發，模擬不可預測的流量
    simulator.schedule_traffic_spike(
        TrafficPatternType.RANDOM_BURSTS,
        delay_seconds=5,
        duration=90,
        peak_users=60,
        base_users=10,
        description="不可預測的隨機流量爆發"
    )
    
    print("5秒後開始隨機流量爆發...")
    simulator.run_traffic_simulation(total_duration=120)

def demo_multiple_spikes():
    """Demo: 多重突波組合 (Multiple Spikes)"""
    print("=== 多重突波組合示範 ===")
    
    simulator = TrafficSpikeSimulator("default")
    
    # 組合多種突波模式
    simulator.schedule_traffic_spike(
        TrafficPatternType.GRADUAL_INCREASE,
        delay_seconds=10,
        duration=60,
        peak_users=30,
        description="早晨漸增流量"
    )
    
    simulator.schedule_traffic_spike(
        TrafficPatternType.SUDDEN_SPIKE,
        delay_seconds=80,
        duration=45,
        peak_users=70,
        description="午餐時間突波"
    )
    
    simulator.schedule_traffic_spike(
        TrafficPatternType.FLASH_CROWD,
        delay_seconds=140,
        duration=30,
        peak_users=100,
        description="下午促銷活動"
    )
    
    print("準備多重突波組合...")
    simulator.run_traffic_simulation(total_duration=200)

def demo_black_friday():
    """Demo: 黑色星期五模式 (Black Friday Pattern)"""
    print("=== 黑色星期五流量模式示範 ===")
    
    simulator = TrafficSpikeSimulator("default")
    PresetTrafficScenarios.create_black_friday(simulator)
    
    print("開始黑色星期五瘋狂購物模式...")
    simulator.run_traffic_simulation(total_duration=300)

def demo_ecommerce_day():
    """Demo: 完整電商日流量 (Full E-commerce Day)"""
    print("=== 完整電商日流量模式示範 ===")
    
    simulator = TrafficSpikeSimulator("default")
    PresetTrafficScenarios.create_ecommerce_day(simulator)
    
    print("模擬完整的電商營業日流量...")
    simulator.run_traffic_simulation(total_duration=600)

def demo_custom_scenario():
    """Demo: 自定義場景 (Custom Scenario)"""
    print("=== 自定義流量場景示範 ===")
    
    simulator = TrafficSpikeSimulator("mobile")
    
    # 自定義場景：社群媒體病毒式傳播
    # 1. 初始小規模分享
    simulator.schedule_traffic_spike(
        TrafficPatternType.GRADUAL_INCREASE,
        delay_seconds=5,
        duration=30,
        peak_users=15,
        description="初始社群分享"
    )
    
    # 2. 病毒式爆發
    simulator.schedule_traffic_spike(
        TrafficPatternType.FLASH_CROWD,
        delay_seconds=40,
        duration=60,
        peak_users=120,
        description="病毒式爆發"
    )
    
    # 3. 持續關注
    simulator.schedule_traffic_spike(
        TrafficPatternType.WAVE_PATTERN,
        delay_seconds=110,
        duration=90,
        peak_users=50,
        description="持續關注波動"
    )
    
    print("開始自定義病毒式傳播場景...")
    simulator.run_traffic_simulation(total_duration=220)

def interactive_demo():
    """互動式示範"""
    print("=== 流量突波模擬器互動示範 ===")
    print()
    print("可用的示範場景：")
    print("1. 突然流量突波")
    print("2. 閃電人群效應")
    print("3. 波浪式流量")
    print("4. 隨機爆發流量")
    print("5. 多重突波組合")
    print("6. 黑色星期五模式")
    print("7. 完整電商日流量")
    print("8. 自定義場景")
    print("0. 退出")
    
    demos = {
        "1": demo_sudden_spike,
        "2": demo_flash_crowd,
        "3": demo_wave_pattern,
        "4": demo_random_bursts,
        "5": demo_multiple_spikes,
        "6": demo_black_friday,
        "7": demo_ecommerce_day,
        "8": demo_custom_scenario
    }
    
    while True:
        try:
            choice = input("\n請選擇要執行的示範 (0-8): ").strip()
            
            if choice == "0":
                print("再見！")
                break
            elif choice in demos:
                print(f"\n執行示範 {choice}...")
                demos[choice]()
                print("\n示範完成！")
            else:
                print("無效選擇，請輸入 0-8")
                
        except KeyboardInterrupt:
            print("\n\n示範被中斷")
            break
        except Exception as e:
            print(f"執行錯誤: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 命令行模式
        demo_name = sys.argv[1]
        demos = {
            "spike": demo_sudden_spike,
            "flash": demo_flash_crowd,
            "wave": demo_wave_pattern,
            "burst": demo_random_bursts,
            "multi": demo_multiple_spikes,
            "blackfriday": demo_black_friday,
            "ecommerce": demo_ecommerce_day,
            "custom": demo_custom_scenario
        }
        
        if demo_name in demos:
            demos[demo_name]()
        else:
            print(f"未知示範: {demo_name}")
            print(f"可用示範: {', '.join(demos.keys())}")
    else:
        # 互動模式
        interactive_demo()