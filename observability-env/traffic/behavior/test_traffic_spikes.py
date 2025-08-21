#!/usr/bin/env python3
"""
Quick test for traffic spike functionality
"""

import sys
import os
import time
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from traffic_spike_simulator import TrafficSpikeSimulator, TrafficPatternType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def quick_spike_test():
    """Quick test of traffic spike functionality."""
    print("=== 快速流量突波測試 ===")
    
    # Create simulator
    simulator = TrafficSpikeSimulator("default")
    
    # Schedule a small spike for testing
    simulator.schedule_traffic_spike(
        TrafficPatternType.SUDDEN_SPIKE,
        delay_seconds=3,
        duration=20,
        peak_users=10,
        description="測試突波"
    )
    
    print("3秒後將開始小規模流量突波測試...")
    print("測試將持續30秒")
    
    try:
        simulator.run_traffic_simulation(total_duration=30)
        print("✅ 流量突波測試成功完成！")
        return True
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

if __name__ == "__main__":
    success = quick_spike_test()
    sys.exit(0 if success else 1)