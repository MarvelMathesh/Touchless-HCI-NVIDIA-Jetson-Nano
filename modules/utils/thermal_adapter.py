import os
import logging

logger = logging.getLogger(__name__)

class ThermalAdapter:
    def __init__(self, config):
        # Uses the config key already defined in requirements/README 
        self.throttle_temp = config.get("performance.thermal_throttle_temp", 70)
        
    def get_temp(self):
        """Reads Jetson Nano thermal zone 0."""
        try:
            with open("/sys/devices/virtual/thermal/thermal_zone0/temp", "r") as f:
                return int(f.read()) / 1000
        except FileNotFoundError:
            return 45.0 # Safe fallback for non-Jetson environments

    def get_target_fps(self, temp):
        if temp >= 80: return 15  # Critical
        if temp >= self.throttle_temp: return 20 # Warning
        return 30 # Optimal