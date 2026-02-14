"""
Thermal management for Jetson Nano.
Reads SoC temperature and adapts pipeline behavior to prevent throttling.

Thermal zones on Jetson Nano:
    /sys/class/thermal/thermal_zone0  -> AO (always-on)
    /sys/class/thermal/thermal_zone1  -> CPU
    /sys/class/thermal/thermal_zone2  -> GPU
    /sys/class/thermal/thermal_zone3  -> PLL
    /sys/class/thermal/thermal_zone4  -> PMIC (Nano B01)

Strategy:
    < 60°C  -> Full speed (target FPS)
    60-70°C -> Throttle (reduce FPS by 25%)
    70-80°C -> Heavy throttle (reduce FPS by 50%, disable GPU preprocessing)
    > 80°C  -> Critical (CPU-only, minimal FPS)
"""

import os
import time
import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)

# Thermal zone paths
THERMAL_ZONE_BASE = "/sys/class/thermal"


class ThermalState(Enum):
    NORMAL = "normal"
    WARM = "warm"
    THROTTLE = "throttle"
    CRITICAL = "critical"


class ThermalManager:
    """Monitors SoC temperature and provides adaptive performance hints."""

    def __init__(self, config: dict):
        self._throttle_temp = config.get("thermal_throttle_temp", 70)
        self._critical_temp = config.get("thermal_critical_temp", 80)
        self._warm_temp = config.get("thermal_warm_temp", 60)
        self._target_fps = config.get("target_fps", 28)
        self._poll_interval = config.get("thermal_poll_interval", 2.0)

        self._current_temp = 0.0
        self._state = ThermalState.NORMAL
        self._thermal_zones = self._discover_zones()
        self._running = False
        self._thread = None

        # Callbacks
        self._state_change_callbacks = []

        if self._thermal_zones:
            logger.info("Thermal monitoring: found %d zones: %s",
                        len(self._thermal_zones),
                        [z["name"] for z in self._thermal_zones])
        else:
            logger.warning("No thermal zones found - thermal management disabled")

    def _discover_zones(self) -> list:
        """Discover available thermal zones."""
        zones = []
        if not os.path.isdir(THERMAL_ZONE_BASE):
            return zones

        for entry in sorted(os.listdir(THERMAL_ZONE_BASE)):
            zone_path = os.path.join(THERMAL_ZONE_BASE, entry)
            temp_path = os.path.join(zone_path, "temp")
            type_path = os.path.join(zone_path, "type")

            if os.path.isfile(temp_path):
                # Read zone type
                zone_type = "unknown"
                try:
                    with open(type_path, "r") as f:
                        zone_type = f.read().strip()
                except (FileNotFoundError, PermissionError):
                    pass

                zones.append({
                    "name": entry,
                    "type": zone_type,
                    "path": temp_path,
                })

        return zones

    def read_temperature(self) -> float:
        """Read current SoC temperature in Celsius.

        Returns the maximum temperature across all zones.
        """
        if not self._thermal_zones:
            return 0.0

        max_temp = 0.0
        for zone in self._thermal_zones:
            try:
                with open(zone["path"], "r") as f:
                    # Temperature is in millidegrees Celsius
                    raw = int(f.read().strip())
                    temp_c = raw / 1000.0
                    max_temp = max(max_temp, temp_c)
            except (ValueError, IOError, PermissionError):
                continue

        self._current_temp = max_temp
        return max_temp

    def get_state(self) -> ThermalState:
        """Get current thermal state based on temperature."""
        temp = self.read_temperature()
        old_state = self._state

        if temp >= self._critical_temp:
            self._state = ThermalState.CRITICAL
        elif temp >= self._throttle_temp:
            self._state = ThermalState.THROTTLE
        elif temp >= self._warm_temp:
            self._state = ThermalState.WARM
        else:
            self._state = ThermalState.NORMAL

        if self._state != old_state:
            logger.info("Thermal state changed: %s -> %s (%.1f°C)",
                        old_state.value, self._state.value, temp)
            for callback in self._state_change_callbacks:
                try:
                    callback(self._state, temp)
                except Exception as e:
                    logger.error("Thermal callback error: %s", e)

        return self._state

    def get_recommended_fps(self) -> int:
        """Get recommended FPS based on thermal state."""
        state = self.get_state()
        if state == ThermalState.CRITICAL:
            return max(5, self._target_fps // 4)
        elif state == ThermalState.THROTTLE:
            return max(10, self._target_fps // 2)
        elif state == ThermalState.WARM:
            return max(15, int(self._target_fps * 0.75))
        return self._target_fps

    def should_use_gpu(self) -> bool:
        """Whether GPU preprocessing should be used."""
        return self._state not in (ThermalState.CRITICAL,)

    def on_state_change(self, callback):
        """Register callback for thermal state changes.

        callback(state: ThermalState, temperature: float)
        """
        self._state_change_callbacks.append(callback)

    def start_monitoring(self):
        """Start background thermal monitoring thread."""
        if not self._thermal_zones:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Thermal monitoring started (interval=%.1fs)", self._poll_interval)

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            self.get_state()
            time.sleep(self._poll_interval)

    @property
    def temperature(self) -> float:
        return self._current_temp

    @property
    def state(self) -> ThermalState:
        return self._state

    @property
    def is_throttling(self) -> bool:
        return self._state in (ThermalState.THROTTLE, ThermalState.CRITICAL)

    def get_zone_temperatures(self) -> dict:
        """Get per-zone temperature readings."""
        temps = {}
        for zone in self._thermal_zones:
            try:
                with open(zone["path"], "r") as f:
                    raw = int(f.read().strip())
                    temps[zone["type"]] = round(raw / 1000.0, 1)
            except (ValueError, IOError, PermissionError):
                temps[zone["type"]] = None
        return temps
