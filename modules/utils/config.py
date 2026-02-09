"""
Centralized configuration manager.
Loads YAML configs and provides typed access with defaults.
"""

import os
import yaml
import logging

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_DIR = os.path.join(_BASE_DIR, "config")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    """Singleton configuration manager."""

    _instance = None
    _data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path=None, gestures_path=None):
        """Load configuration from YAML files."""
        config_path = config_path or os.path.join(_CONFIG_DIR, "config.yaml")
        gestures_path = gestures_path or os.path.join(_CONFIG_DIR, "gestures.yaml")

        try:
            with open(config_path, "r") as f:
                self._data = yaml.safe_load(f) or {}
            logger.info("Loaded config from %s", config_path)
        except FileNotFoundError:
            logger.warning("Config file not found: %s, using defaults", config_path)
            self._data = {}

        try:
            with open(gestures_path, "r") as f:
                gesture_data = yaml.safe_load(f) or {}
            self._data["gestures"] = gesture_data
            logger.info("Loaded gestures from %s", gestures_path)
        except FileNotFoundError:
            logger.warning("Gestures file not found: %s", gestures_path)

        return self

    def get(self, key_path: str, default=None):
        """Get nested config value using dot notation: 'camera.width'."""
        keys = key_path.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_section(self, section: str) -> dict:
        """Get an entire config section."""
        return self._data.get(section, {})

    @property
    def camera(self) -> dict:
        return self._data.get("camera", {})

    @property
    def mediapipe(self) -> dict:
        return self._data.get("mediapipe", {})

    @property
    def recognition(self) -> dict:
        return self._data.get("recognition", {})

    @property
    def debouncing(self) -> dict:
        return self._data.get("debouncing", {})

    @property
    def media(self) -> dict:
        return self._data.get("media", {})

    @property
    def adaptation(self) -> dict:
        return self._data.get("adaptation", {})

    @property
    def performance(self) -> dict:
        return self._data.get("performance", {})

    @property
    def visualization(self) -> dict:
        return self._data.get("visualization", {})

    @property
    def gestures(self) -> dict:
        return self._data.get("gestures", {})

    @property
    def static_gestures(self) -> dict:
        return self.gestures.get("static_gestures", {})

    @property
    def dynamic_gestures(self) -> dict:
        return self.gestures.get("dynamic_gestures", {})

    @property
    def actions(self) -> dict:
        return self.gestures.get("actions", {})

    @property
    def base_dir(self) -> str:
        return _BASE_DIR
