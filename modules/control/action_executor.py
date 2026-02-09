"""
Media action executor for VLC control via xdotool keyboard simulation.
Supports async execution to prevent blocking the main pipeline.
"""

import time
import subprocess
import threading
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MediaAction(Enum):
    PLAY_PAUSE = "play_pause"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    SEEK_FORWARD = "seek_forward"
    SEEK_BACKWARD = "seek_backward"
    MUTE = "mute"
    FULLSCREEN = "fullscreen"
    SMART_PAUSE = "smart_pause"
    SEEK_POSITION = "seek_position"


class ActionExecutor:
    """Executes media control actions via keyboard simulation."""

    def __init__(self, config: dict):
        self._player = config.get("player", "vlc")
        self._method = config.get("control_method", "xdotool")
        self._keybindings = config.get("keybindings", {})
        self._seek_seconds = config.get("seek_seconds", 10)
        self._volume_step = config.get("volume_step", 5)

        self._action_callbacks = []
        self._last_action = None
        self._last_action_time = 0
        self._action_count = 0
        self._xdotool_available = self._check_xdotool()

        if not self._xdotool_available:
            logger.warning("xdotool not found - actions will be simulated (logged only)")

    @staticmethod
    def _check_xdotool() -> bool:
        """Check if xdotool is available."""
        try:
            result = subprocess.run(
                ["which", "xdotool"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=2,
            )
            return result.returncode == 0
        except Exception:
            return False

    def execute(self, action_name: str, async_exec: bool = True) -> bool:
        """Execute a media action.

        Args:
            action_name: Action name from gesture-action mapping
            async_exec: If True, execute in background thread

        Returns:
            True if action was dispatched
        """
        key = self._resolve_key(action_name)
        if key is None:
            logger.debug("No keybinding for action: %s", action_name)
            return False

        if async_exec:
            thread = threading.Thread(target=self._send_key, args=(key, action_name), daemon=True)
            thread.start()
        else:
            self._send_key(key, action_name)

        self._last_action = action_name
        self._last_action_time = time.time()
        self._action_count += 1

        # Notify callbacks
        for callback in self._action_callbacks:
            try:
                callback(action_name)
            except Exception as e:
                logger.error("Action callback error: %s", e)

        return True

    def execute_smart_pause(self):
        """Smart Pause: Pause + rewind 5 seconds."""
        self.execute("play_pause", async_exec=False)
        time.sleep(0.1)
        # Seek backward
        self.execute("seek_backward", async_exec=False)
        logger.info("Smart Pause: paused and rewound")

    def _resolve_key(self, action_name: str) -> str:
        """Map action name to keyboard key."""
        # Handle special composite actions
        if action_name == "smart_pause":
            self.execute_smart_pause()
            return None

        return self._keybindings.get(action_name)

    def _send_key(self, key: str, action_name: str):
        """Send keystroke to VLC via xdotool."""
        if not self._xdotool_available:
            logger.info("[SIMULATED] Action: %s -> key: %s", action_name, key)
            return

        try:
            # Find VLC window and send key
            subprocess.run(
                ["xdotool", "search", "--name", "VLC", "key", "--window", "%@", key],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=1,
            )
            logger.debug("Sent key '%s' for action '%s'", key, action_name)
        except subprocess.TimeoutExpired:
            logger.warning("xdotool timed out for action: %s", action_name)
        except Exception as e:
            logger.error("Failed to send key: %s", e)

    def on_action(self, callback):
        """Register callback for when actions are executed.

        callback(action_name: str)
        """
        self._action_callbacks.append(callback)

    @property
    def last_action(self) -> str:
        return self._last_action

    @property
    def last_action_time(self) -> float:
        return self._last_action_time

    @property
    def action_count(self) -> int:
        return self._action_count
