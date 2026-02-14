"""
Media action executor for VLC control.
Supports xdotool (X11) and D-Bus MPRIS2 backends.
Async execution prevents blocking the main pipeline.

v2 improvements:
    - Fixed smart_pause execution path (no longer returns None)
    - D-Bus MPRIS2 backend for bidirectional VLC control
    - MediaAction enum from core.types
    - Proper callback invocation after action completion
"""

import time
import subprocess
import threading
import logging

from core.types import MediaAction

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes media control actions via keyboard simulation or D-Bus."""

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

        # Check available backends
        self._xdotool_available = self._check_xdotool()
        self._dbus_available = self._check_dbus()

        # Choose best available method
        if self._method == "dbus" and not self._dbus_available:
            logger.warning("D-Bus not available, falling back to xdotool")
            self._method = "xdotool"

        if self._method == "xdotool" and not self._xdotool_available:
            logger.warning("xdotool not found - actions will be simulated (logged only)")

        logger.info("ActionExecutor initialized (method=%s, xdotool=%s, dbus=%s)",
                     self._method, self._xdotool_available, self._dbus_available)

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

    @staticmethod
    def _check_dbus() -> bool:
        """Check if D-Bus python bindings are available."""
        try:
            import dbus
            return True
        except ImportError:
            return False

    def execute(self, action_name: str, async_exec: bool = True) -> bool:
        """Execute a media action.

        Args:
            action_name: Action name from gesture-action mapping
            async_exec: If True, execute in background thread

        Returns:
            True if action was dispatched
        """
        # Handle special composite actions directly
        if action_name == "smart_pause":
            if async_exec:
                thread = threading.Thread(
                    target=self._execute_smart_pause, daemon=True
                )
                thread.start()
            else:
                self._execute_smart_pause()
            self._record_action(action_name)
            return True

        # Resolve key for xdotool or use D-Bus
        if self._method == "dbus":
            if async_exec:
                thread = threading.Thread(
                    target=self._send_dbus, args=(action_name,), daemon=True
                )
                thread.start()
            else:
                self._send_dbus(action_name)
        else:
            key = self._keybindings.get(action_name)
            if key is None:
                logger.debug("No keybinding for action: %s", action_name)
                return False

            if async_exec:
                thread = threading.Thread(
                    target=self._send_key, args=(key, action_name), daemon=True
                )
                thread.start()
            else:
                self._send_key(key, action_name)

        self._record_action(action_name)
        return True

    def _record_action(self, action_name: str):
        """Record action metadata and notify callbacks."""
        self._last_action = action_name
        self._last_action_time = time.time()
        self._action_count += 1

        for callback in self._action_callbacks:
            try:
                callback(action_name)
            except Exception as e:
                logger.error("Action callback error: %s", e)

    def _execute_smart_pause(self):
        """Smart Pause: Pause + rewind 5 seconds."""
        if self._method == "dbus":
            self._send_dbus("play_pause")
            time.sleep(0.1)
            self._send_dbus("seek_backward")
        else:
            key_pause = self._keybindings.get("play_pause")
            key_seek = self._keybindings.get("seek_backward")
            if key_pause:
                self._send_key(key_pause, "play_pause")
            time.sleep(0.1)
            if key_seek:
                self._send_key(key_seek, "seek_backward")
        logger.info("Smart Pause: paused and rewound")

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

    def _send_dbus(self, action_name: str):
        """Send action to VLC via D-Bus MPRIS2 interface."""
        try:
            import dbus
            bus = dbus.SessionBus()

            # Find VLC MPRIS2 service
            player = bus.get_object(
                "org.mpris.MediaPlayer2.vlc",
                "/org/mpris/MediaPlayer2"
            )
            iface = dbus.Interface(player, "org.mpris.MediaPlayer2.Player")
            props = dbus.Interface(player, "org.freedesktop.DBus.Properties")

            dbus_action_map = {
                "play_pause": lambda: iface.PlayPause(),
                "volume_up": lambda: props.Set(
                    "org.mpris.MediaPlayer2.Player", "Volume",
                    min(1.0, props.Get("org.mpris.MediaPlayer2.Player", "Volume") + self._volume_step / 100.0)
                ),
                "volume_down": lambda: props.Set(
                    "org.mpris.MediaPlayer2.Player", "Volume",
                    max(0.0, props.Get("org.mpris.MediaPlayer2.Player", "Volume") - self._volume_step / 100.0)
                ),
                "seek_forward": lambda: iface.Seek(self._seek_seconds * 1_000_000),
                "seek_backward": lambda: iface.Seek(-self._seek_seconds * 1_000_000),
                "mute": lambda: props.Set(
                    "org.mpris.MediaPlayer2.Player", "Volume", 0.0
                ),
                "fullscreen": lambda: props.Set(
                    "org.mpris.MediaPlayer2", "Fullscreen",
                    not bool(props.Get("org.mpris.MediaPlayer2", "Fullscreen"))
                ),
            }

            action_fn = dbus_action_map.get(action_name)
            if action_fn:
                action_fn()
                logger.debug("D-Bus action executed: %s", action_name)
            else:
                logger.warning("No D-Bus mapping for action: %s", action_name)

        except Exception as e:
            logger.warning("D-Bus action failed for '%s': %s. Falling back to xdotool.", action_name, e)
            # Fallback to xdotool
            key = self._keybindings.get(action_name)
            if key:
                self._send_key(key, action_name)

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

    @property
    def control_method(self) -> str:
        return self._method
