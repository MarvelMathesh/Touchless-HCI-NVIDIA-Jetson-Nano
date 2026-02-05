"""
Media Controller Module
========================

VLC media control via keyboard simulation using xdotool.
"""

import subprocess
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable
from enum import Enum, auto

logger = logging.getLogger(__name__)


class MediaAction(Enum):
    """Media control actions."""
    PLAY_PAUSE = auto()
    VOLUME_UP = auto()
    VOLUME_DOWN = auto()
    SEEK_FORWARD = auto()
    SEEK_BACKWARD = auto()
    MUTE = auto()
    FULLSCREEN = auto()
    NEXT_TRACK = auto()
    PREVIOUS_TRACK = auto()
    STOP = auto()


@dataclass
class MediaControllerConfig:
    """Media controller configuration."""
    enabled: bool = True
    controller: str = "xdotool"  # xdotool or subprocess
    
    # VLC keybindings
    keybindings: Dict[str, str] = field(default_factory=lambda: {
        "play_pause": "space",
        "volume_up": "plus",
        "volume_down": "minus",
        "seek_forward": "Right",
        "seek_backward": "Left",
        "mute": "m",
        "fullscreen": "f",
        "next_track": "n",
        "previous_track": "p",
        "stop": "s",
    })
    
    # Feedback options
    visual_feedback: bool = True
    audio_feedback: bool = False
    
    @classmethod
    def from_dict(cls, config: dict) -> "MediaControllerConfig":
        """Create config from dictionary."""
        return cls(
            enabled=config.get("enabled", True),
            controller=config.get("controller", "xdotool"),
            keybindings=config.get("keybindings", cls().keybindings),
            visual_feedback=config.get("visual_feedback", True),
            audio_feedback=config.get("audio_feedback", False),
        )


class MediaController:
    """
    VLC media player controller using keyboard simulation.
    
    Sends keyboard commands to VLC using xdotool for Linux.
    Supports all standard VLC playback controls.
    
    Example:
        >>> controller = MediaController()
        >>> controller.execute("play_pause")  # Toggle play/pause
        >>> controller.execute("volume_up")   # Increase volume
    """
    
    # Mapping from action names to MediaAction enum
    ACTION_MAP = {
        "play_pause": MediaAction.PLAY_PAUSE,
        "volume_up": MediaAction.VOLUME_UP,
        "volume_down": MediaAction.VOLUME_DOWN,
        "seek_forward": MediaAction.SEEK_FORWARD,
        "seek_backward": MediaAction.SEEK_BACKWARD,
        "mute": MediaAction.MUTE,
        "fullscreen": MediaAction.FULLSCREEN,
        "next_track": MediaAction.NEXT_TRACK,
        "previous_track": MediaAction.PREVIOUS_TRACK,
        "stop": MediaAction.STOP,
    }
    
    # Human-readable action names
    ACTION_LABELS = {
        MediaAction.PLAY_PAUSE: "Play/Pause",
        MediaAction.VOLUME_UP: "Volume Up",
        MediaAction.VOLUME_DOWN: "Volume Down",
        MediaAction.SEEK_FORWARD: "Seek Forward",
        MediaAction.SEEK_BACKWARD: "Seek Backward",
        MediaAction.MUTE: "Mute",
        MediaAction.FULLSCREEN: "Fullscreen",
        MediaAction.NEXT_TRACK: "Next Track",
        MediaAction.PREVIOUS_TRACK: "Previous Track",
        MediaAction.STOP: "Stop",
    }
    
    def __init__(self, config: Optional[MediaControllerConfig] = None):
        self.config = config or MediaControllerConfig()
        self._last_action: Optional[MediaAction] = None
        self._last_action_time: float = 0.0
        self._action_callbacks: list[Callable[[MediaAction, bool], None]] = []
        
        # Check if xdotool is available
        self._xdotool_available = self._check_xdotool()
        
        if not self._xdotool_available:
            logger.warning("xdotool not found. Media control will be simulated.")
    
    def _check_xdotool(self) -> bool:
        """Check if xdotool is installed and available."""
        try:
            result = subprocess.run(
                ["which", "xdotool"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Error checking xdotool: {e}")
            return False
    
    def execute(self, action_name: str) -> bool:
        """
        Execute a media control action.
        
        Args:
            action_name: Action name (e.g., "play_pause", "volume_up")
            
        Returns:
            True if action was executed successfully
        """
        if not self.config.enabled:
            logger.debug(f"Media control disabled, ignoring action: {action_name}")
            return False
        
        if action_name not in self.ACTION_MAP:
            logger.warning(f"Unknown action: {action_name}")
            return False
        
        action = self.ACTION_MAP[action_name]
        key = self.config.keybindings.get(action_name)
        
        if not key:
            logger.warning(f"No keybinding for action: {action_name}")
            return False
        
        success = self._send_key(key)
        
        # Track action
        self._last_action = action
        self._last_action_time = time.time()
        
        # Notify callbacks
        for callback in self._action_callbacks:
            try:
                callback(action, success)
            except Exception as e:
                logger.error(f"Error in action callback: {e}")
        
        logger.info(f"Executed action: {self.ACTION_LABELS.get(action, action_name)} (key={key}, success={success})")
        
        return success
    
    def _send_key(self, key: str) -> bool:
        """
        Send a keyboard key to VLC using xdotool.
        
        Args:
            key: Key name (e.g., "space", "Left", "plus")
            
        Returns:
            True if key was sent successfully
        """
        if not self._xdotool_available:
            # Simulation mode
            logger.debug(f"[SIMULATED] Sending key: {key}")
            return True
        
        try:
            # Normalize key names for xdotool
            xdotool_key = self._normalize_key(key)
            
            # Send key to active window
            result = subprocess.run(
                ["xdotool", "key", xdotool_key],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.error(f"xdotool error: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("xdotool command timed out")
            return False
        except Exception as e:
            logger.error(f"Error sending key: {e}")
            return False
    
    def _normalize_key(self, key: str) -> str:
        """
        Normalize key name for xdotool.
        
        xdotool uses specific key names that may differ from config.
        """
        key_map = {
            "plus": "KP_Add",      # Numpad plus (or use "plus" for shift+=)
            "minus": "KP_Subtract", # Numpad minus
            "Left": "Left",
            "Right": "Right",
            "Up": "Up",
            "Down": "Down",
            "space": "space",
        }
        
        # Try direct mapping, otherwise use key as-is
        return key_map.get(key, key)
    
    def execute_action(self, action: MediaAction) -> bool:
        """
        Execute a MediaAction enum value.
        
        Args:
            action: MediaAction enum value
            
        Returns:
            True if action was executed successfully
        """
        # Reverse lookup action name
        for name, act in self.ACTION_MAP.items():
            if act == action:
                return self.execute(name)
        
        logger.warning(f"No mapping for action: {action}")
        return False
    
    def add_callback(self, callback: Callable[[MediaAction, bool], None]) -> None:
        """
        Add callback to be notified when actions are executed.
        
        Args:
            callback: Function(action, success) to call after each action
        """
        self._action_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MediaAction, bool], None]) -> None:
        """Remove a previously added callback."""
        if callback in self._action_callbacks:
            self._action_callbacks.remove(callback)
    
    @property
    def last_action(self) -> Optional[MediaAction]:
        """Get the last executed action."""
        return self._last_action
    
    @property
    def last_action_label(self) -> str:
        """Get human-readable label of last action."""
        if self._last_action is None:
            return "None"
        return self.ACTION_LABELS.get(self._last_action, "Unknown")
    
    @property
    def is_available(self) -> bool:
        """Check if media control is available (xdotool installed)."""
        return self._xdotool_available and self.config.enabled


def test_media_controller():
    """Test media controller with simulated actions."""
    print("Testing Media Controller")
    print("=" * 40)
    
    controller = MediaController()
    print(f"xdotool available: {controller._xdotool_available}")
    print(f"Controller enabled: {controller.config.enabled}")
    
    # Test each action
    actions = ["play_pause", "volume_up", "volume_down", "seek_forward", "mute"]
    
    for action in actions:
        print(f"\nTesting: {action}")
        success = controller.execute(action)
        print(f"  Success: {success}")
        time.sleep(0.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_media_controller()
