"""
Shared domain types for the Touchless Media Control system.

Centralizes enums, data classes, and type definitions used across modules
to eliminate circular imports and ensure type consistency.
"""

import time
from enum import Enum
from typing import Optional, Dict, List
import numpy as np


# =============================================================================
# Gesture Types
# =============================================================================

class GestureType(Enum):
    """All recognized gesture types (static and dynamic)."""
    NONE = "none"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE_SIGN = "peace_sign"
    OK_SIGN = "ok_sign"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    FINGER_POINT = "finger_point"
    I_LOVE_YOU = "i_love_you"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"

    @classmethod
    def from_string(cls, name: str) -> 'GestureType':
        """Convert a string gesture name to GestureType enum, safely."""
        try:
            return cls(name)
        except ValueError:
            return cls.NONE

    @property
    def is_static(self) -> bool:
        return self not in (GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT, GestureType.NONE)

    @property
    def is_dynamic(self) -> bool:
        return self in (GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT)


class MediaAction(Enum):
    """All possible media control actions."""
    PLAY_PAUSE = "play_pause"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    SEEK_FORWARD = "seek_forward"
    SEEK_BACKWARD = "seek_backward"
    MUTE = "mute"
    FULLSCREEN = "fullscreen"
    SMART_PAUSE = "smart_pause"
    SEEK_POSITION = "seek_position"


# =============================================================================
# Gesture ↔ Action Mapping
# =============================================================================

GESTURE_ACTION_MAP: Dict[GestureType, str] = {
    GestureType.THUMBS_UP: "play_pause",
    GestureType.THUMBS_DOWN: "smart_pause",
    GestureType.PEACE_SIGN: "volume_up",
    GestureType.OK_SIGN: "volume_down",
    GestureType.FIST: "mute",
    GestureType.OPEN_PALM: "fullscreen",
    GestureType.FINGER_POINT: "seek_position",
    GestureType.SWIPE_LEFT: "seek_backward",
    GestureType.SWIPE_RIGHT: "seek_forward",
}


# =============================================================================
# Data Containers
# =============================================================================

class GestureResult:
    """Container for gesture classification output.

    Uses __slots__ for memory efficiency on embedded devices.
    """

    __slots__ = ("gesture", "confidence", "action", "is_dynamic", "timestamp", "scores")

    def __init__(self, gesture: GestureType, confidence: float,
                 is_dynamic: bool = False, scores: Optional[Dict] = None):
        self.gesture = gesture
        self.confidence = confidence
        self.action = GESTURE_ACTION_MAP.get(gesture)
        self.is_dynamic = is_dynamic
        self.timestamp = time.time()
        self.scores = scores or {}

    def __repr__(self):
        return f"GestureResult({self.gesture.value}, conf={self.confidence:.2f})"

    @property
    def has_action(self) -> bool:
        return self.action is not None


class HandData:
    """Encapsulates all extracted hand features for a single frame.

    Provides a clean domain object instead of passing raw numpy arrays
    and dictionaries through the pipeline.
    """

    __slots__ = (
        "landmarks", "finger_states", "finger_curls",
        "thumb_direction", "thumb_index_distance", "hand_size",
        "palm_center", "inter_finger_distances", "bounding_box",
        "handedness", "hand_id",
    )

    def __init__(self):
        self.landmarks: Optional[np.ndarray] = None       # (21, 3) normalized
        self.finger_states: Dict[str, bool] = {}           # {finger: extended}
        self.finger_curls: Dict[str, float] = {}           # {finger: 0.0–1.0}
        self.thumb_direction: str = "sideways"             # up/down/sideways
        self.thumb_index_distance: float = 0.0
        self.hand_size: float = 0.0
        self.palm_center: Optional[np.ndarray] = None      # (3,)
        self.inter_finger_distances: Dict[str, float] = {}
        self.bounding_box: Optional[tuple] = None          # (x, y, w, h)
        self.handedness: str = "unknown"
        self.hand_id: int = -1


class FrameBundle:
    """Bundles a frame with its associated metadata through the pipeline.

    Avoids the need to pass multiple arguments between pipeline stages.
    """

    __slots__ = (
        "frame_id", "bgr_frame", "rgb_frame", "timestamp",
        "hand_data", "gesture_result", "filtered_result",
        "should_skip",
    )

    def __init__(self, frame_id: int, bgr_frame: np.ndarray):
        self.frame_id = frame_id
        self.bgr_frame = bgr_frame
        self.rgb_frame: Optional[np.ndarray] = None
        self.timestamp = time.time()
        self.hand_data: Optional[HandData] = None
        self.gesture_result: Optional[GestureResult] = None
        self.filtered_result: Optional[dict] = None
        self.should_skip = False


class PipelineState:
    """Shared mutable state for the entire pipeline, observed by the dashboard.

    Thread-safe access via properties is optional here since the main
    pipeline reads/writes from the same thread in the current design.
    """

    def __init__(self):
        self.current_gesture: Optional[str] = None
        self.current_confidence: float = 0.0
        self.last_action: Optional[str] = None
        self.hand_detected: bool = False
        self.hand_count: int = 0
        self.hand_bbox: Optional[tuple] = None
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.mode: str = "control"
        self.calibrating: bool = False
        self.calibration_progress: float = 0.0
        self.frame_count: int = 0
        self.thermal_temp: float = 0.0
        self.gpu_utilization: float = 0.0

    def to_dashboard_dict(self) -> dict:
        """Convert to the dict format expected by Dashboard.render()."""
        return {
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "gesture_name": self.current_gesture,
            "gesture_confidence": self.current_confidence,
            "hand_detected": self.hand_detected,
            "hand_bbox": self.hand_bbox,
            "hand_count": self.hand_count,
            "mode": self.mode,
            "calibrating": self.calibrating,
            "calibration_progress": self.calibration_progress,
        }
