"""
Hybrid gesture classifier combining rule-based geometric analysis
for static gestures and trajectory matching for dynamic gestures.
"""

import time
import logging
from enum import Enum
from collections import deque
import numpy as np

from modules.detection.landmark_extractor import (
    LandmarkExtractor, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    WRIST, INDEX_MCP, PINKY_MCP,
)

logger = logging.getLogger(__name__)


class GestureType(Enum):
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


# Maps gesture types to media actions
GESTURE_ACTION_MAP = {
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


class GestureResult:
    """Container for gesture classification result."""

    __slots__ = ("gesture", "confidence", "action", "is_dynamic", "timestamp")

    def __init__(self, gesture: GestureType, confidence: float, is_dynamic: bool = False):
        self.gesture = gesture
        self.confidence = confidence
        self.action = GESTURE_ACTION_MAP.get(gesture)
        self.is_dynamic = is_dynamic
        self.timestamp = time.time()

    def __repr__(self):
        return f"GestureResult({self.gesture.value}, conf={self.confidence:.2f})"


class GestureClassifier:
    """Classifies hand gestures using landmark geometry and trajectory analysis."""

    def __init__(self, config: dict):
        self._extractor = LandmarkExtractor()
        self._thresholds = config.get("confidence_thresholds", {})
        self._default_threshold = self._thresholds.get("default", 0.80)

        # Dynamic gesture tracking
        dyn_config = config.get("dynamic", {})
        self._min_velocity = dyn_config.get("min_velocity", 200)
        self._trajectory_window = dyn_config.get("trajectory_window", 10)
        self._swipe_min_dist = dyn_config.get("swipe_min_distance", 80)
        self._trajectory = deque(maxlen=self._trajectory_window)

        # User adaptation offsets (updated by intelligence module)
        self._adaptation_offsets = {}

    def set_frame_size(self, width: int, height: int):
        """Set frame dimensions for pixel coordinate calculations."""
        self._extractor.set_frame_size(width, height)

    def classify(self, landmarks: np.ndarray) -> GestureResult:
        """Classify gesture from hand landmarks.

        Args:
            landmarks: np.ndarray of shape (21, 3) with normalized coordinates

        Returns:
            GestureResult with gesture type, confidence, and action
        """
        # Track trajectory for dynamic gestures
        palm_center = self._extractor.get_palm_center(landmarks)
        self._trajectory.append((palm_center.copy(), time.time()))

        # Try static gesture classification first
        static_result = self._classify_static(landmarks)

        # Try dynamic gesture classification
        dynamic_result = self._classify_dynamic()

        # Return whichever has higher confidence
        if dynamic_result and dynamic_result.confidence > 0.7:
            if not static_result or dynamic_result.confidence > static_result.confidence:
                return dynamic_result

        if static_result:
            return static_result

        return GestureResult(GestureType.NONE, 0.0)

    def _classify_static(self, landmarks: np.ndarray) -> GestureResult:
        """Classify static (pose-based) gestures using geometric rules."""
        finger_states = self._extractor.get_finger_states(landmarks)
        thumb_dir = self._extractor.get_thumb_direction(landmarks)
        thumb_idx_dist = self._extractor.get_thumb_index_distance(landmarks)
        hand_size = self._extractor.get_hand_size(landmarks)

        # Normalize thumb-index distance by hand size
        norm_thumb_idx = thumb_idx_dist / max(hand_size, 0.001)

        scores = {}

        # --- THUMBS UP ---
        if (finger_states["thumb"] and
                not finger_states["index"] and
                not finger_states["middle"] and
                not finger_states["ring"] and
                not finger_states["pinky"] and
                thumb_dir == "up"):
            score = 0.9
            # Bonus for clear upward direction
            thumb_tip = landmarks[THUMB_TIP]
            thumb_mcp = landmarks[1]  # THUMB_CMC
            verticality = abs(thumb_tip[1] - thumb_mcp[1]) / max(abs(thumb_tip[0] - thumb_mcp[0]) + 0.01, 0.001)
            score += min(0.1, verticality * 0.02)
            scores[GestureType.THUMBS_UP] = min(score, 1.0)

        # --- THUMBS DOWN ---
        if (finger_states["thumb"] and
                not finger_states["index"] and
                not finger_states["middle"] and
                not finger_states["ring"] and
                not finger_states["pinky"] and
                thumb_dir == "down"):
            scores[GestureType.THUMBS_DOWN] = 0.90

        # --- PEACE SIGN / VICTORY ---
        if (not finger_states["thumb"] and
                finger_states["index"] and
                finger_states["middle"] and
                not finger_states["ring"] and
                not finger_states["pinky"]):
            score = 0.88
            # Check V-spread between index and middle
            idx_mid_dist = self._extractor.get_inter_finger_distances(landmarks).get("index_middle", 0)
            if idx_mid_dist > 0.1:
                score += 0.07
            scores[GestureType.PEACE_SIGN] = min(score, 1.0)

        # --- OK SIGN ---
        if norm_thumb_idx < 0.15:  # Thumb and index touching
            # Other fingers should be extended
            if (finger_states["middle"] and
                    finger_states["ring"] and
                    finger_states["pinky"]):
                score = 0.85
                # Tighter circle = higher confidence
                score += max(0, 0.15 - norm_thumb_idx) * 0.5
                scores[GestureType.OK_SIGN] = min(score, 1.0)

        # --- CLOSED FIST ---
        if (not finger_states["thumb"] and
                not finger_states["index"] and
                not finger_states["middle"] and
                not finger_states["ring"] and
                not finger_states["pinky"]):
            score = 0.90
            # Check compactness - all tips close to palm
            palm = self._extractor.get_palm_center(landmarks)
            tip_dists = [
                float(np.linalg.norm(landmarks[t][:2] - palm[:2]))
                for t in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
            ]
            avg_tip_dist = np.mean(tip_dists) / max(hand_size, 0.001)
            if avg_tip_dist < 0.3:
                score += 0.08
            scores[GestureType.FIST] = min(score, 1.0)

        # --- OPEN PALM ---
        if (finger_states["thumb"] and
                finger_states["index"] and
                finger_states["middle"] and
                finger_states["ring"] and
                finger_states["pinky"]):
            score = 0.88
            # Check finger spread
            distances = self._extractor.get_inter_finger_distances(landmarks)
            avg_spread = np.mean(list(distances.values()))
            if avg_spread > 0.08:
                score += 0.07
            scores[GestureType.OPEN_PALM] = min(score, 1.0)

        # --- FINGER POINT (index only) ---
        if (not finger_states["thumb"] and
                finger_states["index"] and
                not finger_states["middle"] and
                not finger_states["ring"] and
                not finger_states["pinky"]):
            scores[GestureType.FINGER_POINT] = 0.85

        # --- I LOVE YOU (thumb + index + pinky) ---
        if (finger_states["thumb"] and
                finger_states["index"] and
                not finger_states["middle"] and
                not finger_states["ring"] and
                finger_states["pinky"]):
            scores[GestureType.I_LOVE_YOU] = 0.88

        if not scores:
            return None

        # Apply adaptation offsets
        for gesture, offset in self._adaptation_offsets.items():
            if gesture in scores:
                scores[gesture] = min(1.0, max(0.0, scores[gesture] + offset))

        # Pick highest scoring gesture
        best_gesture = max(scores, key=scores.get)
        best_score = scores[best_gesture]

        # Check against threshold
        threshold = self._thresholds.get(best_gesture.value, self._default_threshold)
        if best_score >= threshold:
            return GestureResult(best_gesture, best_score)

        return None

    def _classify_dynamic(self) -> GestureResult:
        """Classify dynamic (motion-based) gestures from trajectory."""
        if len(self._trajectory) < 5:
            return None

        positions = [p for p, _ in self._trajectory]
        timestamps = [t for _, t in self._trajectory]

        # Calculate displacement
        start_pos = positions[0]
        end_pos = positions[-1]
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        duration = timestamps[-1] - timestamps[0]
        if duration < 0.05:
            return None

        # Calculate velocity (normalized coords per second)
        velocity = abs(dx) / max(duration, 0.001)

        # Convert to approximate pixels for threshold comparison
        velocity_px = velocity * self._extractor._frame_width

        # Check for horizontal swipe
        if velocity_px > self._min_velocity and abs(dx) > abs(dy) * 2:
            displacement_px = abs(dx) * self._extractor._frame_width
            if displacement_px > self._swipe_min_dist:
                confidence = min(1.0, 0.7 + velocity_px / 1000.0)
                if dx > 0:
                    self._trajectory.clear()
                    return GestureResult(GestureType.SWIPE_RIGHT, confidence, is_dynamic=True)
                else:
                    self._trajectory.clear()
                    return GestureResult(GestureType.SWIPE_LEFT, confidence, is_dynamic=True)

        return None

    def apply_adaptation(self, offsets: dict):
        """Apply user-specific adaptation offsets to gesture thresholds.

        Args:
            offsets: dict mapping GestureType -> float offset
        """
        self._adaptation_offsets = offsets
        logger.info("Applied adaptation offsets for %d gestures", len(offsets))

    def reset_trajectory(self):
        """Clear trajectory history."""
        self._trajectory.clear()
