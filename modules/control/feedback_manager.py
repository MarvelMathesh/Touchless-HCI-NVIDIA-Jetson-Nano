"""
Visual and audio feedback for executed actions.
Provides confirmation overlay and optional sound effects.
"""

import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Manages visual and audio feedback for gesture-triggered actions."""

    def __init__(self, config: dict = None):
        config = config or {}
        actions_config = config.get("actions", {})

        self._active_feedback = None
        self._feedback_duration = 1.0  # seconds
        self._fade_duration = 0.3

        # Action display info
        self._action_display = {
            "play_pause": {"icon": "||", "label": "Play/Pause", "color": (0, 255, 0)},
            "volume_up": {"icon": "VOL+", "label": "Volume Up", "color": (0, 255, 255)},
            "volume_down": {"icon": "VOL-", "label": "Volume Down", "color": (0, 255, 255)},
            "seek_forward": {"icon": ">>", "label": "Forward 10s", "color": (255, 200, 0)},
            "seek_backward": {"icon": "<<", "label": "Back 10s", "color": (255, 200, 0)},
            "mute": {"icon": "MUTE", "label": "Mute/Unmute", "color": (0, 0, 255)},
            "fullscreen": {"icon": "[ ]", "label": "Fullscreen", "color": (255, 0, 255)},
            "smart_pause": {"icon": "<<||", "label": "Smart Pause", "color": (255, 100, 0)},
            "seek_position": {"icon": "|>", "label": "Seek", "color": (200, 200, 0)},
        }

    def trigger(self, action_name: str):
        """Trigger visual feedback for an action."""
        display = self._action_display.get(action_name, {
            "icon": "?", "label": action_name, "color": (255, 255, 255)
        })
        self._active_feedback = {
            "action": action_name,
            "icon": display["icon"],
            "label": display["label"],
            "color": display["color"],
            "start_time": time.time(),
        }

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render action feedback overlay on frame.

        Args:
            frame: BGR frame to draw on

        Returns:
            Frame with feedback overlay
        """
        if self._active_feedback is None:
            return frame

        elapsed = time.time() - self._active_feedback["start_time"]
        if elapsed > self._feedback_duration:
            self._active_feedback = None
            return frame

        # Calculate opacity (fade out)
        if elapsed > self._feedback_duration - self._fade_duration:
            fade_progress = (elapsed - (self._feedback_duration - self._fade_duration)) / self._fade_duration
            opacity = 1.0 - fade_progress
        else:
            opacity = 1.0

        h, w = frame.shape[:2]
        fb = self._active_feedback

        # Draw action feedback box at center
        box_w, box_h = 250, 80
        x1 = (w - box_w) // 2
        y1 = h // 2 - box_h - 20

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (40, 40, 40), -1)
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), fb["color"], 2)
        cv2.addWeighted(overlay, opacity * 0.8, frame, 1 - opacity * 0.8, 0, frame)

        if opacity > 0.3:
            # Icon
            cv2.putText(
                frame, fb["icon"],
                (x1 + 15, y1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, fb["color"], 3,
            )
            # Label
            cv2.putText(
                frame, fb["label"],
                (x1 + 90, y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        return frame

    @property
    def is_active(self) -> bool:
        if self._active_feedback is None:
            return False
        elapsed = time.time() - self._active_feedback["start_time"]
        return elapsed <= self._feedback_duration
