"""
Anomaly detection for gesture recognition errors.
Detects unusual patterns that indicate misclassification or system issues.
"""

import time
import logging
from collections import deque, Counter

logger = logging.getLogger(__name__)


class ErrorDetector:
    """Detects anomalous gesture patterns indicating recognition errors."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._gesture_history = deque(maxlen=window_size)
        self._action_history = deque(maxlen=window_size)
        self._error_events = []
        self._flap_threshold = 5  # Max gesture changes in 2 seconds

    def record(self, gesture_name: str, confidence: float, action_taken: bool = False):
        """Record a gesture detection event."""
        now = time.time()
        self._gesture_history.append({
            "gesture": gesture_name,
            "confidence": confidence,
            "action": action_taken,
            "time": now,
        })

    def check_anomalies(self) -> list:
        """Check for anomalous patterns.

        Returns:
            List of anomaly descriptions (empty if none detected)
        """
        anomalies = []

        if len(self._gesture_history) < 5:
            return anomalies

        recent = list(self._gesture_history)

        # 1. Gesture flapping (rapid switching)
        flap_count = self._detect_flapping(recent)
        if flap_count > self._flap_threshold:
            anomalies.append(f"gesture_flapping: {flap_count} changes in 2s")

        # 2. Low confidence streak
        low_conf_streak = self._detect_low_confidence(recent)
        if low_conf_streak > 5:
            anomalies.append(f"low_confidence_streak: {low_conf_streak} frames")

        # 3. No detection streak
        no_detect = self._detect_no_hands(recent)
        if no_detect > 10:
            anomalies.append(f"no_detection: {no_detect} frames")

        if anomalies:
            for a in anomalies:
                logger.warning("Anomaly detected: %s", a)
                self._error_events.append({"anomaly": a, "time": time.time()})

        return anomalies

    def _detect_flapping(self, history: list) -> int:
        """Count gesture type changes in the last 2 seconds."""
        now = time.time()
        recent = [h for h in history if now - h["time"] < 2.0]
        if len(recent) < 2:
            return 0

        changes = 0
        for i in range(1, len(recent)):
            if recent[i]["gesture"] != recent[i - 1]["gesture"]:
                changes += 1
        return changes

    def _detect_low_confidence(self, history: list) -> int:
        """Count consecutive low-confidence detections."""
        streak = 0
        for h in reversed(history):
            if h["confidence"] < 0.6 and h["gesture"] != "none":
                streak += 1
            else:
                break
        return streak

    def _detect_no_hands(self, history: list) -> int:
        """Count consecutive no-detection frames."""
        streak = 0
        for h in reversed(history):
            if h["gesture"] == "none" or h["confidence"] == 0:
                streak += 1
            else:
                break
        return streak

    @property
    def error_count(self) -> int:
        return len(self._error_events)

    @property
    def recent_errors(self) -> list:
        """Get errors from last 60 seconds."""
        now = time.time()
        return [e for e in self._error_events if now - e["time"] < 60]
