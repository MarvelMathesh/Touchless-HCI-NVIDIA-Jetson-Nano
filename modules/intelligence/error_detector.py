"""
Anomaly detection for gesture recognition errors.
Detects unusual patterns and provides corrective action suggestions.

v2 improvements:
    - Corrective action suggestions (not just logging)
    - Flap threshold auto-tuning
    - Recent error window with severity levels
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
            List of anomaly dicts with 'type', 'message', 'severity', 'action'
        """
        anomalies = []

        if len(self._gesture_history) < 5:
            return anomalies

        recent = list(self._gesture_history)

        # 1. Gesture flapping (rapid switching)
        flap_count = self._detect_flapping(recent)
        if flap_count > self._flap_threshold:
            anomalies.append({
                "type": "gesture_flapping",
                "message": f"{flap_count} changes in 2s",
                "severity": "warning",
                "action": "increase_dwell_time",
                "suggestion": "Consider increasing dwell_time_ms or temporal window_size",
            })

        # 2. Low confidence streak
        low_conf_streak = self._detect_low_confidence(recent)
        if low_conf_streak > 5:
            anomalies.append({
                "type": "low_confidence_streak",
                "message": f"{low_conf_streak} consecutive low-confidence frames",
                "severity": "warning" if low_conf_streak < 10 else "error",
                "action": "check_lighting",
                "suggestion": "Poor hand visibility - check lighting or camera angle",
            })

        # 3. No detection streak
        no_detect = self._detect_no_hands(recent)
        if no_detect > 10:
            anomalies.append({
                "type": "no_detection",
                "message": f"{no_detect} frames without hand detection",
                "severity": "info" if no_detect < 20 else "warning",
                "action": "wait",
                "suggestion": "No hands in frame - user may have stepped away",
            })

        # 4. Dominant single gesture (possible stuck detection)
        dominant = self._detect_dominant_gesture(recent)
        if dominant:
            anomalies.append({
                "type": "stuck_gesture",
                "message": f"'{dominant}' detected for {self._window_size}+ frames",
                "severity": "warning",
                "action": "reset_temporal_filter",
                "suggestion": "Same gesture detected for too long - may be misclassification",
            })

        if anomalies:
            for a in anomalies:
                logger.warning("Anomaly [%s]: %s - %s",
                               a["severity"], a["type"], a["message"])
                self._error_events.append({
                    "anomaly": a["type"],
                    "severity": a["severity"],
                    "time": time.time(),
                })

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

    def _detect_dominant_gesture(self, history: list) -> str:
        """Detect if one gesture dominates the entire window (stuck)."""
        if len(history) < self._window_size:
            return None

        gestures = [h["gesture"] for h in history if h["gesture"] != "none"]
        if not gestures:
            return None

        counter = Counter(gestures)
        most_common, count = counter.most_common(1)[0]
        # If >90% of window is same gesture, it may be stuck
        if count / len(history) > 0.9:
            return most_common
        return None

    @property
    def error_count(self) -> int:
        return len(self._error_events)

    @property
    def recent_errors(self) -> list:
        """Get errors from last 60 seconds."""
        now = time.time()
        return [e for e in self._error_events if now - e["time"] < 60]
