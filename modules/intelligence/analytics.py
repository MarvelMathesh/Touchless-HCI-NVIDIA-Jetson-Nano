"""
Usage pattern analytics for gesture recognition insights.
Tracks gesture distribution, timing patterns, and session statistics.
"""

import time
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class Analytics:
    """Collects and analyzes gesture usage patterns."""

    def __init__(self):
        self._session_start = time.time()
        self._gesture_counts = Counter()
        self._action_counts = Counter()
        self._gesture_times = defaultdict(list)  # gesture -> list of timestamps
        self._confidence_history = defaultdict(list)
        self._total_frames = 0
        self._detection_frames = 0

    def record_gesture(self, gesture_name: str, confidence: float):
        """Record a gesture detection."""
        self._gesture_counts[gesture_name] += 1
        self._gesture_times[gesture_name].append(time.time())
        self._confidence_history[gesture_name].append(confidence)

    def record_action(self, action_name: str):
        """Record an executed action."""
        self._action_counts[action_name] += 1

    def record_frame(self, hand_detected: bool):
        """Record a processed frame."""
        self._total_frames += 1
        if hand_detected:
            self._detection_frames += 1

    @property
    def session_duration(self) -> float:
        return time.time() - self._session_start

    @property
    def detection_rate(self) -> float:
        """Percentage of frames with hand detection."""
        if self._total_frames == 0:
            return 0.0
        return self._detection_frames / self._total_frames * 100

    def get_summary(self) -> dict:
        """Generate comprehensive session analytics summary."""
        duration = self.session_duration
        avg_confidences = {}
        for gesture, confs in self._confidence_history.items():
            if confs:
                avg_confidences[gesture] = round(sum(confs) / len(confs), 3)

        return {
            "session_duration_s": round(duration, 1),
            "total_frames": self._total_frames,
            "detection_frames": self._detection_frames,
            "detection_rate_pct": round(self.detection_rate, 1),
            "gesture_counts": dict(self._gesture_counts),
            "action_counts": dict(self._action_counts),
            "avg_confidences": avg_confidences,
            "most_used_gesture": self._gesture_counts.most_common(1)[0][0] if self._gesture_counts else None,
            "most_used_action": self._action_counts.most_common(1)[0][0] if self._action_counts else None,
            "gestures_per_minute": round(sum(self._gesture_counts.values()) / max(duration / 60, 0.1), 1),
        }

    def print_summary(self):
        """Print formatted analytics summary."""
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("SESSION ANALYTICS")
        logger.info("=" * 60)
        logger.info("Duration:        %.1fs", summary["session_duration_s"])
        logger.info("Total Frames:    %d", summary["total_frames"])
        logger.info("Detection Rate:  %.1f%%", summary["detection_rate_pct"])
        logger.info("Gestures/min:    %.1f", summary["gestures_per_minute"])
        logger.info("-" * 40)
        logger.info("Gesture Counts:")
        for gesture, count in sorted(summary["gesture_counts"].items(), key=lambda x: -x[1]):
            conf = summary["avg_confidences"].get(gesture, 0)
            logger.info("  %-18s %4d  (avg conf: %.2f)", gesture, count, conf)
        logger.info("-" * 40)
        logger.info("Action Counts:")
        for action, count in sorted(summary["action_counts"].items(), key=lambda x: -x[1]):
            logger.info("  %-18s %4d", action, count)
        logger.info("=" * 60)
