"""
Probabilistic confidence scoring with per-gesture threshold tuning.
Handles ambiguity detection and confidence gap analysis.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Manages confidence thresholds and provides execution decisions."""

    def __init__(self, config: dict):
        thresholds = config.get("confidence_thresholds", {})
        self._default_threshold = thresholds.get("default", 0.80)
        self._thresholds = {k: v for k, v in thresholds.items() if k != "default"}
        self._min_confidence_gap = 0.1

        # Track confidence history per gesture for analytics
        self._history = {}

    def should_execute(self, gesture_name: str, confidence: float) -> bool:
        """Determine if a gesture should trigger its action.

        Args:
            gesture_name: Name of the detected gesture
            confidence: Detection confidence (0-1)

        Returns:
            True if confidence exceeds gesture-specific threshold
        """
        threshold = self._thresholds.get(gesture_name, self._default_threshold)
        return confidence >= threshold

    def get_threshold(self, gesture_name: str) -> float:
        """Get the confidence threshold for a gesture."""
        return self._thresholds.get(gesture_name, self._default_threshold)

    def classify_confidence(self, confidence: float) -> str:
        """Classify confidence level for visualization.

        Returns:
            'high' (>0.85), 'medium' (0.65-0.85), or 'low' (<0.65)
        """
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.65:
            return "medium"
        return "low"

    def check_ambiguity(self, scores: dict) -> bool:
        """Check if top two gesture scores are too close (ambiguous).

        Args:
            scores: dict mapping gesture names to confidence scores

        Returns:
            True if detection is ambiguous
        """
        if len(scores) < 2:
            return False

        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        return gap < self._min_confidence_gap

    def record(self, gesture_name: str, confidence: float):
        """Record confidence for analytics."""
        if gesture_name not in self._history:
            self._history[gesture_name] = []
        self._history[gesture_name].append(confidence)
        # Keep last 200 entries per gesture
        if len(self._history[gesture_name]) > 200:
            self._history[gesture_name] = self._history[gesture_name][-200:]

    def get_stats(self, gesture_name: str) -> dict:
        """Get confidence statistics for a gesture."""
        values = self._history.get(gesture_name, [])
        if not values:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values)
        return {
            "count": len(values),
            "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
        }

    def get_all_stats(self) -> dict:
        """Get stats for all recorded gestures."""
        return {name: self.get_stats(name) for name in self._history}
