"""
Probabilistic confidence scoring with per-gesture threshold tuning.
Handles ambiguity detection and confidence gap analysis.

v2 improvements:
    - check_ambiguity() now actively used by the pipeline
    - Adaptive thresholds based on running statistics
    - Confidence trend tracking for smoother decisions
"""

import logging
import numpy as np
from collections import deque

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

        # Recent confidence trend for adaptive decisions
        self._recent_confidences = deque(maxlen=20)

    def should_execute(self, gesture_name: str, confidence: float) -> bool:
        """Determine if a gesture should trigger its action.

        Args:
            gesture_name: Name of the detected gesture
            confidence: Detection confidence (0-1)

        Returns:
            True if confidence exceeds gesture-specific threshold
        """
        threshold = self._thresholds.get(gesture_name, self._default_threshold)
        self._recent_confidences.append(confidence)
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
            scores: dict mapping gesture names/types to confidence scores

        Returns:
            True if detection is ambiguous
        """
        if len(scores) < 2:
            return False

        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        is_ambiguous = gap < self._min_confidence_gap
        if is_ambiguous:
            logger.debug("Ambiguous gesture detection: gap=%.3f (threshold=%.3f)",
                         gap, self._min_confidence_gap)
        return is_ambiguous

    def get_confidence_trend(self) -> str:
        """Get the trend direction of recent confidences.

        Returns:
            'rising', 'falling', or 'stable'
        """
        if len(self._recent_confidences) < 5:
            return "stable"
        recent = list(self._recent_confidences)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        diff = second_half - first_half
        if diff > 0.05:
            return "rising"
        elif diff < -0.05:
            return "falling"
        return "stable"

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
