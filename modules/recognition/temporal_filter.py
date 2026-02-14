"""
Multi-frame temporal filtering for gesture stability.
Implements weighted consensus voting with correct recency bias
and hysteresis to prevent dwell-time reset on transient flickers.
"""

import time
import logging
from collections import deque

from core.types import GestureType

logger = logging.getLogger(__name__)


class TemporalFilter:
    """Smooths gesture recognition across frames using consensus voting.

    Prevents flickering between gestures and requires sustained
    detection before triggering actions.

    Fixes from v1:
        - Recency weight now correctly favors NEWER frames (reversed index)
        - Hysteresis prevents dwell-time reset on transient 1-2 frame flickers
        - GestureType imported from core.types (no circular dependency)
    """

    def __init__(self, config: dict):
        temporal_cfg = config.get("temporal", {})
        self._window_size = temporal_cfg.get("window_size", 5)
        self._consensus_ratio = temporal_cfg.get("consensus_ratio", 0.6)
        self._recency_weight = temporal_cfg.get("recency_weight", 0.2)
        self._dwell_time_ms = config.get("dwell_time_ms", 400)

        # Window of recent detections: (gesture_type, confidence, timestamp)
        self._window = deque(maxlen=self._window_size)

        # Track gesture onset for dwell time
        self._current_gesture = None
        self._gesture_onset = None

        # Hysteresis: allow transient mismatches without resetting dwell
        self._mismatch_count = 0
        self._max_mismatch_grace = 2  # Tolerate up to 2 consecutive mismatches

        # Track last emitted gesture to avoid duplicate events
        self._last_emitted = None
        self._last_emit_time = 0

    def update(self, gesture_result) -> dict:
        """Process a new gesture detection through temporal filter.

        Args:
            gesture_result: GestureResult from classifier

        Returns:
            dict with:
                - gesture: filtered GestureType (or None)
                - confidence: weighted confidence
                - stable: bool - whether gesture has met dwell time
                - dwell_ms: float - elapsed dwell time
                - raw: original gesture_result
        """
        gesture = gesture_result.gesture
        confidence = gesture_result.confidence
        now = time.time()

        self._window.append((gesture, confidence, now))

        # Weighted consensus voting with CORRECT recency bias
        # Newer frames get higher weight (index 0 = oldest, N-1 = newest)
        votes = {}
        total_weight = 0.0
        window_len = len(self._window)

        for i, (g, c, t) in enumerate(self._window):
            # FIX: Recency increases WITH index (newer = higher i in deque)
            # Newest frame (i = window_len-1) gets highest weight
            recency = 1.0 + self._recency_weight * (i / max(window_len - 1, 1))
            weight = c * recency
            votes[g] = votes.get(g, 0.0) + weight
            total_weight += weight

        if total_weight == 0:
            return {
                "gesture": None, "confidence": 0.0,
                "stable": False, "dwell_ms": 0, "raw": gesture_result,
            }

        # Find consensus gesture
        best_gesture = max(votes, key=votes.get)
        best_ratio = votes[best_gesture] / total_weight

        if best_ratio < self._consensus_ratio:
            # No consensus — increment mismatch counter but don't reset immediately
            self._mismatch_count += 1
            if self._mismatch_count > self._max_mismatch_grace:
                # Grace period exhausted — reset tracking
                self._current_gesture = None
                self._gesture_onset = None
                self._mismatch_count = 0
            return {
                "gesture": None, "confidence": best_ratio,
                "stable": False, "dwell_ms": 0, "raw": gesture_result,
            }

        weighted_confidence = best_ratio

        # Hysteresis: Track gesture onset for dwell time
        if best_gesture != self._current_gesture:
            self._mismatch_count += 1
            if self._mismatch_count > self._max_mismatch_grace:
                # Genuine gesture change — reset dwell timer
                self._current_gesture = best_gesture
                self._gesture_onset = now
                self._mismatch_count = 0
            # Else: within grace period, keep current gesture and onset
        else:
            # Matches — reset mismatch counter
            self._mismatch_count = 0

        # If we haven't set a current gesture yet, initialize it
        if self._current_gesture is None:
            self._current_gesture = best_gesture
            self._gesture_onset = now

        # Check dwell time
        dwell_elapsed_ms = (now - self._gesture_onset) * 1000 if self._gesture_onset else 0
        stable = dwell_elapsed_ms >= self._dwell_time_ms

        # Don't emit "none"
        if best_gesture == GestureType.NONE:
            return {
                "gesture": None, "confidence": 0.0,
                "stable": False, "dwell_ms": 0, "raw": gesture_result,
            }

        return {
            "gesture": best_gesture,
            "confidence": weighted_confidence,
            "stable": stable,
            "dwell_ms": dwell_elapsed_ms,
            "raw": gesture_result,
        }

    def reset(self):
        """Clear filter state."""
        self._window.clear()
        self._current_gesture = None
        self._gesture_onset = None
        self._mismatch_count = 0
        self._last_emitted = None

    @property
    def current_gesture(self):
        return self._current_gesture

    @property
    def window_fill(self) -> float:
        """How full the window is (0.0 - 1.0)."""
        return len(self._window) / self._window_size
