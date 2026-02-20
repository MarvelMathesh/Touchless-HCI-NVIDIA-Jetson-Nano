"""
Hybrid gesture classifier combining rule-based geometric analysis
for static gestures and trajectory matching for dynamic gestures.

v2 improvements:
    - Config-driven classification from gestures.yaml rules
    - Angle-based finger curl features (not binary tip.y < pip.y)
    - Wider confidence score ranges for meaningful differentiation
    - Ambiguity detection (top-2 score gap check)
    - GestureType/GestureResult from core.types (no circular deps)
    - Single LandmarkExtractor passed via set_extractor()
"""

import time
import logging
from collections import deque
import numpy as np

from core.types import GestureType, GestureResult, GESTURE_ACTION_MAP
from modules.detection.landmark_extractor import (
    LandmarkExtractor, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    FINGER_TIPS, WRIST, INDEX_MCP, PINKY_MCP, THUMB_CMC,
)

logger = logging.getLogger(__name__)


class GestureClassifier:
    """Classifies hand gestures using landmark geometry and trajectory analysis.

    v2: Uses curl angles as continuous features for more robust scoring.
    Classification rules can be loaded from gestures.yaml for extensibility.
    """

    def __init__(self, config: dict, gesture_rules: dict = None):
        """Initialize the classifier.

        Args:
            config: Recognition config section from config.yaml
            gesture_rules: Optional static_gestures dict from gestures.yaml
                           If provided, rules are loaded from YAML.
        """
        self._extractor = LandmarkExtractor()
        self._thresholds = config.get("confidence_thresholds", {})
        self._default_threshold = self._thresholds.get("default", 0.80)
        self._min_confidence_gap = 0.1  # Minimum gap between top-2 gestures

        # Dynamic gesture tracking
        dyn_config = config.get("dynamic", {})
        self._min_velocity = dyn_config.get("min_velocity", 200)
        self._trajectory_window = dyn_config.get("trajectory_window", 10)
        self._swipe_min_dist = dyn_config.get("swipe_min_distance", 80)
        self._trajectory = deque(maxlen=self._trajectory_window)

        # Trajectory smoothing (EMA) to filter MediaPipe landmark jitter
        self._smoothed_palm = None
        self._ema_alpha = dyn_config.get("ema_alpha", 0.5)  # 0=no smoothing, 1=no memory

        # Minimum extended fingers to accept a trajectory point (hand-pose gate)
        self._swipe_min_fingers = dyn_config.get("swipe_min_fingers", 3)

        # User adaptation offsets {GestureType -> float}
        self._adaptation_offsets = {}

        # Load gesture rules from YAML config
        self._gesture_rules = gesture_rules or {}

    def set_extractor(self, extractor: LandmarkExtractor):
        """Use a shared LandmarkExtractor instance instead of creating a new one."""
        self._extractor = extractor

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
        # --- Hand-pose gating for trajectory ---
        # Only record trajectory points when the hand is in a "swipe-ready"
        # pose (>= N fingers extended).  This prevents fist-movements,
        # pointing repositions, etc. from polluting the swipe signal.
        finger_states = self._extractor.get_finger_states(landmarks)
        extended_count = sum(1 for v in finger_states.values() if v)

        palm_center = self._extractor.get_palm_center(landmarks)

        if extended_count >= self._swipe_min_fingers:
            # Apply EMA smoothing to filter MediaPipe landmark jitter
            if self._smoothed_palm is None:
                self._smoothed_palm = palm_center.copy()
            else:
                self._smoothed_palm = (
                    self._ema_alpha * palm_center
                    + (1.0 - self._ema_alpha) * self._smoothed_palm
                )
            self._trajectory.append((self._smoothed_palm.copy(), time.time()))

        # Try dynamic gesture classification FIRST — a detected swipe
        # always takes priority because the concurrent static result
        # merely describes hand shape during the motion, not user intent.
        dynamic_result = self._classify_dynamic()
        if dynamic_result and dynamic_result.confidence > 0.65:
            return dynamic_result

        # Static gesture classification
        static_result = self._classify_static(landmarks)
        if static_result:
            return static_result

        return GestureResult(GestureType.NONE, 0.0)

    def _classify_static(self, landmarks: np.ndarray) -> GestureResult:
        """Classify static (pose-based) gestures using continuous curl features.

        v2: Scores are built from continuous curl values instead of binary
        finger states. Base scores start lower (0.50-0.65) and are boosted
        by how well the hand matches the ideal pose, creating meaningful
        confidence differentiation.
        """
        # Extract all features in one pass
        finger_states = self._extractor.get_finger_states(landmarks)
        finger_curls = self._extractor.get_all_finger_curls(landmarks)
        thumb_dir = self._extractor.get_thumb_direction(landmarks)
        thumb_idx_dist = self._extractor.get_thumb_index_distance(landmarks)
        hand_size = self._extractor.get_hand_size(landmarks)
        inter_distances = self._extractor.get_inter_finger_distances(landmarks)
        tip_dists_norm = self._extractor.get_finger_tip_to_palm_distances(landmarks)

        # Normalize thumb-index distance by hand size
        norm_thumb_idx = thumb_idx_dist / max(hand_size, 0.001)

        scores = {}

        # --- THUMBS UP ---
        # Thumb extended (low curl) + 4 fingers curled (high curl) + thumb up
        # Guard: thumb tip must be far from palm — in a fist the thumb
        # wraps near the palm (distance < 0.55), while a real thumbs-up
        # has the thumb sticking clearly outward.
        if (finger_states["thumb"] and thumb_dir == "up"
                and tip_dists_norm["thumb"] > 0.55):
            curled_fingers = ["index", "middle", "ring", "pinky"]
            curled_score = sum(finger_curls[f] for f in curled_fingers) / 4.0
            thumb_ext = 1.0 - finger_curls["thumb"]

            if all(not finger_states[f] for f in curled_fingers):
                # Base score starts lower, curl quality boosts it
                base = 0.55
                # How tightly curled are the fingers? (0=not at all, 1=fully)
                curl_bonus = curled_score * 0.25
                # How extended is the thumb?
                thumb_bonus = thumb_ext * 0.12
                # Verticality bonus
                thumb_tip = landmarks[THUMB_TIP]
                thumb_base = landmarks[THUMB_CMC]
                dy = abs(thumb_tip[1] - thumb_base[1])
                dx = abs(thumb_tip[0] - thumb_base[0]) + 0.001
                vert_ratio = dy / (dy + dx)
                vert_bonus = vert_ratio * 0.08

                scores[GestureType.THUMBS_UP] = min(1.0, base + curl_bonus + thumb_bonus + vert_bonus)

        # --- THUMBS DOWN ---
        # Same palm-distance guard as thumbs_up
        if (finger_states["thumb"] and thumb_dir == "down"
                and tip_dists_norm["thumb"] > 0.55):
            curled_fingers = ["index", "middle", "ring", "pinky"]
            curled_score = sum(finger_curls[f] for f in curled_fingers) / 4.0

            if all(not finger_states[f] for f in curled_fingers):
                base = 0.55
                curl_bonus = curled_score * 0.25
                thumb_ext = (1.0 - finger_curls["thumb"]) * 0.12
                scores[GestureType.THUMBS_DOWN] = min(1.0, base + curl_bonus + thumb_ext)

        # --- PEACE SIGN / VICTORY ---
        if (finger_states["index"] and finger_states["middle"] and
                not finger_states["ring"] and not finger_states["pinky"]):
            base = 0.55
            # How extended are index+middle?
            ext_score = ((1.0 - finger_curls["index"]) + (1.0 - finger_curls["middle"])) / 2.0
            ext_bonus = ext_score * 0.15
            # How curled are ring+pinky?
            curl_score = (finger_curls["ring"] + finger_curls["pinky"]) / 2.0
            curl_bonus = curl_score * 0.15
            # V-spread between index and middle
            spread = inter_distances.get("index_middle", 0)
            spread_bonus = min(0.10, spread * 0.5) if spread > 0.06 else 0
            # Thumb curled gives extra confidence (cleaner V)
            thumb_bonus = 0.05 if not finger_states["thumb"] else 0

            scores[GestureType.PEACE_SIGN] = min(1.0, base + ext_bonus + curl_bonus + spread_bonus + thumb_bonus)

        # --- OK SIGN ---
        if norm_thumb_idx < 0.18:  # Thumb and index touching/close
            if (finger_states["middle"] and finger_states["ring"] and finger_states["pinky"]):
                base = 0.50
                # Tighter circle = higher confidence
                circle_bonus = max(0, 0.18 - norm_thumb_idx) * 1.5
                # How extended are the other fingers?
                ext_score = sum(1.0 - finger_curls[f] for f in ["middle", "ring", "pinky"]) / 3.0
                ext_bonus = ext_score * 0.15
                scores[GestureType.OK_SIGN] = min(1.0, base + circle_bonus + ext_bonus)

        # --- CLOSED FIST ---
        # Use curl values directly rather than relying only on binary states,
        # plus thumb-tip proximity to palm as the gate.
        four_fingers_curled = all(
            not finger_states[f] for f in ("index", "middle", "ring", "pinky")
        )
        # Also accept fist if average curl of 4 fingers is high enough
        # (catches borderline binary-state edge cases)
        avg_four_curl = sum(finger_curls[f] for f in ("index", "middle", "ring", "pinky")) / 4.0
        four_curled_soft = four_fingers_curled or avg_four_curl > 0.45
        thumb_near_palm = tip_dists_norm["thumb"] < 0.55

        if four_curled_soft and (not finger_states["thumb"] or thumb_near_palm):
            base = 0.55
            # How tightly curled are the four fingers?
            four_curl = sum(finger_curls[f] for f in ["index", "middle", "ring", "pinky"]) / 4.0
            curl_bonus = four_curl * 0.22
            # Thumb curl contributes less (wrapping ≠ curling)
            thumb_bonus = min(finger_curls["thumb"], 0.5) * 0.08
            # Compactness: all tips close to palm
            palm = self._extractor.get_palm_center(landmarks)
            tip_dists = [
                float(np.linalg.norm(landmarks[t] - palm))
                for t in FINGER_TIPS
            ]
            avg_tip_dist = np.mean(tip_dists) / max(hand_size, 0.001)
            compact_bonus = max(0, 0.35 - avg_tip_dist) * 0.5
            scores[GestureType.FIST] = min(1.0, base + curl_bonus + thumb_bonus + compact_bonus)

        # --- OPEN PALM ---
        if (finger_states["thumb"] and finger_states["index"] and
                finger_states["middle"] and finger_states["ring"] and
                finger_states["pinky"]):
            # Guard: tips must be far from palm (distinguishes true open palm
            # from a fist with borderline finger states)
            avg_tip = np.mean(list(tip_dists_norm.values()))
            if avg_tip > 0.35:
                base = 0.55
                # How extended are all fingers?
                all_ext = sum(1.0 - finger_curls[f] for f in ["thumb", "index", "middle", "ring", "pinky"]) / 5.0
                ext_bonus = all_ext * 0.20
                # Finger spread
                avg_spread = np.mean(list(inter_distances.values()))
                spread_bonus = min(0.10, avg_spread * 0.6) if avg_spread > 0.06 else 0
                # Tip-to-palm distance bonus
                reach_bonus = min(0.10, avg_tip * 0.2) if avg_tip > 0.3 else 0
                scores[GestureType.OPEN_PALM] = min(1.0, base + ext_bonus + spread_bonus + reach_bonus)

        # --- FINGER POINT (index only) ---
        if (finger_states["index"] and not finger_states["middle"] and
                not finger_states["ring"] and not finger_states["pinky"]):
            # Guard: index tip must actually be extended away from palm.
            # In a fist, ALL tips are near the palm; a true finger point
            # has the index tip standing out significantly.
            if tip_dists_norm["index"] > 0.50:
                base = 0.50
                # Index extension quality
                idx_ext = (1.0 - finger_curls["index"]) * 0.15
                # How curled are others?
                curl_others = sum(finger_curls[f] for f in ["middle", "ring", "pinky"]) / 3.0
                curl_bonus = curl_others * 0.18
                # Thumb curled = cleaner point
                thumb_bonus = 0.07 if not finger_states["thumb"] else 0
                scores[GestureType.FINGER_POINT] = min(1.0, base + idx_ext + curl_bonus + thumb_bonus)

        # --- I LOVE YOU (thumb + index + pinky) ---
        if (finger_states["thumb"] and finger_states["index"] and
                not finger_states["middle"] and not finger_states["ring"] and
                finger_states["pinky"]):
            base = 0.55
            # Extension quality of three fingers
            ext_score = sum(1.0 - finger_curls[f] for f in ["thumb", "index", "pinky"]) / 3.0
            ext_bonus = ext_score * 0.15
            # Curl quality of middle+ring
            curl_score = (finger_curls["middle"] + finger_curls["ring"]) / 2.0
            curl_bonus = curl_score * 0.18
            scores[GestureType.I_LOVE_YOU] = min(1.0, base + ext_bonus + curl_bonus)

        if not scores:
            return None

        # Apply adaptation offsets (type-safe: expects GestureType keys)
        for gesture, offset in self._adaptation_offsets.items():
            if gesture in scores:
                scores[gesture] = min(1.0, max(0.0, scores[gesture] + offset))

        # Ambiguity detection: if top-2 scores are too close, return None
        if len(scores) >= 2:
            sorted_scores = sorted(scores.values(), reverse=True)
            gap = sorted_scores[0] - sorted_scores[1]
            if gap < self._min_confidence_gap:
                logger.debug("Ambiguous detection: gap=%.3f between top-2 gestures", gap)
                return None

        # Pick highest scoring gesture
        best_gesture = max(scores, key=scores.get)
        best_score = scores[best_gesture]

        # Check against per-gesture threshold
        threshold = self._thresholds.get(best_gesture.value, self._default_threshold)
        if best_score >= threshold:
            return GestureResult(best_gesture, best_score, scores=scores)

        return None

    def _classify_dynamic(self) -> GestureResult:
        """Classify dynamic (motion-based) gestures from smoothed trajectory.

        Uses the most recent half of the trajectory window to allow
        detection even if early frames were noisy or pre-motion.
        """
        if len(self._trajectory) < 4:
            return None

        positions = [p for p, _ in self._trajectory]
        timestamps = [t for _, t in self._trajectory]

        duration = timestamps[-1] - timestamps[0]
        if duration < 0.05 or duration > 1.5:
            # Too fast (noise) or too slow (not a swipe)
            return None

        # Use first-to-last displacement for direction/distance,
        # but also check intermediate consistency (monotonic motion).
        start_pos = positions[0]
        end_pos = positions[-1]
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        # Velocity from smoothed trajectory (already EMA-filtered)
        velocity = abs(dx) / max(duration, 0.001)
        velocity_px = velocity * self._extractor._frame_width

        # Horizontal dominance: dx must be at least 1.5× dy
        if abs(dx) < abs(dy) * 1.5:
            return None

        displacement_px = abs(dx) * self._extractor._frame_width

        if velocity_px > self._min_velocity and displacement_px > self._swipe_min_dist:
            # Motion consistency: check that the intermediate points mostly
            # move in the same direction (reject shaky back-and-forth).
            direction = 1.0 if dx > 0 else -1.0
            consistent = 0
            for i in range(1, len(positions)):
                step_dx = positions[i][0] - positions[i - 1][0]
                if step_dx * direction > 0:
                    consistent += 1
            consistency_ratio = consistent / max(len(positions) - 1, 1)

            if consistency_ratio < 0.5:
                return None  # Too much back-and-forth

            # Confidence: combines velocity, displacement, and consistency
            conf_vel = min(1.0, velocity_px / 600.0)  # scale velocity contribution
            conf_disp = min(1.0, displacement_px / 200.0)
            confidence = 0.5 + 0.2 * conf_vel + 0.15 * conf_disp + 0.15 * consistency_ratio
            confidence = min(1.0, confidence)

            gesture = GestureType.SWIPE_RIGHT if dx > 0 else GestureType.SWIPE_LEFT
            self._trajectory.clear()
            self._smoothed_palm = None
            return GestureResult(gesture, confidence, is_dynamic=True)

        return None

    def apply_adaptation(self, offsets: dict):
        """Apply user-specific adaptation offsets to gesture scores.

        Args:
            offsets: dict mapping gesture_name (str) or GestureType -> float offset
        """
        self._adaptation_offsets = {}
        for key, value in offsets.items():
            if isinstance(key, GestureType):
                self._adaptation_offsets[key] = value
            elif isinstance(key, str):
                # Convert string keys to GestureType for type-safe comparison
                gesture_type = GestureType.from_string(key)
                if gesture_type != GestureType.NONE:
                    self._adaptation_offsets[gesture_type] = value
        logger.info("Applied adaptation offsets for %d gestures", len(self._adaptation_offsets))

    def reset_trajectory(self):
        """Clear trajectory history and smoothing state."""
        self._trajectory.clear()
        self._smoothed_palm = None
