"""
Multi-hand tracking with persistent ID assignment.
Tracks hands across frames and handles hand entrance/exit.
"""

import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TrackedHand:
    """Represents a tracked hand with persistent ID."""

    def __init__(self, hand_id: int, landmarks: np.ndarray, handedness: str = "unknown"):
        self.hand_id = hand_id
        self.landmarks = landmarks
        self.handedness = handedness
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_tracked = 1
        self.is_primary = False

    def update(self, landmarks: np.ndarray):
        self.landmarks = landmarks
        self.last_seen = time.time()
        self.frames_tracked += 1

    @property
    def age_seconds(self) -> float:
        return time.time() - self.first_seen

    @property
    def time_since_last_update(self) -> float:
        return time.time() - self.last_seen


class HandTracker:
    """Assigns persistent IDs to detected hands across frames."""

    def __init__(self, max_hands: int = 2, lost_timeout: float = 0.5):
        self._max_hands = max_hands
        self._lost_timeout = lost_timeout
        self._tracked_hands = {}
        self._next_id = 0
        self._primary_hand_id = None

    def update(self, results, landmark_extractor) -> list:
        """Update tracking with new detection results.

        Args:
            results: MediaPipe detection results
            landmark_extractor: LandmarkExtractor instance

        Returns:
            List of TrackedHand objects
        """
        current_detections = []

        if results and results.multi_hand_landmarks:
            for i, hand_lm in enumerate(results.multi_hand_landmarks):
                landmarks = landmark_extractor.extract_landmarks(hand_lm)

                # Reject false-positive detections (e.g. face misread as hand)
                # A real hand has reasonable spread between landmarks
                if not self._is_valid_hand(landmarks):
                    continue

                handedness = "unknown"
                if results.multi_handedness and i < len(results.multi_handedness):
                    handedness = results.multi_handedness[i].classification[0].label.lower()
                current_detections.append((landmarks, handedness))

        # Match current detections to tracked hands
        matched_ids = set()
        for landmarks, handedness in current_detections:
            best_id = self._find_closest_hand(landmarks, matched_ids)
            if best_id is not None:
                self._tracked_hands[best_id].update(landmarks)
                self._tracked_hands[best_id].handedness = handedness
                matched_ids.add(best_id)
            else:
                # New hand
                new_id = self._next_id
                self._next_id += 1
                hand = TrackedHand(new_id, landmarks, handedness)
                self._tracked_hands[new_id] = hand
                matched_ids.add(new_id)
                logger.debug("New hand tracked: ID=%d, handedness=%s", new_id, handedness)

        # Remove lost hands
        lost_ids = []
        for hand_id, hand in self._tracked_hands.items():
            if hand_id not in matched_ids:
                if hand.time_since_last_update > self._lost_timeout:
                    lost_ids.append(hand_id)

        for hand_id in lost_ids:
            logger.debug("Hand lost: ID=%d", hand_id)
            del self._tracked_hands[hand_id]

        # Update primary hand (largest / closest)
        self._update_primary()

        return list(self._tracked_hands.values())

    def _find_closest_hand(self, landmarks: np.ndarray, exclude_ids: set) -> int:
        """Find the tracked hand closest to the given landmarks."""
        palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
        best_id = None
        best_dist = float("inf")

        for hand_id, hand in self._tracked_hands.items():
            if hand_id in exclude_ids:
                continue
            existing_center = np.mean(hand.landmarks[[0, 5, 9, 13, 17]], axis=0)
            dist = float(np.linalg.norm(palm_center[:2] - existing_center[:2]))
            if dist < best_dist and dist < 0.2:  # Max 20% of frame for matching
                best_dist = dist
                best_id = hand_id

        return best_id

    @staticmethod
    def _is_valid_hand(landmarks: np.ndarray) -> bool:
        """Reject unlikely hand detections (e.g. face features misread as hand).

        Checks that the 21-point skeleton has a reasonable spatial layout:
        - Sufficient overall spread (not all points collapsed)
        - Wrist-to-middle-fingertip distance is realistic
        - Aspect ratio not too extreme (hands are roughly square-ish)
        """
        # Bounding box of all landmarks (normalized 0..1 coords)
        xy = landmarks[:, :2]
        bbox_span = np.ptp(xy, axis=0)  # (x_spread, y_spread)
        total_spread = bbox_span.sum()

        # Too tiny = noise / distant face feature
        if total_spread < 0.05:
            return False

        # Wrist (0) to middle fingertip (12) should be a reasonable length
        wrist_to_tip = float(np.linalg.norm(landmarks[0, :2] - landmarks[12, :2]))
        if wrist_to_tip < 0.03:
            return False

        # Aspect ratio check: hand shouldn't be >4:1 thin sliver
        min_span = max(bbox_span.min(), 1e-6)
        aspect = bbox_span.max() / min_span
        if aspect > 4.0:
            return False

        return True

    def _update_primary(self):
        """Set primary hand (largest bounding box = closest to camera)."""
        if not self._tracked_hands:
            self._primary_hand_id = None
            return

        # Pick hand with largest spread (closest to camera)
        best_id = None
        best_size = 0.0
        for hand_id, hand in self._tracked_hands.items():
            hand.is_primary = False
            spread = float(np.ptp(hand.landmarks[:, :2], axis=0).sum())
            if spread > best_size:
                best_size = spread
                best_id = hand_id

        if best_id is not None:
            self._tracked_hands[best_id].is_primary = True
            self._primary_hand_id = best_id

    def get_primary_hand(self):
        """Get the primary (dominant) tracked hand."""
        if self._primary_hand_id in self._tracked_hands:
            return self._tracked_hands[self._primary_hand_id]
        return None

    @property
    def hand_count(self) -> int:
        return len(self._tracked_hands)

    @property
    def tracked_hands(self) -> dict:
        return self._tracked_hands.copy()

    def reset(self):
        """Clear all tracking state."""
        self._tracked_hands.clear()
        self._primary_hand_id = None
