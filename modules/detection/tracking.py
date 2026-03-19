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
    """Assigns persistent IDs to detected hands across frames.

    Features:
        - Primary Hand Lock: prevents hand-switching to passerby.
          A hand must be tracked for `_lock_after_frames` before
          becoming primary. Once locked, a new hand only takes over
          after the locked hand is lost for `_lock_lost_timeout`.
        - Interaction Zone: optional ROI filter — rejects hands
          whose palm center falls outside the configured rectangle.
    """

    def __init__(self, max_hands: int = 2, lost_timeout: float = 0.5,
                 interaction_zone: dict = None):
        self._max_hands = max_hands
        self._lost_timeout = lost_timeout
        self._tracked_hands = {}
        self._next_id = 0
        self._primary_hand_id = None

        # --- Primary Hand Lock ---
        self._lock_after_frames = 3     # Frames before a hand can become primary
        self._lock_lost_timeout = 2.0   # Seconds to keep lock after hand is lost
        self._locked_hand_id = None     # Currently locked primary hand ID
        self._lock_lost_time = None     # When the locked hand was last seen

        # --- Interaction Zone ---
        zone = interaction_zone or {}
        self._zone_enabled = zone.get("enabled", False)
        self._zone_x = zone.get("x", 0.0)
        self._zone_y = zone.get("y", 0.0)
        self._zone_w = zone.get("width", 1.0)
        self._zone_h = zone.get("height", 1.0)

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

                # Interaction zone filter: reject hands outside ROI
                if self._zone_enabled and not self._is_in_zone(landmarks):
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
        - Finger tip topology: tips should be further from wrist than MCPs
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

        # Topology check: finger tips should be further from wrist
        # than their MCP joints in at least 3 of 4 fingers.
        # A face/object misdetection has scrambled topology.
        wrist = landmarks[0, :2]
        # (MCP_index, TIP_index) for index, middle, ring, pinky
        finger_pairs = [(5, 8), (9, 12), (13, 16), (17, 20)]
        topology_ok = 0
        for mcp_idx, tip_idx in finger_pairs:
            mcp_dist = float(np.linalg.norm(landmarks[mcp_idx, :2] - wrist))
            tip_dist = float(np.linalg.norm(landmarks[tip_idx, :2] - wrist))
            if tip_dist > mcp_dist * 0.85:  # Allow slight margin
                topology_ok += 1
        if topology_ok < 2:  # At least 2 of 4 fingers must have correct topology
            return False

        return True

    def _update_primary(self):
        """Set primary hand with lock-based stability.

        Lock behaviour:
        - A new hand must survive `_lock_after_frames` before becoming primary.
        - Once locked, a different hand cannot take over until the locked
          hand is lost for `_lock_lost_timeout` seconds.
        """
        if not self._tracked_hands:
            # All hands lost
            if self._locked_hand_id is not None:
                # Start loss timer if not already running
                if self._lock_lost_time is None:
                    self._lock_lost_time = time.time()
                # Keep lock alive within timeout
                if time.time() - self._lock_lost_time < self._lock_lost_timeout:
                    self._primary_hand_id = None
                    return  # Lock still held, just no hand this frame
                else:
                    # Lock expired — release
                    logger.debug("Primary hand lock expired (ID=%d)", self._locked_hand_id)
                    self._locked_hand_id = None
                    self._lock_lost_time = None
            self._primary_hand_id = None
            return

        # Reset all primary flags
        for hand in self._tracked_hands.values():
            hand.is_primary = False

        # If locked hand is still tracked, keep it primary
        if self._locked_hand_id is not None and self._locked_hand_id in self._tracked_hands:
            self._tracked_hands[self._locked_hand_id].is_primary = True
            self._primary_hand_id = self._locked_hand_id
            self._lock_lost_time = None  # Reset loss timer
            return

        # Locked hand not present — check if within loss timeout
        if self._locked_hand_id is not None:
            if self._lock_lost_time is None:
                self._lock_lost_time = time.time()
            if time.time() - self._lock_lost_time < self._lock_lost_timeout:
                # Don't promote a new hand yet — wait for lock to expire
                # Pick the largest hand as temporary display, but don't lock
                best_id = max(
                    self._tracked_hands,
                    key=lambda hid: float(np.ptp(self._tracked_hands[hid].landmarks[:, :2], axis=0).sum())
                )
                self._tracked_hands[best_id].is_primary = True
                self._primary_hand_id = best_id
                return
            else:
                # Lock expired — release
                logger.debug("Primary hand lock expired (ID=%d)", self._locked_hand_id)
                self._locked_hand_id = None
                self._lock_lost_time = None

        # No lock — pick best hand (largest spread) if it has enough frames
        best_id = None
        best_size = 0.0
        for hand_id, hand in self._tracked_hands.items():
            # Only promote hands that have survived stability gate
            if hand.frames_tracked < self._lock_after_frames:
                continue
            spread = float(np.ptp(hand.landmarks[:, :2], axis=0).sum())
            if spread > best_size:
                best_size = spread
                best_id = hand_id

        if best_id is not None:
            self._tracked_hands[best_id].is_primary = True
            self._primary_hand_id = best_id
            self._locked_hand_id = best_id
            self._lock_lost_time = None
            logger.debug("Primary hand locked: ID=%d (after %d frames)",
                         best_id, self._tracked_hands[best_id].frames_tracked)
        else:
            # No hand has enough frames yet — pick the most-tracked as tentative
            tentative_id = max(
                self._tracked_hands,
                key=lambda hid: self._tracked_hands[hid].frames_tracked
            )
            self._tracked_hands[tentative_id].is_primary = True
            self._primary_hand_id = tentative_id

    def get_primary_hand(self):
        """Get the primary (dominant) tracked hand."""
        if self._primary_hand_id in self._tracked_hands:
            return self._tracked_hands[self._primary_hand_id]
        return None

    def _is_in_zone(self, landmarks: np.ndarray) -> bool:
        """Check if hand palm center is within the interaction zone."""
        palm = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
        px, py = palm[0], palm[1]  # Normalized 0..1
        return (self._zone_x <= px <= self._zone_x + self._zone_w and
                self._zone_y <= py <= self._zone_y + self._zone_h)

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
        self._locked_hand_id = None
        self._lock_lost_time = None
