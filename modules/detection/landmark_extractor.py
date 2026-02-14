"""
21-point hand landmark extraction and geometric feature computation.
Provides the feature set for gesture classification.

v2 improvements:
    - Angle-based finger extension detection (robust to hand rotation/tilt)
    - Full 3D angle computation (no longer discards z-coordinate)
    - get_finger_curl() now used as the primary finger state signal
    - HandData domain object population helper
"""

import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Finger joint chains: (MCP/CMC, PIP/IP, DIP, TIP) for curl/angle computation
FINGER_JOINTS = {
    "thumb":  (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index":  (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring":   (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky":  (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# Thresholds for angle-based extension detection
_FINGER_EXTENDED_ANGLE_THRESHOLD = 150.0   # degrees — above this = extended
_FINGER_CURLED_ANGLE_THRESHOLD = 100.0     # degrees — below this = curled
_THUMB_EXTENDED_ANGLE_THRESHOLD = 140.0    # thumb is more flexible


class LandmarkExtractor:
    """Extracts geometric features from MediaPipe hand landmarks."""

    def __init__(self):
        self._frame_width = 640
        self._frame_height = 480

    def set_frame_size(self, width: int, height: int):
        """Set frame dimensions for pixel coordinate conversion."""
        self._frame_width = width
        self._frame_height = height

    def extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array of (x, y, z).

        Returns:
            np.ndarray of shape (21, 3) with normalized coordinates
        """
        landmarks = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]
        return landmarks

    def to_pixel_coords(self, landmarks: np.ndarray) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates.

        Returns:
            np.ndarray of shape (21, 2) with pixel x, y
        """
        pixels = np.zeros((21, 2), dtype=np.int32)
        pixels[:, 0] = (landmarks[:, 0] * self._frame_width).astype(np.int32)
        pixels[:, 1] = (landmarks[:, 1] * self._frame_height).astype(np.int32)
        return pixels

    # =========================================================================
    # Finger State Detection — Angle-Based (v2)
    # =========================================================================

    def get_finger_states(self, landmarks: np.ndarray) -> dict:
        """Determine which fingers are extended using angle-based detection.

        Uses true consecutive-joint angles (MCP→PIP→DIP, PIP→DIP→TIP)
        to determine extension, robust to hand rotation, tilt, and depth.

        Returns:
            dict with finger names -> bool (True = extended)
        """
        states = {}
        curls = self.get_all_finger_curls(landmarks)

        # Thumb uses a wider threshold — a fist wraps the thumb sideways
        # producing curl values ~0.40-0.50.  At 0.50 we avoid false
        # "extended" readings that trigger thumbs_up/thumbs_down.
        states["thumb"] = curls["thumb"] < 0.38  # clearly extended only

        # Other fingers: use curl value with standard threshold
        for finger in ("index", "middle", "ring", "pinky"):
            states[finger] = curls[finger] < 0.35  # ~150° angle

        return states

    def get_all_finger_curls(self, landmarks: np.ndarray) -> dict:
        """Get curl values for ALL fingers including thumb.

        Uses TRUE joint angles (consecutive joints only) so the signal
        is monotonic: extended → 0, fully curled → 1.

        Previous bug: angles like MCP→PIP→TIP *skipped* the DIP joint.
        When fingertips curl past DIP and wrap near MCP (tight fist),
        the MCP-PIP-TIP angle becomes large (~straight), falsely reading
        as "extended".  Proper angles MCP→PIP→DIP and PIP→DIP→TIP
        measure the actual joint bend and avoid this.

        Returns:
            dict {finger_name: float} where 0.0=extended, 1.0=fully curled
        """
        curls = {}
        for finger_name, (base, mid1, mid2, tip) in FINGER_JOINTS.items():
            if finger_name == "thumb":
                # MCP joint angle: CMC→MCP→IP  (true joint bend)
                angle_mcp = self._angle_between_3d(
                    landmarks[base], landmarks[mid1], landmarks[mid2]
                )
                # IP joint angle: MCP→IP→TIP  (true joint bend)
                angle_ip = self._angle_between_3d(
                    landmarks[mid1], landmarks[mid2], landmarks[tip]
                )
                avg_angle = (angle_mcp + angle_ip) / 2.0
                curl = 1.0 - (avg_angle / 180.0)
            else:
                # PIP joint angle: MCP→PIP→DIP  (true joint bend)
                angle_pip = self._angle_between_3d(
                    landmarks[base], landmarks[mid1], landmarks[mid2]
                )
                # DIP joint angle: PIP→DIP→TIP  (true joint bend)
                angle_dip = self._angle_between_3d(
                    landmarks[mid1], landmarks[mid2], landmarks[tip]
                )
                # PIP has larger ROM; both contribute meaningfully
                combined_angle = angle_pip * 0.6 + angle_dip * 0.4
                curl = 1.0 - (combined_angle / 180.0)

            curls[finger_name] = max(0.0, min(1.0, curl))
        return curls

    def get_finger_curl(self, landmarks: np.ndarray, finger: str) -> float:
        """Get curl amount for a single finger (0=extended, 1=fully curled).

        Now uses full 3D angle computation.
        """
        if finger not in FINGER_JOINTS:
            return 0.0
        curls = self.get_all_finger_curls(landmarks)
        return curls.get(finger, 0.0)

    def get_thumb_direction(self, landmarks: np.ndarray) -> str:
        """Determine if thumb points up, down, or sideways.

        Uses the vector from thumb CMC to thumb TIP for more
        accurate direction than MCP→TIP.

        Returns:
            'up', 'down', or 'sideways'
        """
        thumb_tip = landmarks[THUMB_TIP]
        thumb_base = landmarks[THUMB_CMC]

        dy = thumb_tip[1] - thumb_base[1]  # Positive = down (screen coords)
        dx = abs(thumb_tip[0] - thumb_base[0])

        if abs(dy) > dx * 0.8:  # More lenient than strict > dx
            return "down" if dy > 0 else "up"
        return "sideways"

    def get_thumb_index_distance(self, landmarks: np.ndarray) -> float:
        """Distance between thumb tip and index tip (for OK sign detection)."""
        return self._distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])

    def get_palm_center(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate palm center from wrist and MCP joints."""
        points = [landmarks[WRIST], landmarks[INDEX_MCP],
                  landmarks[MIDDLE_MCP], landmarks[RING_MCP], landmarks[PINKY_MCP]]
        return np.mean(points, axis=0)

    def get_hand_size(self, landmarks: np.ndarray) -> float:
        """Estimate hand size as distance from wrist to middle finger tip."""
        return self._distance(landmarks[WRIST], landmarks[MIDDLE_TIP])

    def get_bounding_box(self, landmarks: np.ndarray, padding: float = 0.1) -> tuple:
        """Get bounding box around hand landmarks.

        Returns:
            (x, y, w, h) in pixel coordinates
        """
        pixels = self.to_pixel_coords(landmarks)
        x_min, y_min = pixels.min(axis=0)
        x_max, y_max = pixels.max(axis=0)

        w = x_max - x_min
        h = y_max - y_min
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        return (
            max(0, x_min - pad_x),
            max(0, y_min - pad_y),
            w + 2 * pad_x,
            h + 2 * pad_y,
        )

    def get_inter_finger_distances(self, landmarks: np.ndarray) -> dict:
        """Distances between adjacent finger tips, normalized by hand size."""
        tips = {
            "thumb_index": (THUMB_TIP, INDEX_TIP),
            "index_middle": (INDEX_TIP, MIDDLE_TIP),
            "middle_ring": (MIDDLE_TIP, RING_TIP),
            "ring_pinky": (RING_TIP, PINKY_TIP),
        }
        distances = {}
        hand_size = self.get_hand_size(landmarks)
        for name, (a, b) in tips.items():
            distances[name] = self._distance(landmarks[a], landmarks[b]) / max(hand_size, 0.001)
        return distances

    def get_finger_tip_to_palm_distances(self, landmarks: np.ndarray) -> dict:
        """Distance from each fingertip to palm center, normalized by hand size.

        Useful for fist/open-palm confidence scoring.
        """
        palm = self.get_palm_center(landmarks)
        hand_size = self.get_hand_size(landmarks)
        distances = {}
        for name, tip_idx in zip(
            ("thumb", "index", "middle", "ring", "pinky"), FINGER_TIPS
        ):
            dist = self._distance(landmarks[tip_idx], palm)
            distances[name] = dist / max(hand_size, 0.001)
        return distances

    def extract_hand_data(self, landmarks: np.ndarray) -> 'HandData':
        """Extract all features into a HandData domain object.

        This avoids multiple calls from the classifier by computing
        everything in one pass.
        """
        from core.types import HandData

        data = HandData()
        data.landmarks = landmarks
        data.finger_states = self.get_finger_states(landmarks)
        data.finger_curls = self.get_all_finger_curls(landmarks)
        data.thumb_direction = self.get_thumb_direction(landmarks)
        data.thumb_index_distance = self.get_thumb_index_distance(landmarks)
        data.hand_size = self.get_hand_size(landmarks)
        data.palm_center = self.get_palm_center(landmarks)
        data.inter_finger_distances = self.get_inter_finger_distances(landmarks)
        data.bounding_box = self.get_bounding_box(landmarks)
        return data

    # =========================================================================
    # Math Helpers
    # =========================================================================

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Euclidean distance between two 3D points."""
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def _angle_between_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at point b formed by points a-b-c, in degrees.

        Uses full 3D coordinates for rotation-invariant computation.
        """
        ba = a - b
        bc = c - b
        dot = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        cos_angle = dot / (norm_ba * norm_bc + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at point b formed by points a-b-c, in degrees (2D fallback)."""
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
