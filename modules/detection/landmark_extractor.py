"""
21-point hand landmark extraction and geometric feature computation.
Provides the feature set for gesture classification.
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

    def get_finger_states(self, landmarks: np.ndarray) -> dict:
        """Determine which fingers are extended/curled.

        Uses comparison of tip position relative to PIP joint:
        - For thumb: compare x-distance from wrist (handles left/right hand)
        - For other fingers: tip.y < pip.y means extended (screen coords)

        Returns:
            dict with finger names -> bool (True = extended)
        """
        states = {}

        # Thumb: special case - compare tip x-distance vs IP x-distance from palm center
        wrist = landmarks[WRIST]
        thumb_tip = landmarks[THUMB_TIP]
        thumb_ip = landmarks[THUMB_IP]
        thumb_mcp = landmarks[THUMB_MCP]

        # Determine hand orientation: if index MCP is to the right of pinky MCP, it's right hand
        index_mcp = landmarks[INDEX_MCP]
        pinky_mcp = landmarks[PINKY_MCP]
        is_right_hand = index_mcp[0] < pinky_mcp[0]  # In mirrored view

        if is_right_hand:
            states["thumb"] = thumb_tip[0] < thumb_ip[0]
        else:
            states["thumb"] = thumb_tip[0] > thumb_ip[0]

        # Other fingers: tip y < pip y means extended (normalized coords, y increases downward)
        finger_names = ["index", "middle", "ring", "pinky"]
        tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

        for name, tip_idx, pip_idx in zip(finger_names, tips, pips):
            states[name] = landmarks[tip_idx][1] < landmarks[pip_idx][1]

        return states

    def get_thumb_direction(self, landmarks: np.ndarray) -> str:
        """Determine if thumb points up or down.

        Returns:
            'up', 'down', or 'sideways'
        """
        thumb_tip = landmarks[THUMB_TIP]
        thumb_mcp = landmarks[THUMB_MCP]

        dy = thumb_tip[1] - thumb_mcp[1]  # Positive = down (screen coords)
        dx = abs(thumb_tip[0] - thumb_mcp[0])

        if abs(dy) > dx:
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
        """Distances between adjacent finger tips."""
        tips = {
            "thumb_index": (THUMB_TIP, INDEX_TIP),
            "index_middle": (INDEX_TIP, MIDDLE_TIP),
            "middle_ring": (MIDDLE_TIP, RING_TIP),
            "ring_pinky": (RING_TIP, PINKY_TIP),
        }
        distances = {}
        hand_size = self.get_hand_size(landmarks)
        for name, (a, b) in tips.items():
            # Normalize by hand size for scale invariance
            distances[name] = self._distance(landmarks[a], landmarks[b]) / max(hand_size, 0.001)
        return distances

    def get_finger_curl(self, landmarks: np.ndarray, finger: str) -> float:
        """Get curl amount for a finger (0=fully extended, 1=fully curled).

        Based on angle at PIP joint.
        """
        finger_map = {
            "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
            "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
            "ring": (RING_MCP, RING_PIP, RING_TIP),
            "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
        }
        if finger not in finger_map:
            return 0.0

        mcp_idx, pip_idx, tip_idx = finger_map[finger]
        angle = self._angle_between(
            landmarks[mcp_idx], landmarks[pip_idx], landmarks[tip_idx]
        )
        # Straight finger ~180 degrees, curled ~60 degrees
        curl = 1.0 - (angle / 180.0)
        return max(0.0, min(1.0, curl))

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Euclidean distance between two 3D points."""
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at point b formed by points a-b-c, in degrees."""
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
