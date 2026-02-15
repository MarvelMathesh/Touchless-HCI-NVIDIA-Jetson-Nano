"""
Feature extraction pipeline: 21-point hand landmarks → ML feature vector.

Converts raw MediaPipe landmarks into a normalized, position/scale-invariant
feature vector suitable for neural network classification.

Feature layout (81 dimensions):
    [0:63]   Centered + scaled landmarks (21 × 3)
    [63:68]  Finger curl values (thumb, index, middle, ring, pinky)
    [68:73]  Finger tip-to-palm distances (normalized by hand size)
    [73:77]  Inter-finger tip distances (normalized by hand size)
    [77:78]  Thumb-index distance (normalized by hand size)
    [78:81]  Thumb direction one-hot (up, down, sideways)

Compatible with Python 3.6+ and NumPy >= 1.16.
"""

import numpy as np

# MediaPipe landmark indices (duplicated here so this module is standalone)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Joint chains: (base, mid1, mid2, tip)
FINGER_JOINTS = {
    "thumb":  (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index":  (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring":   (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky":  (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

FEATURE_DIM = 81


class GestureFeatureExtractor:
    """Converts raw hand landmarks to a fixed-size ML feature vector.

    All features are normalized to be position, scale, and
    (partially) rotation invariant so the MLP can generalise
    across users and camera positions.
    """

    def __init__(self):
        self._feature_dim = FEATURE_DIM

    @property
    def feature_dim(self):
        return self._feature_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, landmarks):
        """Convert (21, 3) landmarks → (81,) feature vector.

        Args:
            landmarks: np.ndarray of shape (21, 3) with normalized
                       (x, y, z) coordinates from MediaPipe.

        Returns:
            np.ndarray of shape (81,), dtype float32
        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (21, 3):
            raise ValueError("Expected (21, 3) landmarks, got %s" % str(landmarks.shape))

        features = np.zeros(self._feature_dim, dtype=np.float32)

        # --- Centred + scaled landmarks (63 dims) --------------------
        wrist = landmarks[WRIST].copy()
        hand_size = self._hand_size(landmarks)
        scale = max(hand_size, 1e-6)
        centred = (landmarks - wrist) / scale
        features[0:63] = centred.flatten()

        # --- Finger curls (5 dims) ------------------------------------
        curls = self._all_finger_curls(landmarks)
        for i, name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            features[63 + i] = curls[name]

        # --- Tip-to-palm distances (5 dims) ---------------------------
        palm = self._palm_center(landmarks)
        for i, tip_idx in enumerate(FINGER_TIPS):
            dist = float(np.linalg.norm(landmarks[tip_idx] - palm))
            features[68 + i] = dist / scale

        # --- Inter-finger tip distances (4 dims) ----------------------
        pairs = [
            (THUMB_TIP, INDEX_TIP),
            (INDEX_TIP, MIDDLE_TIP),
            (MIDDLE_TIP, RING_TIP),
            (RING_TIP, PINKY_TIP),
        ]
        for i, (a, b) in enumerate(pairs):
            dist = float(np.linalg.norm(landmarks[a] - landmarks[b]))
            features[73 + i] = dist / scale

        # --- Thumb-index distance (1 dim) -----------------------------
        features[77] = float(np.linalg.norm(
            landmarks[THUMB_TIP] - landmarks[INDEX_TIP]
        )) / scale

        # --- Thumb direction one-hot (3 dims) -------------------------
        thumb_dir = self._thumb_direction(landmarks)
        dir_map = {"up": 0, "down": 1, "sideways": 2}
        features[78 + dir_map.get(thumb_dir, 2)] = 1.0

        return features

    def extract_batch(self, landmarks_batch):
        """Vectorised extraction for a batch of samples.

        Args:
            landmarks_batch: np.ndarray of shape (N, 21, 3)

        Returns:
            np.ndarray of shape (N, 81)
        """
        batch_size = landmarks_batch.shape[0]
        out = np.zeros((batch_size, self._feature_dim), dtype=np.float32)
        for i in range(batch_size):
            out[i] = self.extract(landmarks_batch[i])
        return out

    # ------------------------------------------------------------------
    # Internal helpers (self-contained — no dependency on LandmarkExtractor)
    # ------------------------------------------------------------------

    @staticmethod
    def _hand_size(landmarks):
        return float(np.linalg.norm(landmarks[WRIST] - landmarks[MIDDLE_TIP]))

    @staticmethod
    def _palm_center(landmarks):
        pts = [landmarks[WRIST], landmarks[INDEX_MCP],
               landmarks[MIDDLE_MCP], landmarks[RING_MCP], landmarks[PINKY_MCP]]
        return np.mean(pts, axis=0)

    @staticmethod
    def _angle_3d(a, b, c):
        """Angle at b in the chain a-b-c, in degrees."""
        ba = a - b
        bc = c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def _all_finger_curls(self, landmarks):
        curls = {}
        for name, (base, mid1, mid2, tip) in FINGER_JOINTS.items():
            if name == "thumb":
                a1 = self._angle_3d(landmarks[base], landmarks[mid1], landmarks[mid2])
                a2 = self._angle_3d(landmarks[mid1], landmarks[mid2], landmarks[tip])
                avg = (a1 + a2) / 2.0
                curl = 1.0 - (avg / 180.0)
            else:
                a_pip = self._angle_3d(landmarks[base], landmarks[mid1], landmarks[mid2])
                a_dip = self._angle_3d(landmarks[mid1], landmarks[mid2], landmarks[tip])
                combined = a_pip * 0.6 + a_dip * 0.4
                curl = 1.0 - (combined / 180.0)
            curls[name] = max(0.0, min(1.0, curl))
        return curls

    @staticmethod
    def _thumb_direction(landmarks):
        tip = landmarks[THUMB_TIP]
        base = landmarks[THUMB_CMC]
        dy = tip[1] - base[1]
        dx = abs(tip[0] - base[0])
        if abs(dy) > dx * 0.8:
            return "down" if dy > 0 else "up"
        return "sideways"
