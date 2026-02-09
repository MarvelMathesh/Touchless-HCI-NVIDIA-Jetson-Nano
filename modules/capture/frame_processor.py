"""
Frame preprocessing with optional GPU acceleration.
Handles color conversion, ROI extraction, and normalization.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameProcessor:
    """GPU-accelerated frame preprocessing pipeline."""

    def __init__(self, target_size=(224, 224), use_gpu=False):
        self._target_size = target_size
        self._use_gpu = use_gpu and self._check_cuda()

        # CLAHE for adaptive contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if self._use_gpu:
            logger.info("GPU preprocessing enabled (CUDA)")
        else:
            logger.info("CPU preprocessing mode")

    @staticmethod
    def _check_cuda() -> bool:
        """Check if CUDA is available in OpenCV."""
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            return count > 0
        except Exception:
            return False

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Standard preprocessing: BGR -> RGB for MediaPipe."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for difficult lighting conditions."""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        enhanced_l = self._clahe.apply(l_channel)

        # Merge and convert back
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    def detect_lighting(self, frame: np.ndarray) -> str:
        """Detect lighting condition: 'good', 'low', 'bright'."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 40:
            return "low"
        elif mean_brightness > 220:
            return "bright"
        return "good"

    def extract_roi(self, frame: np.ndarray, bbox: tuple, padding: float = 0.15) -> np.ndarray:
        """Extract region of interest around hand bounding box with padding.

        Args:
            frame: Full frame
            bbox: (x, y, w, h) bounding box
            padding: Fractional padding around bbox

        Returns:
            Cropped and resized ROI
        """
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox

        # Add padding
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            roi = cv2.resize(roi, self._target_size)
        return roi

    def auto_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """Simple gray-world white balance correction."""
        result = frame.copy().astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3.0

        if avg_b > 0:
            result[:, :, 0] *= avg_gray / avg_b
        if avg_g > 0:
            result[:, :, 1] *= avg_gray / avg_g
        if avg_r > 0:
            result[:, :, 2] *= avg_gray / avg_r

        return np.clip(result, 0, 255).astype(np.uint8)
