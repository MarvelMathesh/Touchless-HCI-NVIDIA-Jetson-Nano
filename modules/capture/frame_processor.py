"""
Frame preprocessing with optional GPU acceleration.
Handles color conversion, ROI extraction, enhancement, and normalization.

v2 improvements:
    - Actual CUDA paths using cv2.cuda_GpuMat for color conversion
    - Auto-enhancement driven by calibration results
    - Cached GpuMat to reduce allocation overhead
    - Lighting detection wired into preprocessing pipeline
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

        # Enhancement state (set via calibration)
        self._enhance_enabled = False
        self._lighting_condition = "good"
        self._lighting_check_interval = 60  # Check every N frames
        self._frame_counter = 0

        # GPU mat cache to reduce allocation
        self._gpu_src = None
        self._gpu_dst = None

        if self._use_gpu:
            try:
                # Pre-allocate GPU mats for 640x480 BGR
                self._gpu_src = cv2.cuda_GpuMat()
                self._gpu_dst = cv2.cuda_GpuMat()
                logger.info("GPU preprocessing enabled (CUDA with GpuMat cache)")
            except Exception as e:
                logger.warning("GPU mat allocation failed, falling back to CPU: %s", e)
                self._use_gpu = False
        else:
            logger.info("CPU preprocessing mode")

    @staticmethod
    def _check_cuda() -> bool:
        """Check if CUDA is available in OpenCV."""
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                # Verify we can actually use cuda_GpuMat
                test = cv2.cuda_GpuMat()
                return True
            return False
        except (AttributeError, Exception):
            return False

    def set_enhancement(self, enabled: bool):
        """Enable/disable auto-enhancement (called by calibration)."""
        self._enhance_enabled = enabled
        logger.info("Frame enhancement %s", "enabled" if enabled else "disabled")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Main preprocessing: optional enhancement + BGR -> RGB.

        Uses GPU for color conversion when available.
        Auto-enhances if calibration detected poor lighting.
        """
        self._frame_counter += 1

        # Periodic lighting check
        if self._frame_counter % self._lighting_check_interval == 0:
            self._lighting_condition = self.detect_lighting(frame)
            # Auto-enable enhancement in poor lighting
            if self._lighting_condition in ("low", "bright") and not self._enhance_enabled:
                self._enhance_enabled = True
                logger.info("Auto-enabled enhancement (lighting: %s)", self._lighting_condition)
            elif self._lighting_condition == "good" and self._enhance_enabled:
                self._enhance_enabled = False
                logger.info("Auto-disabled enhancement (lighting improved)")

        # Apply CLAHE enhancement if needed
        if self._enhance_enabled:
            frame = self.enhance(frame)

        # BGR -> RGB conversion
        if self._use_gpu:
            return self._preprocess_gpu(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _preprocess_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-accelerated BGR -> RGB using CUDA."""
        try:
            self._gpu_src.upload(frame)
            cv2.cuda.cvtColor(self._gpu_src, cv2.COLOR_BGR2RGB, self._gpu_dst)
            return self._gpu_dst.download()
        except Exception as e:
            # Fallback to CPU on any GPU error
            logger.debug("GPU cvtColor failed, falling back to CPU: %s", e)
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

    @property
    def is_gpu_enabled(self) -> bool:
        return self._use_gpu

    @property
    def lighting_condition(self) -> str:
        return self._lighting_condition

    @property
    def is_enhancing(self) -> bool:
        return self._enhance_enabled
