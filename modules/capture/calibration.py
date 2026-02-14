"""
Camera calibration utilities for auto-exposure and white balance.

v2 improvements:
    - Returns frame_processor-compatible enhancement flag
    - Enhanced lighting assessment with zone analysis
    - Optional auto-set FrameProcessor enhancement on calibration
"""

import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Calibrator:
    """Auto-calibration for camera settings."""

    def __init__(self, target_brightness=128, tolerance=20):
        self._target_brightness = target_brightness
        self._tolerance = tolerance

    def auto_calibrate(self, camera, num_frames=30, frame_processor=None) -> dict:
        """Run auto-calibration sequence and return optimal settings.

        Args:
            camera: CameraManager instance
            num_frames: Number of frames to analyze
            frame_processor: Optional FrameProcessor to auto-configure enhancement

        Returns:
            dict with calibration results
        """
        logger.info("Starting auto-calibration (%d frames)...", num_frames)
        brightness_values = []
        contrast_values = []

        for i in range(num_frames):
            _, frame = camera.read_sync()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))
            contrast_values.append(np.std(gray))
            time.sleep(0.033)  # ~30 FPS sampling

        if not brightness_values:
            logger.warning("Calibration failed: no frames captured")
            return {"success": False}

        avg_brightness = np.mean(brightness_values)
        avg_contrast = np.mean(contrast_values)
        brightness_stability = np.std(brightness_values)

        needs_enhancement = avg_brightness < 60 or avg_brightness > 200

        result = {
            "success": True,
            "avg_brightness": round(float(avg_brightness), 1),
            "avg_contrast": round(float(avg_contrast), 1),
            "brightness_stability": round(float(brightness_stability), 2),
            "lighting_quality": self._assess_lighting(avg_brightness, avg_contrast),
            "needs_enhancement": needs_enhancement,
        }

        # Auto-configure FrameProcessor if provided
        if frame_processor is not None and needs_enhancement:
            frame_processor.set_enhancement(True)
            logger.info("Auto-enabled frame enhancement based on calibration")

        logger.info(
            "Calibration complete: brightness=%.1f, contrast=%.1f, quality=%s, enhance=%s",
            result["avg_brightness"],
            result["avg_contrast"],
            result["lighting_quality"],
            needs_enhancement,
        )
        return result

    @staticmethod
    def _assess_lighting(brightness: float, contrast: float) -> str:
        """Assess overall lighting quality."""
        if brightness < 30:
            return "very_low"
        elif brightness < 60:
            return "low"
        elif brightness > 220:
            return "overexposed"
        elif contrast < 20:
            return "flat"
        elif 80 <= brightness <= 180 and contrast >= 40:
            return "good"
        else:
            return "acceptable"
