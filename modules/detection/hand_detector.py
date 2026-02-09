"""
MediaPipe hand detection wrapper with optimized settings for Jetson Nano.
"""

import logging
import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)


class HandDetector:
    """MediaPipe Hands wrapper optimized for real-time performance."""

    def __init__(self, config: dict):
        self._model_complexity = config.get("model_complexity", 0)
        self._max_hands = config.get("max_num_hands", 2)
        self._min_detect_conf = config.get("min_detection_confidence", 0.6)
        self._min_track_conf = config.get("min_tracking_confidence", 0.5)

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils

        # drawing_styles was added in MediaPipe 0.8.9+; graceful fallback for 0.8.5
        try:
            self._mp_drawing_styles = mp.solutions.drawing_styles
        except AttributeError:
            self._mp_drawing_styles = None

        self._hands = None
        self._initialized = False

    def initialize(self):
        """Initialize MediaPipe Hands solution."""
        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                model_complexity=self._model_complexity,
                max_num_hands=self._max_hands,
                min_detection_confidence=self._min_detect_conf,
                min_tracking_confidence=self._min_track_conf,
            )
        except TypeError:
            # model_complexity not supported in older MediaPipe versions
            logger.warning("model_complexity not supported, falling back without it")
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=self._min_detect_conf,
                min_tracking_confidence=self._min_track_conf,
            )
        self._initialized = True
        logger.info(
            "MediaPipe Hands initialized (complexity=%d, max_hands=%d, "
            "detect_conf=%.2f, track_conf=%.2f)",
            self._model_complexity, self._max_hands,
            self._min_detect_conf, self._min_track_conf,
        )

    def detect(self, rgb_frame: np.ndarray):
        """Run hand detection on an RGB frame.

        Args:
            rgb_frame: Frame in RGB color space

        Returns:
            MediaPipe results object or None
        """
        if not self._initialized:
            self.initialize()

        # Set frame as non-writable for performance
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        return results

    def get_hand_count(self, results) -> int:
        """Get number of detected hands."""
        if results and results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks)
        return 0

    def draw_landmarks(self, frame: np.ndarray, results, color=(0, 255, 0)):
        """Draw hand landmarks and connections on BGR frame."""
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if self._mp_drawing_styles is not None:
                    self._mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._mp_drawing_styles.get_default_hand_landmarks_style(),
                        self._mp_drawing_styles.get_default_hand_connections_style(),
                    )
                else:
                    # Fallback for MediaPipe < 0.8.9 (no drawing_styles)
                    landmark_spec = self._mp_drawing.DrawingSpec(
                        color=color, thickness=2, circle_radius=2
                    )
                    connection_spec = self._mp_drawing.DrawingSpec(
                        color=(200, 200, 200), thickness=1
                    )
                    self._mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self._mp_hands.HAND_CONNECTIONS,
                        landmark_spec,
                        connection_spec,
                    )
        return frame

    def close(self):
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()
            self._initialized = False
            logger.info("MediaPipe Hands closed")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.close()
