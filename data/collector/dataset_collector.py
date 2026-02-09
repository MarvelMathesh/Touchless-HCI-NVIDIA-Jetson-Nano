"""
Interactive gesture dataset collection tool.
Captures hand images and landmarks for training/validation.
Supports augmentation and diversity tracking.
"""

import os
import json
import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

GESTURE_KEYS = {
    ord("1"): "thumbs_up",
    ord("2"): "thumbs_down",
    ord("3"): "peace_sign",
    ord("4"): "ok_sign",
    ord("5"): "fist",
    ord("6"): "open_palm",
    ord("7"): "finger_point",
    ord("8"): "i_love_you",
    ord("9"): "swipe_left",
    ord("0"): "swipe_right",
}


class DatasetCollector:
    """Interactive tool for collecting gesture training data."""

    def __init__(self, output_dir: str = "data/gestures", config: dict = None):
        config = config or {}
        self._output_dir = output_dir
        self._image_format = config.get("image_format", "jpg")
        self._save_landmarks = config.get("save_landmarks", True)
        self._save_metadata = config.get("save_metadata", True)
        self._auto_augment = config.get("auto_augment", True)

        self._session_id = f"session_{int(time.time())}"
        self._counts = {}
        self._total_saved = 0

        # Create output directories
        for gesture_name in GESTURE_KEYS.values():
            gesture_dir = os.path.join(self._output_dir, gesture_name)
            os.makedirs(gesture_dir, exist_ok=True)
            existing = len([f for f in os.listdir(gesture_dir)
                           if f.endswith(f".{self._image_format}")])
            self._counts[gesture_name] = existing

    def save_sample(self, gesture_name: str, frame: np.ndarray,
                    landmarks: np.ndarray = None, metadata: dict = None) -> str:
        """Save a gesture sample (image + landmarks + metadata).

        Args:
            gesture_name: Name of the gesture class
            frame: BGR frame from camera
            landmarks: Optional (21, 3) landmark array
            metadata: Optional metadata dict

        Returns:
            Path to saved image
        """
        gesture_dir = os.path.join(self._output_dir, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)

        count = self._counts.get(gesture_name, 0)
        basename = f"{gesture_name}_{self._session_id}_{count:04d}"

        # Save image
        img_path = os.path.join(gesture_dir, f"{basename}.{self._image_format}")
        cv2.imwrite(img_path, frame)

        # Save landmarks
        if self._save_landmarks and landmarks is not None:
            lm_path = os.path.join(gesture_dir, f"{basename}_landmarks.npy")
            np.save(lm_path, landmarks)

        # Save metadata
        if self._save_metadata and metadata:
            meta_path = os.path.join(gesture_dir, f"{basename}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

        # Auto-augmentation
        if self._auto_augment:
            self._augment_and_save(frame, gesture_dir, basename)

        self._counts[gesture_name] = count + 1
        self._total_saved += 1

        return img_path

    def _augment_and_save(self, frame: np.ndarray, gesture_dir: str, basename: str):
        """Generate augmented versions of the sample."""
        augmentations = []

        # 1. Brightness adjustment (+30%)
        bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=10)
        augmentations.append((bright, "bright"))

        # 2. Darker version (-30%)
        dark = cv2.convertScaleAbs(frame, alpha=0.7, beta=-10)
        augmentations.append((dark, "dark"))

        # 3. Gaussian noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        noisy = cv2.add(frame, noise)
        augmentations.append((noisy, "noise"))

        # 4. Horizontal flip
        flipped = cv2.flip(frame, 1)
        augmentations.append((flipped, "flip"))

        for aug_frame, suffix in augmentations:
            aug_path = os.path.join(gesture_dir, f"{basename}_aug_{suffix}.{self._image_format}")
            cv2.imwrite(aug_path, aug_frame)

    def get_status(self) -> dict:
        """Get collection status for all gesture classes."""
        return {
            "session_id": self._session_id,
            "total_saved": self._total_saved,
            "per_gesture": dict(self._counts),
        }

    def print_status(self):
        """Print formatted collection status."""
        status = self.get_status()
        logger.info("=" * 50)
        logger.info("DATASET COLLECTION STATUS")
        logger.info("=" * 50)
        logger.info("Session: %s", status["session_id"])
        logger.info("Total samples: %d", status["total_saved"])
        logger.info("-" * 30)
        for gesture, count in sorted(status["per_gesture"].items()):
            bar = "#" * min(count // 10, 30)
            logger.info("  %-15s %4d %s", gesture, count, bar)
        logger.info("=" * 50)

    def run_interactive(self, camera, hand_detector, landmark_extractor, frame_processor):
        """Run interactive collection session with live camera.

        Key bindings:
            1-0: Select gesture class and capture
            s: Print status
            q: Quit collection
        """
        logger.info("Starting interactive dataset collection")
        logger.info("Keys: 1-0 to capture gesture, s=status, q=quit")
        for key_code, name in sorted(GESTURE_KEYS.items()):
            logger.info("  %s -> %s", chr(key_code), name)

        while True:
            frame_id, frame = camera.read_sync()
            if frame is None:
                continue

            # Run hand detection
            rgb = frame_processor.preprocess(frame)
            results = hand_detector.detect(rgb)

            # Draw landmarks
            display = frame.copy()
            hand_detector.draw_landmarks(display, results)

            # Show gesture counts
            y_offset = 30
            for gesture, count in sorted(self._counts.items()):
                cv2.putText(
                    display, f"{gesture}: {count}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
                y_offset += 20

            cv2.putText(
                display, "Press 1-0 to capture, s=status, q=quit",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

            cv2.imshow("Dataset Collector", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                self.print_status()
            elif key in GESTURE_KEYS:
                gesture_name = GESTURE_KEYS[key]
                landmarks = None
                if results and results.multi_hand_landmarks:
                    landmarks = landmark_extractor.extract_landmarks(
                        results.multi_hand_landmarks[0]
                    )

                metadata = {
                    "session": self._session_id,
                    "timestamp": time.time(),
                    "frame_id": frame_id,
                    "hand_detected": landmarks is not None,
                }

                path = self.save_sample(gesture_name, frame, landmarks, metadata)
                logger.info("Saved: %s (%d total for %s)", path, self._counts[gesture_name], gesture_name)

        cv2.destroyWindow("Dataset Collector")
        self.print_status()
