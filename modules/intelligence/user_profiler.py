"""
Gesture Adaptation Engine - learns and adapts to each user's gesture style.
30-second calibration creates personalized gesture profiles.
"""

import os
import json
import time
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class UserProfile:
    """Stores a user's personalized gesture calibration data."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.created_at = time.time()
        self.last_updated = time.time()
        self.gesture_samples = defaultdict(list)  # gesture_name -> list of feature vectors
        self.confidence_offsets = {}  # gesture_name -> threshold adjustment
        self.gesture_counts = defaultdict(int)
        self.total_gestures = 0

    def add_sample(self, gesture_name: str, features: dict, confidence: float):
        """Add a calibration sample for a gesture."""
        self.gesture_samples[gesture_name].append({
            "features": features,
            "confidence": confidence,
            "timestamp": time.time(),
        })
        self.gesture_counts[gesture_name] += 1
        self.total_gestures += 1
        self.last_updated = time.time()

        # Keep last 100 samples per gesture
        if len(self.gesture_samples[gesture_name]) > 100:
            self.gesture_samples[gesture_name] = self.gesture_samples[gesture_name][-100:]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "confidence_offsets": self.confidence_offsets,
            "gesture_counts": dict(self.gesture_counts),
            "total_gestures": self.total_gestures,
        }

    @classmethod
    def from_dict(cls, data: dict):
        profile = cls(data.get("user_id", "default"))
        profile.created_at = data.get("created_at", time.time())
        profile.last_updated = data.get("last_updated", time.time())
        profile.confidence_offsets = data.get("confidence_offsets", {})
        profile.gesture_counts = defaultdict(int, data.get("gesture_counts", {}))
        profile.total_gestures = data.get("total_gestures", 0)
        return profile


class UserProfiler:
    """Manages user profiles and adaptive gesture learning."""

    def __init__(self, config: dict):
        self._enabled = config.get("enabled", True)
        self._calibration_duration = config.get("calibration_duration_sec", 30)
        self._storage_path = config.get("profile_storage", "data/profiles")
        self._max_profiles = config.get("max_profiles", 10)
        self._learning_rate = config.get("learning_rate", 0.1)
        self._adaptation_window = config.get("adaptation_window", 50)

        self._current_profile = UserProfile()
        self._calibrating = False
        self._calibration_start = None
        self._recent_results = []

        # Ensure storage directory exists
        os.makedirs(self._storage_path, exist_ok=True)

    def start_calibration(self, user_id: str = "default"):
        """Start a calibration session for a user."""
        if not self._enabled:
            return

        self._current_profile = UserProfile(user_id)
        self._calibrating = True
        self._calibration_start = time.time()
        logger.info("Calibration started for user '%s' (%ds duration)",
                     user_id, self._calibration_duration)

    def update(self, gesture_name: str, confidence: float, finger_states: dict = None):
        """Update the user profile with a new gesture observation.

        Args:
            gesture_name: Detected gesture
            confidence: Detection confidence
            finger_states: Optional dict of finger extension states
        """
        if not self._enabled:
            return

        features = {"confidence": confidence}
        if finger_states:
            features["finger_states"] = finger_states

        self._current_profile.add_sample(gesture_name, features, confidence)
        self._recent_results.append((gesture_name, confidence, time.time()))

        # Trim old results
        if len(self._recent_results) > self._adaptation_window:
            self._recent_results = self._recent_results[-self._adaptation_window:]

        # Check if calibration is complete
        if self._calibrating:
            elapsed = time.time() - self._calibration_start
            if elapsed >= self._calibration_duration:
                self._finish_calibration()

    def _finish_calibration(self):
        """Process calibration data and compute adaptation offsets."""
        self._calibrating = False
        logger.info("Calibration complete for user '%s'", self._current_profile.user_id)

        # Compute per-gesture confidence offsets
        for gesture_name, samples in self._current_profile.gesture_samples.items():
            if len(samples) >= 3:
                confidences = [s["confidence"] for s in samples]
                mean_conf = np.mean(confidences)
                std_conf = np.std(confidences)

                # If user consistently gets lower confidence, lower the threshold
                if mean_conf < 0.85:
                    offset = -(0.85 - mean_conf) * self._learning_rate
                    self._current_profile.confidence_offsets[gesture_name] = round(offset, 3)
                    logger.info(
                        "Adapted '%s': mean_conf=%.2f, offset=%.3f",
                        gesture_name, mean_conf, offset,
                    )

        self.save_profile()

    def get_adaptation_offsets(self) -> dict:
        """Get current confidence offsets for the gesture classifier."""
        return self._current_profile.confidence_offsets.copy()

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    @property
    def calibration_progress(self) -> float:
        """Calibration progress as 0.0-1.0."""
        if not self._calibrating:
            return 1.0
        elapsed = time.time() - self._calibration_start
        return min(1.0, elapsed / self._calibration_duration)

    @property
    def current_profile(self) -> UserProfile:
        return self._current_profile

    def save_profile(self):
        """Save current profile to disk."""
        path = os.path.join(self._storage_path, f"{self._current_profile.user_id}.json")
        try:
            with open(path, "w") as f:
                json.dump(self._current_profile.to_dict(), f, indent=2)
            logger.info("Profile saved: %s", path)
        except Exception as e:
            logger.error("Failed to save profile: %s", e)

    def load_profile(self, user_id: str) -> bool:
        """Load a saved user profile."""
        path = os.path.join(self._storage_path, f"{user_id}.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._current_profile = UserProfile.from_dict(data)
            logger.info("Profile loaded: %s", path)
            return True
        except FileNotFoundError:
            logger.info("No profile found for '%s'", user_id)
            return False
        except Exception as e:
            logger.error("Failed to load profile: %s", e)
            return False

    def list_profiles(self) -> list:
        """List available user profiles."""
        profiles = []
        try:
            for f in os.listdir(self._storage_path):
                if f.endswith(".json"):
                    profiles.append(f.replace(".json", ""))
        except FileNotFoundError:
            pass
        return profiles
