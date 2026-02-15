"""
HybridClassifier — ML-first gesture classifier with rule-based fallback.

Priority order:
    1. TensorRT engine   (fastest, FP16, <0.3 ms on Jetson Nano)
    2. PyTorch model      (CPU/GPU, ~1 ms)
    3. Rule-based engine  (always available, no model needed)

Automatically selects the best available backend at init time.
Falls back to rule-based when ML confidence is below a threshold
or when no trained model exists.

Compatible with Python 3.6+.
"""

import os
import logging
import time
import numpy as np

from core.types import GestureType, GestureResult

logger = logging.getLogger(__name__)

# Class index → GestureType mapping (must match training label order)
DEFAULT_CLASS_MAP = {
    0: GestureType.THUMBS_UP,
    1: GestureType.THUMBS_DOWN,
    2: GestureType.PEACE_SIGN,
    3: GestureType.OK_SIGN,
    4: GestureType.FIST,
    5: GestureType.OPEN_PALM,
    6: GestureType.FINGER_POINT,
    7: GestureType.I_LOVE_YOU,
    8: GestureType.NONE,           # "no gesture" rejection class
}

# Reverse: GestureType → class index
DEFAULT_LABEL_MAP = {v: k for k, v in DEFAULT_CLASS_MAP.items()}


class HybridClassifier:
    """ML-first gesture classifier with automatic fallback.

    Usage::

        hybrid = HybridClassifier(config, rule_classifier)
        result = hybrid.classify(landmarks)
    """

    def __init__(self, config, rule_classifier=None):
        """
        Args:
            config: ``ml_classifier`` section from config.yaml
            rule_classifier: existing :class:`GestureClassifier` instance
                             used as fallback when ML is unavailable or
                             confidence is too low.
        """
        self._config = config
        self._rule_classifier = rule_classifier

        # Paths
        model_dir = config.get("model_dir", "models/weights")
        self._trt_path = os.path.join(model_dir, config.get("trt_engine", "gesture_net.engine"))
        self._onnx_path = os.path.join(model_dir, config.get("onnx_model", "gesture_net.onnx"))
        self._pth_path = os.path.join(model_dir, config.get("pth_model", "gesture_net.pth"))

        # Thresholds
        self._ml_confidence_threshold = config.get("confidence_threshold", 0.60)
        self._fallback_on_low_confidence = config.get("fallback_on_low_confidence", True)

        # Class mapping
        self._class_map = dict(DEFAULT_CLASS_MAP)

        # Feature extractor
        from models.feature_extractor import GestureFeatureExtractor
        self._feature_extractor = GestureFeatureExtractor()

        # Backend selection
        self._backend = "rules"   # default
        self._trt_engine = None
        self._torch_model = None
        self._device = "cpu"

        self._init_backend()

        # Stats
        self._ml_calls = 0
        self._rule_calls = 0
        self._fallback_count = 0

    def _init_backend(self):
        """Try to load TensorRT engine → PyTorch model → fallback to rules."""

        # --- Try TensorRT first (fastest) ---
        if os.path.isfile(self._trt_path):
            try:
                from models.tensorrt_engine import TensorRTEngine, TRT_AVAILABLE
                if TRT_AVAILABLE:
                    self._trt_engine = TensorRTEngine(self._trt_path)
                    self._backend = "tensorrt"
                    logger.info("ML backend: TensorRT FP16 engine loaded")
                    return
            except Exception as e:
                logger.warning("TensorRT load failed: %s — trying PyTorch", e)

        # --- Try PyTorch ---
        if os.path.isfile(self._pth_path):
            try:
                from models.gesture_net import GestureNet, TORCH_AVAILABLE
                if TORCH_AVAILABLE:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._torch_model = GestureNet.load_checkpoint(
                        self._pth_path, device=device
                    )
                    self._device = device
                    self._backend = "pytorch"
                    logger.info("ML backend: PyTorch on %s", device)
                    return
            except Exception as e:
                logger.warning("PyTorch load failed: %s — falling back to rules", e)

        # --- Rule-based fallback ---
        if self._rule_classifier is not None:
            logger.info("ML backend: rule-based (no trained model found)")
        else:
            logger.warning(
                "No ML model and no rule classifier available. "
                "Train a model with: python -m training.train"
            )

    @property
    def backend(self):
        """Current inference backend name."""
        return self._backend

    @property
    def stats(self):
        """Inference statistics."""
        total = self._ml_calls + self._rule_calls
        return {
            "backend": self._backend,
            "ml_calls": self._ml_calls,
            "rule_calls": self._rule_calls,
            "fallback_count": self._fallback_count,
            "ml_ratio": self._ml_calls / max(total, 1),
        }

    # ------------------------------------------------------------------
    # Main classification API
    # ------------------------------------------------------------------

    def classify(self, landmarks):
        """Classify gesture from 21-point hand landmarks.

        Priority: ML backend → rule-based fallback.

        Args:
            landmarks: np.ndarray shape (21, 3)

        Returns:
            GestureResult
        """
        ml_result = None

        # Attempt ML inference
        if self._backend in ("tensorrt", "pytorch"):
            ml_result = self._classify_ml(landmarks)

            if ml_result is not None:
                self._ml_calls += 1

                # Accept ML result if confidence is high enough
                if ml_result.confidence >= self._ml_confidence_threshold:
                    return ml_result

                # Low-confidence ML — optionally fall back
                if self._fallback_on_low_confidence:
                    self._fallback_count += 1
                    logger.debug(
                        "ML confidence %.3f < %.3f, falling back to rules",
                        ml_result.confidence, self._ml_confidence_threshold,
                    )
                else:
                    return ml_result

        # Rule-based fallback
        if self._rule_classifier is not None:
            self._rule_calls += 1
            return self._rule_classifier.classify(landmarks)

        # No backend available
        return GestureResult(GestureType.NONE, 0.0)

    def _classify_ml(self, landmarks):
        """Run ML inference and return GestureResult or None."""
        try:
            features = self._feature_extractor.extract(landmarks)
        except Exception as e:
            logger.debug("Feature extraction failed: %s", e)
            return None

        try:
            if self._backend == "tensorrt":
                probs = self._trt_engine.predict_proba(features)
            else:
                import torch
                tensor = torch.from_numpy(features).unsqueeze(0).to(self._device)
                probs = self._torch_model.predict_proba(tensor)
                probs = probs.cpu().numpy().squeeze()

            # Decode prediction
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            gesture_type = self._class_map.get(class_idx, GestureType.NONE)

            # Build scores dict
            scores = {}
            for idx, prob in enumerate(probs):
                gt = self._class_map.get(idx)
                if gt is not None and gt != GestureType.NONE:
                    scores[gt] = float(prob)

            return GestureResult(gesture_type, confidence, scores=scores)

        except Exception as e:
            logger.warning("ML inference error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Model hot-swap (for retraining without restart)
    # ------------------------------------------------------------------

    def reload_model(self):
        """Re-initialise backend (e.g. after retraining).

        Call this after running training to pick up the new model
        without restarting the main application.
        """
        logger.info("Reloading ML model...")
        old_backend = self._backend
        self._backend = "rules"
        self._trt_engine = None
        self._torch_model = None
        self._init_backend()
        logger.info("Backend changed: %s → %s", old_backend, self._backend)

    # ------------------------------------------------------------------
    # Forwarding methods — duck-type compatible with GestureClassifier
    # so HybridClassifier is a drop-in replacement in the Pipeline.
    # ------------------------------------------------------------------

    def set_extractor(self, extractor):
        """Forward to rule-based classifier."""
        if self._rule_classifier is not None:
            self._rule_classifier.set_extractor(extractor)

    def set_frame_size(self, width, height):
        """Forward to rule-based classifier."""
        if self._rule_classifier is not None:
            self._rule_classifier.set_frame_size(width, height)

    def apply_adaptation(self, offsets):
        """Forward to rule-based classifier."""
        if self._rule_classifier is not None:
            self._rule_classifier.apply_adaptation(offsets)

    def reset_trajectory(self):
        """Forward to rule-based classifier."""
        if self._rule_classifier is not None:
            self._rule_classifier.reset_trajectory()

    def destroy(self):
        """Release GPU resources."""
        if self._trt_engine is not None:
            self._trt_engine.destroy()
            self._trt_engine = None
        self._torch_model = None
