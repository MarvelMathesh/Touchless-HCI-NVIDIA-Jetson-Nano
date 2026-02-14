"""
Core pipeline orchestrator for the gesture recognition system.
Encapsulates the detect -> recognize -> act pipeline with queue-based
stage decoupling, frame skipping, and thermal-adaptive FPS control.

Architecture:
    Camera -> FrameProcessor -> HandDetector -> LandmarkExtractor
    -> GestureClassifier -> TemporalFilter -> ConfidenceScorer
    -> Debouncer -> ActionExecutor

This module replaces the monolithic _process_frame_pipeline() in main.py
with a composable, testable pipeline.
"""

import time
import logging
import numpy as np

from core.types import GestureType, GestureResult, GESTURE_ACTION_MAP
from core.events import EventBus, Events

logger = logging.getLogger(__name__)


class PipelineResult:
    """Result of a single pipeline iteration."""

    __slots__ = (
        "frame", "rgb_frame", "hand_detected", "hand_count",
        "gesture_name", "gesture_confidence", "gesture_scores",
        "action_name", "action_executed", "latency_ms",
        "frame_id", "timestamp",
    )

    def __init__(self):
        self.frame = None
        self.rgb_frame = None
        self.hand_detected = False
        self.hand_count = 0
        self.gesture_name = None
        self.gesture_confidence = 0.0
        self.gesture_scores = {}
        self.action_name = None
        self.action_executed = False
        self.latency_ms = 0.0
        self.frame_id = 0
        self.timestamp = 0.0


class Pipeline:
    """Composable gesture recognition pipeline.

    Manages the full detect-recognize-act cycle with:
    - Frame skip under load
    - Thermal-adaptive FPS
    - Event bus integration
    - Per-stage timing
    """

    def __init__(
        self,
        camera,
        frame_processor,
        detector,
        extractor,
        tracker,
        classifier,
        temporal_filter,
        confidence_scorer,
        debouncer,
        executor,
        profiler,
        error_detector,
        analytics,
        performance_monitor,
        thermal_manager=None,
        event_bus=None,
        config=None,
    ):
        # Store all modules
        self._camera = camera
        self._frame_processor = frame_processor
        self._detector = detector
        self._extractor = extractor
        self._tracker = tracker
        self._classifier = classifier
        self._temporal_filter = temporal_filter
        self._confidence_scorer = confidence_scorer
        self._debouncer = debouncer
        self._executor = executor
        self._profiler = profiler
        self._error_detector = error_detector
        self._analytics = analytics
        self._perf = performance_monitor
        self._thermal = thermal_manager
        self._bus = event_bus or EventBus()

        # Config
        config = config or {}
        self._use_threading = config.get("enable_threading", True)
        self._frame_skip_threshold_ms = config.get("frame_skip_threshold", 50)
        self._anomaly_check_interval = config.get("anomaly_check_interval", 30)

        # State
        self._frame_count = 0
        self._skip_count = 0
        self._current_gesture = None
        self._current_confidence = 0.0
        self._mode = config.get("mode", "control")

        # Share extractor with classifier
        self._classifier.set_extractor(self._extractor)

    def set_mode(self, mode: str):
        """Set pipeline mode: 'control', 'demo', 'benchmark'."""
        self._mode = mode
        logger.info("Pipeline mode set to: %s", mode)

    def tick(self) -> PipelineResult:
        """Execute one full pipeline iteration.

        Returns:
            PipelineResult with all detection/recognition/action data
        """
        result = PipelineResult()
        result.timestamp = time.time()

        # --- 1. Frame Capture ---
        with self._perf.measure("capture"):
            if self._use_threading:
                frame_id, frame = self._camera.read()
            else:
                frame_id, frame = self._camera.read_sync()

        if frame is None:
            return result

        result.frame_id = frame_id
        self._frame_count += 1

        # --- 2. Frame Skip Check ---
        # If previous frame took too long, consider skipping
        if self._perf.total_latency_ms > self._frame_skip_threshold_ms:
            # Skip every other frame when overloaded
            if self._frame_count % 2 == 0:
                self._skip_count += 1
                result.frame = frame
                return result

        # --- 3. Preprocess ---
        with self._perf.measure("preprocess"):
            rgb_frame = self._frame_processor.preprocess(frame)

        result.rgb_frame = rgb_frame

        # --- 4. Hand Detection ---
        with self._perf.measure("detection"):
            detection_results = self._detector.detect(rgb_frame)

        # --- 5. Tracking & Landmark Extraction ---
        tracked_hands = self._tracker.update(detection_results, self._extractor)
        primary_hand = self._tracker.get_primary_hand()
        result.hand_detected = primary_hand is not None
        result.hand_count = self._tracker.hand_count

        self._analytics.record_frame(result.hand_detected)

        if result.hand_detected:
            self._bus.emit(Events.HAND_DETECTED, hand=primary_hand)

        # --- 6. Gesture Classification ---
        with self._perf.measure("classification"):
            if primary_hand is not None:
                raw_result = self._classifier.classify(primary_hand.landmarks)

                # Temporal filtering with hysteresis
                filtered = self._temporal_filter.update(raw_result)

                if filtered["gesture"] and filtered["stable"]:
                    gesture_name = filtered["gesture"].value
                    confidence = filtered["confidence"]

                    self._current_gesture = gesture_name
                    self._current_confidence = confidence
                    result.gesture_name = gesture_name
                    result.gesture_confidence = confidence
                    result.gesture_scores = raw_result.scores if raw_result else {}

                    # Record across modules
                    self._analytics.record_gesture(gesture_name, confidence)
                    self._confidence_scorer.record(gesture_name, confidence)
                    self._error_detector.record(gesture_name, confidence)
                    self._profiler.update(gesture_name, confidence)

                    self._bus.emit(Events.GESTURE_STABLE,
                                   gesture=gesture_name, confidence=confidence)
                else:
                    self._current_gesture = None
                    self._current_confidence = filtered.get("confidence", 0)
                    result.gesture_confidence = self._current_confidence
                    self._error_detector.record("none", 0)

                    # End any held actions when gesture is lost
                    self._debouncer.end_all_holds()
            else:
                self._current_gesture = None
                self._current_confidence = 0.0
                self._temporal_filter.reset()
                self._classifier.reset_trajectory()
                self._error_detector.record("none", 0)
                self._debouncer.end_all_holds()

        # --- 7. Action Execution (control mode only) ---
        if self._mode == "control" and result.gesture_name:
            with self._perf.measure("action"):
                action_result = self._try_execute(result.gesture_name, result.gesture_confidence)
                result.action_name = action_result.get("action")
                result.action_executed = action_result.get("executed", False)

        # --- 8. Draw landmarks ---
        self._detector.draw_landmarks(frame, detection_results)
        result.frame = frame

        # --- 9. Periodic anomaly check ---
        if self._frame_count % self._anomaly_check_interval == 0:
            anomalies = self._error_detector.check_anomalies()
            if anomalies:
                logger.warning("Anomalies: %s", anomalies)
                self._bus.emit(Events.ANOMALY_DETECTED, anomalies=anomalies)

        # --- 10. Timing ---
        self._perf.tick()
        result.latency_ms = self._perf.total_latency_ms

        return result

    def _try_execute(self, gesture_name: str, confidence: float) -> dict:
        """Attempt to execute the action for a detected gesture.

        Returns:
            {"action": action_name, "executed": bool}
        """
        # Map gesture to action
        try:
            gesture_type = GestureType(gesture_name)
            action = GESTURE_ACTION_MAP.get(gesture_type)
        except ValueError:
            return {"action": None, "executed": False}

        if action is None:
            return {"action": None, "executed": False}

        # Confidence check
        if not self._confidence_scorer.should_execute(gesture_name, confidence):
            return {"action": action, "executed": False}

        # Ambiguity check (use scores if available)
        # Already handled in classifier, but double-check with scorer
        # if self._confidence_scorer.check_ambiguity(scores):
        #     return {"action": action, "executed": False}

        # Debouncing
        if not self._debouncer.can_execute(action):
            return {"action": action, "executed": False}

        # Handle holdable actions
        if self._debouncer.is_holdable(action):
            self._debouncer.start_hold(action)

        # Execute
        success = self._executor.execute(action)

        if success:
            self._debouncer.record(action)
            self._bus.emit(Events.ACTION_EXECUTED, action=action, gesture=gesture_name)
            return {"action": action, "executed": True}

        return {"action": action, "executed": False}

    def build_state(self) -> dict:
        """Build state dict for dashboard rendering."""
        primary = self._tracker.get_primary_hand()
        bbox = None
        if primary is not None:
            bbox = self._extractor.get_bounding_box(primary.landmarks)

        state = {
            "fps": self._perf.fps,
            "latency_ms": self._perf.total_latency_ms,
            "gesture_name": self._current_gesture,
            "gesture_confidence": self._current_confidence,
            "hand_detected": primary is not None,
            "hand_bbox": bbox,
            "hand_count": self._tracker.hand_count,
            "mode": self._mode,
            "calibrating": self._profiler.is_calibrating,
            "calibration_progress": self._profiler.calibration_progress,
        }

        # Add thermal info if available
        if self._thermal:
            state["temperature"] = self._thermal.temperature
            state["thermal_state"] = self._thermal.state.value

        return state

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def skip_count(self) -> int:
        return self._skip_count

    @property
    def current_gesture(self) -> str:
        return self._current_gesture

    @property
    def current_confidence(self) -> float:
        return self._current_confidence
