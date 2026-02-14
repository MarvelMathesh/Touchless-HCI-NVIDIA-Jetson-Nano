#!/usr/bin/env python3
"""
Touchless Media Control - Edge AI on Jetson Nano
Main application entry point and pipeline orchestrator.

Usage:
    python main.py                    # Default control mode
    python main.py --mode demo        # Demo mode (no VLC control)
    python main.py --mode benchmark   # Performance benchmark
    python main.py --mode collect     # Dataset collection mode
    python main.py --calibrate        # Run user calibration first
"""

import sys
import os
import time
import signal
import argparse
import logging

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules.utils.config import Config
from modules.utils.logger import setup_logging, GestureLogger
from modules.utils.performance_monitor import PerformanceMonitor
from modules.capture.camera_manager import CameraManager
from modules.capture.frame_processor import FrameProcessor
from modules.capture.calibration import Calibrator
from modules.detection.hand_detector import HandDetector
from modules.detection.landmark_extractor import LandmarkExtractor
from modules.detection.tracking import HandTracker
from modules.recognition.gesture_classifier import GestureClassifier, GestureType
from modules.recognition.temporal_filter import TemporalFilter
from modules.recognition.confidence_scorer import ConfidenceScorer
from modules.control.action_executor import ActionExecutor
from modules.control.debouncer import Debouncer
from modules.control.feedback_manager import FeedbackManager
from modules.intelligence.user_profiler import UserProfiler
from modules.intelligence.error_detector import ErrorDetector
from modules.intelligence.analytics import Analytics
from modules.visualization.dashboard import Dashboard

logger = logging.getLogger(__name__)


class TouchlessMediaControl:
    """Main application orchestrating the gesture-controlled media system."""

    def __init__(self, config: Config, mode: str = "control", profile: str = "full"):
        self._config = config
        self._mode = mode
        self._running = False
        self._profile = profile

        # --- Initialize all modules ---

        # Capture
        self._camera = CameraManager(config.camera)
        self._frame_processor = FrameProcessor(
            use_gpu=config.get("performance.enable_gpu", True)
        )
        self._calibrator = Calibrator()

        # Detection
        self._detector = HandDetector(config.mediapipe)
        self._extractor = LandmarkExtractor()
        self._tracker = HandTracker(
            max_hands=config.get("mediapipe.max_num_hands", 2)
        )

        # Recognition
        self._classifier = GestureClassifier(config.recognition)
        self._temporal_filter = TemporalFilter(config.recognition)
        self._confidence_scorer = ConfidenceScorer(config.recognition)

        # Control
        self._executor = ActionExecutor(config.media)
        self._debouncer = Debouncer(config.debouncing)
        self._feedback = FeedbackManager(config.get_section("gestures"))

        # Intelligence
        # Intelligence (conditional based on profile)
        if self._profile == "full":
            self._profiler = UserProfiler(config.adaptation)
            self._error_detector = ErrorDetector()
            self._analytics = Analytics()
        else:
            self._profiler = None
            self._error_detector = None
            self._analytics = None


        # Visualization
        self._dashboard = Dashboard(config.visualization)

        # Performance
        self._perf = PerformanceMonitor(
            window_size=config.get("performance.metrics_window", 100)
        )
        self._gesture_logger = GestureLogger()

        # State
        self._current_gesture = None
        self._current_confidence = 0.0
        self._last_action = None
        self._frame_count = 0

        # Register action feedback callback
        self._executor.on_action(self._on_action_executed)

        logger.info("TouchlessMediaControl initialized (mode=%s)", mode)

    def _on_action_executed(self, action_name: str):
        """Callback when a media action is executed."""
        self._feedback.trigger(action_name)
        if self._analytics:
            self._analytics.record_action(action_name)
        self._gesture_logger.log_action(action_name)

    def start(self, calibrate: bool = False):
        """Start the main application loop."""
        # Open camera
        if not self._camera.open():
            logger.error("Failed to open camera. Check connection and permissions.")
            return False

        # Start async capture
        if self._config.get("performance.enable_threading", True):
            self._camera.start_async()
            time.sleep(0.5)  # Let async capture stabilize

        # Initialize detector
        self._detector.initialize()

        # Set frame size for landmark extractor
        w, h = self._camera.resolution
        self._extractor.set_frame_size(w, h)
        self._classifier.set_frame_size(w, h)

        # Camera calibration
        logger.info("Running camera calibration...")
        cal_result = self._calibrator.auto_calibrate(self._camera)
        if cal_result.get("needs_enhancement"):
            logger.info("Low light detected - enabling image enhancement")

        # User calibration
        if calibrate:
            self._run_calibration()

        # Load user profile if available
        if self._profiler and self._profiler.load_profile("default"):
            offsets = self._profiler.get_adaptation_offsets()
            if offsets:
                self._classifier.apply_adaptation(offsets)
                logger.info("Loaded user adaptation profile")

        # Route to appropriate mode
        self._running = True
        logger.info("Starting main loop (mode=%s)", self._mode)

        if self._mode == "collect":
            self._run_collection_mode()
        elif self._mode == "benchmark":
            self._run_benchmark_mode()
        else:
            self._run_main_loop()

        return True

    def _run_calibration(self):
        """Run the 30-second user calibration sequence."""
	if not self._profiler:
            logger.warning("Calibration not available in this profile.")
            return
        self._profiler.start_calibration("default")
        calibration_duration = self._config.get("adaptation.calibration_duration_sec", 30)
        calibration_start = time.time()
        logger.info("=== USER CALIBRATION ===")
        logger.info("Perform various gestures naturally for %d seconds", calibration_duration)

        while self._profiler.is_calibrating:
            # Safety timeout: force-end calibration if profiler hasn't ended it
            if time.time() - calibration_start > calibration_duration + 2:
                logger.warning("Calibration safety timeout reached, finishing")
                break

            frame = self._process_frame_pipeline()
            if frame is not None:
                # Always update profiler during calibration (even without stable gesture)
                self._profiler.update(
                    self._current_gesture or "none",
                    self._current_confidence,
                )

                state = {
                    "calibrating": True,
                    "calibration_progress": self._profiler.calibration_progress,
                    "fps": self._perf.fps,
                    "latency_ms": self._perf.total_latency_ms,
                    "hand_detected": self._current_gesture is not None,
                    "gesture_name": self._current_gesture,
                    "gesture_confidence": self._current_confidence,
                    "mode": "calibration",
                    "hand_count": self._tracker.hand_count,
                }
                frame = self._dashboard.render(frame, state)
                cv2.imshow(self._config.get("visualization.window_name", "Touchless Media Control"), frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Apply calibration results
        offsets = self._profiler.get_adaptation_offsets()
        if offsets:
            self._classifier.apply_adaptation(offsets)
            logger.info("Calibration offsets applied: %s", offsets)

    def _run_main_loop(self):
        """Main control/demo loop."""
        window_name = self._config.get("visualization.window_name", "Touchless Media Control")

        while self._running:
            with self._perf.measure("total"):
                frame = self._process_frame_pipeline()

                if frame is None:
                    continue

                # Execute action (control mode only)
                if self._mode == "control" and self._current_gesture:
                    self._try_execute_action()

                # Render dashboard
                if self._config.get("visualization.enabled", True):
                    state = self._build_state()
                    frame = self._dashboard.render(frame, state)
                    frame = self._feedback.render(frame)
                    cv2.imshow(window_name, frame)

                self._perf.tick()

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._running = False
            elif key == ord("m"):
                self._mode = "demo" if self._mode == "control" else "control"
                logger.info("Mode switched to: %s", self._mode)
            elif key == ord("p"):
                self._perf.print_report()
                self._analytics.print_summary()
            elif key == ord("c"):
                self._run_calibration()

        self._shutdown()

    def _process_frame_pipeline(self) -> np.ndarray:
        """Run the full detection+recognition pipeline on one frame.

        Returns:
            Processed BGR frame, or None if no frame available
        """
        # 1. Capture
        with self._perf.measure("capture"):
            if self._config.get("performance.enable_threading", True):
                frame_id, frame = self._camera.read()
            else:
                frame_id, frame = self._camera.read_sync()

        if frame is None:
            return None

        self._frame_count += 1

        # 2. Preprocess
        with self._perf.measure("preprocess"):
            rgb_frame = self._frame_processor.preprocess(frame)

        # 3. Hand detection
        with self._perf.measure("detection"):
            results = self._detector.detect(rgb_frame)

        # 4. Landmark extraction & tracking
        tracked_hands = self._tracker.update(results, self._extractor)
        primary_hand = self._tracker.get_primary_hand()
        hand_detected = primary_hand is not None

        if self._analytics:
            self._analytics.record_frame(hand_detected)


        # 5. Gesture classification
        with self._perf.measure("classification"):
            if primary_hand is not None:
                raw_result = self._classifier.classify(primary_hand.landmarks)

                # Temporal filtering
                filtered = self._temporal_filter.update(raw_result)

                if filtered["gesture"] and filtered["stable"]:
                    self._current_gesture = filtered["gesture"].value
                    self._current_confidence = filtered["confidence"]

                    # Record for analytics and adaptation
                    # Record for analytics and adaptation
                    if self._analytics:
                        self._analytics.record_gesture(self._current_gesture, self._current_confidence)
                    self._confidence_scorer.record(self._current_gesture, self._current_confidence)

                    if self._error_detector:
                        self._error_detector.record(self._current_gesture, self._current_confidence)

                    if self._profiler:
                        self._profiler.update(self._current_gesture, self._current_confidence)

                else:
                    self._current_gesture = None
                    self._current_confidence = filtered.get("confidence", 0)
                    if self._error_detector:
                        self._error_detector.record("none", 0)
            else:
                self._current_gesture = None
                self._current_confidence = 0.0
                self._temporal_filter.reset()
                self._classifier.reset_trajectory()
                
                if self._error_detector:
                    self._error_detector.record("none", 0)

        # Draw landmarks on frame
        self._detector.draw_landmarks(frame, results)

        # Check for anomalies periodically
        if self._error_detector and self._frame_count % 30 == 0:
            anomalies = self._error_detector.check_anomalies()
            if anomalies:
                logger.warning("Anomalies detected: %s", anomalies)

        return frame

    def _try_execute_action(self):
        """Attempt to execute the action for the current gesture."""
        if not self._current_gesture:
            return

        action = self._get_action_for_gesture(self._current_gesture)
        if action is None:
            return

        # Check confidence threshold
        if not self._confidence_scorer.should_execute(self._current_gesture, self._current_confidence):
            return

        # Check debouncing
        if not self._debouncer.can_execute(action):
            return

        # Execute
        with self._perf.measure("action"):
            success = self._executor.execute(action)

        if success:
            self._debouncer.record(action)
            self._last_action = action
            self._gesture_logger.log_gesture(
                self._current_gesture, self._current_confidence, action,
                self._perf.total_latency_ms,
            )

    @staticmethod
    def _get_action_for_gesture(gesture_name: str) -> str:
        """Map gesture name to media action."""
        from modules.recognition.gesture_classifier import GESTURE_ACTION_MAP, GestureType
        try:
            gesture_type = GestureType(gesture_name)
            return GESTURE_ACTION_MAP.get(gesture_type)
        except ValueError:
            return None

    def _build_state(self) -> dict:
        """Build state dict for dashboard rendering."""
        primary = self._tracker.get_primary_hand()
        bbox = None
        if primary is not None:
            bbox = self._extractor.get_bounding_box(primary.landmarks)

        return {
            "fps": self._perf.fps,
            "latency_ms": self._perf.total_latency_ms,
            "gesture_name": self._current_gesture,
            "gesture_confidence": self._current_confidence,
            "hand_detected": primary is not None,
            "hand_bbox": bbox,
            "hand_count": self._tracker.hand_count,
            "mode": self._mode,
            "calibrating": self._profiler.is_calibrating if self._profiler else False,
            "calibration_progress": self._profiler.calibration_progress if self._profiler else 0,

        }

    def _run_benchmark_mode(self):
        """Run performance benchmark (no VLC, measure pipeline speed)."""
        logger.info("=== BENCHMARK MODE ===")
        logger.info("Running 300-frame benchmark...")
        benchmark_frames = 300

        for i in range(benchmark_frames):
            with self._perf.measure("total"):
                frame = self._process_frame_pipeline()
                if frame is not None:
                    self._perf.tick()

            if i % 50 == 0:
                logger.info("Benchmark progress: %d/%d (FPS: %.1f)", i, benchmark_frames, self._perf.fps)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        self._perf.print_report()
        self._shutdown()

    def _run_collection_mode(self):
        """Run dataset collection mode."""
        from data.collector.dataset_collector import DatasetCollector
        collector = DatasetCollector(
            output_dir=self._config.get("data_collection.output_dir", "data/gestures"),
            config=self._config.get_section("data_collection"),
        )
        collector.run_interactive(
            self._camera, self._detector, self._extractor, self._frame_processor,
        )
        self._shutdown()

    def _shutdown(self):
        """Clean shutdown of all modules."""
        logger.info("Shutting down...")
        self._running = False
        self._camera.stop()
        self._detector.close()
        cv2.destroyAllWindows()

        # Print final reports
        self._perf.print_report()
        if self._analytics:
            self._analytics.print_summary()

        logger.info("Shutdown complete.")

    def handle_signal(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Signal %d received, shutting down...", signum)
        self._running = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Touchless Media Control - Edge AI on Jetson Nano"
    )

    parser.add_argument(
        "--mode", choices=["control", "demo", "benchmark", "collect"],
        default="control", help="Operating mode"
    )

    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run user calibration before starting"
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml"
    )

    parser.add_argument(
        "--gestures", type=str, default=None,
        help="Path to gestures.yaml"
    )

    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device ID"
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "desktop"],
        help="Execution profile (full, minimal, desktop)"
    )

    return parser.parse_args()



def main():
    args = parse_args()

    # Load configuration
    config = Config()
    config.load(config_path=args.config, gestures_path=args.gestures)

    # Override camera if specified
    if args.camera is not None:
        config._data.setdefault("camera", {})["device_id"] = args.camera

    # Setup logging
    log_cfg = config.get_section("logging")
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
        max_size_mb=log_cfg.get("max_size_mb", 10),
        backup_count=log_cfg.get("backup_count", 3),
    )

    logger.info("=" * 60)
    logger.info("  TOUCHLESS MEDIA CONTROL - Edge AI on Jetson Nano")
    logger.info("  Version: %s", config.get("system.version", "2.0.0"))
    logger.info("  Mode: %s", args.mode)
    logger.info("=" * 60)

    # Create and start application
    app = TouchlessMediaControl(
        config,
        mode=args.mode,
        profile=args.profile
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, app.handle_signal)
    signal.signal(signal.SIGTERM, app.handle_signal)

    # Start
    app.start(calibrate=args.calibrate)


if __name__ == "__main__":
    main()
