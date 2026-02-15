#!/usr/bin/env python3
"""
Touchless Media Control - Edge AI on Jetson Nano
Main application entry point and pipeline orchestrator.

v2.0 Architecture:
    - core.Pipeline handles detect→recognize→act cycle
    - core.EventBus for decoupled module communication
    - ThermalManager for adaptive FPS control
    - Proper calibration → enhancement wiring
    - No God Object — TouchlessMediaControl delegates to Pipeline

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
from modules.recognition.gesture_classifier import GestureClassifier
from modules.recognition.temporal_filter import TemporalFilter
from modules.recognition.confidence_scorer import ConfidenceScorer
from modules.control.action_executor import ActionExecutor
from modules.control.debouncer import Debouncer
from modules.control.feedback_manager import FeedbackManager
from modules.intelligence.user_profiler import UserProfiler
from modules.intelligence.error_detector import ErrorDetector
from modules.intelligence.analytics import Analytics
from modules.intelligence.thermal_manager import ThermalManager
from modules.visualization.dashboard import Dashboard

from core.events import EventBus, Events
from core.pipeline import Pipeline
from models.hybrid_classifier import HybridClassifier

logger = logging.getLogger(__name__)


class TouchlessMediaControl:
    """Main application orchestrating the gesture-controlled media system.

    Delegates pipeline logic to core.Pipeline and uses EventBus for
    decoupled module communication.
    """

    def __init__(self, config: Config, mode: str = "control"):
        self._config = config
        self._mode = mode
        self._running = False

        # --- Event Bus ---
        self._bus = EventBus()

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

        # ML Hybrid Classifier — wraps rule-based with ML model
        # Uses TensorRT > PyTorch > rule-based (auto-selection)
        ml_config = config.get_section("ml_classifier") or {}
        if ml_config.get("enabled", True):
            self._hybrid = HybridClassifier(
                config=ml_config,
                rule_classifier=self._classifier,
            )
            logger.info("ML Hybrid classifier active (backend: %s)",
                        self._hybrid.backend)
        else:
            self._hybrid = self._classifier
            logger.info("ML classifier disabled — using rule-based only")

        # Control
        self._executor = ActionExecutor(config.media)
        self._debouncer = Debouncer(config.debouncing)
        self._feedback = FeedbackManager(config.get_section("gestures"))

        # Intelligence
        self._profiler = UserProfiler(config.adaptation)
        self._error_detector = ErrorDetector()
        self._analytics = Analytics()
        self._thermal = ThermalManager(config.performance)

        # Visualization
        self._dashboard = Dashboard(config.visualization)

        # Performance
        self._perf = PerformanceMonitor(
            window_size=config.get("performance.metrics_window", 100)
        )
        self._gesture_logger = GestureLogger()

        # --- Build Pipeline ---
        self._pipeline = Pipeline(
            camera=self._camera,
            frame_processor=self._frame_processor,
            detector=self._detector,
            extractor=self._extractor,
            tracker=self._tracker,
            classifier=self._hybrid,  # ML hybrid (auto-fallback to rules)
            temporal_filter=self._temporal_filter,
            confidence_scorer=self._confidence_scorer,
            debouncer=self._debouncer,
            executor=self._executor,
            profiler=self._profiler,
            error_detector=self._error_detector,
            analytics=self._analytics,
            performance_monitor=self._perf,
            thermal_manager=self._thermal,
            event_bus=self._bus,
            config={
                "enable_threading": config.get("performance.enable_threading", True),
                "frame_skip_threshold": config.get("performance.frame_skip_threshold", 50),
                "mode": mode,
            },
        )

        # --- Wire Event Callbacks ---
        self._bus.subscribe(Events.ACTION_EXECUTED, self._on_action_executed)
        self._bus.subscribe(Events.GESTURE_STABLE, self._on_gesture_stable)
        self._bus.subscribe(Events.THERMAL_WARNING, self._on_thermal_warning)

        # Legacy callback for feedback
        self._executor.on_action(lambda action: self._feedback.trigger(action))

        logger.info("TouchlessMediaControl initialized (mode=%s)", mode)

    def _on_action_executed(self, **kwargs):
        """Handle action executed event."""
        action = kwargs.get("action", "")
        gesture = kwargs.get("gesture", "")
        self._analytics.record_action(action)
        self._gesture_logger.log_action(action)
        self._gesture_logger.log_gesture(
            gesture, self._pipeline.current_confidence, action,
            self._perf.total_latency_ms,
        )

    def _on_gesture_stable(self, **kwargs):
        """Handle stable gesture detection event."""
        # Could trigger visual feedback, sound, etc.
        pass

    def _on_thermal_warning(self, **kwargs):
        """Handle thermal warning event."""
        state = kwargs.get("state", "")
        temp = kwargs.get("temperature", 0)
        logger.warning("Thermal warning: %s at %.1f°C", state, temp)

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
            self._frame_processor.set_enhancement(True)

        # Start thermal monitoring
        self._thermal.start_monitoring()
        self._thermal.on_state_change(
            lambda state, temp: self._bus.emit(
                Events.THERMAL_WARNING, state=state.value, temperature=temp
            )
        )

        # User calibration
        if calibrate:
            self._run_calibration()

        # Load user profile if available
        if self._profiler.load_profile("default"):
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
        self._profiler.start_calibration("default")
        calibration_duration = self._config.get("adaptation.calibration_duration_sec", 30)
        calibration_start = time.time()
        window_name = self._config.get("visualization.window_name", "Touchless Media Control")

        logger.info("=== USER CALIBRATION ===")
        logger.info("Perform various gestures naturally for %d seconds", calibration_duration)

        while self._profiler.is_calibrating:
            # Safety timeout
            if time.time() - calibration_start > calibration_duration + 2:
                logger.warning("Calibration safety timeout reached, finishing")
                break

            with self._perf.measure("total"):
                result = self._pipeline.tick()

            if result.frame is not None:
                # Update profiler during calibration
                self._profiler.update(
                    result.gesture_name or "none",
                    result.gesture_confidence,
                )

                state = self._pipeline.build_state()
                state["calibrating"] = True
                state["calibration_progress"] = self._profiler.calibration_progress
                state["mode"] = "calibration"

                frame = self._dashboard.render(result.frame, state)
                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Apply calibration results
        offsets = self._profiler.get_adaptation_offsets()
        if offsets:
            self._classifier.apply_adaptation(offsets)
            logger.info("Calibration offsets applied: %s", offsets)

    def _run_main_loop(self):
        """Main control/demo loop using the Pipeline."""
        window_name = self._config.get("visualization.window_name", "Touchless Media Control")

        while self._running:
            with self._perf.measure("total"):
                result = self._pipeline.tick()

                if result.frame is None:
                    continue

                # Render dashboard
                if self._config.get("visualization.enabled", True):
                    state = self._pipeline.build_state()
                    frame = self._dashboard.render(result.frame, state)
                    frame = self._feedback.render(frame)
                    cv2.imshow(window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._running = False
            elif key == ord("m"):
                new_mode = "demo" if self._mode == "control" else "control"
                self._mode = new_mode
                self._pipeline.set_mode(new_mode)
                logger.info("Mode switched to: %s", new_mode)
            elif key == ord("p"):
                self._perf.print_report()
                self._analytics.print_summary()
            elif key == ord("c"):
                self._run_calibration()

        self._shutdown()

    def _run_benchmark_mode(self):
        """Run performance benchmark."""
        logger.info("=== BENCHMARK MODE ===")
        logger.info("Running 300-frame benchmark...")
        benchmark_frames = 300

        for i in range(benchmark_frames):
            with self._perf.measure("total"):
                result = self._pipeline.tick()

            if i % 50 == 0:
                logger.info("Benchmark progress: %d/%d (FPS: %.1f)",
                            i, benchmark_frames, self._perf.fps)

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
        self._thermal.stop_monitoring()
        self._camera.stop()
        self._detector.close()
        cv2.destroyAllWindows()

        # Print final reports
        self._perf.print_report()
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
        "--profile", type=str, default=None,
        help="User profile to load"
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
    app = TouchlessMediaControl(config, mode=args.mode)

    # Register signal handlers
    signal.signal(signal.SIGINT, app.handle_signal)
    signal.signal(signal.SIGTERM, app.handle_signal)

    # Start
    app.start(calibrate=args.calibrate)


if __name__ == "__main__":
    main()
