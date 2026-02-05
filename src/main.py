"""
Touchless HCI Media Control System - Main Application
=======================================================

Entry point for the hand gesture media control system.
Orchestrates camera, detection, recognition, and control modules.
"""

import cv2
import yaml
import logging
import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Import local modules
from capture.camera import Camera, CameraConfig
from detection.hand_detector import HandDetector, HandDetectorConfig
from recognition.gesture_classifier import GestureClassifier, GestureClassifierConfig, Gesture
from recognition.dynamic_gestures import DynamicGestureDetector, DynamicGestureConfig
from recognition.gesture_buffer import GestureBuffer, GestureBufferConfig
from control.media_controller import MediaController, MediaControllerConfig
from utils.performance import PerformanceMonitor
from utils.visualization import Visualizer, VisualizerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration container."""
    camera: CameraConfig
    mediapipe: HandDetectorConfig
    recognition: GestureClassifierConfig
    dynamic: DynamicGestureConfig
    buffer: GestureBufferConfig
    media_control: MediaControllerConfig
    visualization: VisualizerConfig
    performance_target_fps: float = 25.0
    performance_target_latency: float = 30.0


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_app_config(config_dict: dict) -> AppConfig:
    """Create AppConfig from configuration dictionary."""
    return AppConfig(
        camera=CameraConfig.from_dict(config_dict.get("camera", {})),
        mediapipe=HandDetectorConfig.from_dict(config_dict.get("mediapipe", {})),
        recognition=GestureClassifierConfig.from_dict(config_dict.get("recognition", {})),
        dynamic=DynamicGestureConfig.from_dict(config_dict.get("recognition", {})),
        buffer=GestureBufferConfig.from_dict(config_dict.get("recognition", {})),
        media_control=MediaControllerConfig.from_dict(config_dict.get("media_control", {})),
        visualization=VisualizerConfig.from_dict(config_dict.get("visualization", {})),
        performance_target_fps=config_dict.get("performance", {}).get("target_fps", 25.0),
        performance_target_latency=config_dict.get("performance", {}).get("target_latency_ms", 30.0),
    )


class HCIApplication:
    """
    Main application class for Touchless HCI Media Control.
    
    Coordinates all components:
    - Camera capture
    - Hand detection (MediaPipe)
    - Gesture recognition (static + dynamic)
    - Media control (VLC)
    - Performance monitoring
    - Visualization
    
    Modes:
    - control: Full media control mode
    - demo: Visualization only (no media control)
    - benchmark: Performance benchmarking
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        # Initialize components
        self.camera = Camera(config.camera)
        self.detector = HandDetector(config.mediapipe)
        self.classifier = GestureClassifier(config.recognition)
        self.dynamic_detector = DynamicGestureDetector(config.dynamic)
        self.buffer = GestureBuffer(config.buffer)
        self.media_controller = MediaController(config.media_control)
        self.visualizer = Visualizer(config.visualization)
        self.performance = PerformanceMonitor()
        
        # Set performance targets
        self.performance.target_fps = config.performance_target_fps
        self.performance.target_latency_ms = config.performance_target_latency
        
        # State
        self._running = False
        self._mode = "control"
        self._last_action_display = ""
        self._action_display_time = 0.0
    
    def start(self) -> bool:
        """Start all components."""
        logger.info("Starting HCI Application...")
        
        if not self.camera.start():
            logger.error("Failed to start camera")
            return False
        
        self.detector.start()
        self.performance.start()
        
        self._running = True
        logger.info("HCI Application started successfully")
        return True
    
    def stop(self) -> None:
        """Stop all components."""
        logger.info("Stopping HCI Application...")
        self._running = False
        
        self.camera.stop()
        self.detector.stop()
        self.performance.stop()
        
        cv2.destroyAllWindows()
        logger.info("HCI Application stopped")
    
    def run(self, mode: str = "control") -> None:
        """
        Run the main application loop.
        
        Args:
            mode: Operating mode ("control", "demo", "benchmark")
        """
        self._mode = mode
        logger.info(f"Running in {mode} mode")
        
        if not self.start():
            return
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            self._main_loop()
        finally:
            self.stop()
            self._print_final_report()
    
    def _main_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            self.performance.frame_start()
            
            # === Capture Stage ===
            with self.performance.measure("capture"):
                frame = self.camera.read()
            
            if frame is None:
                continue
            
            # === Detection Stage ===
            with self.performance.measure("detection"):
                rgb_image = frame.rgb
                hands = self.detector.detect(rgb_image)
            
            # === Recognition Stage ===
            gesture: Optional[Gesture] = None
            should_act = False
            
            with self.performance.measure("recognition"):
                if hands:
                    # Try static recognition first
                    raw_gesture = self.classifier.classify(hands[0])
                    
                    # Try dynamic recognition
                    dynamic_gesture = self.dynamic_detector.update(hands[0])
                    
                    # Prefer dynamic if detected, otherwise use static
                    if dynamic_gesture and dynamic_gesture.is_valid:
                        # Dynamic gestures bypass buffer (immediate action)
                        gesture = dynamic_gesture
                        should_act = True
                    else:
                        # Use buffer for static gestures
                        gesture, should_act = self.buffer.update(raw_gesture)
                else:
                    # No hands - update buffer with empty
                    self.buffer.update(Gesture.none())
                    self.dynamic_detector.reset()
            
            # === Action Stage ===
            if should_act and gesture and self._mode == "control":
                success = self.media_controller.execute(gesture.action)
                if success:
                    self._last_action_display = gesture.action.upper()
                    self._action_display_time = time.time()
                    logger.info(f"Action executed: {gesture.action}")
            
            # === Visualization Stage ===
            display = frame.image.copy()
            
            # Draw hands
            if hands:
                self.visualizer.draw_hands(display, hands)
            
            # Draw gesture
            if gesture and gesture.is_valid:
                self.visualizer.draw_gesture(
                    display,
                    gesture.name,
                    gesture.confidence,
                    gesture.action
                )
            
            # Draw action feedback (fade out after 1 second)
            if self._last_action_display and (time.time() - self._action_display_time) < 1.0:
                self.visualizer.draw_action_feedback(display, self._last_action_display)
            
            # Draw performance
            self.visualizer.draw_performance(
                display,
                fps=self.performance.fps,
                latency_ms=self.performance.total_latency_ms
            )
            
            # Draw mode indicator
            cv2.putText(display, f"Mode: {self._mode.upper()}", 
                       (display.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Display
            cv2.imshow("Touchless HCI Media Control", display)
            
            # Mark frame complete
            self.performance.frame_complete()
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self._running = False
            elif key == ord('m'):
                # Toggle mode
                self._mode = "demo" if self._mode == "control" else "control"
                logger.info(f"Switched to {self._mode} mode")
            elif key == ord('p'):
                # Print performance report
                print(self.performance.get_report())
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False
    
    def _print_final_report(self) -> None:
        """Print final performance report."""
        print("\n" + "=" * 50)
        print("FINAL PERFORMANCE REPORT")
        print("=" * 50)
        print(self.performance.get_report())
        print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Touchless HCI Media Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  control   - Full media control mode (default)
  demo      - Visualization only, no media control
  benchmark - Performance benchmarking mode

Keyboard Controls:
  q/ESC     - Quit
  m         - Toggle control/demo mode
  p         - Print performance report

Examples:
  python main.py --mode control
  python main.py --mode demo
  python main.py --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["control", "demo", "benchmark"],
        default="control",
        help="Operating mode"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        config_dict = load_config(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config_dict = {}
    
    # Create application config
    app_config = create_app_config(config_dict)
    
    # Print banner
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        TOUCHLESS HCI MEDIA CONTROL SYSTEM                    ║
║              NVIDIA Jetson Nano Edition                       ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Create and run application
    app = HCIApplication(app_config)
    app.run(mode=args.mode)


if __name__ == "__main__":
    main()
