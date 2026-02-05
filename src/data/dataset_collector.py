"""
Dataset Collector Module
=========================

Interactive tool for capturing hand gesture training data.
"""

import cv2
import os
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
from pathlib import Path

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from capture.camera import Camera, CameraConfig
from detection.hand_detector import HandDetector, HandDetectorConfig, HandLandmarks

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata for a data collection session."""
    session_id: str
    participant_id: str
    timestamp: str
    lighting: str
    background: str
    notes: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GestureSample:
    """A single gesture sample."""
    gesture_name: str
    image_path: str
    landmarks_path: str
    timestamp: float
    confidence: float = 0.0
    handedness: str = ""


class DatasetCollector:
    """
    Interactive dataset collection tool for gesture training data.
    
    Features:
    - Keyboard-triggered capture (hold key to record)
    - Automatic timestamping and organization
    - Landmark data export alongside images
    - Session metadata tracking
    
    Gesture Key Mappings:
        1: closed_fist
        2: open_palm
        3: thumb_up
        4: thumb_down
        5: victory
        6: pointing_up
        7: i_love_you
        8: swipe_left (record trajectory)
        9: swipe_right (record trajectory)
        0: circle (record trajectory)
    
    Example:
        >>> collector = DatasetCollector("data/gestures")
        >>> collector.start_session("participant_001", "indoor", "office")
        >>> collector.run()  # Interactive capture loop
    """
    
    # Key to gesture mapping
    KEY_GESTURE_MAP = {
        ord('1'): "closed_fist",
        ord('2'): "open_palm",
        ord('3'): "thumb_up",
        ord('4'): "thumb_down",
        ord('5'): "victory",
        ord('6'): "pointing_up",
        ord('7'): "i_love_you",
        ord('8'): "swipe_left",
        ord('9'): "swipe_right",
        ord('0'): "circle",
    }
    
    def __init__(
        self,
        output_dir: str = "data/gestures",
        camera_config: Optional[CameraConfig] = None,
        save_images: bool = True,
        save_landmarks: bool = True,
        image_format: str = "jpg",
        quality: int = 95
    ):
        self.output_dir = Path(output_dir)
        self.camera_config = camera_config or CameraConfig()
        self.save_images = save_images
        self.save_landmarks = save_landmarks
        self.image_format = image_format
        self.quality = quality
        
        self._camera: Optional[Camera] = None
        self._detector: Optional[HandDetector] = None
        self._session: Optional[SessionMetadata] = None
        self._session_dir: Optional[Path] = None
        self._samples: List[GestureSample] = []
        self._recording: bool = False
        self._current_gesture: Optional[str] = None
    
    def start_session(
        self,
        participant_id: str,
        lighting: str = "indoor",
        background: str = "default",
        notes: str = ""
    ) -> Path:
        """
        Start a new data collection session.
        
        Args:
            participant_id: Unique identifier for participant
            lighting: Lighting condition description
            background: Background description
            notes: Additional notes
            
        Returns:
            Path to session directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp}_{participant_id}"
        
        self._session = SessionMetadata(
            session_id=session_id,
            participant_id=participant_id,
            timestamp=timestamp,
            lighting=lighting,
            background=background,
            notes=notes,
        )
        
        # Create session directory structure
        self._session_dir = self.output_dir / session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create gesture subdirectories
        for gesture_name in self.KEY_GESTURE_MAP.values():
            (self._session_dir / gesture_name).mkdir(exist_ok=True)
        
        # Save session metadata
        metadata_path = self._session_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._session.to_dict(), f, indent=2)
        
        self._samples = []
        
        logger.info(f"Started session: {session_id}")
        logger.info(f"Session directory: {self._session_dir}")
        
        return self._session_dir
    
    def run(self) -> None:
        """
        Run interactive data collection loop.
        
        Controls:
            1-0: Hold to record corresponding gesture
            q: Quit collection
            s: Show session statistics
            ESC: Exit
        """
        if self._session is None:
            raise RuntimeError("No session started. Call start_session() first.")
        
        # Initialize camera and detector
        self._camera = Camera(self.camera_config)
        self._detector = HandDetector(HandDetectorConfig())
        
        self._camera.start()
        self._detector.start()
        
        logger.info("Starting data collection. Press keys 1-0 to record gestures, 'q' to quit.")
        self._print_instructions()
        
        try:
            while True:
                # Read frame
                frame = self._camera.read()
                if frame is None:
                    continue
                
                # Detect hands
                rgb_image = frame.rgb
                hands = self._detector.detect(rgb_image)
                
                # Draw visualization
                display = frame.image.copy()
                
                if hands:
                    self._detector.draw_landmarks(display, hands[0])
                
                # Draw status
                self._draw_status(display, hands)
                
                # Handle recording
                if self._recording and self._current_gesture and hands:
                    self._save_sample(frame.image, hands[0], self._current_gesture)
                
                # Display
                cv2.imshow("Dataset Collector", display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('s'):
                    self._print_statistics()
                elif key in self.KEY_GESTURE_MAP:
                    # Start recording
                    if not self._recording:
                        self._current_gesture = self.KEY_GESTURE_MAP[key]
                        self._recording = True
                        logger.info(f"Recording: {self._current_gesture}")
                else:
                    # Stop recording when key released
                    if self._recording:
                        self._recording = False
                        logger.info(f"Stopped recording. Captured {len(self._samples)} total samples.")
        
        finally:
            self._camera.stop()
            self._detector.stop()
            cv2.destroyAllWindows()
            self._save_session_summary()
    
    def _save_sample(
        self,
        image: any,
        hand: HandLandmarks,
        gesture_name: str
    ) -> None:
        """Save a single gesture sample."""
        timestamp = time.time()
        sample_id = f"{gesture_name}_{int(timestamp * 1000)}"
        
        gesture_dir = self._session_dir / gesture_name
        
        # Save image
        image_path = ""
        if self.save_images:
            image_path = gesture_dir / f"{sample_id}.{self.image_format}"
            if self.image_format == "jpg":
                cv2.imwrite(str(image_path), image, 
                           [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:
                cv2.imwrite(str(image_path), image)
        
        # Save landmarks
        landmarks_path = ""
        if self.save_landmarks:
            landmarks_path = gesture_dir / f"{sample_id}.json"
            landmarks_data = {
                "landmarks": [[lm.x, lm.y, lm.z] for lm in hand.landmarks],
                "handedness": hand.handedness,
                "confidence": hand.confidence,
                "timestamp": timestamp,
            }
            with open(landmarks_path, "w") as f:
                json.dump(landmarks_data, f)
        
        # Track sample
        sample = GestureSample(
            gesture_name=gesture_name,
            image_path=str(image_path),
            landmarks_path=str(landmarks_path),
            timestamp=timestamp,
            confidence=hand.confidence,
            handedness=hand.handedness,
        )
        self._samples.append(sample)
    
    def _draw_status(self, image: any, hands: List[HandLandmarks]) -> None:
        """Draw status overlay on image."""
        height, width = image.shape[:2]
        
        # Recording indicator
        if self._recording:
            cv2.circle(image, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(image, f"RECORDING: {self._current_gesture}", 
                       (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Ready - Hold 1-0 to record", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hand detection status
        if hands:
            cv2.putText(image, f"Hand detected: {hands[0].handedness}", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(image, "No hand detected", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Sample count
        cv2.putText(image, f"Samples: {len(self._samples)}", 
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _print_instructions(self) -> None:
        """Print collection instructions."""
        print("\n" + "=" * 50)
        print("GESTURE DATA COLLECTION")
        print("=" * 50)
        print("\nKey mappings:")
        for key, gesture in self.KEY_GESTURE_MAP.items():
            print(f"  {chr(key)}: {gesture}")
        print("\nControls:")
        print("  Hold key: Record gesture continuously")
        print("  Release key: Stop recording")
        print("  s: Show statistics")
        print("  q/ESC: Quit")
        print("=" * 50 + "\n")
    
    def _print_statistics(self) -> None:
        """Print current session statistics."""
        print("\n" + "-" * 40)
        print("SESSION STATISTICS")
        print("-" * 40)
        
        # Count samples per gesture
        counts: Dict[str, int] = {}
        for sample in self._samples:
            counts[sample.gesture_name] = counts.get(sample.gesture_name, 0) + 1
        
        for gesture in sorted(self.KEY_GESTURE_MAP.values()):
            count = counts.get(gesture, 0)
            bar = "█" * min(count // 5, 20)
            print(f"  {gesture:15s}: {count:4d} {bar}")
        
        print(f"\n  Total samples: {len(self._samples)}")
        print("-" * 40 + "\n")
    
    def _save_session_summary(self) -> None:
        """Save session summary at end of collection."""
        if self._session_dir is None:
            return
        
        summary = {
            "session": self._session.to_dict() if self._session else {},
            "total_samples": len(self._samples),
            "samples_per_gesture": {},
        }
        
        for sample in self._samples:
            g = sample.gesture_name
            summary["samples_per_gesture"][g] = summary["samples_per_gesture"].get(g, 0) + 1
        
        summary_path = self._session_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Session summary saved to {summary_path}")
        self._print_statistics()


def main():
    """Command-line entry point for dataset collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect gesture training data")
    parser.add_argument("--output", "-o", default="data/gestures", help="Output directory")
    parser.add_argument("--participant", "-p", required=True, help="Participant ID")
    parser.add_argument("--lighting", "-l", default="indoor", help="Lighting condition")
    parser.add_argument("--background", "-b", default="default", help="Background description")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    collector = DatasetCollector(output_dir=args.output)
    collector.start_session(
        participant_id=args.participant,
        lighting=args.lighting,
        background=args.background,
    )
    collector.run()


if __name__ == "__main__":
    main()
