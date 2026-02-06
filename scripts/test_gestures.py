#!/usr/bin/env python3
"""
Quick gesture test - shows what gestures are being detected in real-time.
Useful for debugging and tuning recognition parameters.
"""

import cv2
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capture.camera import Camera, CameraConfig
from detection.hand_detector import HandDetector, HandDetectorConfig
from recognition.gesture_classifier import GestureClassifier, GestureClassifierConfig

def main():
    print("Gesture Recognition Test")
    print("=" * 50)
    print("Shows detected gestures in real-time")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    camera = Camera(CameraConfig.from_dict(config.get("camera", {})))
    detector = HandDetector(HandDetectorConfig.from_dict(config.get("mediapipe", {})))
    classifier = GestureClassifier(GestureClassifierConfig.from_dict(config.get("recognition", {})))
    
    camera.start()
    detector.start()
    
    print("\nCamera started. Showing detections...\n")
    
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            
            # Detect hands
            hands = detector.detect(frame.rgb)
            
            # Draw on frame
            display = frame.image.copy()
            
            if hands:
                hand = hands[0]
                
                # Draw landmarks
                detector.draw_landmarks(display, hand)
                
                # Classify gesture
                gesture = classifier.classify(hand)
                
                # Draw gesture info
                text = "{} ({:.0f}%)".format(
                    gesture.name.replace("_", " ").title(),
                    gesture.confidence * 100
                )
                
                # Color based on confidence
                if gesture.confidence > 0.75:
                    color = (0, 255, 0)  # Green - high confidence
                elif gesture.confidence > 0.5:
                    color = (0, 255, 255)  # Yellow - medium
                else:
                    color = (0, 0, 255)  # Red - low
                
                cv2.putText(display, text, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                # Show finger states
                fingers = classifier._get_finger_states(hand)
                finger_text = "Fingers: "
                for name, extended in fingers.items():
                    if extended:
                        finger_text += "{} ".format(name[0].upper())
                
                cv2.putText(display, finger_text, (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Print to console
                if gesture.is_valid:
                    print("Detected: {} (conf={:.2f}, action={})".format(
                        gesture.name, gesture.confidence, gesture.action))
            else:
                cv2.putText(display, "No hand detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            cv2.imshow("Gesture Test", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.stop()
        detector.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
