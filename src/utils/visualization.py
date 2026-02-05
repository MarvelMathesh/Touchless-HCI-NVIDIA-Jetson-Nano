"""
Visualization Module
=====================

Debug overlays and visual feedback for the HCI system.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

# Import local types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from detection.hand_detector import HandLandmarks, LandmarkIndex


@dataclass
class VisualizerConfig:
    """Visualization settings."""
    show_landmarks: bool = True
    show_connections: bool = True
    show_gesture_label: bool = True
    show_confidence: bool = True
    show_fps: bool = True
    show_latency: bool = True
    show_bounding_box: bool = False
    
    # Colors (BGR format)
    landmark_color: Tuple[int, int, int] = (0, 255, 0)      # Green
    connection_color: Tuple[int, int, int] = (255, 255, 255) # White
    text_color: Tuple[int, int, int] = (0, 255, 255)        # Yellow
    bbox_color: Tuple[int, int, int] = (255, 0, 0)          # Blue
    gesture_color: Tuple[int, int, int] = (0, 255, 0)       # Green
    warning_color: Tuple[int, int, int] = (0, 0, 255)       # Red
    
    # Font settings
    font_scale: float = 0.7
    font_thickness: int = 2
    
    @classmethod
    def from_dict(cls, config: dict) -> "VisualizerConfig":
        """Create config from dictionary."""
        colors = config.get("colors", {})
        return cls(
            show_landmarks=config.get("show_landmarks", True),
            show_connections=config.get("show_connections", True),
            show_gesture_label=config.get("show_gesture_label", True),
            show_confidence=config.get("show_confidence", True),
            show_fps=config.get("show_fps", True),
            show_latency=config.get("show_latency", True),
            show_bounding_box=config.get("show_bounding_box", False),
            landmark_color=tuple(colors.get("landmarks", [0, 255, 0])),
            connection_color=tuple(colors.get("connections", [255, 255, 255])),
            text_color=tuple(colors.get("text", [0, 255, 255])),
            bbox_color=tuple(colors.get("bounding_box", [255, 0, 0])),
            font_scale=config.get("font_scale", 0.7),
            font_thickness=config.get("font_thickness", 2),
        )


class Visualizer:
    """
    Visualization overlay for debug and feedback display.
    
    Draws hand landmarks, gesture labels, FPS counter, and other
    debug information on camera frames.
    
    Example:
        >>> viz = Visualizer(VisualizerConfig())
        >>> 
        >>> while running:
        ...     frame = camera.read()
        ...     hands = detector.detect(frame.rgb)
        ...     gesture = classifier.classify(hands[0])
        ...     
        ...     # Draw overlays
        ...     viz.draw_hands(frame.image, hands)
        ...     viz.draw_gesture(frame.image, gesture)
        ...     viz.draw_performance(frame.image, fps=30, latency=25)
        ...     
        ...     cv2.imshow("HCI", frame.image)
    """
    
    # Hand connection pairs for drawing skeleton
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17),           # Palm
    ]
    
    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()
        self._font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_hands(
        self, 
        image: np.ndarray, 
        hands: List[HandLandmarks]
    ) -> np.ndarray:
        """
        Draw all detected hands on image.
        
        Args:
            image: BGR image to draw on
            hands: List of detected hands
            
        Returns:
            Image with hands drawn
        """
        for hand in hands:
            self.draw_hand(image, hand)
        return image
    
    def draw_hand(
        self, 
        image: np.ndarray, 
        hand: HandLandmarks
    ) -> np.ndarray:
        """
        Draw single hand landmarks and connections.
        
        Args:
            image: BGR image to draw on
            hand: Hand landmarks to draw
            
        Returns:
            Image with hand drawn
        """
        height, width = image.shape[:2]
        
        # Draw bounding box if enabled
        if self.config.show_bounding_box:
            x, y, w, h = hand.bounding_box
            cv2.rectangle(image, (x, y), (x + w, y + h), 
                         self.config.bbox_color, 2)
        
        # Draw connections
        if self.config.show_connections:
            for start_idx, end_idx in self.HAND_CONNECTIONS:
                start = hand.get_pixel(LandmarkIndex(start_idx))
                end = hand.get_pixel(LandmarkIndex(end_idx))
                cv2.line(image, start, end, self.config.connection_color, 2)
        
        # Draw landmarks
        if self.config.show_landmarks:
            for i, lm in enumerate(hand.landmarks):
                x, y = lm.to_pixel(width, height)
                
                # Use different colors for fingertips
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(image, (x, y), 8, (0, 0, 255), -1)  # Red
                else:
                    cv2.circle(image, (x, y), 5, self.config.landmark_color, -1)
        
        # Draw handedness label
        if self.config.show_confidence:
            palm_x, palm_y = hand.palm_center_pixel
            label = f"{hand.handedness} ({hand.confidence:.2f})"
            cv2.putText(image, label, (palm_x - 50, palm_y - 30),
                       self._font, 0.5, self.config.text_color, 1)
        
        return image
    
    def draw_gesture(
        self, 
        image: np.ndarray,
        gesture_name: str,
        confidence: float = 1.0,
        action: str = "",
        position: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Draw recognized gesture label.
        
        Args:
            image: BGR image to draw on
            gesture_name: Name of recognized gesture
            confidence: Gesture confidence score
            action: Associated action (e.g., "Play/Pause")
            position: Custom position (default: bottom-left)
            
        Returns:
            Image with gesture label drawn
        """
        if not self.config.show_gesture_label:
            return image
        
        height, width = image.shape[:2]
        
        if position is None:
            position = (20, height - 60)
        
        x, y = position
        
        # Draw gesture name
        color = self.config.gesture_color if confidence > 0.8 else self.config.warning_color
        
        text = f"Gesture: {gesture_name}"
        if self.config.show_confidence:
            text += f" ({confidence:.0%})"
        
        cv2.putText(image, text, (x, y),
                   self._font, self.config.font_scale, 
                   color, self.config.font_thickness)
        
        # Draw action if specified
        if action:
            cv2.putText(image, f"Action: {action}", (x, y + 30),
                       self._font, self.config.font_scale,
                       self.config.text_color, self.config.font_thickness)
        
        return image
    
    def draw_performance(
        self,
        image: np.ndarray,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        extra_info: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Draw performance metrics overlay.
        
        Args:
            image: BGR image to draw on
            fps: Frames per second
            latency_ms: End-to-end latency in milliseconds
            extra_info: Additional key-value pairs to display
            
        Returns:
            Image with performance overlay drawn
        """
        x, y = 20, 30
        line_height = 25
        
        # FPS indicator
        if self.config.show_fps:
            fps_color = self.config.gesture_color if fps >= 25 else self.config.warning_color
            cv2.putText(image, f"FPS: {fps:.1f}", (x, y),
                       self._font, self.config.font_scale,
                       fps_color, self.config.font_thickness)
            y += line_height
        
        # Latency indicator
        if self.config.show_latency:
            lat_color = self.config.gesture_color if latency_ms <= 30 else self.config.warning_color
            cv2.putText(image, f"Latency: {latency_ms:.1f}ms", (x, y),
                       self._font, self.config.font_scale,
                       lat_color, self.config.font_thickness)
            y += line_height
        
        # Extra info
        if extra_info:
            for key, value in extra_info.items():
                cv2.putText(image, f"{key}: {value}", (x, y),
                           self._font, 0.5, self.config.text_color, 1)
                y += 20
        
        return image
    
    def draw_action_feedback(
        self,
        image: np.ndarray,
        action: str,
        duration_ms: float = 500
    ) -> np.ndarray:
        """
        Draw action confirmation feedback (large centered text).
        
        Args:
            image: BGR image to draw on
            action: Action name to display
            duration_ms: Not used directly (for future animation)
            
        Returns:
            Image with action feedback drawn
        """
        height, width = image.shape[:2]
        
        # Large centered text
        font_scale = 1.5
        thickness = 3
        
        text_size = cv2.getTextSize(action, self._font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        y = (height + text_size[1]) // 2
        
        # Draw shadow
        cv2.putText(image, action, (x + 2, y + 2),
                   self._font, font_scale, (0, 0, 0), thickness + 2)
        
        # Draw text
        cv2.putText(image, action, (x, y),
                   self._font, font_scale, (0, 255, 0), thickness)
        
        return image
    
    def draw_instructions(
        self,
        image: np.ndarray,
        instructions: List[str],
        position: str = "bottom-right"
    ) -> np.ndarray:
        """
        Draw instruction text overlay.
        
        Args:
            image: BGR image to draw on
            instructions: List of instruction lines
            position: Corner position ("top-left", "bottom-right", etc.)
            
        Returns:
            Image with instructions drawn
        """
        height, width = image.shape[:2]
        line_height = 20
        margin = 10
        
        if position == "bottom-right":
            x = width - 200
            y = height - len(instructions) * line_height - margin
        else:  # top-left default
            x = margin
            y = 100
        
        for i, line in enumerate(instructions):
            cv2.putText(image, line, (x, y + i * line_height),
                       self._font, 0.5, self.config.text_color, 1)
        
        return image
