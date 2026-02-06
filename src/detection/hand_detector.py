"""
MediaPipe Hand Detection Module
================================

Optimized hand landmark detection for NVIDIA Jetson Nano.
Wraps MediaPipe Hands with configuration and performance tuning.

Python 3.6 compatible version.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, List, Tuple, NamedTuple
from enum import IntEnum

# Python 3.6 compatibility: dataclasses backport
try:
    from dataclasses import dataclass, field
except ImportError:
    # Fallback for Python 3.6 - install python3-dataclasses
    print("WARNING: dataclasses not found. Install with: pip install dataclasses")
    # Minimal replacement
    def dataclass(cls):
        return cls
    def field(**kwargs):
        return None

logger = logging.getLogger(__name__)


class LandmarkIndex(IntEnum):
    """MediaPipe hand landmark indices for reference."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class HandDetectorConfig:
    """MediaPipe Hands configuration."""
    model_complexity: int = 0  # 0=lite (fast), 1=full (accurate)
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    
    @classmethod
    def from_dict(cls, config):
        """Create config from dictionary."""
        return cls(
            model_complexity=config.get("model_complexity", 0),
            max_num_hands=config.get("max_num_hands", 2),
            min_detection_confidence=config.get("min_detection_confidence", 0.7),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.5),
            static_image_mode=config.get("static_image_mode", False),
        )


class Landmark(NamedTuple):
    """Single hand landmark with position and visibility."""
    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    z: float  # Relative depth
    visibility: float = 1.0
    
    def to_pixel(self, width, height):
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


class HandLandmarks:
    """
    Container for detected hand landmarks with helper methods.
    
    Provides easy access to landmarks and geometric calculations
    needed for gesture recognition.
    """
    
    def __init__(self, landmarks, handedness, confidence, image_width, image_height):
        self.landmarks = landmarks
        self.handedness = handedness
        self.confidence = confidence
        self.image_width = image_width
        self.image_height = image_height
        self._landmark_dict = {i: lm for i, lm in enumerate(self.landmarks)}
    
    def get(self, index):
        """Get landmark by index."""
        return self._landmark_dict[int(index)]
    
    def get_pixel(self, index):
        """Get landmark pixel coordinates."""
        lm = self.get(index)
        return lm.to_pixel(self.image_width, self.image_height)
    
    @property
    def wrist(self):
        """Get wrist landmark."""
        return self.get(LandmarkIndex.WRIST)
    
    @property
    def palm_center(self):
        """
        Calculate palm center as average of wrist and MCP joints.
        Returns normalized coordinates.
        """
        wrist = self.get(LandmarkIndex.WRIST)
        index_mcp = self.get(LandmarkIndex.INDEX_MCP)
        middle_mcp = self.get(LandmarkIndex.MIDDLE_MCP)
        ring_mcp = self.get(LandmarkIndex.RING_MCP)
        pinky_mcp = self.get(LandmarkIndex.PINKY_MCP)
        
        x = (wrist.x + index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 5
        y = (wrist.y + index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 5
        
        return (x, y)
    
    @property
    def palm_center_pixel(self):
        """Get palm center in pixel coordinates."""
        x, y = self.palm_center
        return (int(x * self.image_width), int(y * self.image_height))
    
    def finger_tip(self, finger):
        """Get fingertip landmark by name."""
        tips = {
            "thumb": LandmarkIndex.THUMB_TIP,
            "index": LandmarkIndex.INDEX_TIP,
            "middle": LandmarkIndex.MIDDLE_TIP,
            "ring": LandmarkIndex.RING_TIP,
            "pinky": LandmarkIndex.PINKY_TIP,
        }
        return self.get(tips[finger.lower()])
    
    def finger_mcp(self, finger):
        """Get finger MCP (knuckle) landmark by name."""
        mcps = {
            "thumb": LandmarkIndex.THUMB_MCP,
            "index": LandmarkIndex.INDEX_MCP,
            "middle": LandmarkIndex.MIDDLE_MCP,
            "ring": LandmarkIndex.RING_MCP,
            "pinky": LandmarkIndex.PINKY_MCP,
        }
        return self.get(mcps[finger.lower()])
    
    def distance(self, idx1, idx2):
        """Calculate Euclidean distance between two landmarks (normalized)."""
        lm1 = self.get(idx1)
        lm2 = self.get(idx2)
        return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
    
    def distance_2d(self, idx1, idx2):
        """Calculate 2D distance between landmarks (ignoring depth)."""
        lm1 = self.get(idx1)
        lm2 = self.get(idx2)
        return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)
    
    @property
    def bounding_box(self):
        """
        Get bounding box around hand in pixel coordinates.
        Returns (x, y, width, height).
        """
        xs = [lm.x for lm in self.landmarks]
        ys = [lm.y for lm in self.landmarks]
        
        min_x = int(min(xs) * self.image_width)
        max_x = int(max(xs) * self.image_width)
        min_y = int(min(ys) * self.image_height)
        max_y = int(max(ys) * self.image_height)
        
        # Add padding
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(self.image_width, max_x + padding)
        max_y = min(self.image_height, max_y + padding)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def to_numpy(self):
        """Convert landmarks to numpy array of shape (21, 3)."""
        return np.array([[lm.x, lm.y, lm.z] for lm in self.landmarks])


class HandDetector:
    """
    MediaPipe Hands wrapper optimized for Jetson Nano.
    
    Compatible with MediaPipe 0.8.5 and Python 3.6.
    """
    
    def __init__(self, config=None):
        self.config = config or HandDetectorConfig()
        self._hands = None
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
    
    def start(self):
        """Initialize MediaPipe Hands."""
        logger.info("Initializing MediaPipe Hands (complexity={}, max_hands={})".format(
            self.config.model_complexity, self.config.max_num_hands))
        
        # MediaPipe 0.8.5 API
        self._hands = self._mp_hands.Hands(
            static_image_mode=self.config.static_image_mode,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        
        logger.info("MediaPipe Hands initialized")
    
    def stop(self):
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()
            self._hands = None
        logger.info("MediaPipe Hands stopped")
    
    def detect(self, image):
        """
        Detect hands in image.
        
        Args:
            image: RGB image (numpy array)
            
        Returns:
            List of HandLandmarks for each detected hand
        """
        if self._hands is None:
            logger.warning("HandDetector not started")
            return []
        
        height, width = image.shape[:2]
        
        # Process image
        results = self._hands.process(image)
        
        if not results.multi_hand_landmarks:
            return []
        
        hands = []
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness
            handedness = "Unknown"
            confidence = 0.0
            if results.multi_handedness:
                hand_info = results.multi_handedness[idx]
                handedness = hand_info.classification[0].label
                confidence = hand_info.classification[0].score
            
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=getattr(lm, 'visibility', 1.0)
                ))
            
            hands.append(HandLandmarks(
                landmarks=landmarks,
                handedness=handedness,
                confidence=confidence,
                image_width=width,
                image_height=height,
            ))
        
        # Sort by confidence (highest first)
        hands.sort(key=lambda h: h.confidence, reverse=True)
        
        return hands
    
    def draw_landmarks(self, image, hand, color=(0, 255, 0), thickness=2):
        """
        Draw hand landmarks on image.
        
        Args:
            image: BGR image to draw on
            hand: HandLandmarks to draw
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with landmarks drawn
        """
        # Draw connections
        connections = self._mp_hands.HAND_CONNECTIONS
        
        for connection in connections:
            start_idx, end_idx = connection
            start = hand.get_pixel(LandmarkIndex(start_idx))
            end = hand.get_pixel(LandmarkIndex(end_idx))
            cv2.line(image, start, end, (255, 255, 255), thickness)
        
        # Draw landmarks
        for i, lm in enumerate(hand.landmarks):
            x, y = lm.to_pixel(hand.image_width, hand.image_height)
            cv2.circle(image, (x, y), 5, color, -1)
        
        return image
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
