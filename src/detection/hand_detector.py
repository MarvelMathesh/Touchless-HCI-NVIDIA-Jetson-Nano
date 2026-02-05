"""
Hand Detection Module - Modern MediaPipe Tasks API
====================================================

Uses the new MediaPipe HandLandmarker (Tasks API) instead of legacy mp.solutions.hands.
Optimized for real-time detection on Jetson Nano.
"""

import cv2
import numpy as np
import logging
import urllib.request
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, NamedTuple
from enum import IntEnum
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

logger = logging.getLogger(__name__)

# Model download URL
HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "hand_landmarker.task"


class LandmarkIndex(IntEnum):
    """Hand landmark indices following MediaPipe convention."""
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


class Landmark(NamedTuple):
    """A single landmark point with normalized coordinates."""
    x: float  # 0.0 to 1.0, normalized by image width
    y: float  # 0.0 to 1.0, normalized by image height
    z: float  # Depth relative to wrist
    
    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


@dataclass
class HandDetectorConfig:
    """Configuration for hand detector."""
    model_path: str = ""
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    running_mode: str = "VIDEO"  # IMAGE, VIDEO, or LIVE_STREAM
    
    @classmethod
    def from_dict(cls, d: dict) -> "HandDetectorConfig":
        """Create config from dictionary."""
        return cls(
            model_path=d.get("model_path", ""),
            max_num_hands=d.get("max_num_hands", 1),
            min_detection_confidence=d.get("min_detection_confidence", 0.5),
            min_tracking_confidence=d.get("min_tracking_confidence", 0.5),
            min_presence_confidence=d.get("min_presence_confidence", 0.5),
            running_mode=d.get("running_mode", "VIDEO"),
        )


@dataclass
class HandLandmarks:
    """Container for detected hand landmarks with utility methods."""
    landmarks: List[Landmark]
    handedness: str  # "Left" or "Right"
    confidence: float
    image_width: int = 1280
    image_height: int = 720
    
    def get(self, index: LandmarkIndex) -> Landmark:
        """Get landmark by index."""
        return self.landmarks[index]
    
    def get_pixel(self, index: LandmarkIndex) -> Tuple[int, int]:
        """Get landmark as pixel coordinates."""
        lm = self.get(index)
        return lm.to_pixel(self.image_width, self.image_height)
    
    @property
    def palm_center(self) -> Tuple[float, float]:
        """Calculate palm center from wrist and finger MCPs."""
        wrist = self.get(LandmarkIndex.WRIST)
        index_mcp = self.get(LandmarkIndex.INDEX_MCP)
        middle_mcp = self.get(LandmarkIndex.MIDDLE_MCP)
        ring_mcp = self.get(LandmarkIndex.RING_MCP)
        pinky_mcp = self.get(LandmarkIndex.PINKY_MCP)
        
        center_x = (wrist.x + index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 5
        center_y = (wrist.y + index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 5
        
        return (center_x, center_y)
    
    @property
    def palm_center_pixel(self) -> Tuple[int, int]:
        """Get palm center in pixel coordinates."""
        x, y = self.palm_center
        return (int(x * self.image_width), int(y * self.image_height))
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box (x, y, width, height) in pixels."""
        xs = [lm.x for lm in self.landmarks]
        ys = [lm.y for lm in self.landmarks]
        
        min_x = int(min(xs) * self.image_width)
        min_y = int(min(ys) * self.image_height)
        max_x = int(max(xs) * self.image_width)
        max_y = int(max(ys) * self.image_height)
        
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(self.image_width, max_x + padding)
        max_y = min(self.image_height, max_y + padding)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def distance(self, idx1: LandmarkIndex, idx2: LandmarkIndex) -> float:
        """Calculate Euclidean distance between two landmarks."""
        lm1 = self.get(idx1)
        lm2 = self.get(idx2)
        return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
    
    def to_numpy(self) -> np.ndarray:
        """Convert landmarks to numpy array of shape (21, 3)."""
        return np.array([[lm.x, lm.y, lm.z] for lm in self.landmarks])


def download_model(url: str, save_path: Path) -> bool:
    """Download the hand landmarker model if not present."""
    if save_path.exists():
        logger.info(f"Model already exists at {save_path}")
        return True
    
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading hand landmarker model to {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        logger.info("Model download complete!")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


class HandDetector:
    """
    Hand detection wrapper using MediaPipe Tasks API (HandLandmarker).
    
    This is the modern API recommended by Google, replacing the legacy
    mp.solutions.hands module.
    
    Example:
        >>> config = HandDetectorConfig()
        >>> detector = HandDetector(config)
        >>> detector.start()
        >>> hands = detector.detect(rgb_image)  # RGB format!
        >>> detector.stop()
    """
    
    # Hand connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (5, 9), (9, 10), (10, 11), (11, 12), # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17),                               # Palm base
    ]
    
    def __init__(self, config: Optional[HandDetectorConfig] = None):
        self.config = config or HandDetectorConfig()
        self._landmarker: Optional[vision.HandLandmarker] = None
        self._frame_timestamp = 0
    
    def start(self) -> bool:
        """Initialize the hand landmarker."""
        try:
            # Determine model path
            model_path = self.config.model_path or str(DEFAULT_MODEL_PATH)
            
            # Download model if needed
            if not Path(model_path).exists():
                if not download_model(HAND_LANDMARKER_MODEL_URL, Path(model_path)):
                    logger.error("Could not download hand landmarker model")
                    return False
            
            # Determine running mode
            if self.config.running_mode == "IMAGE":
                running_mode = vision.RunningMode.IMAGE
            elif self.config.running_mode == "VIDEO":
                running_mode = vision.RunningMode.VIDEO
            else:
                running_mode = vision.RunningMode.VIDEO  # Default to VIDEO for real-time
            
            # Create options
            base_options = python.BaseOptions(model_asset_path=model_path)
            
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_hands=self.config.max_num_hands,
                min_hand_detection_confidence=self.config.min_detection_confidence,
                min_hand_presence_confidence=self.config.min_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            
            # Create the landmarker
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            
            logger.info(f"HandLandmarker initialized with model: {model_path}")
            logger.info(f"Running mode: {self.config.running_mode}, Max hands: {self.config.max_num_hands}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HandLandmarker: {e}")
            return False
    
    def stop(self) -> None:
        """Release resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        logger.info("HandLandmarker stopped")
    
    def detect(self, image: np.ndarray, timestamp_ms: Optional[int] = None) -> List[HandLandmarks]:
        """
        Detect hands in the given image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            timestamp_ms: Timestamp in milliseconds (required for VIDEO mode)
            
        Returns:
            List of HandLandmarks for each detected hand
        """
        if self._landmarker is None:
            logger.warning("HandLandmarker not initialized. Call start() first.")
            return []
        
        height, width = image.shape[:2]
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect based on running mode
        if self.config.running_mode == "IMAGE":
            result = self._landmarker.detect(mp_image)
        else:
            # VIDEO mode requires timestamp
            if timestamp_ms is None:
                self._frame_timestamp += 33  # ~30 FPS
                timestamp_ms = self._frame_timestamp
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Convert to HandLandmarks
        hands = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            # Get handedness
            handedness = "Right"
            if result.handedness and len(result.handedness) > i:
                handedness = result.handedness[i][0].category_name
            
            # Get confidence
            confidence = 0.0
            if result.handedness and len(result.handedness) > i:
                confidence = result.handedness[i][0].score
            
            # Convert landmarks
            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ]
            
            hands.append(HandLandmarks(
                landmarks=landmarks,
                handedness=handedness,
                confidence=confidence,
                image_width=width,
                image_height=height,
            ))
        
        return hands
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        hand: HandLandmarks,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 255, 255),
        landmark_radius: int = 5,
        connection_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw hand landmarks and connections on an image.
        
        Args:
            image: BGR image to draw on
            hand: HandLandmarks to visualize
            
        Returns:
            Image with landmarks drawn
        """
        # Draw connections
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            start = hand.get_pixel(LandmarkIndex(start_idx))
            end = hand.get_pixel(LandmarkIndex(end_idx))
            cv2.line(image, start, end, connection_color, connection_thickness)
        
        # Draw landmarks
        for i, lm in enumerate(hand.landmarks):
            pos = lm.to_pixel(hand.image_width, hand.image_height)
            
            # Different colors for fingertips
            if i in [4, 8, 12, 16, 20]:
                color = (0, 0, 255)  # Red for fingertips
            else:
                color = landmark_color
            
            cv2.circle(image, pos, landmark_radius, color, -1)
        
        return image
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
