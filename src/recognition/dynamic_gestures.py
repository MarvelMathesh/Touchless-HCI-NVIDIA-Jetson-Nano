"""
Dynamic Gesture Detector
=========================

Temporal gesture recognition for motion-based gestures.
Tracks hand trajectory over time to detect swipes and circles.
"""

import time
import math
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque
from enum import Enum, auto

# Import local types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from detection.hand_detector import HandLandmarks
from .gesture_classifier import GestureType, Gesture

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Single point in hand trajectory."""
    x: float  # Normalized x
    y: float  # Normalized y
    timestamp: float
    
    def distance_to(self, other: "TrajectoryPoint") -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class DynamicGestureConfig:
    """Dynamic gesture detection configuration."""
    # Swipe detection
    swipe_min_distance: float = 0.15      # Minimum swipe distance (normalized)
    swipe_max_time: float = 0.5           # Maximum swipe duration (seconds)
    swipe_min_velocity: float = 0.3       # Minimum swipe velocity (units/sec)
    swipe_direction_threshold: float = 0.7 # Ratio for horizontal vs vertical
    
    # Circle detection
    circle_min_points: int = 15           # Minimum points for circle
    circle_min_radius: float = 0.05       # Minimum circle radius (normalized)
    circle_max_duration: float = 2.0      # Maximum time to complete circle
    circularity_threshold: float = 0.7    # How circular the path must be
    
    # Trajectory settings
    max_trajectory_points: int = 50       # Maximum stored points
    trajectory_timeout: float = 1.0       # Clear trajectory after inactivity
    
    @classmethod
    def from_dict(cls, config: dict) -> "DynamicGestureConfig":
        """Create config from dictionary."""
        return cls(
            swipe_min_distance=config.get("swipe_min_distance", 0.15) / 1000,  # Convert from pixels
            swipe_max_time=config.get("swipe_max_time", 0.5),
            swipe_min_velocity=config.get("swipe_min_velocity", 0.3),
            circle_min_points=config.get("circle_min_points", 15),
            circle_min_radius=config.get("circle_min_radius", 0.05),
            circle_max_duration=config.get("circle_max_duration", 2.0),
            circularity_threshold=config.get("circularity_threshold", 0.7),
        )


class DynamicGestureDetector:
    """
    Motion-based gesture detector using hand trajectory analysis.
    
    Tracks palm center movement over time to detect:
    - Swipe Left/Right: Quick horizontal hand movement
    - Swipe Up/Down: Quick vertical hand movement
    - Circle: Circular hand motion
    
    Example:
        >>> detector = DynamicGestureDetector()
        >>> 
        >>> while running:
        ...     hands = hand_detector.detect(frame)
        ...     if hands:
        ...         gesture = detector.update(hands[0])
        ...         if gesture and gesture.is_valid:
        ...             print(f"Dynamic gesture: {gesture.name}")
    """
    
    def __init__(self, config: Optional[DynamicGestureConfig] = None):
        self.config = config or DynamicGestureConfig()
        self._trajectory = deque(maxlen=self.config.max_trajectory_points)  # type: deque
        self._last_update: float = 0.0
        self._last_gesture_time: float = 0.0
        self._gesture_cooldown: float = 0.5  # Prevent rapid re-detection
    
    def update(self, hand: HandLandmarks) -> Optional[Gesture]:
        """
        Update trajectory with new hand position and check for gestures.
        
        Args:
            hand: Current hand landmarks
            
        Returns:
            Detected dynamic gesture or None
        """
        current_time = time.time()
        
        # Clear trajectory on timeout
        if current_time - self._last_update > self.config.trajectory_timeout:
            self._trajectory.clear()
        
        self._last_update = current_time
        
        # Get palm center position
        palm_x, palm_y = hand.palm_center
        
        # Add point to trajectory
        self._trajectory.append(TrajectoryPoint(
            x=palm_x,
            y=palm_y,
            timestamp=current_time
        ))
        
        # Check cooldown
        if current_time - self._last_gesture_time < self._gesture_cooldown:
            return None
        
        # Try to detect gestures
        gesture = self._detect_swipe()
        if gesture:
            self._on_gesture_detected()
            return gesture
        
        gesture = self._detect_circle()
        if gesture:
            self._on_gesture_detected()
            return gesture
        
        return None
    
    def _on_gesture_detected(self) -> None:
        """Called when a gesture is detected."""
        self._last_gesture_time = time.time()
        self._trajectory.clear()
    
    def _detect_swipe(self) -> Optional[Gesture]:
        """
        Detect swipe gestures from trajectory.
        
        Analyzes recent trajectory for quick directional movement.
        """
        if len(self._trajectory) < 5:
            return None
        
        # Get recent points
        points = list(self._trajectory)
        recent_points = points[-15:]  # Last 15 points
        
        if len(recent_points) < 5:
            return None
        
        # Calculate movement from first to last point
        first = recent_points[0]
        last = recent_points[-1]
        
        dx = last.x - first.x
        dy = last.y - first.y
        distance = math.sqrt(dx**2 + dy**2)
        duration = last.timestamp - first.timestamp
        
        # Check if movement is fast and far enough
        if duration <= 0 or duration > self.config.swipe_max_time:
            return None
        
        if distance < self.config.swipe_min_distance:
            return None
        
        velocity = distance / duration
        if velocity < self.config.swipe_min_velocity:
            return None
        
        # Determine direction (horizontal vs vertical)
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        if abs_dx > abs_dy * self.config.swipe_direction_threshold:
            # Horizontal swipe
            if dx > 0:
                return Gesture(
                    gesture_type=GestureType.SWIPE_RIGHT,
                    name="swipe_right",
                    confidence=min(0.95, 0.7 + velocity * 0.3),
                    action="seek_forward"
                )
            else:
                return Gesture(
                    gesture_type=GestureType.SWIPE_LEFT,
                    name="swipe_left",
                    confidence=min(0.95, 0.7 + velocity * 0.3),
                    action="seek_backward"
                )
        
        return None
    
    def _detect_circle(self) -> Optional[Gesture]:
        """
        Detect circular motion gesture.
        
        Analyzes trajectory for circular pattern using variance from centroid.
        """
        if len(self._trajectory) < self.config.circle_min_points:
            return None
        
        points = list(self._trajectory)
        
        # Check duration
        duration = points[-1].timestamp - points[0].timestamp
        if duration > self.config.circle_max_duration:
            return None
        
        # Calculate centroid
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        
        # Calculate distances from centroid
        distances = [
            math.sqrt((p.x - centroid_x)**2 + (p.y - centroid_y)**2)
            for p in points
        ]
        
        avg_radius = sum(distances) / len(distances)
        
        # Check minimum radius
        if avg_radius < self.config.circle_min_radius:
            return None
        
        # Calculate circularity (variance of distances from centroid)
        variance = sum((d - avg_radius)**2 for d in distances) / len(distances)
        std_dev = math.sqrt(variance)
        
        # Low standard deviation = more circular
        circularity = 1.0 - min(1.0, std_dev / avg_radius)
        
        if circularity >= self.config.circularity_threshold:
            # Check if path returns close to start
            start_end_dist = points[0].distance_to(points[-1])
            closed_path = start_end_dist < avg_radius * 0.5
            
            if closed_path or circularity > 0.85:
                return Gesture(
                    gesture_type=GestureType.CIRCLE,
                    name="circle",
                    confidence=circularity,
                    action="fullscreen"
                )
        
        return None
    
    def reset(self) -> None:
        """Clear trajectory and reset detector state."""
        self._trajectory.clear()
        self._last_update = 0.0
    
    @property
    def trajectory_points(self) -> List[Tuple[float, float]]:
        """Get current trajectory as list of (x, y) tuples for visualization."""
        return [(p.x, p.y) for p in self._trajectory]
