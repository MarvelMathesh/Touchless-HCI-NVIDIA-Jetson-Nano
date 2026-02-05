"""
Gesture Buffer
===============

Multi-frame consensus voting for robust gesture recognition.
Reduces false positives and stabilizes gesture detection.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque
from collections import deque, Counter

# Import local types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from .gesture_classifier import Gesture, GestureType

logger = logging.getLogger(__name__)


@dataclass
class GestureBufferConfig:
    """Gesture buffer configuration."""
    buffer_size: int = 5            # Number of frames for consensus
    min_consensus: int = 3          # Minimum votes required
    action_cooldown: float = 0.8    # Seconds between actions
    gesture_hold_time: float = 0.3  # Seconds gesture must be held
    
    @classmethod
    def from_dict(cls, config: dict) -> "GestureBufferConfig":
        """Create config from dictionary."""
        return cls(
            buffer_size=config.get("buffer_size", 5),
            min_consensus=config.get("min_consensus", 3),
            action_cooldown=config.get("action_cooldown", 0.8),
            gesture_hold_time=config.get("gesture_hold_time", 0.3),
        )


@dataclass
class BufferedGesture:
    """Gesture with timestamp for buffer tracking."""
    gesture: Gesture
    timestamp: float
    
    @property
    def gesture_type(self) -> GestureType:
        return self.gesture.gesture_type


class GestureBuffer:
    """
    Multi-frame gesture buffer for robust recognition.
    
    Implements consensus voting across multiple frames to:
    - Reduce false positives from momentary misdetections
    - Stabilize gesture recognition output
    - Provide debouncing for action execution
    
    Example:
        >>> buffer = GestureBuffer(GestureBufferConfig())
        >>> 
        >>> while running:
        ...     raw_gesture = classifier.classify(hand)
        ...     
        ...     # Buffer returns gesture only when consensus reached
        ...     confirmed_gesture, should_act = buffer.update(raw_gesture)
        ...     
        ...     if should_act:
        ...         execute_action(confirmed_gesture.action)
    """
    
    def __init__(self, config: Optional[GestureBufferConfig] = None):
        self.config = config or GestureBufferConfig()
        
        self._buffer: Deque[BufferedGesture] = deque(maxlen=self.config.buffer_size)
        self._last_confirmed: Optional[Gesture] = None
        self._last_action_time: float = 0.0
        self._gesture_start_time: Dict[GestureType, float] = {}
        self._consecutive_none: int = 0
    
    def update(self, gesture: Gesture) -> tuple[Optional[Gesture], bool]:
        """
        Add gesture to buffer and check for consensus.
        
        Args:
            gesture: Raw gesture from classifier
            
        Returns:
            Tuple of (confirmed_gesture, should_execute_action)
            - confirmed_gesture: The recognized gesture if consensus reached
            - should_execute_action: True if action should be executed (not in cooldown)
        """
        current_time = time.time()
        
        # Add to buffer
        self._buffer.append(BufferedGesture(
            gesture=gesture,
            timestamp=current_time
        ))
        
        # Track consecutive NONE for clearing state
        if gesture.gesture_type == GestureType.NONE:
            self._consecutive_none += 1
            if self._consecutive_none > self.config.buffer_size:
                self._gesture_start_time.clear()
        else:
            self._consecutive_none = 0
        
        # Check consensus
        confirmed = self._check_consensus()
        
        if confirmed is None:
            return None, False
        
        # Check if gesture has been held long enough
        if confirmed.gesture_type not in self._gesture_start_time:
            self._gesture_start_time[confirmed.gesture_type] = current_time
        
        hold_duration = current_time - self._gesture_start_time[confirmed.gesture_type]
        
        if hold_duration < self.config.gesture_hold_time:
            return confirmed, False
        
        # Check cooldown
        in_cooldown = (current_time - self._last_action_time) < self.config.action_cooldown
        
        should_act = False
        if not in_cooldown:
            # Only act if this is a new gesture or gesture changed
            if (self._last_confirmed is None or 
                self._last_confirmed.gesture_type != confirmed.gesture_type):
                should_act = True
                self._last_action_time = current_time
        
        self._last_confirmed = confirmed
        
        return confirmed, should_act
    
    def _check_consensus(self) -> Optional[Gesture]:
        """
        Check if buffer has consensus on a gesture.
        
        Returns the gesture with majority votes if threshold met.
        """
        if len(self._buffer) < self.config.min_consensus:
            return None
        
        # Count gesture types (excluding NONE)
        type_counts: Counter = Counter()
        gesture_by_type: Dict[GestureType, Gesture] = {}
        
        for bg in self._buffer:
            if bg.gesture.gesture_type != GestureType.NONE:
                type_counts[bg.gesture.gesture_type] += 1
                # Keep highest confidence instance
                if (bg.gesture.gesture_type not in gesture_by_type or
                    bg.gesture.confidence > gesture_by_type[bg.gesture.gesture_type].confidence):
                    gesture_by_type[bg.gesture.gesture_type] = bg.gesture
        
        if not type_counts:
            return None
        
        # Get most common gesture type
        most_common_type, count = type_counts.most_common(1)[0]
        
        # Check if meets threshold
        if count >= self.config.min_consensus:
            return gesture_by_type[most_common_type]
        
        return None
    
    def force_action(self) -> None:
        """Force reset cooldown to allow immediate action."""
        self._last_action_time = 0.0
    
    def reset(self) -> None:
        """Clear buffer and reset state."""
        self._buffer.clear()
        self._last_confirmed = None
        self._last_action_time = 0.0
        self._gesture_start_time.clear()
        self._consecutive_none = 0
    
    @property
    def current_gesture(self) -> Optional[Gesture]:
        """Get current confirmed gesture without triggering action."""
        return self._check_consensus()
    
    @property
    def buffer_contents(self) -> List[str]:
        """Get buffer contents as gesture names for debugging."""
        return [bg.gesture.name for bg in self._buffer]
    
    @property
    def time_since_last_action(self) -> float:
        """Get seconds since last action was executed."""
        return time.time() - self._last_action_time
    
    @property
    def cooldown_remaining(self) -> float:
        """Get seconds remaining in cooldown (0 if not in cooldown)."""
        remaining = self.config.action_cooldown - self.time_since_last_action
        return max(0.0, remaining)
