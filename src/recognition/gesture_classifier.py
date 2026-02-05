"""
Static Gesture Classifier
==========================

Rule-based gesture recognition using hand landmark geometry.
Analyzes finger positions, orientations, and spatial relationships.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto

# Import local types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from detection.hand_detector import HandLandmarks, LandmarkIndex

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Recognized gesture types."""
    NONE = auto()
    CLOSED_FIST = auto()
    OPEN_PALM = auto()
    THUMB_UP = auto()
    THUMB_DOWN = auto()
    VICTORY = auto()
    POINTING_UP = auto()
    I_LOVE_YOU = auto()
    # Dynamic gestures (detected by DynamicGestureDetector)
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    CIRCLE = auto()


@dataclass
class Gesture:
    """Container for recognized gesture with metadata."""
    gesture_type: GestureType
    name: str
    confidence: float
    action: str = ""
    handedness: str = ""
    
    @property
    def is_valid(self) -> bool:
        """Check if gesture is valid (not NONE and high confidence)."""
        return self.gesture_type != GestureType.NONE and self.confidence > 0.5
    
    @staticmethod
    def none() -> "Gesture":
        """Create empty/no gesture result."""
        return Gesture(
            gesture_type=GestureType.NONE,
            name="none",
            confidence=0.0
        )


@dataclass
class GestureClassifierConfig:
    """Gesture classifier configuration."""
    # Finger extension threshold (higher = more extended required)
    finger_threshold: float = 0.6
    # Minimum confidence to report gesture
    confidence_threshold: float = 0.75
    # Enable detailed logging
    debug: bool = False
    
    @classmethod
    def from_dict(cls, config: dict) -> "GestureClassifierConfig":
        """Create config from dictionary."""
        return cls(
            finger_threshold=config.get("finger_threshold", 0.6),
            confidence_threshold=config.get("confidence_threshold", 0.75),
            debug=config.get("debug", False),
        )


class GestureClassifier:
    """
    Rule-based static gesture classifier.
    
    Analyzes hand landmarks to recognize predefined gestures using
    geometric relationships (finger extensions, angles, distances).
    
    Gesture Recognition Logic:
    - Finger extension: Compare fingertip distance from wrist vs MCP distance
    - Thumb orientation: Analyze thumb tip position relative to wrist
    - Hand shape: Combine finger states to identify gestures
    
    Example:
        >>> classifier = GestureClassifier()
        >>> gesture = classifier.classify(hand_landmarks)
        >>> if gesture.is_valid:
        ...     print(f"Detected: {gesture.name} -> {gesture.action}")
    """
    
    # Mapping from gesture type to action
    GESTURE_ACTIONS = {
        GestureType.CLOSED_FIST: "play_pause",
        GestureType.OPEN_PALM: "play_pause",
        GestureType.THUMB_UP: "volume_up",
        GestureType.THUMB_DOWN: "volume_down",
        GestureType.VICTORY: "seek_forward",
        GestureType.POINTING_UP: "seek_backward",
        GestureType.I_LOVE_YOU: "mute",
    }
    
    def __init__(self, config: Optional[GestureClassifierConfig] = None):
        self.config = config or GestureClassifierConfig()
    
    def classify(self, hand: HandLandmarks) -> Gesture:
        """
        Classify hand gesture from landmarks.
        
        Args:
            hand: Hand landmarks to analyze
            
        Returns:
            Recognized gesture with confidence score
        """
        # Get finger states
        fingers = self._get_finger_states(hand)
        
        # Count extended fingers
        extended_count = sum(1 for f, s in fingers.items() if s and f != "thumb")
        thumb_extended = fingers.get("thumb", False)
        
        # Get thumb orientation
        thumb_up, thumb_down = self._get_thumb_orientation(hand)
        
        # Recognize gesture based on finger pattern
        gesture_type = GestureType.NONE
        confidence = 0.0
        
        # Check each gesture pattern
        if self._is_closed_fist(fingers):
            gesture_type = GestureType.CLOSED_FIST
            confidence = self._calculate_fist_confidence(hand, fingers)
            
        elif self._is_open_palm(fingers):
            gesture_type = GestureType.OPEN_PALM
            confidence = self._calculate_palm_confidence(hand, fingers)
            
        elif self._is_thumb_up(fingers, thumb_up):
            gesture_type = GestureType.THUMB_UP
            confidence = 0.9 if thumb_up else 0.7
            
        elif self._is_thumb_down(fingers, thumb_down):
            gesture_type = GestureType.THUMB_DOWN
            confidence = 0.9 if thumb_down else 0.7
            
        elif self._is_victory(fingers):
            gesture_type = GestureType.VICTORY
            confidence = self._calculate_victory_confidence(hand, fingers)
            
        elif self._is_pointing_up(fingers):
            gesture_type = GestureType.POINTING_UP
            confidence = self._calculate_pointing_confidence(hand, fingers)
            
        elif self._is_i_love_you(fingers):
            gesture_type = GestureType.I_LOVE_YOU
            confidence = 0.85
        
        # Apply confidence threshold
        if confidence < self.config.confidence_threshold:
            return Gesture.none()
        
        # Get action mapping
        action = self.GESTURE_ACTIONS.get(gesture_type, "")
        
        return Gesture(
            gesture_type=gesture_type,
            name=gesture_type.name.lower(),
            confidence=confidence,
            action=action,
            handedness=hand.handedness,
        )
    
    def _get_finger_states(self, hand: HandLandmarks) -> Dict[str, bool]:
        """
        Determine if each finger is extended or curled.
        
        Uses distance ratios to be robust to hand size variations.
        """
        fingers = {}
        
        # Wrist as reference point
        wrist = hand.get(LandmarkIndex.WRIST)
        
        # Check each non-thumb finger
        for finger, (tip_idx, mcp_idx, pip_idx) in [
            ("index", (LandmarkIndex.INDEX_TIP, LandmarkIndex.INDEX_MCP, LandmarkIndex.INDEX_PIP)),
            ("middle", (LandmarkIndex.MIDDLE_TIP, LandmarkIndex.MIDDLE_MCP, LandmarkIndex.MIDDLE_PIP)),
            ("ring", (LandmarkIndex.RING_TIP, LandmarkIndex.RING_MCP, LandmarkIndex.RING_PIP)),
            ("pinky", (LandmarkIndex.PINKY_TIP, LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP)),
        ]:
            tip = hand.get(tip_idx)
            mcp = hand.get(mcp_idx)
            pip = hand.get(pip_idx)
            
            # Finger is extended if tip is farther from wrist than PIP
            tip_dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            pip_dist = np.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)
            
            # Also check if tip is above PIP (for pointing up gestures)
            # Use y coordinate (lower y = higher in image)
            tip_above_pip = tip.y < pip.y
            
            # Combined criteria
            extended = tip_dist > pip_dist * self.config.finger_threshold
            fingers[finger] = extended
        
        # Thumb needs special handling (moves differently)
        thumb_tip = hand.get(LandmarkIndex.THUMB_TIP)
        thumb_ip = hand.get(LandmarkIndex.THUMB_IP)
        thumb_mcp = hand.get(LandmarkIndex.THUMB_MCP)
        index_mcp = hand.get(LandmarkIndex.INDEX_MCP)
        
        # Thumb extended if tip is far from palm center
        palm_x, palm_y = hand.palm_center
        thumb_tip_dist = np.sqrt((thumb_tip.x - palm_x)**2 + (thumb_tip.y - palm_y)**2)
        thumb_mcp_dist = np.sqrt((thumb_mcp.x - palm_x)**2 + (thumb_mcp.y - palm_y)**2)
        
        # Also check distance from index MCP
        thumb_to_index = np.sqrt((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2)
        
        fingers["thumb"] = thumb_tip_dist > thumb_mcp_dist * 0.8 or thumb_to_index > 0.1
        
        if self.config.debug:
            logger.debug(f"Finger states: {fingers}")
        
        return fingers
    
    def _get_thumb_orientation(self, hand: HandLandmarks) -> Tuple[bool, bool]:
        """
        Determine if thumb is pointing up or down.
        
        Returns:
            (thumb_up, thumb_down): Tuple of orientation flags
        """
        thumb_tip = hand.get(LandmarkIndex.THUMB_TIP)
        thumb_mcp = hand.get(LandmarkIndex.THUMB_MCP)
        wrist = hand.get(LandmarkIndex.WRIST)
        
        # Check vertical position relative to wrist
        # Lower y = higher in image
        thumb_above_wrist = thumb_tip.y < wrist.y - 0.05
        thumb_below_wrist = thumb_tip.y > wrist.y + 0.05
        
        # Also check that thumb is extended vertically
        vertical_extension = abs(thumb_tip.y - thumb_mcp.y) > 0.08
        
        thumb_up = thumb_above_wrist and vertical_extension
        thumb_down = thumb_below_wrist and vertical_extension
        
        return thumb_up, thumb_down
    
    def _is_closed_fist(self, fingers: Dict[str, bool]) -> bool:
        """Check if gesture is closed fist."""
        return (not fingers["index"] and not fingers["middle"] and 
                not fingers["ring"] and not fingers["pinky"])
    
    def _is_open_palm(self, fingers: Dict[str, bool]) -> bool:
        """Check if gesture is open palm."""
        return (fingers["thumb"] and fingers["index"] and fingers["middle"] and
                fingers["ring"] and fingers["pinky"])
    
    def _is_thumb_up(self, fingers: Dict[str, bool], thumb_up: bool) -> bool:
        """Check if gesture is thumb up."""
        other_fingers_closed = (not fingers["index"] and not fingers["middle"] and
                               not fingers["ring"] and not fingers["pinky"])
        return fingers["thumb"] and other_fingers_closed and thumb_up
    
    def _is_thumb_down(self, fingers: Dict[str, bool], thumb_down: bool) -> bool:
        """Check if gesture is thumb down."""
        other_fingers_closed = (not fingers["index"] and not fingers["middle"] and
                               not fingers["ring"] and not fingers["pinky"])
        return fingers["thumb"] and other_fingers_closed and thumb_down
    
    def _is_victory(self, fingers: Dict[str, bool]) -> bool:
        """Check if gesture is victory/peace sign."""
        return (fingers["index"] and fingers["middle"] and
                not fingers["ring"] and not fingers["pinky"] and not fingers["thumb"])
    
    def _is_pointing_up(self, fingers: Dict[str, bool]) -> bool:
        """Check if gesture is pointing up (index only)."""
        return (fingers["index"] and not fingers["middle"] and
                not fingers["ring"] and not fingers["pinky"] and not fingers["thumb"])
    
    def _is_i_love_you(self, fingers: Dict[str, bool]) -> bool:
        """Check if gesture is I Love You (thumb, index, pinky)."""
        return (fingers["thumb"] and fingers["index"] and not fingers["middle"] and
                not fingers["ring"] and fingers["pinky"])
    
    def _calculate_fist_confidence(
        self, 
        hand: HandLandmarks, 
        fingers: Dict[str, bool]
    ) -> float:
        """Calculate confidence for closed fist detection."""
        # Higher confidence if fingers are very close to palm
        palm_x, palm_y = hand.palm_center
        
        total_dist = 0.0
        for tip_idx in [LandmarkIndex.INDEX_TIP, LandmarkIndex.MIDDLE_TIP,
                       LandmarkIndex.RING_TIP, LandmarkIndex.PINKY_TIP]:
            tip = hand.get(tip_idx)
            dist = np.sqrt((tip.x - palm_x)**2 + (tip.y - palm_y)**2)
            total_dist += dist
        
        avg_dist = total_dist / 4
        
        # Lower distance = higher confidence (inverted, capped)
        confidence = max(0.5, min(0.95, 1.0 - avg_dist * 3))
        return confidence
    
    def _calculate_palm_confidence(
        self, 
        hand: HandLandmarks, 
        fingers: Dict[str, bool]
    ) -> float:
        """Calculate confidence for open palm detection."""
        # Higher confidence if all fingers are clearly extended
        extended_count = sum(1 for s in fingers.values() if s)
        return 0.6 + (extended_count / 5) * 0.35
    
    def _calculate_victory_confidence(
        self,
        hand: HandLandmarks,
        fingers: Dict[str, bool]
    ) -> float:
        """Calculate confidence for victory sign."""
        # Check if index and middle form a V shape
        index_tip = hand.get(LandmarkIndex.INDEX_TIP)
        middle_tip = hand.get(LandmarkIndex.MIDDLE_TIP)
        
        # Distance between fingertips (V shape = some separation)
        separation = np.sqrt((index_tip.x - middle_tip.x)**2 + 
                            (index_tip.y - middle_tip.y)**2)
        
        # Good V shape has moderate separation
        if 0.03 < separation < 0.15:
            return 0.9
        elif separation <= 0.03:
            return 0.7  # Fingers too close
        else:
            return 0.75  # Fingers too far
    
    def _calculate_pointing_confidence(
        self,
        hand: HandLandmarks,
        fingers: Dict[str, bool]
    ) -> float:
        """Calculate confidence for pointing gesture."""
        index_tip = hand.get(LandmarkIndex.INDEX_TIP)
        index_mcp = hand.get(LandmarkIndex.INDEX_MCP)
        
        # Check if index is pointing upward
        vertical_dist = index_mcp.y - index_tip.y  # Positive if pointing up
        
        if vertical_dist > 0.1:
            return 0.9
        elif vertical_dist > 0.05:
            return 0.8
        else:
            return 0.7
