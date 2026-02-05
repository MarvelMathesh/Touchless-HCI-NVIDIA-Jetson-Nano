"""
Tests for Gesture Recognition Module
=====================================
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection.hand_detector import HandLandmarks, Landmark, LandmarkIndex
from recognition.gesture_classifier import GestureClassifier, GestureClassifierConfig, GestureType


def create_mock_landmarks(
    finger_states: dict,
    image_width: int = 1280,
    image_height: int = 720
) -> HandLandmarks:
    """
    Create mock hand landmarks for testing.
    
    Args:
        finger_states: Dict of finger -> "up" or "down"
        
    Returns:
        Mock HandLandmarks object
    """
    # Base positions for a hand at center of frame
    base_x, base_y = 0.5, 0.6  # Wrist position
    
    landmarks = []
    
    # Create 21 landmarks
    # Wrist
    landmarks.append(Landmark(x=base_x, y=base_y, z=0.0))
    
    # Thumb (indices 1-4)
    thumb_up = finger_states.get("thumb", "down") == "up"
    thumb_offset = -0.15 if thumb_up else -0.05
    landmarks.append(Landmark(x=base_x + thumb_offset * 0.3, y=base_y - 0.02, z=0.0))  # CMC
    landmarks.append(Landmark(x=base_x + thumb_offset * 0.6, y=base_y - 0.04, z=0.0))  # MCP
    landmarks.append(Landmark(x=base_x + thumb_offset * 0.8, y=base_y - 0.06, z=0.0))  # IP
    landmarks.append(Landmark(x=base_x + thumb_offset, y=base_y - 0.08, z=0.0))        # TIP
    
    # Index finger (indices 5-8)
    index_up = finger_states.get("index", "down") == "up"
    for i, y_off in enumerate([0.08, 0.14, 0.20, 0.28 if index_up else 0.10]):
        landmarks.append(Landmark(x=base_x - 0.05, y=base_y - y_off, z=0.0))
    
    # Middle finger (indices 9-12)
    middle_up = finger_states.get("middle", "down") == "up"
    for i, y_off in enumerate([0.09, 0.16, 0.23, 0.32 if middle_up else 0.11]):
        landmarks.append(Landmark(x=base_x, y=base_y - y_off, z=0.0))
    
    # Ring finger (indices 13-16)
    ring_up = finger_states.get("ring", "down") == "up"
    for i, y_off in enumerate([0.08, 0.14, 0.20, 0.28 if ring_up else 0.10]):
        landmarks.append(Landmark(x=base_x + 0.05, y=base_y - y_off, z=0.0))
    
    # Pinky finger (indices 17-20)
    pinky_up = finger_states.get("pinky", "down") == "up"
    for i, y_off in enumerate([0.06, 0.11, 0.16, 0.22 if pinky_up else 0.08]):
        landmarks.append(Landmark(x=base_x + 0.1, y=base_y - y_off, z=0.0))
    
    return HandLandmarks(
        landmarks=landmarks,
        handedness="Right",
        confidence=0.95,
        image_width=image_width,
        image_height=image_height
    )


class TestGestureClassifier:
    """Test suite for static gesture classifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier with default config."""
        return GestureClassifier(GestureClassifierConfig(
            finger_threshold=0.6,
            confidence_threshold=0.5,  # Lower threshold for testing
            debug=True
        ))
    
    def test_closed_fist(self, classifier):
        """Test closed fist detection."""
        hand = create_mock_landmarks({
            "thumb": "down",
            "index": "down",
            "middle": "down",
            "ring": "down",
            "pinky": "down"
        })
        
        gesture = classifier.classify(hand)
        assert gesture.gesture_type == GestureType.CLOSED_FIST
        assert gesture.action == "play_pause"
        assert gesture.confidence > 0.5
    
    def test_open_palm(self, classifier):
        """Test open palm detection."""
        hand = create_mock_landmarks({
            "thumb": "up",
            "index": "up",
            "middle": "up",
            "ring": "up",
            "pinky": "up"
        })
        
        gesture = classifier.classify(hand)
        assert gesture.gesture_type == GestureType.OPEN_PALM
        assert gesture.action == "play_pause"
    
    def test_victory_sign(self, classifier):
        """Test victory/peace sign detection."""
        hand = create_mock_landmarks({
            "thumb": "down",
            "index": "up",
            "middle": "up",
            "ring": "down",
            "pinky": "down"
        })
        
        gesture = classifier.classify(hand)
        assert gesture.gesture_type == GestureType.VICTORY
        assert gesture.action == "seek_forward"
    
    def test_pointing_up(self, classifier):
        """Test pointing up gesture detection."""
        hand = create_mock_landmarks({
            "thumb": "down",
            "index": "up",
            "middle": "down",
            "ring": "down",
            "pinky": "down"
        })
        
        gesture = classifier.classify(hand)
        assert gesture.gesture_type == GestureType.POINTING_UP
        assert gesture.action == "seek_backward"
    
    def test_i_love_you(self, classifier):
        """Test I Love You sign detection."""
        hand = create_mock_landmarks({
            "thumb": "up",
            "index": "up",
            "middle": "down",
            "ring": "down",
            "pinky": "up"
        })
        
        gesture = classifier.classify(hand)
        assert gesture.gesture_type == GestureType.I_LOVE_YOU
        assert gesture.action == "mute"
    
    def test_gesture_has_correct_handedness(self, classifier):
        """Test that gesture preserves handedness info."""
        hand = create_mock_landmarks({"index": "up", "middle": "up"})
        hand.handedness = "Left"
        
        gesture = classifier.classify(hand)
        assert gesture.handedness == "Left"


class TestHandLandmarks:
    """Test suite for HandLandmarks helper methods."""
    
    @pytest.fixture
    def hand(self):
        """Create sample hand landmarks."""
        return create_mock_landmarks({
            "thumb": "up",
            "index": "up",
            "middle": "down",
            "ring": "down",
            "pinky": "down"
        })
    
    def test_palm_center(self, hand):
        """Test palm center calculation."""
        palm_x, palm_y = hand.palm_center
        
        # Palm center should be roughly in middle of hand
        assert 0.3 < palm_x < 0.7
        assert 0.4 < palm_y < 0.8
    
    def test_bounding_box(self, hand):
        """Test bounding box calculation."""
        x, y, w, h = hand.bounding_box
        
        # Bounding box should have positive dimensions
        assert w > 0
        assert h > 0
        assert x >= 0
        assert y >= 0
    
    def test_to_numpy(self, hand):
        """Test conversion to numpy array."""
        arr = hand.to_numpy()
        
        assert arr.shape == (21, 3)
        assert isinstance(arr, np.ndarray)
    
    def test_get_landmark(self, hand):
        """Test landmark access by index."""
        wrist = hand.get(LandmarkIndex.WRIST)
        
        assert isinstance(wrist, Landmark)
        assert hasattr(wrist, 'x')
        assert hasattr(wrist, 'y')
        assert hasattr(wrist, 'z')


class TestLandmark:
    """Test suite for Landmark class."""
    
    def test_to_pixel(self):
        """Test conversion to pixel coordinates."""
        lm = Landmark(x=0.5, y=0.5, z=0.0)
        
        px, py = lm.to_pixel(1280, 720)
        
        assert px == 640
        assert py == 360
    
    def test_to_pixel_edge_cases(self):
        """Test pixel conversion at edges."""
        # Top-left corner
        lm1 = Landmark(x=0.0, y=0.0, z=0.0)
        assert lm1.to_pixel(100, 100) == (0, 0)
        
        # Bottom-right corner
        lm2 = Landmark(x=1.0, y=1.0, z=0.0)
        assert lm2.to_pixel(100, 100) == (100, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
