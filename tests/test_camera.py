"""
Tests for Camera Module
========================
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capture.camera import Camera, CameraConfig, Frame


class TestCameraConfig:
    """Test suite for CameraConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = CameraConfig()
        
        assert config.device_id == 0
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30
        assert config.buffer_size == 1
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "device_id": 1,
            "width": 640,
            "height": 480,
            "fps": 60,
        }
        
        config = CameraConfig.from_dict(config_dict)
        
        assert config.device_id == 1
        assert config.width == 640
        assert config.height == 480
        assert config.fps == 60
    
    def test_from_dict_partial(self):
        """Test creating config from partial dictionary."""
        config_dict = {"device_id": 2}
        
        config = CameraConfig.from_dict(config_dict)
        
        assert config.device_id == 2
        assert config.width == 1280  # Default


class TestFrame:
    """Test suite for Frame class."""
    
    def test_frame_creation(self):
        """Test frame creation."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(image=image, timestamp=1234.5, frame_number=42)
        
        assert frame.frame_number == 42
        assert frame.timestamp == 1234.5
        assert frame.image.shape == (480, 640, 3)
    
    def test_rgb_conversion(self):
        """Test BGR to RGB conversion."""
        # Create BGR image with blue pixel
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]  # Blue in BGR
        
        frame = Frame(image=image, timestamp=0, frame_number=0)
        rgb = frame.rgb
        
        # Should be red in RGB
        assert rgb[0, 0, 0] == 0    # R
        assert rgb[0, 0, 1] == 0    # G
        assert rgb[0, 0, 2] == 255  # B


class TestCamera:
    """Test suite for Camera class."""
    
    @pytest.fixture
    def mock_cv2(self):
        """Mock OpenCV VideoCapture."""
        with patch('capture.camera.cv2') as mock:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
            mock_cap.get.return_value = 30.0
            mock.VideoCapture.return_value = mock_cap
            yield mock
    
    def test_camera_init(self):
        """Test camera initialization."""
        config = CameraConfig(device_id=0)
        camera = Camera(config)
        
        assert camera.config.device_id == 0
        assert not camera.is_running
    
    def test_start_success(self, mock_cv2):
        """Test successful camera start."""
        camera = Camera(CameraConfig(warmup_frames=0, threaded=False))
        
        result = camera.start()
        
        assert result is True
        assert camera.is_running
        
        camera.stop()
    
    def test_resolution_property(self):
        """Test resolution property."""
        config = CameraConfig(width=800, height=600)
        camera = Camera(config)
        
        assert camera.resolution == (800, 600)
    
    def test_context_manager(self, mock_cv2):
        """Test camera as context manager."""
        config = CameraConfig(warmup_frames=0, threaded=False)
        
        with Camera(config) as camera:
            assert camera.is_running
        
        assert not camera.is_running


class TestCameraIntegration:
    """Integration tests requiring real camera (marked as slow)."""
    
    @pytest.mark.skip(reason="Requires physical camera")
    def test_real_camera_capture(self):
        """Test capturing from real camera."""
        camera = Camera(CameraConfig(warmup_frames=5))
        
        try:
            if camera.start():
                frame = camera.read()
                
                assert frame is not None
                assert frame.image.shape[0] > 0
                assert frame.image.shape[1] > 0
        finally:
            camera.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
