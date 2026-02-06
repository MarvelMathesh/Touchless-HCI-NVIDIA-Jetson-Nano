"""
High-Performance Camera Capture Module
=======================================

Optimized camera capture for NVIDIA Jetson Nano with minimal latency.
Supports threaded capture for non-blocking operation.
"""

import cv2
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1  # Minimal buffering for low latency
    threaded: bool = True
    flip_horizontal: bool = False
    warmup_frames: int = 5  # Reduced from 30 for faster startup
    
    @classmethod
    def from_dict(cls, config: dict) -> "CameraConfig":
        """Create config from dictionary (YAML parsed)."""
        return cls(
            device_id=config.get("device_id", 0),
            width=config.get("width", 1280),
            height=config.get("height", 720),
            fps=config.get("fps", 30),
            buffer_size=config.get("buffer_size", 1),
            warmup_frames=config.get("warmup_frames", 30),
            flip_horizontal=config.get("flip_horizontal", True),
            threaded=config.get("threaded", True),
        )


@dataclass
class Frame:
    """Container for captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int
    
    @property
    def rgb(self) -> np.ndarray:
        """Convert BGR to RGB."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)


class Camera:
    """
    High-performance camera capture with optional threading.
    
    Features:
    - Minimal buffering for low latency
    - Threaded capture for non-blocking reads
    - Frame timestamping for latency measurement
    - Automatic warmup to stabilize camera exposure
    
    Example:
        >>> camera = Camera(CameraConfig())
        >>> camera.start()
        >>> frame = camera.read()
        >>> if frame:
        ...     process(frame.image)
        >>> camera.stop()
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_number = 0
        self._running = False
        
        # Threading components
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[Frame] = None
        
        # Performance tracking
        self._capture_times = deque(maxlen=30)  # type: deque
        
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if camera started successfully
        """
        logger.info("Starting camera (device={}, {}x{}@{}fps)".format(
            self.config.device_id, self.config.width, self.config.height, self.config.fps))
        
        # Try V4L2 backend first (works better on Jetson with USB cameras)
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            if backend == cv2.CAP_V4L2:
                logger.info("Trying V4L2 backend...")
                self._cap = cv2.VideoCapture(self.config.device_id, backend)
            else:
                logger.info("Trying default backend...")
                self._cap = cv2.VideoCapture(self.config.device_id)
            
            if not self._cap.isOpened():
                logger.warning("Backend failed, trying next...")
                continue
            
            # Try to set resolution and FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            # Verify we can actually read frames
            test_ret, test_frame = self._cap.read()
            if test_ret and test_frame is not None:
                # Success!
                break
            else:
                logger.warning("Can't read frames, trying next backend...")
                self._cap.release()
                self._cap = None
        
        if self._cap is None or not self._cap.isOpened():
            logger.error("Failed to open camera device {}".format(self.config.device_id))
            return False
        
        # Get actual resolution and FPS
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        logger.info("Camera initialized: {}x{}@{}fps".format(actual_width, actual_height, actual_fps))
        
        # Warm up camera
        if self.config.warmup_frames > 0:
            logger.info("Warming up camera ({} frames)...".format(self.config.warmup_frames))
            for _ in range(self.config.warmup_frames):
                self._cap.read()
        
        self._running = True
        self._frame_number = 0
        
        # Start threaded capture if enabled
        if self.config.threaded:
            logger.info("Started threaded capture")
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop camera capture and release resources."""
        logger.info("Stopping camera...")
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        logger.info("Camera stopped")
    
    def read(self) -> Optional[Frame]:
        """
        Read the latest frame.
        
        In threaded mode, returns the most recent captured frame.
        In synchronous mode, captures a new frame.
        
        Returns:
            Frame or None if capture failed
        """
        if not self._running:
            return None
        
        if self.config.threaded:
            with self._lock:
                return self._latest_frame
        else:
            return self._capture_frame()
    
    def _capture_frame(self) -> Optional[Frame]:
        """Capture a single frame from the camera."""
        if not self._cap:
            return None
        
        start_time = time.perf_counter()
        ret, image = self._cap.read()
        capture_time = time.perf_counter() - start_time
        
        if not ret or image is None:
            logger.warning("Failed to capture frame")
            return None
        
        # Flip horizontally for mirror effect (intuitive for user)
        if self.config.flip_horizontal:
            image = cv2.flip(image, 1)
        
        self._frame_number += 1
        self._capture_times.append(capture_time)
        
        return Frame(
            image=image,
            timestamp=time.time(),
            frame_number=self._frame_number
        )
    
    def _capture_loop(self) -> None:
        """Background thread for continuous frame capture."""
        while self._running:
            frame = self._capture_frame()
            if frame:
                with self._lock:
                    self._latest_frame = frame
    
    @property
    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self._running
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current camera resolution."""
        return (self.config.width, self.config.height)
    
    @property
    def avg_capture_time_ms(self) -> float:
        """Get average frame capture time in milliseconds."""
        if not self._capture_times:
            return 0.0
        return (sum(self._capture_times) / len(self._capture_times)) * 1000
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
