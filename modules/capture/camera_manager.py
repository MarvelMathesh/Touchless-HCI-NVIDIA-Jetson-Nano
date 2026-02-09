"""
Async camera capture with threaded buffering for minimal latency.
Optimized for USB cameras on Jetson Nano with V4L2 backend.
"""

import time
import threading
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """High-performance camera capture with threaded frame acquisition."""

    def __init__(self, config: dict):
        self._device_id = config.get("device_id", 0)
        self._width = config.get("width", 640)
        self._height = config.get("height", 480)
        self._fps = config.get("fps", 30)
        self._backend = config.get("backend", "v4l2")
        self._buffer_size = config.get("buffer_size", 1)
        self._flip_h = config.get("flip_horizontal", True)
        self._warmup_frames = config.get("warmup_frames", 15)
        self._timeout_ms = config.get("capture_timeout_ms", 100)

        self._cap = None
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._capture_times = []

    def open(self) -> bool:
        """Open camera with optimized settings."""
        backend_map = {
            "v4l2": cv2.CAP_V4L2,
            "gstreamer": cv2.CAP_GSTREAMER,
            "auto": cv2.CAP_ANY,
        }
        backend = backend_map.get(self._backend, cv2.CAP_ANY)

        self._cap = cv2.VideoCapture(self._device_id, backend)
        if not self._cap.isOpened():
            logger.error("Failed to open camera %d with backend %s", self._device_id, self._backend)
            return False

        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        # Verify actual settings
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera opened: %dx%d @ %.0f FPS (requested %dx%d @ %d)",
            actual_w, actual_h, actual_fps,
            self._width, self._height, self._fps,
        )

        # Warmup - let auto-exposure stabilize
        logger.info("Camera warmup: discarding %d frames...", self._warmup_frames)
        for _ in range(self._warmup_frames):
            self._cap.read()

        return True

    def start_async(self):
        """Start threaded frame capture."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Async capture started")

    def _capture_loop(self):
        """Background capture thread - always holds the latest frame."""
        while self._running:
            start = time.perf_counter()
            ret, frame = self._cap.read()
            elapsed_ms = (time.perf_counter() - start) * 1000

            if ret and frame is not None:
                if self._flip_h:
                    frame = cv2.flip(frame, 1)
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._capture_times.append(elapsed_ms)
                    if len(self._capture_times) > 100:
                        self._capture_times = self._capture_times[-100:]
            else:
                time.sleep(0.001)

    def read(self):
        """Get the latest frame (non-blocking).

        Returns:
            tuple: (frame_id, numpy array) or (None, None) if no frame
        """
        with self._lock:
            if self._frame is not None:
                return self._frame_id, self._frame.copy()
            return None, None

    def read_sync(self):
        """Synchronous frame read (for non-threaded mode).

        Returns:
            tuple: (frame_id, numpy array) or (None, None)
        """
        if self._cap is None:
            return None, None
        ret, frame = self._cap.read()
        if ret and frame is not None:
            if self._flip_h:
                frame = cv2.flip(frame, 1)
            self._frame_id += 1
            return self._frame_id, frame
        return None, None

    @property
    def avg_capture_time_ms(self) -> float:
        """Average frame capture time in ms."""
        if not self._capture_times:
            return 0.0
        return sum(self._capture_times) / len(self._capture_times)

    @property
    def resolution(self) -> tuple:
        return (self._width, self._height)

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def stop(self):
        """Stop async capture and release camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.stop()
