"""
Performance Monitoring Module
==============================

Real-time performance metrics for FPS, latency, and system resources.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from collections import deque
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class Timer:
    """
    High-precision timer for measuring code execution time.
    
    Can be used as a context manager or decorator.
    
    Example:
        >>> timer = Timer("inference")
        >>> with timer:
        ...     model.predict(x)
        >>> print(f"Took {timer.elapsed_ms:.2f}ms")
        
        >>> @Timer.decorate("process_frame")
        ... def process_frame(frame):
        ...     pass
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._elapsed: float = 0.0
    
    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        self._end_time = time.perf_counter()
        if self._start_time:
            self._elapsed = self._end_time - self._start_time
        return self._elapsed
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._elapsed
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    @staticmethod
    def decorate(name: str = ""):
        """Decorator factory for timing functions."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                with Timer(name or func.__name__) as t:
                    result = func(*args, **kwargs)
                logger.debug(f"{t.name}: {t.elapsed_ms:.2f}ms")
                return result
            return wrapper
        return decorator


@dataclass
class PerformanceMetrics:
    """Container for performance metrics snapshot."""
    fps: float = 0.0
    frame_time_ms: float = 0.0
    latency_ms: float = 0.0
    capture_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    recognition_time_ms: float = 0.0
    total_frames: int = 0
    dropped_frames: int = 0


class PerformanceMonitor:
    """
    Real-time performance monitoring for the HCI pipeline.
    
    Tracks:
    - FPS (frames per second)
    - Per-stage latency (capture, detection, recognition)
    - End-to-end latency
    - Frame drops
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> monitor.start()
        >>> 
        >>> while running:
        ...     with monitor.measure("capture"):
        ...         frame = camera.read()
        ...     with monitor.measure("detection"):
        ...         hands = detector.detect(frame)
        ...     monitor.frame_complete()
        ...     
        ...     if monitor.fps < 25:
        ...         print("Performance warning!")
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of frames for rolling average
        """
        self.window_size = window_size
        self._frame_times: deque = deque(maxlen=window_size)
        self._stage_times: Dict[str, deque] = {}
        self._frame_start: Optional[float] = None
        self._total_frames: int = 0
        self._dropped_frames: int = 0
        self._running: bool = False
        self._lock = threading.Lock()
        
        # Target metrics
        self.target_fps: float = 25.0
        self.target_latency_ms: float = 30.0
    
    def start(self) -> None:
        """Start performance monitoring."""
        self._running = True
        self._total_frames = 0
        self._dropped_frames = 0
        self._frame_times.clear()
        self._stage_times.clear()
        logger.info("Performance monitor started")
    
    def stop(self) -> None:
        """Stop performance monitoring."""
        self._running = False
        logger.info(f"Performance monitor stopped. "
                   f"Total frames: {self._total_frames}, "
                   f"Dropped: {self._dropped_frames}")
    
    def frame_start(self) -> None:
        """Mark the start of frame processing."""
        self._frame_start = time.perf_counter()
    
    def frame_complete(self) -> None:
        """Mark frame processing complete and update metrics."""
        if self._frame_start is None:
            return
        
        frame_time = time.perf_counter() - self._frame_start
        
        with self._lock:
            self._frame_times.append(frame_time)
            self._total_frames += 1
            
            # Check for frame drops (taking too long)
            if frame_time > (1.0 / self.target_fps):
                self._dropped_frames += 1
        
        self._frame_start = None
    
    @contextmanager
    def measure(self, stage: str):
        """
        Context manager to measure a processing stage.
        
        Args:
            stage: Name of the stage (e.g., "capture", "detection")
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                if stage not in self._stage_times:
                    self._stage_times[stage] = deque(maxlen=self.window_size)
                self._stage_times[stage].append(elapsed)
    
    @property
    def fps(self) -> float:
        """Get current FPS (rolling average)."""
        with self._lock:
            if not self._frame_times:
                return 0.0
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    @property
    def frame_time_ms(self) -> float:
        """Get average frame time in milliseconds."""
        with self._lock:
            if not self._frame_times:
                return 0.0
            return (sum(self._frame_times) / len(self._frame_times)) * 1000
    
    def stage_time_ms(self, stage: str) -> float:
        """Get average time for a specific stage in milliseconds."""
        with self._lock:
            if stage not in self._stage_times or not self._stage_times[stage]:
                return 0.0
            times = self._stage_times[stage]
            return (sum(times) / len(times)) * 1000
    
    @property
    def total_latency_ms(self) -> float:
        """Get total end-to-end latency in milliseconds."""
        return self.frame_time_ms
    
    @property
    def is_meeting_targets(self) -> bool:
        """Check if performance targets are being met."""
        return (self.fps >= self.target_fps and 
                self.total_latency_ms <= self.target_latency_ms)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot."""
        return PerformanceMetrics(
            fps=self.fps,
            frame_time_ms=self.frame_time_ms,
            latency_ms=self.total_latency_ms,
            capture_time_ms=self.stage_time_ms("capture"),
            detection_time_ms=self.stage_time_ms("detection"),
            recognition_time_ms=self.stage_time_ms("recognition"),
            total_frames=self._total_frames,
            dropped_frames=self._dropped_frames,
        )
    
    def get_report(self) -> str:
        """Get formatted performance report string."""
        metrics = self.get_metrics()
        
        status = "✓" if self.is_meeting_targets else "✗"
        
        return (
            f"Performance Report {status}\n"
            f"{'=' * 40}\n"
            f"FPS: {metrics.fps:.1f} (target: ≥{self.target_fps})\n"
            f"Total Latency: {metrics.latency_ms:.1f}ms (target: ≤{self.target_latency_ms}ms)\n"
            f"\nPer-Stage Breakdown:\n"
            f"  Capture: {metrics.capture_time_ms:.2f}ms\n"
            f"  Detection: {metrics.detection_time_ms:.2f}ms\n"
            f"  Recognition: {metrics.recognition_time_ms:.2f}ms\n"
            f"\nFrame Stats:\n"
            f"  Total: {metrics.total_frames}\n"
            f"  Dropped: {metrics.dropped_frames} ({100*metrics.dropped_frames/max(1,metrics.total_frames):.1f}%)\n"
        )
