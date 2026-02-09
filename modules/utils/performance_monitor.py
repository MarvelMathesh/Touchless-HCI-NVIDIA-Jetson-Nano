"""
Real-time performance monitoring with per-stage latency tracking.
Thread-safe metrics collection with rolling windows.
"""

import time
import threading
import logging
from collections import deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Tracks FPS, per-stage latency, and system performance metrics."""

    def __init__(self, window_size=100):
        self._window_size = window_size
        self._lock = threading.Lock()

        # Frame timing
        self._frame_times = deque(maxlen=window_size)
        self._last_frame_time = None

        # Per-stage latency tracking
        self._stage_times = {}
        self._stage_names = [
            "capture", "preprocess", "detection",
            "classification", "action", "visualization", "total"
        ]
        for name in self._stage_names:
            self._stage_times[name] = deque(maxlen=window_size)

        # Counters
        self._frame_count = 0
        self._dropped_frames = 0
        self._start_time = time.time()

    @contextmanager
    def measure(self, stage_name: str):
        """Context manager to measure a pipeline stage's duration."""
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        with self._lock:
            if stage_name not in self._stage_times:
                self._stage_times[stage_name] = deque(maxlen=self._window_size)
            self._stage_times[stage_name].append(elapsed_ms)

    def tick(self):
        """Call once per frame to track FPS."""
        now = time.perf_counter()
        with self._lock:
            if self._last_frame_time is not None:
                self._frame_times.append(now - self._last_frame_time)
            self._last_frame_time = now
            self._frame_count += 1

    def record_drop(self):
        """Record a dropped frame."""
        with self._lock:
            self._dropped_frames += 1

    @property
    def fps(self) -> float:
        """Current frames per second (rolling average)."""
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            avg_interval = sum(self._frame_times) / len(self._frame_times)
            return 1.0 / avg_interval if avg_interval > 0 else 0.0

    @property
    def total_latency_ms(self) -> float:
        """Average total pipeline latency in ms."""
        return self._get_stage_avg("total")

    def get_stage_latency(self, stage_name: str) -> float:
        """Get average latency for a specific stage in ms."""
        return self._get_stage_avg(stage_name)

    def _get_stage_avg(self, stage_name: str) -> float:
        with self._lock:
            times = self._stage_times.get(stage_name, [])
            if not times:
                return 0.0
            return sum(times) / len(times)

    def get_all_latencies(self) -> dict:
        """Get average latency for all stages."""
        result = {}
        with self._lock:
            for name, times in self._stage_times.items():
                if times:
                    result[name] = sum(times) / len(times)
                else:
                    result[name] = 0.0
        return result

    def get_report(self) -> dict:
        """Generate a comprehensive performance report."""
        uptime = time.time() - self._start_time
        latencies = self.get_all_latencies()
        return {
            "fps": round(self.fps, 1),
            "total_frames": self._frame_count,
            "dropped_frames": self._dropped_frames,
            "drop_rate": round(
                self._dropped_frames / max(self._frame_count, 1) * 100, 2
            ),
            "uptime_seconds": round(uptime, 1),
            "latencies_ms": {k: round(v, 2) for k, v in latencies.items()},
        }

    def print_report(self):
        """Print formatted performance report."""
        report = self.get_report()
        logger.info("=" * 60)
        logger.info("PERFORMANCE REPORT")
        logger.info("=" * 60)
        logger.info("FPS:            %.1f", report["fps"])
        logger.info("Total Frames:   %d", report["total_frames"])
        logger.info("Dropped Frames: %d (%.2f%%)", report["dropped_frames"], report["drop_rate"])
        logger.info("Uptime:         %.1fs", report["uptime_seconds"])
        logger.info("-" * 40)
        logger.info("Stage Latencies (avg ms):")
        for stage, latency in report["latencies_ms"].items():
            logger.info("  %-18s %7.2f ms", stage, latency)
        logger.info("=" * 60)

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._frame_times.clear()
            self._last_frame_time = None
            for name in self._stage_times:
                self._stage_times[name].clear()
            self._frame_count = 0
            self._dropped_frames = 0
            self._start_time = time.time()
