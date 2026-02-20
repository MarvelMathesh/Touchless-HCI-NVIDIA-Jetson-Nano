"""
Structured logging with performance and gesture event logging.
"""

import os
import logging
import logging.handlers
import time
from functools import wraps


def setup_logging(level="INFO", log_file=None, max_size_mb=10, backup_count=3):
    """Configure structured logging for the application."""
    # Clean console format — compact and readable
    console_format = "%(asctime)s  %(levelname)-5s  %(message)s"
    # Verbose format for log file
    file_format = "%(asctime)s [%(levelname)-7s] %(name)-25s | %(message)s"
    date_format = "%H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler — only INFO+ and clean format
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(console_format, datefmt=date_format))
    root_logger.addHandler(console)

    # File handler (rotating)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format, datefmt=date_format))
        root_logger.addHandler(file_handler)

    return root_logger


class GestureLogger:
    """Specialized logger for gesture events and analytics."""

    def __init__(self):
        self.logger = logging.getLogger("gesture_events")
        self._gesture_history = []

    def log_gesture(self, gesture_name, confidence, action=None, latency_ms=None):
        """Log a recognized gesture event."""
        entry = {
            "timestamp": time.time(),
            "gesture": gesture_name,
            "confidence": confidence,
            "action": action,
            "latency_ms": latency_ms,
        }
        self._gesture_history.append(entry)
        self.logger.info(
            "Gesture: %-15s | Confidence: %.2f | Action: %-15s | Latency: %s",
            gesture_name,
            confidence,
            action or "none",
            f"{latency_ms:.1f}ms" if latency_ms else "N/A",
        )

    def log_action(self, action_name, success=True, detail=""):
        """Log an executed media action."""
        self.logger.info(
            "Action: %-15s | Success: %s | %s",
            action_name,
            success,
            detail,
        )

    def get_history(self, last_n=None):
        """Get recent gesture history."""
        if last_n:
            return self._gesture_history[-last_n:]
        return self._gesture_history.copy()

    @property
    def total_gestures(self):
        return len(self._gesture_history)


def log_timing(func):
    """Decorator to log function execution time."""
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("%s took %.2fms", func.__name__, elapsed)
        return result

    return wrapper
