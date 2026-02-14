"""
Lightweight event bus for decoupled inter-module communication.

Replaces direct method calls between modules with a publish/subscribe
pattern, enabling loose coupling and plugin extensibility.

Usage:
    bus = EventBus()
    bus.subscribe("gesture_detected", my_handler)
    bus.emit("gesture_detected", gesture_name="thumbs_up", confidence=0.92)
"""

import time
import logging
import threading
from collections import defaultdict
from typing import Callable, Any

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe publish/subscribe event bus.

    Supports synchronous event dispatch with priority ordering
    and optional async dispatch for non-critical listeners.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern — one bus per application."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._listeners = defaultdict(list)  # event_name -> [(priority, callback)]
        self._lock = threading.Lock()
        self._event_history = []
        self._max_history = 100
        self._enabled = True
        self._initialized = True

    def subscribe(self, event_name: str, callback: Callable, priority: int = 0):
        """Register a listener for an event.

        Args:
            event_name: Event to listen for
            callback: Function to call. Receives **kwargs from emit().
            priority: Higher priority callbacks run first (default 0)
        """
        with self._lock:
            self._listeners[event_name].append((priority, callback))
            # Sort by priority descending (highest first)
            self._listeners[event_name].sort(key=lambda x: -x[0])
        logger.debug("Subscribed to '%s': %s (priority=%d)",
                      event_name, callback.__name__, priority)

    def unsubscribe(self, event_name: str, callback: Callable):
        """Remove a listener for an event."""
        with self._lock:
            self._listeners[event_name] = [
                (p, cb) for p, cb in self._listeners[event_name] if cb is not callback
            ]

    def emit(self, event_name: str, **kwargs):
        """Emit an event to all registered listeners.

        Args:
            event_name: Event name to emit
            **kwargs: Data passed to all listeners
        """
        if not self._enabled:
            return

        with self._lock:
            listeners = list(self._listeners.get(event_name, []))

        # Record in history
        self._event_history.append({
            "event": event_name,
            "time": time.time(),
            "data_keys": list(kwargs.keys()),
        })
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Dispatch to listeners
        for priority, callback in listeners:
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error("Event handler error [%s -> %s]: %s",
                             event_name, callback.__name__, e)

    def emit_async(self, event_name: str, **kwargs):
        """Emit an event in a separate daemon thread (non-blocking).

        Use for non-critical events (analytics, logging) that shouldn't
        block the real-time pipeline.
        """
        thread = threading.Thread(
            target=self.emit,
            args=(event_name,),
            kwargs=kwargs,
            daemon=True,
        )
        thread.start()

    def clear(self, event_name: str = None):
        """Remove all listeners, optionally for a specific event."""
        with self._lock:
            if event_name:
                self._listeners.pop(event_name, None)
            else:
                self._listeners.clear()

    @property
    def registered_events(self) -> list:
        """List all events with registered listeners."""
        with self._lock:
            return list(self._listeners.keys())

    @property
    def listener_count(self) -> int:
        """Total number of registered listeners."""
        with self._lock:
            return sum(len(cbs) for cbs in self._listeners.values())

    def get_history(self, last_n: int = 10) -> list:
        """Get recent event history."""
        return self._event_history[-last_n:]

    def reset(self):
        """Reset singleton state (for testing)."""
        self._listeners.clear()
        self._event_history.clear()
        self._enabled = True


# =============================================================================
# Standard Event Names (constants to avoid typos)
# =============================================================================

class Events:
    """Standard event names used throughout the system."""

    # Pipeline events
    FRAME_CAPTURED = "frame_captured"
    HAND_DETECTED = "hand_detected"
    HAND_LOST = "hand_lost"
    GESTURE_DETECTED = "gesture_detected"
    GESTURE_STABLE = "gesture_stable"

    # Action events
    ACTION_REQUESTED = "action_requested"
    ACTION_EXECUTED = "action_executed"
    ACTION_FAILED = "action_failed"

    # System events
    ANOMALY_DETECTED = "anomaly_detected"
    THERMAL_WARNING = "thermal_warning"
    THERMAL_CRITICAL = "thermal_critical"
    CAMERA_ERROR = "camera_error"
    CALIBRATION_STARTED = "calibration_started"
    CALIBRATION_COMPLETE = "calibration_complete"

    # Lifecycle
    SYSTEM_STARTED = "system_started"
    SYSTEM_SHUTDOWN = "system_shutdown"
    MODE_CHANGED = "mode_changed"
