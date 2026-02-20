"""
Fire-once debouncer with gesture lifecycle awareness.

Xbox Kinect-style UX:
    - Toggle actions (fullscreen, mute, play_pause): fire ONCE when gesture
      is first detected, then block until the user releases the gesture
      (hand down or different gesture) before it can fire again.
    - Holdable actions (volume_up, volume_down): repeat at a controlled
      rate while the gesture is held, with acceleration after 2s.

Lifecycle (called by Pipeline):
    gesture_changed(name)  - stable gesture changed to a new one
    gesture_lost()         - gesture became unstable or hand lost
"""

import time
import logging

logger = logging.getLogger(__name__)


class Debouncer:
    """Fire-once debouncer with gesture lifecycle awareness."""

    def __init__(self, config: dict):
        self._hold_repeat_ms = config.get("hold_repeat_ms", 200)
        self._post_release_cooldown_ms = config.get("post_release_cooldown_ms", 400)

        # Per-action timing
        self._last_action_times = {}
        self._hold_states = {}  # action -> hold_start_time

        # Fire-once tracking
        self._fired_actions = set()  # Actions that fired, awaiting gesture release
        self._active_gesture = None  # Current stable gesture name
        self._release_time = 0  # ms timestamp of last gesture release

        # Holdable actions (repeat while held)
        self._holdable_actions = {"volume_up", "volume_down"}

    def can_execute(self, action_name: str) -> bool:
        """Check if an action should fire.

        Toggle actions: blocked after first fire until gesture release.
        Holdable actions: repeat at controlled rate while held.
        """
        now = time.time() * 1000

        # Fire-once: block if this action already fired this gesture cycle
        if action_name in self._fired_actions:
            return False

        # Post-release cooldown for toggle actions (prevents flicker re-trigger)
        if action_name not in self._holdable_actions:
            if self._release_time and (now - self._release_time) < self._post_release_cooldown_ms:
                return False

        # Holdable actions use repeat rate with acceleration
        if action_name in self._holdable_actions:
            last_time = self._last_action_times.get(action_name, 0)
            elapsed = now - last_time
            if self.is_held(action_name):
                hold_duration = (time.time() - self._hold_states[action_name]) * 1000
                repeat_rate = self._hold_repeat_ms
                if hold_duration > 2000:
                    repeat_rate = max(100, self._hold_repeat_ms // 2)
                if elapsed < repeat_rate:
                    return False
            else:
                if elapsed < self._hold_repeat_ms:
                    return False

        return True

    def record(self, action_name: str):
        """Record that an action fired. Toggle actions get locked until release."""
        self._last_action_times[action_name] = time.time() * 1000
        if action_name not in self._holdable_actions:
            self._fired_actions.add(action_name)
            logger.debug("Action '%s' fired — locked until gesture release", action_name)

    def gesture_changed(self, new_gesture: str):
        """Called by pipeline when the stable gesture changes to a different one.

        Re-arms all toggle actions so the new gesture's action can fire.
        """
        if new_gesture != self._active_gesture:
            old = self._active_gesture
            self._active_gesture = new_gesture
            self._fired_actions.clear()
            self._hold_states.clear()
            self._release_time = time.time() * 1000
            logger.debug("Gesture changed: %s -> %s, actions re-armed", old, new_gesture)

    def gesture_lost(self):
        """Called by pipeline when gesture becomes unstable or hand is lost.

        Re-arms all toggle actions so the same gesture can fire again
        next time it is detected.
        """
        if self._active_gesture is not None or self._fired_actions:
            logger.debug("Gesture lost (was %s), re-arming: %s",
                         self._active_gesture, self._fired_actions or "none")
            self._active_gesture = None
            self._fired_actions.clear()
            self._hold_states.clear()
            self._release_time = time.time() * 1000

    def start_hold(self, action_name: str):
        """Mark a holdable action as being held."""
        if action_name not in self._hold_states:
            self._hold_states[action_name] = time.time()
            logger.debug("Hold started: %s", action_name)

    def end_hold(self, action_name: str):
        """End a specific held action."""
        if action_name in self._hold_states:
            duration = time.time() - self._hold_states[action_name]
            del self._hold_states[action_name]
            logger.debug("Hold ended: %s (%.1fs)", action_name, duration)

    def is_held(self, action_name: str) -> bool:
        """Check if an action is currently being held."""
        return action_name in self._hold_states

    def is_holdable(self, action_name: str) -> bool:
        """Check if an action supports hold-repeat."""
        return action_name in self._holdable_actions

    def reset(self):
        """Clear all state."""
        self._last_action_times.clear()
        self._hold_states.clear()
        self._fired_actions.clear()
        self._active_gesture = None
        self._release_time = 0
