"""
Action debouncing to prevent rapid repeated triggers.
Supports holdable gestures (volume) with controlled repeat rates.

v2 improvements:
    - Hold gesture lifecycle properly tracked from pipeline
    - Hold actions use accelerating repeat rate
    - Configurable holdable action set
"""

import time
import logging

logger = logging.getLogger(__name__)


class Debouncer:
    """Prevents rapid re-triggering of media actions with smart cooldowns."""

    def __init__(self, config: dict):
        self._cooldown_ms = config.get("cooldown_ms", 500)
        self._repeat_cooldown_ms = config.get("repeat_cooldown_ms", 300)
        self._hold_repeat_ms = config.get("hold_repeat_ms", 200)

        # Track per-action timing
        self._last_action_times = {}
        self._hold_states = {}  # action_name -> hold_start_time

        # Holdable actions (volume up/down can repeat while held)
        self._holdable_actions = {"volume_up", "volume_down"}

    def can_execute(self, action_name: str) -> bool:
        """Check if an action should be allowed to execute.

        Args:
            action_name: Name of the action to check

        Returns:
            True if the action should proceed
        """
        now = time.time() * 1000  # Convert to ms
        last_time = self._last_action_times.get(action_name, 0)
        elapsed = now - last_time

        if action_name in self._holdable_actions:
            # Holdable actions use a faster repeat rate
            # Accelerate repeat rate the longer the gesture is held
            if self.is_held(action_name):
                hold_duration = (time.time() - self._hold_states[action_name]) * 1000
                # After 2 seconds of holding, speed up repeat by 2x
                repeat_rate = self._hold_repeat_ms
                if hold_duration > 2000:
                    repeat_rate = max(100, self._hold_repeat_ms // 2)
                if elapsed < repeat_rate:
                    return False
            else:
                if elapsed < self._hold_repeat_ms:
                    return False
        else:
            # Check if this is a repeat of the same action
            last_any = max(self._last_action_times.values()) if self._last_action_times else 0
            same_action = (action_name == self._get_last_action())

            if same_action and elapsed < self._repeat_cooldown_ms:
                return False
            elif not same_action and (now - last_any) < self._cooldown_ms:
                return False

        return True

    def record(self, action_name: str):
        """Record that an action was executed."""
        self._last_action_times[action_name] = time.time() * 1000

    def _get_last_action(self) -> str:
        """Get the most recently executed action."""
        if not self._last_action_times:
            return None
        return max(self._last_action_times, key=self._last_action_times.get)

    def start_hold(self, action_name: str):
        """Mark an action as being held (for continuous gestures like volume)."""
        if action_name not in self._hold_states:
            self._hold_states[action_name] = time.time()
            logger.debug("Hold started: %s", action_name)

    def end_hold(self, action_name: str):
        """Mark end of a held action."""
        if action_name in self._hold_states:
            duration = time.time() - self._hold_states[action_name]
            del self._hold_states[action_name]
            logger.debug("Hold ended: %s (%.1fs)", action_name, duration)

    def end_all_holds(self):
        """End all held actions (e.g., when hand is lost)."""
        if self._hold_states:
            logger.debug("Ending all holds: %s", list(self._hold_states.keys()))
            self._hold_states.clear()

    def is_held(self, action_name: str) -> bool:
        """Check if an action is currently being held."""
        return action_name in self._hold_states

    def is_holdable(self, action_name: str) -> bool:
        """Check if an action supports hold-repeat."""
        return action_name in self._holdable_actions

    def reset(self):
        """Clear all debouncing state."""
        self._last_action_times.clear()
        self._hold_states.clear()
