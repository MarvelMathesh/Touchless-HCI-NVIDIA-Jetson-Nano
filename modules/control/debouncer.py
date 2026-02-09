"""
Action debouncing to prevent rapid repeated triggers.
Supports holdable gestures (volume) with controlled repeat rates.
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
        self._hold_states = {}

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
        """Mark an action as being held (for continuous gestures)."""
        self._hold_states[action_name] = time.time()

    def end_hold(self, action_name: str):
        """Mark end of a held action."""
        self._hold_states.pop(action_name, None)

    def is_held(self, action_name: str) -> bool:
        """Check if an action is currently being held."""
        return action_name in self._hold_states

    def reset(self):
        """Clear all debouncing state."""
        self._last_action_times.clear()
        self._hold_states.clear()
