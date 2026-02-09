"""
Real-time visualization dashboard with gesture confidence overlay,
performance metrics, hand landmarks, and action feedback.
"""

import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Dashboard:
    """Renders real-time visualization overlay for the gesture control system."""

    def __init__(self, config: dict):
        self._show_landmarks = config.get("show_landmarks", True)
        self._show_fps = config.get("show_fps", True)
        self._show_latency = config.get("show_latency", True)
        self._show_gesture = config.get("show_gesture", True)
        self._show_confidence_bar = config.get("show_confidence_bar", True)
        self._show_legend = config.get("show_gesture_legend", True)
        self._show_bbox = config.get("show_hand_bbox", True)

        colors = config.get("colors", {})
        self._color_text = tuple(colors.get("text", [255, 255, 255]))
        self._color_fps_good = tuple(colors.get("fps_good", [0, 255, 0]))
        self._color_fps_warn = tuple(colors.get("fps_warn", [0, 255, 255]))
        self._color_fps_bad = tuple(colors.get("fps_bad", [0, 0, 255]))
        self._color_conf_high = tuple(colors.get("confidence_high", [0, 255, 0]))
        self._color_conf_mid = tuple(colors.get("confidence_mid", [0, 255, 255]))
        self._color_conf_low = tuple(colors.get("confidence_low", [0, 0, 255]))
        self._color_bbox = tuple(colors.get("bbox", [0, 255, 255]))
        self._color_action = tuple(colors.get("action_feedback", [255, 200, 0]))

        # Dashboard background
        dash_cfg = config.get("dashboard", {})
        self._dash_opacity = dash_cfg.get("opacity", 0.7)
        self._dash_height = dash_cfg.get("height", 80)

        # Gesture legend
        self._gesture_legend = [
            ("Thumbs Up", "Play/Pause"),
            ("Peace", "Vol+"),
            ("OK", "Vol-"),
            ("Fist", "Mute"),
            ("Palm", "Fullscreen"),
            ("Swipe", "Seek"),
        ]

    def render(self, frame: np.ndarray, state: dict) -> np.ndarray:
        """Render full dashboard overlay.

        Args:
            frame: BGR frame to draw on
            state: dict with current system state:
                - fps: float
                - latency_ms: float
                - gesture_name: str or None
                - gesture_confidence: float
                - hand_detected: bool
                - hand_bbox: tuple or None
                - mode: str
                - calibrating: bool
                - calibration_progress: float

        Returns:
            Frame with dashboard overlay
        """
        h, w = frame.shape[:2]

        # Top dashboard bar
        self._draw_dashboard_bar(frame, w, state)

        # Hand bounding box
        if self._show_bbox and state.get("hand_bbox"):
            self._draw_bbox(frame, state["hand_bbox"])

        # Current gesture display
        if self._show_gesture and state.get("gesture_name"):
            self._draw_gesture(frame, w, state)

        # Confidence bar
        if self._show_confidence_bar and state.get("gesture_confidence", 0) > 0:
            self._draw_confidence_bar(frame, h, state["gesture_confidence"])

        # Gesture legend
        if self._show_legend:
            self._draw_legend(frame, h)

        # Calibration progress
        if state.get("calibrating"):
            self._draw_calibration(frame, w, h, state.get("calibration_progress", 0))

        # No hand warning
        if not state.get("hand_detected", True):
            self._draw_no_hand_warning(frame, w, h)

        return frame

    def _draw_dashboard_bar(self, frame, w, state):
        """Draw top performance metrics bar."""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self._dash_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, self._dash_opacity, frame, 1 - self._dash_opacity, 0, frame)

        # FPS
        if self._show_fps:
            fps = state.get("fps", 0)
            if fps >= 25:
                fps_color = self._color_fps_good
            elif fps >= 15:
                fps_color = self._color_fps_warn
            else:
                fps_color = self._color_fps_bad

            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2,
            )

        # Latency
        if self._show_latency:
            latency = state.get("latency_ms", 0)
            lat_color = self._color_fps_good if latency < 30 else self._color_fps_warn
            cv2.putText(
                frame, f"Latency: {latency:.1f}ms",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lat_color, 1,
            )

        # Mode indicator
        mode = state.get("mode", "control")
        cv2.putText(
            frame, f"Mode: {mode.upper()}",
            (w - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self._color_text, 2,
        )

        # Hand count
        hand_count = state.get("hand_count", 0)
        cv2.putText(
            frame, f"Hands: {hand_count}",
            (w - 200, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._color_text, 1,
        )

    def _draw_gesture(self, frame, w, state):
        """Draw current gesture name and confidence."""
        gesture = state.get("gesture_name", "")
        confidence = state.get("gesture_confidence", 0)

        # Color based on confidence
        if confidence >= 0.85:
            color = self._color_conf_high
        elif confidence >= 0.65:
            color = self._color_conf_mid
        else:
            color = self._color_conf_low

        text = f"{gesture} ({confidence:.0%})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        x = (w - text_size[0]) // 2

        cv2.putText(
            frame, text,
            (x, self._dash_height + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
        )

    def _draw_confidence_bar(self, frame, h, confidence):
        """Draw vertical confidence bar on left side."""
        bar_x = 10
        bar_w = 15
        bar_h = 200
        bar_y = h - bar_h - 40

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

        # Fill based on confidence
        fill_h = int(confidence * bar_h)
        fill_y = bar_y + bar_h - fill_h

        if confidence >= 0.85:
            color = self._color_conf_high
        elif confidence >= 0.65:
            color = self._color_conf_mid
        else:
            color = self._color_conf_low

        cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h), color, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        # Label
        cv2.putText(
            frame, "CONF", (bar_x - 2, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._color_text, 1,
        )

    def _draw_bbox(self, frame, bbox):
        """Draw hand bounding box."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), self._color_bbox, 2)

    def _draw_legend(self, frame, h):
        """Draw gesture legend at bottom."""
        y = h - 15
        x = 40
        for gesture, action in self._gesture_legend:
            text = f"{gesture}={action}"
            cv2.putText(
                frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1,
            )
            x += 110

    def _draw_calibration(self, frame, w, h, progress):
        """Draw calibration progress overlay."""
        # Progress bar
        bar_w = 300
        bar_h = 25
        x = (w - bar_w) // 2
        y = h // 2

        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
        fill_w = int(progress * bar_w)
        cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), (0, 200, 255), -1)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 1)

        cv2.putText(
            frame, f"Calibrating... {progress:.0%}",
            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )
        cv2.putText(
            frame, "Show different gestures naturally",
            (x - 30, y + bar_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

    def _draw_no_hand_warning(self, frame, w, h):
        """Draw 'no hand detected' warning."""
        text = "Show hand to control"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x = (w - text_size[0]) // 2
        y = h - 50

        cv2.putText(
            frame, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2,
        )
