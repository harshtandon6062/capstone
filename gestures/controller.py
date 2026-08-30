"""Centralized gesture command routing and duplicate suppression."""

import time


class GestureController:
    """Resolve static/dynamic gesture events into application-level commands."""

    def __init__(self, dynamic_classes=None, cooldown=0.35):
        self.dynamic_classes = dynamic_classes or ["grasp", "none", "sweep", "tilt", "wrist_rotation"]
        self.cooldown = cooldown
        self.last_dynamic_event_time = 0.0
        self.last_static_event = None
        self.last_dynamic_event = None
        self.last_command = None

    def _make_event(self, source, gesture, confidence=1.0):
        return {
            "type": gesture,
            "source": source,
            "confidence": float(confidence),
            "timestamp": time.time(),
        }

    def handle_static_gesture(self, gesture, confidence=1.0):
        if gesture is None:
            return None

        event = self._make_event("static", gesture, confidence)
        self.last_static_event = event
        if gesture == "unknown":
            return None
        return event

    def handle_dynamic_gesture(self, gesture, confidence=1.0):
        if gesture is None:
            return None

        gesture_name = str(gesture)
        if gesture_name not in self.dynamic_classes:
            return None

        now = time.time()
        if gesture_name == "none":
            self.last_dynamic_event = None
            self.last_dynamic_event_time = now
            return None

        if self.last_dynamic_event is not None and self.last_dynamic_event["type"] == gesture_name:
            if now - self.last_dynamic_event_time < self.cooldown:
                return None

        event = self._make_event("dynamic", gesture_name, confidence)
        self.last_dynamic_event = event
        self.last_dynamic_event_time = now
        return event

    def resolve_command(self, static_event=None, dynamic_event=None):
        """Return a single command/Event while preserving safety gating later in the stack."""
        if static_event is None and dynamic_event is None:
            return None

        if static_event is not None and dynamic_event is not None:
            if static_event["confidence"] >= dynamic_event["confidence"]:
                selected = static_event
            else:
                selected = dynamic_event
        else:
            selected = static_event or dynamic_event

        command = {**selected, "command": selected["type"]}
        self.last_command = command
        return command
