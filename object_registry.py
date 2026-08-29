"""The registry of what is in the workspace.

This is the spine between perception, the UI and the robot. Perception fills it,
the UI reads it to decide what can be selected, and the robot reads it to decide
where to move. Colour lives here once, so the panel and the simulation can never
disagree.

Nothing here knows about PyBullet or OpenCV.
"""

from dataclasses import dataclass, field


def rgba_to_bgr(rgba):
    """Convert an RGBA float colour (0-1) into an OpenCV BGR tuple (0-255)."""
    red, green, blue = rgba[0], rgba[1], rgba[2]
    return (
        int(round(blue * 255)),
        int(round(green * 255)),
        int(round(red * 255)),
    )


@dataclass
class SceneObject:
    """One thing in the workspace that the operator can refer to."""

    handle: int
    label: str
    color_name: str
    color_rgba: list
    position: list
    kind: str
    consumed: bool = False

    @property
    def color_bgr(self):
        """Panel colour derived from the same value the simulation renders."""
        return rgba_to_bgr(self.color_rgba)

    @property
    def available(self):
        return not self.consumed


class ObjectRegistry:
    """Live view of the workspace, kept in sync from a perception source."""

    def __init__(self, perception):
        self.perception = perception
        self._objects = {}
        self._order = []
        self.refresh()

    # ── syncing ────────────────────────────────────────────────

    def refresh(self):
        """Pull the latest observations, preserving per-object consumed state."""
        for observation in self.perception.detect():
            handle = observation["handle"]
            existing = self._objects.get(handle)
            if existing is None:
                self._objects[handle] = SceneObject(
                    handle=handle,
                    label=observation["label"],
                    color_name=observation["color_name"],
                    color_rgba=list(observation["color_rgba"]),
                    position=list(observation["position"]),
                    kind=observation["kind"],
                )
                self._order.append(handle)
            else:
                existing.position = list(observation["position"])
        return self

    # ── reading ────────────────────────────────────────────────

    def all(self, kind=None):
        objects = [self._objects[h] for h in self._order]
        if kind is None:
            return objects
        return [obj for obj in objects if obj.kind == kind]

    @property
    def sources(self):
        return self.all("source")

    @property
    def destinations(self):
        return self.all("destination")

    def available(self, kind):
        return [obj for obj in self.all(kind) if obj.available]

    def by_handle(self, handle):
        return self._objects.get(handle)

    def index_of(self, kind, handle):
        objects = self.all(kind)
        for position, obj in enumerate(objects):
            if obj.handle == handle:
                return position
        return None

    # ── navigation ─────────────────────────────────────────────

    def next_available(self, kind, current_handle, step):
        """Return the next selectable object's handle, or None if there are none.

        Always terminates: it inspects each object at most once. Returning None
        is the correct answer when every object of this kind has been used, and
        callers must handle it rather than spinning.
        """
        objects = self.all(kind)
        if not objects:
            return None

        start = self.index_of(kind, current_handle)
        if start is None:
            start = 0

        count = len(objects)
        for offset in range(1, count + 1):
            candidate = objects[(start + step * offset) % count]
            if candidate.available:
                return candidate.handle
        return None

    def first_available(self, kind):
        for obj in self.all(kind):
            if obj.available:
                return obj.handle
        return None

    # ── state ──────────────────────────────────────────────────

    def consume(self, handle):
        obj = self._objects.get(handle)
        if obj is not None:
            obj.consumed = True
        return obj

    def release(self, handle):
        obj = self._objects.get(handle)
        if obj is not None:
            obj.consumed = False
        return obj

    def reset(self):
        for obj in self._objects.values():
            obj.consumed = False
        return self

    # ── convenience ────────────────────────────────────────────

    def __len__(self):
        return len(self._order)

    def count(self, kind):
        return len(self.all(kind))
