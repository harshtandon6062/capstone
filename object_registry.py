"""The registry of what is in the workspace.

This is the spine between perception, the UI and the robot. Perception fills it,
the UI reads it to decide what can be selected, and the robot reads it to decide
where to move. Colour lives here once, so the panel and the simulation can never
disagree.

Two things are deliberately *derived* rather than remembered:

  * whether a spot is occupied, which is worked out from where the tubes actually
    are. Remembering it with a flag went wrong every time the world changed by
    some other route - an undo, a scene reset, or a tube being moved somewhere
    else - and left spots looking used when nothing was on them.
  * what a tube contains, which is a property of the tube and not of how many
    times it has been handled. A tube that has been moved is still a full tube.

Nothing here knows about PyBullet or OpenCV.
"""

import math
from dataclasses import dataclass

try:
    from config import SPOT_OCCUPANCY_RADIUS
except ImportError:  # keep this module importable on its own
    SPOT_OCCUPANCY_RADIUS = 0.06


# What a tube looks like once its contents have been poured out.
EMPTY_LIQUID_RGBA = [0.82, 0.82, 0.84, 1.0]

# Every step at which the operator chooses something, and what may be chosen.
PURPOSES = ("move_source", "pour_source", "move_target", "pour_target")


def rgba_to_bgr(rgba):
    """Convert an RGBA float colour (0-1) into an OpenCV BGR tuple (0-255)."""
    red, green, blue = rgba[0], rgba[1], rgba[2]
    return (
        int(round(blue * 255)),
        int(round(green * 255)),
        int(round(red * 255)),
    )


def mix_colors(first, second):
    """Blend two liquids. Crude on purpose - it only has to be visibly neither."""
    return [(a + b) / 2 for a, b in zip(first[:3], second[:3])] + [1.0]


@dataclass
class SceneObject:
    """One thing in the workspace that the operator can refer to."""

    handle: int
    label: str
    color_name: str
    color_rgba: list
    position: list
    kind: str
    empty: bool = False

    def __post_init__(self):
        # What this object looked like when the scene was built, so a reset can
        # put its contents back rather than leaving it permanently empty.
        self._initial = (list(self.color_rgba), self.label, self.color_name)

    @property
    def color_bgr(self):
        """Panel colour derived from the same value the simulation renders."""
        return rgba_to_bgr(self.color_rgba)

    def restore(self):
        """Put the contents back the way the scene started."""
        colour, label, name = self._initial
        self.color_rgba = list(colour)
        self.label = label
        self.color_name = name
        self.empty = False


class ObjectRegistry:
    """Live view of the workspace, kept in sync from a perception source."""

    def __init__(self, perception):
        self.perception = perception
        self._objects = {}
        self._order = []
        self.refresh()

    # ── syncing ────────────────────────────────────────────────

    def refresh(self):
        """Pull the latest observations, preserving what each object contains."""
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

    def by_handle(self, handle):
        return self._objects.get(handle)

    def index_of(self, kind, handle):
        for position, obj in enumerate(self.all(kind)):
            if obj.handle == handle:
                return position
        return None

    # ── occupancy, worked out rather than remembered ───────────

    def occupancy(self):
        """Map of spot handle -> the tube standing on it.

        Each tube claims only its nearest spot, so a tube sitting between two of
        them cannot make both look occupied.
        """
        taken = {}
        spots = self.destinations
        if not spots:
            return taken
        for tube in self.sources:
            nearest = min(
                spots,
                key=lambda spot: math.hypot(
                    tube.position[0] - spot.position[0],
                    tube.position[1] - spot.position[1],
                ),
            )
            gap = math.hypot(
                tube.position[0] - nearest.position[0],
                tube.position[1] - nearest.position[1],
            )
            if gap <= SPOT_OCCUPANCY_RADIUS:
                taken.setdefault(nearest.handle, tube.handle)
        return taken

    def occupant_of(self, spot_handle):
        return self.occupancy().get(spot_handle)

    def spot_under(self, tube_handle):
        for spot, tube in self.occupancy().items():
            if tube == tube_handle:
                return spot
        return None

    # ── what may be chosen, and when ───────────────────────────

    def selectable(self, purpose, exclude=None):
        """The objects that may legally be chosen at this step, in row order.

        move_source   every tube, however many times it has already been moved
        pour_source   tubes that still have something in them
        move_target   spots with no tube standing on them
        pour_target   any other tube, full or empty
        """
        if purpose == "move_source":
            return [t for t in self.sources if t.handle != exclude]
        if purpose == "pour_source":
            return [t for t in self.sources if not t.empty and t.handle != exclude]
        if purpose == "pour_target":
            return [t for t in self.sources if t.handle != exclude]
        if purpose == "move_target":
            taken = self.occupancy()
            return [
                s for s in self.destinations
                if s.handle not in taken and s.handle != exclude
            ]
        raise ValueError(f"unknown selection purpose {purpose!r}")

    def next_selectable(self, purpose, current_handle, step, exclude=None):
        """The next choosable handle, or None when there is nothing to choose.

        Always terminates: it walks a fixed list rather than searching until it
        finds something, so an exhausted workspace returns None instead of
        spinning.
        """
        handles = [obj.handle for obj in self.selectable(purpose, exclude)]
        if not handles:
            return None
        if current_handle not in handles:
            return handles[0]
        return handles[(handles.index(current_handle) + step) % len(handles)]

    def first_selectable(self, purpose, exclude=None):
        options = self.selectable(purpose, exclude)
        return options[0].handle if options else None

    def can_pour(self, source_handle):
        """Whether this tube has anything to pour, and anywhere to pour it."""
        source = self._objects.get(source_handle)
        if source is None or source.kind != "source" or source.empty:
            return False
        return bool(self.selectable("pour_target", exclude=source_handle))

    def can_move(self, source_handle=None):
        """Whether there is any free spot to move a tube onto."""
        return bool(self.selectable("move_target"))

    # ── contents ───────────────────────────────────────────────

    def transfer_contents(self, source_handle, target_handle):
        """Pour one tube into another. Returns (source, target), or None.

        Refuses when there is nothing to pour. An empty tube takes the contents
        as they are rather than averaging with them: "empty" is not a liquid, and
        blending with it would tint every pour toward the empty-tube colour.
        """
        source = self._objects.get(source_handle)
        target = self._objects.get(target_handle)
        if source is None or target is None or source is target:
            return None
        if source.empty:
            return None

        if target.empty:
            target.color_rgba = list(source.color_rgba)
            target.color_name = source.color_name
            target.label = source.label
            target.empty = False
        else:
            target.color_rgba = mix_colors(source.color_rgba, target.color_rgba)
            target.color_name = "MIXED"
            target.label = "Mixed tube"

        source.color_rgba = list(EMPTY_LIQUID_RGBA)
        source.color_name = "EMPTY"
        source.label = "Empty tube"
        source.empty = True
        return source, target

    def reset(self):
        """Put every object's contents back to how the scene started."""
        for obj in self._objects.values():
            obj.restore()
        return self

    # ── convenience ────────────────────────────────────────────

    def __len__(self):
        return len(self._order)

    def count(self, kind):
        return len(self.all(kind))
