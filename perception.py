"""Perception sources that report what is currently in the workspace.

A perception source answers one question: what objects are in front of the robot
right now, and where are they? It returns plain dictionaries so that nothing
downstream depends on how the objects were found.

`SimulatedPerception` reads ground-truth poses straight from PyBullet. A future
`CameraPerception` would run detection on a camera frame and return the same
shape, so the rest of the application does not change.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class PerceptionSource(Protocol):
    """Anything that can report the current contents of the workspace."""

    def detect(self):
        """Return a list of observation dicts.

        Each observation must contain:
            handle      backend id used to command the object (PyBullet body id)
            label       human-readable name, e.g. "Red tube"
            color_name  short tag, e.g. "RED"
            color_rgba  [r, g, b, a] floats 0-1 - the single source of truth for colour
            position    [x, y, z] in world coordinates
            kind        "source" or "destination"
        """


class SimulatedPerception:
    """Report workspace contents by reading PyBullet body poses directly.

    This stands in for a camera. Destinations are fixed markers, so their
    positions are supplied once; sources are tracked live because the robot
    moves them.
    """

    def __init__(self, physics, sources=(), destinations=()):
        self.physics = physics
        self._sources = list(sources)
        self._destinations = list(destinations)

    def add_source(self, handle, label, color_name, color_rgba):
        self._sources.append(
            {
                "handle": handle,
                "label": label,
                "color_name": color_name,
                "color_rgba": list(color_rgba),
            }
        )

    def add_destination(self, handle, label, color_name, color_rgba, position):
        self._destinations.append(
            {
                "handle": handle,
                "label": label,
                "color_name": color_name,
                "color_rgba": list(color_rgba),
                "position": list(position),
            }
        )

    def detect(self):
        observations = []

        for entry in self._sources:
            position, _ = self.physics.getBasePositionAndOrientation(entry["handle"])
            observations.append(
                {
                    "handle": entry["handle"],
                    "label": entry["label"],
                    "color_name": entry["color_name"],
                    "color_rgba": entry["color_rgba"],
                    "position": list(position),
                    "kind": "source",
                }
            )

        for entry in self._destinations:
            observations.append(
                {
                    "handle": entry["handle"],
                    "label": entry["label"],
                    "color_name": entry["color_name"],
                    "color_rgba": entry["color_rgba"],
                    "position": list(entry["position"]),
                    "kind": "destination",
                }
            )

        return observations


class StaticPerception:
    """Return a fixed list of observations. Useful for tests and dry runs."""

    def __init__(self, observations):
        self._observations = [dict(entry) for entry in observations]

    def detect(self):
        return [dict(entry) for entry in self._observations]
