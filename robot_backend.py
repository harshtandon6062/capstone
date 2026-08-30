"""The contract between the application and whatever arm is actually moving.

Commands, the state machine and the UI only ever talk to something that looks
like this. `RobotController` implements it against PyBullet; `NiryoArmController`
implements it against a physical Niryo Ned 2. Neither knows about the other.

Every action appears as a `*_steps` generator that yields regularly while the arm
is moving. That is not a stylistic choice: it is what lets the application loop
keep reading the camera during a motion, which is the only reason an operator can
gesture a stop while the arm is in flight rather than only between actions.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class RobotBackend(Protocol):
    """What an arm has to provide for the application to drive it."""

    def get_object_state(self, object_id):
        """Return (position, orientation) for an object in the workspace."""

    @property
    def is_holding(self):
        """True while the gripper is closed on something."""

    def pick_and_place_steps(self, source_object, destination):
        """Generator. Yields while moving; returns True only if the object was placed."""

    def reverse_pick_and_place_steps(self, source_object, saved_position,
                                     saved_orientation):
        """Generator. Put an object back where it started."""

    def pour_steps(self, source_object, target_object, return_position):
        """Generator. Tip one container into another, then set the first one back."""

    def hover_over_steps(self, x, y):
        """Generator. Point at a workspace position without touching it.

        This is how the operator learns which real object the panel means before
        confirming anything. On hardware it is the arm parking above the object;
        the mechanism differs, the promise does not.
        """

    def abort_safely_steps(self):
        """Generator. Give up on the current action without dropping what is held."""


def conforms(candidate):
    """Whether an object provides every operation the application needs.

    isinstance against a runtime_checkable Protocol does not check properties or
    signatures, so this spells the requirement out rather than trusting it.
    """
    required = (
        "get_object_state",
        "is_holding",
        "pick_and_place_steps",
        "reverse_pick_and_place_steps",
        "pour_steps",
        "hover_over_steps",
        "abort_safely_steps",
    )
    return all(hasattr(candidate, name) for name in required)
