from abc import ABC, abstractmethod


class Command(ABC):
    """Executable application action with an inverse operation.

    Each action exists twice: a blocking form, and a `*_steps` generator that
    yields once per simulation step. The application loop drives the generator
    so it can keep reading the camera while the robot moves; everything else
    uses the blocking form.
    """

    @abstractmethod
    def execute(self):
        """Run the action and return True only when it completes."""

    @abstractmethod
    def undo(self):
        """Reverse a previously completed action."""

    @abstractmethod
    def execute_steps(self):
        """Generator form of execute(). Returns the same True/False."""

    @abstractmethod
    def undo_steps(self):
        """Generator form of undo(). Returns the same True/False."""

    # Whether undo() can put the world back the way it was. Pick-and-place can.
    # Anything that mixes or transfers a sample cannot, and the operator has to
    # be told that before they commit, not after.
    reversible = True

    # Shown to the operator when confirming.
    description = "Action"
