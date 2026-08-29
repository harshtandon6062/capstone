from abc import ABC, abstractmethod


class Command(ABC):
    """Executable application action with an inverse operation."""

    @abstractmethod
    def execute(self):
        """Run the action and return True only when it completes."""

    @abstractmethod
    def undo(self):
        """Reverse a previously completed action."""
