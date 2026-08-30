from .base_command import Command


class PourCommand(Command):
    """Transfer the contents of one tube into another.

    This is the action that cannot be taken back. No sequence of robot motions
    separates two mixed samples again, so the command does not offer a fake undo
    - and executing it invalidates every earlier move as well, because the world
    those moves would return to no longer exists.

    The transfer itself is handed in as a callback so this stays independent of
    how the workspace is represented.
    """

    reversible = False
    description = "Pour"

    def __init__(self, source_object, target_object, return_position,
                 robot_controller, transfer):
        self.source_object = source_object
        self.target_object = target_object
        self.return_position = list(return_position)
        self.robot_controller = robot_controller
        self.transfer = transfer
        self._executed = False

    def _finish(self, completed):
        if completed:
            self._executed = True
            self.transfer(self.source_object, self.target_object)
        return completed

    def execute(self):
        if self._executed:
            return False
        return self._finish(
            self.robot_controller.pour(
                self.source_object, self.target_object, self.return_position
            )
        )

    def execute_steps(self):
        if self._executed:
            return False
        completed = yield from self.robot_controller.pour_steps(
            self.source_object, self.target_object, self.return_position
        )
        return self._finish(completed)

    def undo(self):
        """Refused, always. Saying no is the honest answer here."""
        return False

    def undo_steps(self):
        return False
        yield  # never reached; makes this a generator like the other commands
