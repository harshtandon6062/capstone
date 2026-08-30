from .base_command import Command


class MixCommand(Command):
    """Rotate a held tube in place to simulate mixing while staying reversible in the UI.

    The user-visible command is "mix" and it is irreversible in practice because
    it changes the tube contents, but the underlying robot motion is still an
    in-place wrist rotation. This keeps the command API consistent with the other
    actions without requiring a separate undo path.
    """

    reversible = False
    description = "Mix"

    def __init__(self, source_object, robot_controller):
        self.source_object = source_object
        self.robot_controller = robot_controller
        self.transfer = None
        self._executed = False

    def _finish(self, completed):
        if completed:
            self._executed = True
            if self.transfer is not None:
                self.transfer(self.source_object)
        return completed

    def execute(self):
        if self._executed:
            return False
        return self._finish(self.robot_controller.mix_tube(self.source_object))

    def execute_steps(self):
        if self._executed:
            return False
        completed = yield from self.robot_controller.mix_tube_steps(self.source_object)
        return self._finish(completed)

    def undo(self):
        return False

    def undo_steps(self):
        return False
        yield  # never reached; makes this a generator like the other commands
