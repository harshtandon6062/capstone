from .command_history import CommandHistory


class CommandInvoker:
    """Executes commands and records only successful executions."""

    def __init__(self, history=None):
        self.history = history or CommandHistory()
        self.last_undone_command = None

    def execute(self, command):
        completed = command.execute()
        if completed:
            self._record(command)
        return completed

    def execute_steps(self, command):
        """Generator form, driven a few simulation steps at a time by the app loop."""
        completed = yield from command.execute_steps()
        if completed:
            self._record(command)
        return completed

    def _record(self, command):
        # An irreversible action must not sit on the undo stack pretending it can
        # be taken back. It also invalidates everything under it: you cannot undo
        # a move that happened before a sample was consumed.
        if command.reversible:
            self.history.push(command)
        else:
            self.history.clear()

    def undo(self):
        if not self.history.can_undo:
            return False
        self.last_undone_command = self.history.last
        return self.history.undo()

    def undo_steps(self):
        if not self.history.can_undo:
            return False
        self.last_undone_command = self.history.last
        return (yield from self.history.undo_steps())

    @property
    def can_undo(self):
        return self.history.can_undo

    def clear_history(self):
        self.history.clear()
