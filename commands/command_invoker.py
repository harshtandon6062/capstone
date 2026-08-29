from .command_history import CommandHistory


class CommandInvoker:
    """Executes commands and records only successful executions."""

    def __init__(self, history=None):
        self.history = history or CommandHistory()
        self.last_undone_command = None

    def execute(self, command):
        completed = command.execute()
        if completed:
            self.history.push(command)
        return completed

    def undo(self):
        if not self.history.can_undo:
            return False
        self.last_undone_command = self.history._undo_stack[-1]
        return self.history.undo()

    @property
    def can_undo(self):
        return self.history.can_undo

    def clear_history(self):
        self.history.clear()
