class CommandHistory:
    """LIFO history of successfully executed commands."""

    def __init__(self):
        self._undo_stack = []

    @property
    def can_undo(self):
        return bool(self._undo_stack)

    def push(self, command):
        self._undo_stack.append(command)

    def undo(self):
        if not self._undo_stack:
            return False
        command = self._undo_stack[-1]
        if not command.undo():
            return False
        self._undo_stack.pop()
        return True

    def clear(self):
        self._undo_stack.clear()
