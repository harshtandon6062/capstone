class CommandHistory:
    """LIFO history of successfully executed commands."""

    def __init__(self):
        self._undo_stack = []

    @property
    def can_undo(self):
        return bool(self._undo_stack)

    @property
    def last(self):
        return self._undo_stack[-1] if self._undo_stack else None

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

    def undo_steps(self):
        if not self._undo_stack:
            return False
        command = self._undo_stack[-1]
        undone = yield from command.undo_steps()
        if not undone:
            return False
        self._undo_stack.pop()
        return True

    def clear(self):
        self._undo_stack.clear()
