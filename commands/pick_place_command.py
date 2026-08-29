from .base_command import Command


class PickPlaceCommand(Command):
    """Pick one object and place it at a destination through a robot controller."""

    def __init__(self, source_object, destination, robot_controller):
        self.source_object = source_object
        self.destination = destination
        self.robot_controller = robot_controller
        self.saved_position = None
        self.saved_orientation = None
        self._executed = False

    description = "Move"

    def _remember_start(self):
        position, orientation = self.robot_controller.get_object_state(self.source_object)
        self.saved_position = list(position)
        self.saved_orientation = list(orientation)

    def execute(self):
        if self._executed:
            return False
        self._remember_start()
        completed = self.robot_controller.pick_and_place(
            self.source_object,
            self.destination,
        )
        if completed:
            self._executed = True
        return completed

    def execute_steps(self):
        if self._executed:
            return False
        self._remember_start()
        completed = yield from self.robot_controller.pick_and_place_steps(
            self.source_object,
            self.destination,
        )
        if completed:
            self._executed = True
        return completed

    def undo(self):
        if not self._executed or self.saved_position is None:
            return False
        restored = self.robot_controller.reverse_pick_and_place(
            self.source_object,
            self.saved_position,
            self.saved_orientation,
        )
        if restored:
            self._executed = False
        return restored

    def undo_steps(self):
        if not self._executed or self.saved_position is None:
            return False
        restored = yield from self.robot_controller.reverse_pick_and_place_steps(
            self.source_object,
            self.saved_position,
            self.saved_orientation,
        )
        if restored:
            self._executed = False
        return restored
