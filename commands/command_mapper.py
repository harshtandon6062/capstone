from .pick_place_command import PickPlaceCommand


class CommandMapper:
    """Translate a confirmed application action into a command object."""

    def __init__(self, robot_controller):
        self.robot_controller = robot_controller

    def pick_and_place(self, source_object, destination):
        return PickPlaceCommand(source_object, destination, self.robot_controller)
