from .mix_command import MixCommand
from .pick_place_command import PickPlaceCommand
from .pour_command import PourCommand


class CommandMapper:
    """Translate a confirmed application action into a command object."""

    def __init__(self, robot_controller):
        self.robot_controller = robot_controller

    def pick_and_place(self, source_object, destination):
        return PickPlaceCommand(source_object, destination, self.robot_controller)

    def mix(self, source_object):
        return MixCommand(source_object, self.robot_controller)

    def pour(self, source_object, target_object, return_position, transfer):
        return PourCommand(
            source_object,
            target_object,
            return_position,
            self.robot_controller,
            transfer,
        )
