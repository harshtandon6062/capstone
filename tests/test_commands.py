from commands import CommandInvoker, PickPlaceCommand


class FakeRobotController:
    def __init__(self, position=(1, 2, 3), orientation=(0, 0, 0, 1), succeeds=True):
        self.position = list(position)
        self.orientation = list(orientation)
        self.succeeds = succeeds
        self.reverse_calls = []

    def get_object_state(self, object_id):
        return self.position, self.orientation

    def pick_and_place(self, object_id, destination):
        if not self.succeeds:
            return False
        self.position = [destination[0], destination[1], self.position[2]]
        return True

    def reverse_pick_and_place(self, object_id, position, orientation):
        self.reverse_calls.append((object_id, list(position), list(orientation)))
        if not self.succeeds:
            return False
        self.position = list(position)
        self.orientation = list(orientation)
        return True


def test_successful_pick_place_is_undoable():
    controller = FakeRobotController()
    invoker = CommandInvoker()
    command = PickPlaceCommand(7, [9, 8, 0], controller)

    assert invoker.execute(command)
    assert invoker.can_undo
    assert controller.position == [9, 8, 3]
    assert invoker.undo()
    assert controller.position == [1, 2, 3]
    assert controller.reverse_calls == [(7, [1, 2, 3], [0, 0, 0, 1])]
    assert not invoker.can_undo


def test_failed_command_is_not_added_to_history():
    invoker = CommandInvoker()
    command = PickPlaceCommand(7, [9, 8, 0], FakeRobotController(succeeds=False))

    assert not invoker.execute(command)
    assert not invoker.can_undo
    assert not invoker.undo()


def test_failed_undo_remains_available():
    controller = FakeRobotController(succeeds=True)
    invoker = CommandInvoker()
    command = PickPlaceCommand(7, [9, 8, 0], controller)

    assert invoker.execute(command)
    controller.succeeds = False
    assert not invoker.undo()
    assert invoker.can_undo