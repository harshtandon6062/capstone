"""Pouring is the action that cannot be taken back.

Everything else in the system is undoable, which quietly undermines the reason
for having a confirmation step at all. These tests pin down that pouring behaves
differently in the ways that matter: it refuses to offer an undo, it invalidates
the history under it, and it changes the workspace irreversibly.
"""

from commands import CommandInvoker, PickPlaceCommand, PourCommand
from object_registry import EMPTY_LIQUID_RGBA, ObjectRegistry, mix_colors
from perception import StaticPerception


def observation(handle, name, rgba, kind="source"):
    return {
        "handle": handle,
        "label": f"{name.title()} tube",
        "color_name": name,
        "color_rgba": rgba,
        "position": [0.75, -0.45 + handle * 0.12, 0.65],
        "kind": kind,
    }


def make_registry():
    return ObjectRegistry(StaticPerception([
        observation(0, "RED", [1.0, 0.0, 0.0, 1.0]),
        observation(1, "BLUE", [0.0, 0.0, 1.0, 1.0]),
        observation(2, "GREEN", [0.0, 0.7, 0.0, 1.0]),
    ]))


class FakePourController:
    def __init__(self, succeeds=True):
        self.succeeds = succeeds
        self.calls = []

    def get_object_state(self, object_id):
        return [0.75, -0.45, 0.65], [0, 0, 0, 1]

    def pour(self, source_object, target_object, return_position):
        self.calls.append((source_object, target_object, list(return_position)))
        return self.succeeds

    def pour_steps(self, source_object, target_object, return_position):
        self.calls.append((source_object, target_object, list(return_position)))
        yield
        return self.succeeds

    def pick_and_place(self, source_object, destination):
        return True

    def reverse_pick_and_place(self, source_object, position, orientation):
        return True


def test_pouring_empties_the_source_and_mixes_the_target():
    registry = make_registry()
    registry.transfer_contents(0, 1)

    emptied = registry.by_handle(0)
    mixed = registry.by_handle(1)

    assert emptied.empty, "the tube that was poured out still reads as full"
    assert emptied.color_rgba == EMPTY_LIQUID_RGBA
    assert mixed.color_rgba == mix_colors([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0])
    assert mixed.color_rgba != [1.0, 0.0, 0.0, 1.0]
    assert mixed.color_rgba != [0.0, 0.0, 1.0, 1.0]


def test_an_emptied_tube_is_still_something_you_can_move():
    """It has nothing left to pour, but it is still a tube on a table."""
    registry = make_registry()
    registry.transfer_contents(0, 1)
    movable = [obj.handle for obj in registry.selectable("move_source")]
    assert 0 in movable


def test_a_tube_cannot_be_poured_into_itself():
    registry = make_registry()
    assert registry.transfer_contents(0, 0) is None
    assert registry.next_selectable("pour_target", 2, 1, exclude=0) == 1
    assert registry.first_selectable("pour_target", exclude=0) == 1


def test_only_tubes_with_contents_can_be_the_source_of_a_pour():
    registry = make_registry()
    registry.transfer_contents(0, 1)      # 0 is now empty
    registry.transfer_contents(2, 1)      # 2 is now empty too

    pourable = [obj.handle for obj in registry.selectable("pour_source")]
    assert pourable == [1], "only the tube holding everything can still pour"
    assert registry.first_selectable("pour_source") == 1


def test_pour_command_refuses_to_offer_an_undo():
    controller = FakePourController()
    command = PourCommand(0, 1, [0.75, -0.45, 0.65], controller, lambda a, b: None)

    assert command.reversible is False
    assert command.execute()
    assert command.undo() is False, "there is no motion that unmixes two samples"


def test_pour_is_never_added_to_the_undo_history():
    controller = FakePourController()
    invoker = CommandInvoker()

    assert invoker.execute(PourCommand(0, 1, [0, 0, 0], controller, lambda a, b: None))
    assert not invoker.can_undo


def test_pouring_invalidates_the_moves_that_came_before_it():
    """You cannot undo a move back into a world that no longer exists."""
    controller = FakePourController()
    invoker = CommandInvoker()

    assert invoker.execute(PickPlaceCommand(7, [0.95, -0.2, 0.65], controller))
    assert invoker.can_undo

    assert invoker.execute(PourCommand(0, 1, [0, 0, 0], controller, lambda a, b: None))
    assert not invoker.can_undo, "the earlier move must not still look reversible"


def test_contents_only_transfer_when_the_motion_succeeded():
    controller = FakePourController(succeeds=False)
    transfers = []
    command = PourCommand(0, 1, [0, 0, 0], controller, lambda a, b: transfers.append((a, b)))

    assert command.execute() is False
    assert transfers == [], "a failed pour must not change the workspace"


def test_generator_form_transfers_exactly_once():
    controller = FakePourController()
    transfers = []
    command = PourCommand(0, 1, [0, 0, 0], controller, lambda a, b: transfers.append((a, b)))

    runner = command.execute_steps()
    result = None
    while result is None:
        try:
            next(runner)
        except StopIteration as stop:
            result = stop.value

    assert result is True
    assert transfers == [(0, 1)]
