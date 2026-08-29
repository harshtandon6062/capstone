"""What the operator can do while the arm is actually moving.

The application loop used to block inside one call for the whole of a move, so
no camera frame was read and a stop gesture could not be seen until the motion
had already finished. Motions are generators now; these tests pin down what that
buys and what it must not break.
"""

import pytest

from config import GRAB_Z
from commands import CommandInvoker, PickPlaceCommand
from robot_controller import RobotController
from safety_controller import EmergencyStopError, SafetyController
from test_robot_controller import RecordingPhysics, make_controller


def drive_until_holding(controller, runner, limit=5000):
    for _ in range(limit):
        next(runner)
        if controller.is_holding:
            return True
    return False


def test_a_move_yields_once_per_simulation_step():
    """Each yield is a chance for the loop to read the camera."""
    physics, controller = make_controller()
    runner = controller.move_to_steps([0.9, -0.2, 0.97], True, steps=25)

    yields = 0
    while True:
        try:
            next(runner)
        except StopIteration:
            break
        yields += 1

    assert yields == 25
    assert physics.steps == 25


def test_pause_holds_the_arm_without_advancing_the_simulation():
    physics, controller = make_controller()
    runner = controller.move_to_steps([0.9, -0.2, 0.97], True, steps=100)
    next(runner)

    controller.safety.pause()
    before = physics.steps
    for _ in range(20):
        next(runner)
    assert physics.steps == before, "a paused motion must not advance physics"

    controller.safety.resume()
    for _ in range(5):
        next(runner)
    assert physics.steps > before, "resuming must let the motion continue"


def test_emergency_stop_reaches_a_move_already_in_flight():
    physics, controller = make_controller()
    runner = controller.pick_and_place_steps(99, [0.95, -0.21, 0.65])
    next(runner)

    controller.safety.emergency_stop()
    with pytest.raises(EmergencyStopError):
        for _ in range(50):
            next(runner)


def test_a_stop_does_not_make_the_gripper_drop_the_sample():
    """A real arm brakes on a stop. It does not open its hand."""
    physics, controller = make_controller()
    runner = controller.pick_and_place_steps(99, [0.95, -0.21, 0.65])
    assert drive_until_holding(controller, runner), "never reached the grasp"

    controller.safety.emergency_stop()
    with pytest.raises(EmergencyStopError):
        for _ in range(50):
            next(runner)

    assert controller.is_holding, "the stop released the object"
    assert physics.constraints, "the object was detached by the stop"


def test_abort_sets_the_sample_down_instead_of_dropping_it():
    physics, controller = make_controller()
    runner = controller.pick_and_place_steps(99, [0.95, -0.21, 0.65])
    assert drive_until_holding(controller, runner)
    controller.safety.emergency_stop()
    with pytest.raises(EmergencyStopError):
        for _ in range(50):
            next(runner)

    controller.safety.reset_emergency_stop()
    physics.targets.clear()
    assert controller.abort_safely()

    assert not controller.is_holding, "abort must end with the object released"
    descents = [target for target in physics.targets if target[2] == GRAB_Z]
    assert descents, "the arm must lower to table height before opening"

    runner.close()
    assert physics.constraints == [], "abandoning the motion must leave nothing attached"


def test_abort_is_a_no_op_when_the_gripper_is_empty():
    physics, controller = make_controller()
    physics.targets.clear()
    assert controller.abort_safely() is True
    assert physics.targets == [], "nothing held means no motion"


def test_command_generator_records_history_exactly_like_the_blocking_form():
    physics, controller = make_controller()
    invoker = CommandInvoker()
    command = PickPlaceCommand(99, [0.95, -0.21, 0.65], controller)

    runner = invoker.execute_steps(command)
    completed = None
    while completed is None:
        try:
            next(runner)
        except StopIteration as stop:
            completed = stop.value

    assert completed is True
    assert invoker.can_undo
