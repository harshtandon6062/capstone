"""The hardware adapter, exercised against a fake arm.

None of this proves the code works on a real Ned 2 - it cannot, without one. What
it does pin down is the behaviour that would be dangerous to get wrong: that a
stop actually prevents further waypoints, that long moves are cut into short
segments so a stop takes effect promptly, and that nothing ever cuts motor torque
to stop the arm, because that drops whatever is being carried.
"""

import math

import pytest

import robot.backend as robot_backend
from robot.niryo import (
    SEGMENT_LENGTH,
    BlockingCall,
    NiryoArmController,
    WorkspaceTransform,
)
from workspace.registry import ObjectRegistry
from workspace.perception import StaticPerception
from robot.controller import RobotController
from robot.safety import SafetyController
from test_robot_controller import RecordingPhysics


class Pose:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class FakeArm:
    """Stands in for pyniryo's NiryoRobot. Records what it was asked to do."""

    def __init__(self):
        self.waypoints = []
        self.gripper = []
        self.learning_mode_calls = []
        self.pose = Pose(0.25, 0.0, 0.10)

    def get_pose(self):
        return self.pose

    def move_pose(self, x, y, z, roll, pitch, yaw):
        self.waypoints.append((x, y, z, roll, pitch, yaw))
        self.pose = Pose(x, y, z)

    def open_gripper(self):
        self.gripper.append("open")

    def close_gripper(self):
        self.gripper.append("close")

    def set_learning_mode(self, enabled):
        self.learning_mode_calls.append(enabled)

    def close_connection(self):
        pass


def make_controller():
    registry = ObjectRegistry(StaticPerception([
        {"handle": 1, "label": "Red tube", "color_name": "RED",
         "color_rgba": [1, 0, 0, 1], "position": [0.75, -0.45, 0.625], "kind": "source"},
        {"handle": 2, "label": "Blue tube", "color_name": "BLUE",
         "color_rgba": [0, 0, 1, 1], "position": [0.75, -0.09, 0.625], "kind": "source"},
    ]))
    arm = FakeArm()
    safety = SafetyController(RecordingPhysics())
    return arm, NiryoArmController(arm, registry, safety)


def advance(runner, count):
    """Step a generator, tolerating it finishing early."""
    for _ in range(count):
        try:
            next(runner)
        except StopIteration:
            return False
    return True


def drain(runner):
    """Run a generator to completion, counting how often it yielded."""
    yields = 0
    while True:
        try:
            next(runner)
            yields += 1
        except StopIteration as stop:
            return stop.value, yields


def test_workspace_transform_round_trips():
    transform = WorkspaceTransform()
    table = [0.75, -0.33, 0.625]
    assert transform.to_table(transform.to_robot(table)) == pytest.approx(table)


def test_transform_puts_the_table_origin_at_the_robots_working_point():
    transform = WorkspaceTransform()
    assert transform.to_robot(list(transform.origin)) == pytest.approx(
        list(transform.robot_origin)
    )


def test_a_long_move_is_cut_into_short_segments():
    """Stop latency on hardware is bounded by one segment, so they must be short."""
    arm, controller = make_controller()
    result, _ = drain(controller.move_to_steps([0.75, 0.03, 0.625]))

    assert result is True
    assert len(arm.waypoints) > 1, "a long move must not be one blocking command"

    previous = (arm.waypoints[0][0], arm.waypoints[0][1], arm.waypoints[0][2])
    for waypoint in arm.waypoints[1:]:
        step = math.dist(previous, waypoint[:3])
        assert step <= SEGMENT_LENGTH + 1e-9, f"segment of {step:.3f} m is too long"
        previous = waypoint[:3]


def test_a_stop_prevents_any_further_waypoints():
    arm, controller = make_controller()
    runner = controller.move_to_steps([0.75, 0.03, 0.625])

    # One yield reads the current pose, the next sends the first segment.
    assert advance(runner, 2), "the move finished before it could be stopped"
    sent_when_stopped = len(arm.waypoints)
    assert sent_when_stopped >= 1
    controller.safety.emergency_stop()

    result, _ = drain(runner)
    assert result is False, "a stopped move must not report success"
    assert len(arm.waypoints) == sent_when_stopped, (
        "waypoints were still sent after the operator stopped the arm"
    )


def test_pausing_also_holds_the_arm_at_the_next_segment():
    arm, controller = make_controller()
    runner = controller.move_to_steps([0.75, 0.03, 0.625])
    assert advance(runner, 2)
    controller.safety.pause()
    sent = len(arm.waypoints)

    result, _ = drain(runner)
    assert result is False
    assert len(arm.waypoints) == sent


def test_the_arm_is_never_stopped_by_cutting_motor_torque():
    """set_learning_mode(True) makes a Ned 2 go limp and drop its payload."""
    arm, controller = make_controller()
    runner = controller.pick_and_place_steps(1, [0.75, -0.09, 0.625])
    advance(runner, 10)
    controller.safety.emergency_stop()
    drain(runner)

    assert arm.learning_mode_calls == [], "torque must never be cut to stop the arm"


def test_motion_yields_at_least_once_per_waypoint():
    """How often the operator is looked at must not depend on how fast the arm is.

    The fake arm answers instantly, which is the worst case for this: if yielding
    only happened while a call was still running, a fast arm would starve the
    camera completely.
    """
    arm, controller = make_controller()
    _, yields = drain(controller.move_to_steps([0.75, 0.03, 0.625]))

    assert arm.waypoints, "the move sent nothing"
    assert yields >= len(arm.waypoints), (
        "the loop got control back less often than the arm was commanded"
    )


def test_abort_sets_down_what_is_held_and_opens_there():
    arm, controller = make_controller()
    drain(controller._grasp_steps(1))
    assert controller.is_holding

    arm.waypoints.clear()
    result, _ = drain(controller.abort_safely_steps())

    assert result is True
    assert not controller.is_holding
    assert arm.gripper[-1] == "open"
    assert arm.waypoints, "the arm must move down before letting go"


def test_abort_does_nothing_when_the_gripper_is_empty():
    arm, controller = make_controller()
    result, _ = drain(controller.abort_safely_steps())
    assert result is True
    assert arm.waypoints == []


def test_blocking_call_reports_a_failure_rather_than_swallowing_it():
    def explode():
        raise RuntimeError("arm unreachable")

    call = BlockingCall(explode)
    while not call.finished:
        pass
    assert isinstance(call.error, RuntimeError)


def test_both_backends_satisfy_the_same_contract():
    physics = RecordingPhysics()
    simulated = RobotController(
        physics, 1, 2, SafetyController(physics), [0.0] * 7, [0.0] * 8
    )
    _, hardware = make_controller()

    assert robot_backend.conforms(simulated)
    assert robot_backend.conforms(hardware)


def test_hovering_on_hardware_stops_above_the_object():
    """With no rendered scene to draw a marker into, the arm does the pointing.

    It must stop short of the object: this is how the operator checks the machine
    understood them, and it happens before anything is confirmed.
    """
    arm, controller = make_controller()
    result, _ = drain(controller.hover_over_steps(0.75, -0.45))

    assert result is True
    assert arm.waypoints, "hovering commanded no motion"
    lowest = min(waypoint[2] for waypoint in arm.waypoints)
    table = controller.transform.to_robot([0.75, -0.45, controller.transform.origin[2]])
    assert lowest > table[2], "the arm descended to the object instead of over it"
    assert arm.gripper == [], "pointing must not open or close the gripper"


def test_hovering_respects_a_stop_like_any_other_motion():
    arm, controller = make_controller()
    runner = controller.hover_over_steps(0.75, 0.03)
    assert advance(runner, 2)
    sent = len(arm.waypoints)
    controller.safety.emergency_stop()

    result, _ = drain(runner)
    assert result is False
    assert len(arm.waypoints) == sent
