import pytest

from safety_controller import EmergencyStopError, SafetyController, SafetyState


class FakePhysics:
    VELOCITY_CONTROL = 0
    POSITION_CONTROL = 1

    def __init__(self):
        self.steps = 0
        self.stopped = []
        self.commands = []
        self.angles = {}

    def getNumJoints(self, body):
        return 7

    def getJointState(self, body, joint):
        return (self.angles.get((body, joint), 0.25 * joint), 0.0, None, None)

    def setJointMotorControl2(self, body, joint, mode, **kwargs):
        self.stopped.append((body, joint))
        self.commands.append({"body": body, "joint": joint, "mode": mode, **kwargs})

    def stepSimulation(self):
        self.steps += 1


def make(poll_every=1):
    physics = FakePhysics()
    return physics, SafetyController(physics, robot_id=1, gripper_id=2, poll_every=poll_every)


def test_pause_and_resume_toggle():
    _, safety = make()
    assert safety.toggle_pause() is SafetyState.PAUSED
    assert safety.toggle_pause() is SafetyState.RUNNING


def test_emergency_stop_is_latched_against_resume():
    _, safety = make()
    safety.emergency_stop()
    safety.resume()
    safety.toggle_pause()
    assert safety.state is SafetyState.EMERGENCY_STOPPED
    safety.reset_emergency_stop()
    assert safety.state is SafetyState.RUNNING


def test_emergency_stop_brakes_both_bodies():
    physics, safety = make()
    safety.emergency_stop()
    assert {1, 2} <= {body for body, _ in physics.stopped}


def test_emergency_stop_holds_position_instead_of_going_limp():
    """Regression: force=0 removed all torque and the arm collapsed under gravity."""
    physics, safety = make()
    safety.emergency_stop()

    assert physics.commands, "stop must command the joints"
    for command in physics.commands:
        assert command["mode"] == physics.POSITION_CONTROL, "must brake, not free-spin"
        assert command["force"] > 0, "zero force lets gravity drop the arm"
        assert command["targetVelocity"] == 0


def test_stop_holds_each_joint_at_its_current_angle():
    physics, safety = make()
    physics.angles[(1, 3)] = 1.234
    safety.emergency_stop()

    held = [c for c in physics.commands if c["body"] == 1 and c["joint"] == 3]
    assert held and held[0]["targetPosition"] == 1.234


def test_clearing_the_stop_reasserts_control():
    """Regression: reset only flipped the enum, so the arm stayed limp forever."""
    physics, safety = make()
    safety.emergency_stop()
    physics.commands.clear()

    safety.reset_emergency_stop()
    assert safety.state is SafetyState.RUNNING
    assert physics.commands, "clearing the stop must re-establish joint control"
    assert all(c["mode"] == physics.POSITION_CONTROL for c in physics.commands)


def test_step_if_running_never_blocks_while_paused():
    """The main loop uses this so the camera keeps updating while paused."""
    physics, safety = make()
    safety.pause()
    assert safety.step_if_running() is False
    assert physics.steps == 0

    safety.resume()
    assert safety.step_if_running() is True
    assert physics.steps == 1


def test_step_if_running_does_nothing_when_stopped():
    physics, safety = make()
    safety.emergency_stop()
    assert safety.step_if_running() is False
    assert physics.steps == 0


def test_operator_input_is_polled_while_running():
    """The stop must be reachable during motion, not only while already paused."""
    physics, safety = make(poll_every=1)
    calls = []
    for _ in range(10):
        safety.step_simulation(lambda: calls.append(1), raise_on_stop=True)
    assert len(calls) == 10
    assert physics.steps == 10


def test_poll_every_throttles_without_silencing():
    physics, safety = make(poll_every=4)
    calls = []
    for _ in range(12):
        safety.step_simulation(lambda: calls.append(1))
    assert len(calls) == 3


def test_stop_pressed_during_motion_aborts_the_step():
    physics, safety = make(poll_every=1)
    steps_before_stop = []

    def poll():
        steps_before_stop.append(physics.steps)
        if physics.steps >= 3:
            safety.emergency_stop()

    with pytest.raises(EmergencyStopError):
        for _ in range(10):
            safety.step_simulation(poll, raise_on_stop=True)

    assert physics.steps == 3, "simulation must stop advancing once the stop fires"


def test_estop_interrupt_is_reported_once():
    _, safety = make()
    assert safety.consume_estop_interrupt() is False
    safety.emergency_stop()
    assert safety.consume_estop_interrupt() is True
    assert safety.consume_estop_interrupt() is False


def test_quit_request_unblocks_a_paused_motion():
    _, safety = make()
    safety.pause()

    def poll():
        safety.quit_requested = True

    assert safety.step_simulation(poll) is False
