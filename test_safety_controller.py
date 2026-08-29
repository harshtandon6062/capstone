import pytest

from safety_controller import EmergencyStopError, SafetyController, SafetyState


class FakePhysics:
    VELOCITY_CONTROL = 0

    def __init__(self):
        self.steps = 0
        self.stopped = []

    def getNumJoints(self, body):
        return 7

    def setJointMotorControl2(self, body, joint, mode, **kwargs):
        self.stopped.append((body, joint))

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


def test_emergency_stop_zeroes_both_bodies():
    physics, safety = make()
    safety.emergency_stop()
    assert {1, 2} <= {body for body, _ in physics.stopped}


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
