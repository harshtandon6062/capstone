"""Centralized safety state and robot motion gate."""

from enum import Enum
import time


class SafetyState(Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


class EmergencyStopError(RuntimeError):
    """Raised when a safety stop interrupts an active robot operation."""


class SafetyController:
    """Owns safety transitions and prevents robot commands while unsafe."""

    def __init__(self, physics, robot_id=None, gripper_id=None, poll_every=4,
                 hold_force=500.0):
        self.physics = physics
        self.robot_id = robot_id
        self.gripper_id = gripper_id
        self.state = SafetyState.RUNNING
        self.quit_requested = False
        # Torque the joints hold with when stopped. A real arm engages brakes on
        # an emergency stop; it does not drop whatever it is carrying.
        self.hold_force = hold_force
        # How often to read operator input while stepping. Every step is
        # wasteful, never is unsafe.
        self.poll_every = max(1, int(poll_every))
        self._poll_counter = 0
        self.estop_interrupted = False

    @property
    def state_name(self):
        return self.state.value

    @property
    def motion_allowed(self):
        return self.state is SafetyState.RUNNING

    def attach_robot(self, robot_id, gripper_id):
        self.robot_id = robot_id
        self.gripper_id = gripper_id

    def pause(self):
        if self.state is SafetyState.RUNNING:
            self.state = SafetyState.PAUSED
        return self.state

    def resume(self):
        # An emergency stop is latched and cannot be resumed by a pause toggle.
        if self.state is SafetyState.PAUSED:
            self.state = SafetyState.RUNNING
        return self.state

    def toggle_pause(self):
        if self.state is SafetyState.RUNNING:
            return self.pause()
        if self.state is SafetyState.PAUSED:
            return self.resume()
        return self.state

    def emergency_stop(self):
        if self.state is not SafetyState.EMERGENCY_STOPPED:
            self.state = SafetyState.EMERGENCY_STOPPED
            # Latched until someone reads it. Clearing the stop must not silently
            # resume whatever action was pending when it fired.
            self.estop_interrupted = True
            self._stop_robot()
        return self.state

    def reset_emergency_stop(self):
        # Re-enable requires an explicit action; normal resume cannot clear it.
        if self.state is SafetyState.EMERGENCY_STOPPED:
            # Re-assert control at the current pose before releasing the gate, so
            # the arm holds where it is instead of jumping to a stale target.
            self._hold_current_pose()
            self.state = SafetyState.RUNNING
        return self.state

    def step_if_running(self):
        """Advance the simulation only when running. Never blocks.

        The main loop uses this so that a pause keeps the camera and gesture
        pipeline alive. Blocking there would freeze the webcam and make a
        gesture-triggered pause impossible to release by gesture.
        """
        if self.state is SafetyState.RUNNING:
            self.physics.stepSimulation()
            return True
        return False

    def consume_estop_interrupt(self):
        """Return True once after an emergency stop, so callers can re-arm safely."""
        interrupted = self.estop_interrupted
        self.estop_interrupted = False
        return interrupted

    def handle_key(self, key):
        if key == ord("q"):
            self.quit_requested = True
        elif key == ord("p"):
            self.toggle_pause()
        elif key == ord("x"):
            self.emergency_stop()
        elif key == ord("e"):
            self.reset_emergency_stop()
        return self.state

    def handle_gesture(self, gesture, previous_gesture):
        if gesture == previous_gesture:
            return self.state
        if gesture == "open_palm":
            self.toggle_pause()
        elif gesture == "thumbs_down":
            self.emergency_stop()
        return self.state

    def require_motion(self):
        if self.state is SafetyState.EMERGENCY_STOPPED:
            raise EmergencyStopError("Emergency stop is active; press E to reset.")
        return self.motion_allowed

    def wait_until_running(self, poll=None):
        while self.state is SafetyState.PAUSED:
            if poll is not None:
                poll()
            if self.state is SafetyState.EMERGENCY_STOPPED:
                raise EmergencyStopError("Emergency stop interrupted paused motion.")
            if self.quit_requested:
                return False
            time.sleep(0.01)
        self.require_motion()
        return not self.quit_requested

    def step_simulation(self, poll=None, raise_on_stop=False):
        # Poll before deciding anything. A long motion runs entirely inside this
        # method while the state is RUNNING, so if operator input were only read
        # in the paused branch of wait_until_running(), pause and emergency stop
        # would be unreachable during exactly the motion they exist to interrupt.
        if poll is not None:
            self._poll_counter += 1
            if self._poll_counter >= self.poll_every:
                self._poll_counter = 0
                poll()
        try:
            if not self.wait_until_running(poll):
                return False
        except EmergencyStopError:
            if raise_on_stop:
                raise
            return False
        self.physics.stepSimulation()
        return True

    def _stop_robot(self):
        """Brake: hold every joint at the angle it is at right now.

        Commanding zero force instead removes all motor torque, which makes the
        arm go limp and collapse under gravity - and because nothing restores
        control afterwards, it stays limp once the stop is cleared.
        """
        self._hold_current_pose()

    def _hold_current_pose(self):
        for body in (self.robot_id, self.gripper_id):
            if body is None:
                continue
            for joint in range(self.physics.getNumJoints(body)):
                angle = self.physics.getJointState(body, joint)[0]
                self.physics.setJointMotorControl2(
                    body,
                    joint,
                    self.physics.POSITION_CONTROL,
                    targetPosition=angle,
                    targetVelocity=0,
                    force=self.hold_force,
                )
