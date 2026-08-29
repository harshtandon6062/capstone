import math
import time

from config import GRAB_Z, GRASP_TOLERANCE, HOVER_Z, LIFT_Z, TARGET_EULER
from safety_controller import EmergencyStopError, SafetyState


class RobotController:
    """Backend-neutral robot operations used by application commands.

    Every motion exists in two forms. The `*_steps` generators yield once per
    simulation step and are what the application loop drives, so the camera and
    the gesture pipeline keep running for the whole of a move. That is what makes
    an emergency stop by gesture reachable *during* a motion rather than only
    between motions - previously the loop was blocked inside a single call for
    several seconds and no camera frame was read at all.

    The plain methods drain those generators and block, which is what tests and
    one-shot setup paths want.
    """

    def __init__(self, physics, kuka_id, gripper_id, safety, initial_kuka_positions,
                 initial_gripper_positions, safety_poll=None):
        self.physics = physics
        self.kuka_id = kuka_id
        self.gripper_id = gripper_id
        self.safety = safety
        self.initial_kuka_positions = initial_kuka_positions
        self.initial_gripper_positions = initial_gripper_positions
        self.safety_poll = safety_poll or (lambda: None)
        self.target_orientation = physics.getQuaternionFromEuler(TARGET_EULER)
        self.kuka_joint_count = physics.getNumJoints(kuka_id)
        # Owned here rather than inside a motion, so that a motion interrupted by
        # an emergency stop can be abandoned without the object it was carrying
        # falling out of the gripper.
        self._held_constraint = None

    def set_safety_poll(self, safety_poll):
        self.safety_poll = safety_poll

    def get_object_state(self, object_id):
        return self.physics.getBasePositionAndOrientation(object_id)

    # ── running a motion ────────────────────────────────────────────────────
    def _drain(self, runner):
        """Run a motion generator to completion without an application loop.

        A pause blocks here, exactly as it used to, because a caller that is not
        the application loop has nothing else it needs to be doing.
        """
        while True:
            try:
                next(runner)
            except StopIteration as stop:
                return stop.value
            if self.safety.state is SafetyState.PAUSED:
                self.safety_poll()
                time.sleep(0.01)

    def _step_for(self, count):
        """Advance the simulation, yielding after each step.

        Yielding is what hands control back to whoever is driving, so the
        operator stays visible to the system while the arm is moving. While
        paused this yields without stepping: the arm holds position and the
        driver decides when to come back.
        """
        remaining = count
        while remaining > 0:
            if self.safety.state is SafetyState.EMERGENCY_STOPPED:
                raise EmergencyStopError("Emergency stop interrupted the motion.")
            if self.safety.quit_requested:
                return False
            if self.safety.state is SafetyState.PAUSED:
                yield
                continue
            self.physics.stepSimulation()
            remaining -= 1
            yield
        return True

    # ── primitives ──────────────────────────────────────────────────────────
    def _drive_to_steps(self, target_position, gripper_open, steps):
        self.safety.require_motion()
        gripper_value = 0 if gripper_open else 1
        joint_positions = self.physics.calculateInverseKinematics(
            self.kuka_id, 6, target_position, self.target_orientation
        )
        for joint in range(self.kuka_joint_count):
            self.physics.setJointMotorControl2(
                self.kuka_id, joint, self.physics.POSITION_CONTROL, joint_positions[joint]
            )
        self.physics.setJointMotorControl2(
            self.gripper_id, 4, self.physics.POSITION_CONTROL, gripper_value * 0.05, force=100
        )
        self.physics.setJointMotorControl2(
            self.gripper_id, 6, self.physics.POSITION_CONTROL, gripper_value * 0.05, force=100
        )
        return (yield from self._step_for(steps))

    def move_to_steps(self, target_position, gripper_open, steps=150, precise=False):
        """Drive the tool to a target.

        With precise=True the move closes the loop: it measures where the tool
        actually ended up and aims off by the residual until the error is inside
        tolerance. Inverse kinematics plus a fixed number of position-control
        steps consistently undershoots by roughly 20 mm.
        """
        if not (yield from self._drive_to_steps(target_position, gripper_open, steps)):
            return False
        if precise:
            return (yield from self._refine_steps(target_position, gripper_open))
        return True

    def _refine_steps(self, target_position, gripper_open, tolerance=0.004,
                      attempts=3, steps=70):
        """Aim off by the measured residual until the tool is inside tolerance."""
        aim = list(target_position)
        for _ in range(attempts):
            actual = self.tool_position()
            error = [t - a for t, a in zip(target_position, actual)]
            if math.sqrt(sum(e * e for e in error)) <= tolerance:
                return True
            aim = [a + e for a, e in zip(aim, error)]
            if not (yield from self._drive_to_steps(aim, gripper_open, steps)):
                return False
        return True

    def approach_from_above_steps(self, x, y, gripper_open, steps=150):
        """Travel at clearance height first, then descend straight down.

        There is no path planning and no collision checking, so travelling
        directly to grasp height lets the arm sweep sideways through whatever is
        in the way and knock it over. Going up and over first keeps the
        horizontal leg of the motion above the objects.
        """
        if not (yield from self.move_to_steps([x, y, LIFT_Z], gripper_open, steps)):
            return False
        return (yield from self.move_to_steps(
            [x, y, HOVER_Z], gripper_open, max(60, steps // 2)
        ))

    def center_gripper_over_steps(self, x, y, gripper_open, attempts=4, steps=70,
                                  tolerance=0.004):
        """Aim the wrist so the gripper sits over (x, y). Returns the aim used.

        The gripper hangs off the wrist on a constraint, so its centre trails the
        inverse-kinematics target by a pose-dependent horizontal offset. Closing
        without correcting for that means the fingers meet the object off-centre
        and shove it sideways, which is what makes repeated pick and undo cycles
        walk an object across the table.
        """
        aim_x, aim_y = x, y
        for _ in range(attempts):
            grasp = self.grasp_position()
            error_x, error_y = x - grasp[0], y - grasp[1]
            if math.hypot(error_x, error_y) <= tolerance:
                return (aim_x, aim_y)
            aim_x += error_x
            aim_y += error_y
            if not (yield from self._drive_to_steps(
                [aim_x, aim_y, HOVER_Z], gripper_open, steps
            )):
                return None
        return (aim_x, aim_y)

    def reset_robot_steps(self):
        self.safety.require_motion()
        for joint in range(self.kuka_joint_count):
            position = self.initial_kuka_positions[joint]
            self.physics.resetJointState(self.kuka_id, joint, position)
            self.physics.setJointMotorControl2(
                self.kuka_id, joint, self.physics.POSITION_CONTROL, position, 0
            )
        self.physics.resetBasePositionAndOrientation(
            self.gripper_id,
            [0.923103, -0.2, 1.250036],
            [-0.0, 0.964531, -0.000002, -0.263970],
        )
        for joint in range(self.physics.getNumJoints(self.gripper_id)):
            position = self.initial_gripper_positions[joint]
            self.physics.resetJointState(self.gripper_id, joint, position)
            self.physics.setJointMotorControl2(
                self.gripper_id, joint, self.physics.POSITION_CONTROL, position, 0
            )
        return (yield from self._step_for(120))

    # ── measurement ─────────────────────────────────────────────────────────
    def tool_position(self):
        return self.physics.getLinkState(self.kuka_id, 6)[4]

    def grasp_position(self):
        """Midpoint of the two fingers - the point that actually closes on an object.

        Links 4 and 6 are gripper_left and gripper_right. Using link 6 on its own
        put the reference on a single finger, about half the finger gap off to
        one side, so every correction based on it inherited that bias.
        """
        left = self.physics.getLinkState(self.gripper_id, 4)[4]
        right = self.physics.getLinkState(self.gripper_id, 6)[4]
        return [(a + b) / 2 for a, b in zip(left, right)]

    def grasp_is_valid(self, object_id, tolerance=GRASP_TOLERANCE):
        """Are the fingers actually around the object before we call this a grasp?

        _attach_object creates a constraint, which succeeds from any distance and
        silently teleports the object into the gripper. Left unchecked, a move
        that missed by over a metre - which is what happens on the first pick
        after the arm has been displaced - still returned True to the operator.
        """
        target, _ = self.get_object_state(object_id)
        grasp = self.grasp_position()
        return math.hypot(grasp[0] - target[0], grasp[1] - target[1]) <= tolerance

    # ── grasping ────────────────────────────────────────────────────────────
    def _attach_object(self, object_id, desired_orientation=None):
        object_position, object_orientation = self.get_object_state(object_id)
        if desired_orientation is None:
            desired_orientation = object_orientation
        parent_state = self.physics.getLinkState(self.gripper_id, 6)
        parent_position, parent_orientation = parent_state[4], parent_state[5]
        inverse_parent = self.physics.invertTransform(
            parent_position,
            parent_orientation,
        )
        relative_position, relative_orientation = self.physics.multiplyTransforms(
            inverse_parent[0],
            inverse_parent[1],
            object_position,
            desired_orientation,
        )
        self._held_constraint = self.physics.createConstraint(
            self.gripper_id,
            6,
            object_id,
            -1,
            self.physics.JOINT_FIXED,
            [0, 0, 0],
            relative_position,
            [0, 0, 0, 1],
            relative_orientation,
        )
        return self._held_constraint

    def _release_object(self, constraint_id):
        self.physics.removeConstraint(constraint_id)
        if constraint_id == self._held_constraint:
            self._held_constraint = None

    @property
    def is_holding(self):
        return self._held_constraint is not None

    def abort_safely_steps(self):
        """Give up on the current action without dropping the sample.

        An emergency stop leaves the arm holding a tube in mid-air. Abandoning
        the move by simply releasing the constraint would drop it from whatever
        height the arm happened to stop at, so instead the arm lowers to table
        height at its current position and opens there.
        """
        if self._held_constraint is None:
            return True
        tool = self.tool_position()
        if not (yield from self.move_to_steps([tool[0], tool[1], GRAB_Z], False, 120)):
            return False
        if not (yield from self.move_to_steps([tool[0], tool[1], GRAB_Z], True, 80)):
            return False
        if self._held_constraint is not None:
            self._release_object(self._held_constraint)
        return (yield from self.move_to_steps([tool[0], tool[1], HOVER_Z], True, 60))

    def _carry_steps(self, start, end, steps_per_leg=80):
        """Carry whatever is held from one place to the other at clearance height."""
        legs = max(5, int(abs(end[1] - start[1]) / 0.05))
        for leg in range(legs + 1):
            fraction = leg / legs
            position = [
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
                LIFT_Z,
            ]
            if not (yield from self.move_to_steps(position, False, steps_per_leg)):
                return False
        return True

    # ── application actions ─────────────────────────────────────────────────
    def pick_and_place_steps(self, source_object, destination):
        source, _ = self.get_object_state(source_object)
        grasp_constraint = None
        keep_holding = False
        try:
            if not (yield from self.approach_from_above_steps(
                source[0], source[1], True, 200
            )):
                return False
            aim = yield from self.center_gripper_over_steps(source[0], source[1], True)
            if aim is None:
                return False
            if not (yield from self.move_to_steps([aim[0], aim[1], GRAB_Z], False, 150)):
                return False
            if not self.grasp_is_valid(source_object):
                return False
            grasp_constraint = self._attach_object(source_object)
            if not (yield from self.move_to_steps(
                [source[0], source[1], LIFT_Z], False, 200
            )):
                return False
            if not (yield from self._carry_steps(source, destination)):
                return False
            drop = yield from self.center_gripper_over_steps(
                destination[0], destination[1], False
            )
            if drop is None:
                return False
            if not (yield from self.move_to_steps([drop[0], drop[1], GRAB_Z], False, 200)):
                return False
            if not (yield from self.move_to_steps([drop[0], drop[1], GRAB_Z], True, 100)):
                return False
            self._release_object(grasp_constraint)
            grasp_constraint = None
            if not (yield from self.move_to_steps(
                [destination[0], destination[1], HOVER_Z], True, 100
            )):
                return False
            if not (yield from self.reset_robot_steps()):
                return False
        except EmergencyStopError:
            # Keep gripping. A stop should freeze the arm, not make it drop what
            # it is carrying; abort_safely_steps() sets the sample down later.
            keep_holding = True
            raise
        finally:
            if (grasp_constraint is not None
                    and self._held_constraint == grasp_constraint
                    and not keep_holding):
                self._release_object(grasp_constraint)
        return True

    def reverse_pick_and_place_steps(self, source_object, saved_position,
                                     saved_orientation):
        """Physically carry an object from its current location back to its saved pose."""
        self.safety.require_motion()
        current_position, _ = self.get_object_state(source_object)
        grasp_constraint = None
        keep_holding = False
        try:
            if not (yield from self.approach_from_above_steps(
                current_position[0], current_position[1], True, 200
            )):
                return False
            aim = yield from self.center_gripper_over_steps(
                current_position[0], current_position[1], True
            )
            if aim is None:
                return False
            if not (yield from self.move_to_steps([aim[0], aim[1], GRAB_Z], False, 150)):
                return False
            if not self.grasp_is_valid(source_object):
                return False
            grasp_constraint = self._attach_object(source_object, saved_orientation)
            if not (yield from self.move_to_steps(
                [current_position[0], current_position[1], LIFT_Z], False, 200
            )):
                return False
            if not (yield from self._carry_steps(current_position, saved_position)):
                return False
            drop = yield from self.center_gripper_over_steps(
                saved_position[0], saved_position[1], False
            )
            if drop is None:
                return False
            if not (yield from self.move_to_steps([drop[0], drop[1], GRAB_Z], False, 200)):
                return False
            if not (yield from self.move_to_steps([drop[0], drop[1], GRAB_Z], True, 100)):
                return False
            self._release_object(grasp_constraint)
            grasp_constraint = None
            if not (yield from self.move_to_steps(
                [saved_position[0], saved_position[1], HOVER_Z], True, 100
            )):
                return False
            return (yield from self.reset_robot_steps())
        except EmergencyStopError:
            keep_holding = True
            raise
        finally:
            if (grasp_constraint is not None
                    and self._held_constraint == grasp_constraint
                    and not keep_holding):
                self._release_object(grasp_constraint)

    # ── blocking forms, for tests and one-shot setup ────────────────────────
    def move_to(self, target_position, gripper_open, steps=150, precise=False):
        return self._drain(
            self.move_to_steps(target_position, gripper_open, steps, precise)
        )

    def approach_from_above(self, x, y, gripper_open, steps=150):
        return self._drain(self.approach_from_above_steps(x, y, gripper_open, steps))

    def center_gripper_over(self, x, y, gripper_open, attempts=4, steps=70,
                            tolerance=0.004):
        return self._drain(
            self.center_gripper_over_steps(x, y, gripper_open, attempts, steps, tolerance)
        )

    def reset_robot(self):
        return self._drain(self.reset_robot_steps())

    def pick_and_place(self, source_object, destination):
        return self._drain(self.pick_and_place_steps(source_object, destination))

    def abort_safely(self):
        return self._drain(self.abort_safely_steps())

    def reverse_pick_and_place(self, source_object, saved_position, saved_orientation):
        return self._drain(
            self.reverse_pick_and_place_steps(
                source_object, saved_position, saved_orientation
            )
        )
