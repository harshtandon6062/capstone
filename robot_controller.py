import math
import time

from config import GRAB_Z, HOVER_Z, LIFT_Z, TARGET_EULER
from safety_controller import EmergencyStopError


class RobotController:
    """Backend-neutral robot operations used by application commands."""

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

    def set_safety_poll(self, safety_poll):
        self.safety_poll = safety_poll

    def get_object_state(self, object_id):
        return self.physics.getBasePositionAndOrientation(object_id)

    def reset_robot(self):
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
        for _ in range(120):
            if not self.safety.step_simulation(self.safety_poll, raise_on_stop=True):
                return False
        return True

    def move_to(self, target_position, gripper_open, steps=150, precise=False):
        """Drive the tool to a target.

        With precise=True the move closes the loop: it measures where the tool
        actually ended up and aims off by the residual until the error is inside
        tolerance. Inverse kinematics plus a fixed number of position-control
        steps consistently undershoots by roughly 20 mm, and because the error
        has a consistent direction it accumulates over repeated pick and undo
        cycles rather than averaging out.
        """
        if not self.safety.wait_until_running(self.safety_poll):
            return False
        if not self._drive_to(target_position, gripper_open, steps):
            return False
        if precise:
            return self._refine(target_position, gripper_open)
        return True

    def _drive_to(self, target_position, gripper_open, steps):
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
        for _ in range(steps):
            if not self.safety.step_simulation(self.safety_poll, raise_on_stop=True):
                return False
            time.sleep(1 / 480)
        return True

    def tool_position(self):
        return self.physics.getLinkState(self.kuka_id, 6)[4]

    def grasp_position(self):
        """Where the fingers actually are, which is not where the wrist is."""
        return self.physics.getLinkState(self.gripper_id, 6)[4]

    def center_gripper_over(self, x, y, gripper_open, attempts=4, steps=70,
                            tolerance=0.004):
        """Aim the wrist so the gripper sits over (x, y). Returns the aim used.

        The gripper hangs off the wrist on a constraint, so its centre trails the
        inverse-kinematics target by a pose-dependent horizontal offset of a few
        centimetres. Closing without correcting for that means the fingers meet
        the object off-centre and shove it sideways - about 8 mm per grasp, always
        in the same direction. That is what makes repeated pick and undo cycles
        walk an object across the table instead of returning it.
        """
        aim_x, aim_y = x, y
        for _ in range(attempts):
            grasp = self.grasp_position()
            error_x, error_y = x - grasp[0], y - grasp[1]
            if math.hypot(error_x, error_y) <= tolerance:
                return (aim_x, aim_y)
            aim_x += error_x
            aim_y += error_y
            if not self._drive_to([aim_x, aim_y, HOVER_Z], gripper_open, steps):
                return None
        return (aim_x, aim_y)

    def _refine(self, target_position, gripper_open, tolerance=0.004, attempts=3, steps=70):
        """Aim off by the measured residual until the tool is inside tolerance."""
        aim = list(target_position)
        for _ in range(attempts):
            actual = self.tool_position()
            error = [t - a for t, a in zip(target_position, actual)]
            if math.sqrt(sum(e * e for e in error)) <= tolerance:
                return True
            aim = [a + e for a, e in zip(aim, error)]
            if not self._drive_to(aim, gripper_open, steps):
                return False
        return True

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
        return self.physics.createConstraint(
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

    def _release_object(self, constraint_id):
        self.physics.removeConstraint(constraint_id)

    def _move_with_object(self, object_id, target_position, steps=150):
        return self.move_to(target_position, False, steps)

    def approach_from_above(self, x, y, gripper_open, steps=150):
        """Travel at clearance height first, then descend straight down.

        move_to() only sets IK joint targets - there is no path planning and no
        collision checking, so travelling directly to grasp height lets the arm
        sweep sideways through whatever is in the way and knock it over. Going up
        and over first keeps the horizontal leg of the motion above the objects.
        """
        if not self.move_to([x, y, LIFT_Z], gripper_open, steps):
            return False
        return self.move_to([x, y, HOVER_Z], gripper_open, max(60, steps // 2))

    def pick_and_place(self, source_object, destination):
        source, _ = self.get_object_state(source_object)
        grasp_constraint = None
        try:
            if not self.approach_from_above(source[0], source[1], True, 200):
                return False
            aim = self.center_gripper_over(source[0], source[1], True)
            if aim is None:
                return False
            if not self.move_to([aim[0], aim[1], GRAB_Z], False, 150):
                return False
            grasp_constraint = self._attach_object(source_object)
            if not self.move_to([source[0], source[1], LIFT_Z], False, 200):
                return False
            steps = max(5, int(abs(destination[1] - source[1]) / 0.05))
            for step in range(steps + 1):
                fraction = step / steps
                position = [
                    source[0] + fraction * (destination[0] - source[0]),
                    source[1] + fraction * (destination[1] - source[1]),
                    LIFT_Z,
                ]
                if not self.move_to(position, False, 80):
                    return False
            drop = self.center_gripper_over(destination[0], destination[1], False)
            if drop is None:
                return False
            if not self.move_to([drop[0], drop[1], GRAB_Z], False, 200):
                return False
            if not self.move_to([drop[0], drop[1], GRAB_Z], True, 100):
                return False
            self._release_object(grasp_constraint)
            grasp_constraint = None
            if not self.move_to([destination[0], destination[1], HOVER_Z], True, 100):
                return False
            if not self.reset_robot():
                return False
        except EmergencyStopError:
            raise
        finally:
            if grasp_constraint is not None:
                self._release_object(grasp_constraint)
        return True

    def reverse_pick_and_place(self, source_object, saved_position, saved_orientation):
        """Physically carry an object from its current location back to its saved pose."""
        self.safety.require_motion()
        current_position, _ = self.get_object_state(source_object)
        grasp_constraint = None
        try:
            if not self.approach_from_above(
                current_position[0], current_position[1], True, 200
            ):
                return False
            aim = self.center_gripper_over(current_position[0], current_position[1], True)
            if aim is None:
                return False
            if not self.move_to([aim[0], aim[1], GRAB_Z], False, 150):
                return False
            grasp_constraint = self._attach_object(source_object, saved_orientation)
            if not self.move_to(
                [current_position[0], current_position[1], LIFT_Z], False, 200
            ):
                return False

            steps = max(5, int(abs(saved_position[1] - current_position[1]) / 0.05))
            for step in range(steps + 1):
                fraction = step / steps
                position = [
                    current_position[0] + fraction * (saved_position[0] - current_position[0]),
                    current_position[1] + fraction * (saved_position[1] - current_position[1]),
                    LIFT_Z,
                ]
                if not self._move_with_object(source_object, position, 80):
                    return False

            drop = self.center_gripper_over(saved_position[0], saved_position[1], False)
            if drop is None:
                return False
            if not self.move_to([drop[0], drop[1], GRAB_Z], False, 200):
                return False
            if not self.move_to([drop[0], drop[1], GRAB_Z], True, 100):
                return False
            self._release_object(grasp_constraint)
            grasp_constraint = None
            if not self.move_to(
                [saved_position[0], saved_position[1], HOVER_Z], True, 100
            ):
                return False
            return self.reset_robot()
        except EmergencyStopError:
            raise
        finally:
            if grasp_constraint is not None:
                self._release_object(grasp_constraint)
