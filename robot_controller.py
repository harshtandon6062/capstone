import math
import time

from config import (
    GRAB_Z,
    GRASP_TOLERANCE,
    HOVER_STEPS,
    HOVER_Z,
    LIFT_Z,
    POUR_CLEARANCE,
    POUR_AIM_LIMIT,
    POUR_HOLD_STEPS,
    POUR_TILT_STAGES,
    POUR_YAW,
    POUR_SWING,
    POUR_TILT_MAX_FRACTION,
    POUR_WRIST_FLOOR,
    POUR_TILT_RADIANS,
    TARGET_EULER,
    TEST_TUBE_HEIGHT,
    TEST_TUBE_RADIUS,
)
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
        self._held_object = None

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
    def _drive_to_steps(self, target_position, gripper_open, steps, orientation=None):
        self.safety.require_motion()
        gripper_value = 0 if gripper_open else 1
        joint_positions = self.physics.calculateInverseKinematics(
            self.kuka_id, 6, target_position, orientation or self.target_orientation
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

    def move_to_steps(self, target_position, gripper_open, steps=150, precise=False,
                      orientation=None):
        """Drive the tool to a target.

        With precise=True the move closes the loop: it measures where the tool
        actually ended up and aims off by the residual until the error is inside
        tolerance. Inverse kinematics plus a fixed number of position-control
        steps consistently undershoots by roughly 20 mm.
        """
        if not (yield from self._drive_to_steps(
            target_position, gripper_open, steps, orientation
        )):
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

    def pour_orientation_for(self, target_y=None, fraction=1.0):
        """Tool orientation for pouring: rolled over, and yawed a quarter turn.

        The yaw is the important part. Rolled at the default yaw the mouth swings
        along y, straight over the neighbouring tubes, so putting it on the target
        meant dropping the wrist into the middle of the row. Yawed, the swing is
        along x instead - over the destination spots, which are flat markers with
        no collision geometry - so the arm can come down there without touching
        anything and the tube can be tipped properly level.

        target_y is accepted and ignored; the swing no longer depends on which
        side of the robot the target is.
        """
        return self.physics.getQuaternionFromEuler(
            [POUR_TILT_RADIANS * fraction, TARGET_EULER[1], POUR_YAW]
        )

    def _tilt_in_place_steps(self, position, target_y, mouth_target,
                             stages=POUR_TILT_STAGES, steps=50):
        """Roll the wrist over, stopping while the pour is still reachable safely.

        Two separate reasons this is done in stages and by measurement.

        Commanding the full roll in one move makes inverse kinematics pick a
        different arm configuration and position control drives straight through
        the gap, dipping low enough to clip the tube being poured into.

        And rolling swings the tube up and over rather than tipping it down, so
        the steeper the tip, the lower the wrist must go to keep the mouth over a
        tube on the table. Past a point that means putting the wrist at bench
        height, where the arm sweeps everything off the table. So after each
        stage this works out where the wrist would have to be, and stops at the
        last angle that keeps it above POUR_WRIST_FLOOR.

        Returns the fraction of the nominal roll actually used, or None.
        """
        best = None
        for stage in range(1, stages + 1):
            fraction = POUR_TILT_MAX_FRACTION * stage / stages
            orientation = self.pour_orientation_for(target_y, fraction)
            if not (yield from self._drive_to_steps(position, False, steps, orientation)):
                return None

            mouth = self.held_object_tip()
            if mouth is None:
                return None
            wrist = self.tool_position()
            # Where the wrist would have to be for this tip to reach the target.
            needed_z = mouth_target[2] - (mouth[2] - wrist[2])
            if needed_z < POUR_WRIST_FLOOR:
                break
            best = fraction

        if best is None:
            return None
        if best != fraction:
            # Wind back to the last angle that was still reachable.
            if not (yield from self._drive_to_steps(
                position, False, steps, self.pour_orientation_for(target_y, best)
            )):
                return None
        return best

    def _untilt_in_place_steps(self, position, target_y, stages=5, steps=60,
                               from_fraction=None):
        """The reverse of _tilt_in_place_steps, ending upright."""
        top = from_fraction if from_fraction is not None else 1.0
        for stage in range(stages - 1, -1, -1):
            orientation = self.pour_orientation_for(target_y, top * stage / stages)
            if not (yield from self._drive_to_steps(position, False, steps, orientation)):
                return False
        return True

    def center_pour_over_steps(self, mouth_target, start=None, attempts=6, steps=90,
                               tolerance=0.010, tilt_fraction=1.0, gain=0.7):
        """Put the tilted tube's mouth over the target.

        Where the mouth sits relative to the wrist is a rigid offset once the tube
        is tipped, so it is measured once and the wrist is sent straight to
        mouth_target minus that offset. Only the residual is then corrected.

        The earlier version iterated from the start, each pass re-measuring and
        re-aiming. Moving the wrist changes the arm's configuration, which changes
        the offset, so the passes chased each other instead of settling - for some
        targets it wandered a metre away.
        """
        mouth = self.held_object_tip()
        if mouth is None:
            return None
        wrist = self.tool_position()
        offset = [mouth[i] - wrist[i] for i in range(3)]
        origin = list(start) if start is not None else list(wrist)
        aim = self._bounded_pour_aim(
            [mouth_target[i] - offset[i] for i in range(3)], origin
        )
        if not (yield from self._drive_to_steps(
            aim, False, steps, self.pour_orientation_for(None, tilt_fraction)
        )):
            return None

        for _ in range(attempts):
            mouth = self.held_object_tip()
            if mouth is None:
                return None
            error = [mouth_target[i] - mouth[i] for i in range(3)]
            if math.sqrt(sum(e * e for e in error)) <= tolerance:
                return aim
            # Damped: the offset between mouth and wrist shifts as the arm moves,
            # so correcting by the whole residual each time makes the passes
            # overshoot past each other instead of settling.
            aim = self._bounded_pour_aim(
                [aim[i] + gain * error[i] for i in range(3)], origin
            )
            if not (yield from self._drive_to_steps(
                aim, False, steps, self.pour_orientation_for(None, tilt_fraction)
            )):
                return None
        return aim

    def _bounded_pour_aim(self, aim, origin):
        """Keep the pour alignment inside the region it is safe to search.

        Pouring a little high is harmless. Dropping the arm into the bench is not,
        so the height is floored, and sideways drift is capped so a diverging
        correction cannot walk the arm across the workspace.
        """
        limited = [
            min(origin[0] + POUR_AIM_LIMIT, max(origin[0] - POUR_AIM_LIMIT, aim[0])),
            min(origin[1] + POUR_AIM_LIMIT, max(origin[1] - POUR_AIM_LIMIT, aim[1])),
            min(LIFT_Z, max(POUR_WRIST_FLOOR, aim[2])),
        ]
        return limited

    def hover_over_steps(self, x, y, steps=HOVER_STEPS):
        """Park the tool above (x, y) so the operator can see which object is meant.

        The panel can only say "the third square"; this says "this one". It is a
        way of pointing, not a way of picking - it stays at clearance height, so
        it cannot reach or disturb anything it is pointing at.
        """
        return (yield from self.move_to_steps([x, y, LIFT_Z], True, steps))

    def hover_over(self, x, y, steps=HOVER_STEPS):
        return self._drain(self.hover_over_steps(x, y, steps))

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

    def held_object_tilt(self):
        """How far the held tube is off vertical, in degrees.

        Below 90 the mouth still points upward, which means the tube is being
        tipped back rather than poured out.
        """
        if self._held_object is None:
            return 0.0
        _, orientation = self.get_object_state(self._held_object)
        matrix = self.physics.getMatrixFromQuaternion(orientation)
        return math.degrees(math.acos(max(-1.0, min(1.0, matrix[8]))))

    def held_object_tip(self, length=TEST_TUBE_HEIGHT):
        """World position of the open end of whatever is currently held.

        A tube hangs roughly 280 mm below the wrist, so rolling it past
        horizontal swings its mouth about that far sideways. Anything that needs
        to know where the contents will actually go has to ask about the tube,
        not about the wrist.
        """
        if self._held_object is None:
            return None
        position, orientation = self.get_object_state(self._held_object)
        matrix = self.physics.getMatrixFromQuaternion(orientation)
        axis = (matrix[2], matrix[5], matrix[8])
        return [position[i] + axis[i] * length for i in range(3)]

    def pour_is_aimed(self, mouth_target, tolerance=TEST_TUBE_RADIUS):
        """Is the mouth actually over the target before anything is transferred?

        The same principle as grasp_is_valid. Lining the pour up depends on how
        far the arm managed to tip the tube, which varies with its configuration,
        so occasionally it cannot be aimed at all. Pouring anyway would put the
        contents on the bench and report success; refusing says so instead.
        """
        mouth = self.held_object_tip()
        if mouth is None:
            return False
        return math.hypot(
            mouth[0] - mouth_target[0], mouth[1] - mouth_target[1]
        ) <= tolerance

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
        self._held_object = object_id
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
            self._held_object = None
        self._held_object = None

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

    def pour_steps(self, source_object, target_object, return_position):
        """Tip one tube over another, then put the tube back where it came from.

        Every part of the motion where the tube is tilted is either a rotation in
        place or a straight vertical move. Nothing is carried across the table
        while tipped over, which is what made an earlier version look like it was
        about to spill the contents on the way past.

        The order is the one a person would use: bring the tube over upright, tip
        it, lower it into the target, pour, lift straight out, straighten up, and
        only then carry it back.
        """
        source, _ = self.get_object_state(source_object)
        target, _ = self.get_object_state(target_object)
        mouth_target = [
            target[0],
            target[1],
            target[2] + TEST_TUBE_HEIGHT + POUR_CLEARANCE,
        ]
        # Park the wrist off to the side by however far the mouth will swing, so
        # that tipping lands the mouth on the target instead of beside it.
        # The wrist parks toward the robot by however far the mouth will swing,
        # so that tipping brings the mouth onto the target. That puts it over the
        # spot row, which is clear of anything solid.
        staging = [target[0] + POUR_SWING, target[1], LIFT_Z]

        grasp_constraint = None
        keep_holding = False
        poured = False
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

            # Carried upright, at clearance height, to one side of the target.
            if not (yield from self._carry_steps(source, staging)):
                return False
            # Tipped in place, high and clear of everything. This reports how far
            # it actually had to roll, which everything downstream has to keep
            # commanding or the tube springs back upright.
            tilt_fraction = yield from self._tilt_in_place_steps(
                staging, target[1], mouth_target
            )
            if not tilt_fraction:
                return False
            # Lowered onto the target. center_pour_over_steps corrects what the
            # swing estimate got wrong, which is now centimetres rather than the
            # whole 0.26 m.
            pour_aim = yield from self.center_pour_over_steps(
                mouth_target, start=staging, tilt_fraction=tilt_fraction
            )
            if pour_aim is None:
                return False
            # Nothing is transferred unless the mouth really is over the target.
            # A refused pour still has to set the tube down: returning here with
            # it gripped and tipped would drop it from mid-air.
            if not self.pour_is_aimed(mouth_target):
                if not (yield from self._put_tube_back_steps(
                    pour_aim, return_position, target[1], tilt_fraction
                )):
                    yield from self.abort_safely_steps()
                grasp_constraint = None
                return False
            if not (yield from self._step_for(POUR_HOLD_STEPS)):
                return False
            poured = True

            if not (yield from self._put_tube_back_steps(
                pour_aim, return_position, target[1], tilt_fraction
            )):
                # Carrying it home failed. Do not fall through still holding it -
                # set it down wherever the arm is instead of dropping it.
                yield from self.abort_safely_steps()
                grasp_constraint = None
                return False
            grasp_constraint = None
        except EmergencyStopError:
            keep_holding = True
            raise
        finally:
            if (grasp_constraint is not None
                    and self._held_constraint == grasp_constraint
                    and not keep_holding):
                self._release_object(grasp_constraint)
        return poured

    def _put_tube_back_steps(self, pour_aim, return_position, target_y, tilt_fraction):
        """Straighten up and set the held tube back down where it came from.

        Used by the successful pour and by a refused one alike, because either
        way the arm is left holding a tipped tube in mid-air and letting go there
        would drop it.
        """
        upright = [pour_aim[0], pour_aim[1], LIFT_Z]
        if not (yield from self._drive_to_steps(
            upright, False, 200, self.pour_orientation_for(target_y, tilt_fraction)
        )):
            return False
        if not (yield from self._untilt_in_place_steps(
            upright, target_y, from_fraction=tilt_fraction
        )):
            return False
        if not (yield from self._carry_steps(upright, return_position)):
            return False
        drop = yield from self.center_gripper_over_steps(
            return_position[0], return_position[1], False
        )
        if drop is None:
            return False
        if not (yield from self.move_to_steps([drop[0], drop[1], GRAB_Z], False, 200)):
            return False
        if not (yield from self.move_to_steps([drop[0], drop[1], GRAB_Z], True, 100)):
            return False
        if self._held_constraint is not None:
            self._release_object(self._held_constraint)
        if not (yield from self.move_to_steps(
            [return_position[0], return_position[1], HOVER_Z], True, 100
        )):
            return False
        return (yield from self.reset_robot_steps())

    def pour(self, source_object, target_object, return_position):
        return self._drain(
            self.pour_steps(source_object, target_object, return_position)
        )

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
    def move_to(self, target_position, gripper_open, steps=150, precise=False,
                orientation=None):
        return self._drain(
            self.move_to_steps(target_position, gripper_open, steps, precise, orientation)
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
