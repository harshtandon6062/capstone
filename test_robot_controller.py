"""Motion-shape tests for RobotController. No simulator required."""

from config import GRAB_Z, HOVER_Z, LIFT_Z
from robot_controller import RobotController
from safety_controller import SafetyController


class RecordingPhysics:
    """Minimal PyBullet stand-in that records the targets it is asked to reach."""

    VELOCITY_CONTROL = 0
    POSITION_CONTROL = 1
    JOINT_FIXED = 4

    def __init__(self):
        self.targets = []
        self.steps = 0
        self.constraints = []
        self.tool = [0.0, 0.0, 1.0]

    # ── queried during construction ──
    def getQuaternionFromEuler(self, euler):
        return [0, 0, 0, 1]

    def getNumJoints(self, body):
        return 7

    def getJointState(self, body, joint):
        return (0.0, 0.0, None, None)

    # ── motion ──
    def calculateInverseKinematics(self, body, link, position, orientation):
        self.targets.append(list(position))
        # Pretend the arm tracks perfectly, so closed-loop refinement converges
        # on the first check instead of chasing a stationary tool.
        self.tool = list(position)
        return [0.0] * 7

    def setJointMotorControl2(self, *args, **kwargs):
        pass

    def resetJointState(self, *args, **kwargs):
        pass

    def resetBasePositionAndOrientation(self, *args, **kwargs):
        pass

    def stepSimulation(self):
        self.steps += 1

    # ── grasping ──
    def getBasePositionAndOrientation(self, body):
        return ([0.75, -0.45, 0.625], [0, 0, 0, 1])

    def getLinkState(self, body, link):
        return (None, None, None, None, list(self.tool), [0, 0, 0, 1])

    def invertTransform(self, position, orientation):
        return ([0, 0, 0], [0, 0, 0, 1])

    def multiplyTransforms(self, pa, oa, pb_, ob):
        return ([0, 0, 0], [0, 0, 0, 1])

    def createConstraint(self, *args, **kwargs):
        self.constraints.append(args)
        return len(self.constraints)

    def removeConstraint(self, cid):
        self.constraints.pop()


def make_controller():
    physics = RecordingPhysics()
    safety = SafetyController(physics, robot_id=1, gripper_id=2)
    controller = RobotController(
        physics, 1, 2, safety,
        initial_kuka_positions=[0.0] * 7,
        initial_gripper_positions=[0.0] * 8,
    )
    return physics, controller


def test_approach_travels_high_then_descends():
    """Horizontal travel must happen above the objects, not at grasp height."""
    physics, controller = make_controller()
    controller.approach_from_above(0.9, -0.2, gripper_open=True)

    assert len(physics.targets) == 2
    travel, descend = physics.targets
    assert travel[:2] == [0.9, -0.2]
    assert travel[2] == LIFT_Z, "first leg must be at clearance height"
    assert descend[:2] == [0.9, -0.2]
    assert descend[2] == HOVER_Z, "second leg must be a straight vertical descent"
    assert travel[2] > descend[2]


def test_pick_and_place_never_travels_sideways_at_grasp_height():
    """Regression: the arm used to sweep through tubes and shove them aside.

    Any move that changes x or y must be at or above HOVER_Z. Only pure vertical
    moves are allowed to reach grasp height.
    """
    physics, controller = make_controller()
    controller.pick_and_place(99, [0.95, -0.21, 0.65])

    previous = None
    for target in physics.targets:
        if previous is not None:
            moved_laterally = (
                abs(target[0] - previous[0]) > 1e-9 or abs(target[1] - previous[1]) > 1e-9
            )
            if moved_laterally:
                assert target[2] >= HOVER_Z, (
                    f"lateral move to {target} happened below clearance height"
                )
        previous = target


def test_reverse_also_approaches_from_above():
    physics, controller = make_controller()
    physics.targets.clear()
    controller.reverse_pick_and_place(99, [0.75, -0.45, 0.625], [0, 0, 0, 1])

    assert physics.targets, "reverse should command motion"
    assert physics.targets[0][2] == LIFT_Z, "reverse must also travel high first"


def test_grasp_constraint_is_released_when_a_move_fails():
    physics, controller = make_controller()
    controller.safety.emergency_stop()
    try:
        controller.pick_and_place(99, [0.95, -0.21, 0.65])
    except Exception:
        pass
    assert physics.constraints == [], "an aborted pick must not leave the object attached"


def test_precise_move_stops_once_inside_tolerance():
    """A perfectly tracking arm needs no correction passes."""
    physics, controller = make_controller()
    controller.move_to([0.9, -0.2, 0.97], gripper_open=False, steps=10, precise=True)
    assert len(physics.targets) == 1, "no refinement needed when the tool is on target"


def test_precise_move_corrects_a_systematic_undershoot():
    """Regression: a repeatable undershoot compounded across pick/undo cycles."""
    physics, controller = make_controller()

    real_ik = physics.calculateInverseKinematics
    def undershooting_ik(body, link, position, orientation):
        result = real_ik(body, link, position, orientation)
        # The arm always stops 20 mm short in x, exactly as measured in sim.
        physics.tool = [position[0] - 0.02, position[1], position[2]]
        return result
    physics.calculateInverseKinematics = undershooting_ik

    target = [0.9, -0.2, 0.97]
    controller.move_to(target, gripper_open=False, steps=10, precise=True)

    assert len(physics.targets) > 1, "an undershoot must trigger a correction"
    assert physics.targets[-1][0] > target[0], "correction must aim past the target"
    final_error = abs(physics.tool[0] - target[0])
    assert final_error < 0.02, f"residual {final_error:.4f} should beat the raw undershoot"


class MissingPhysics(RecordingPhysics):
    """The arm is nowhere near the object - what a displaced arm looks like."""

    def getLinkState(self, body, link):
        return (None, None, None, None, [2.0, 2.0, 1.0], [0, 0, 0, 1])


def test_grasp_position_is_the_midpoint_of_the_two_fingers():
    physics = RecordingPhysics()
    positions = {4: [0.0, 0.0, 1.0], 6: [1.0, 2.0, 3.0]}
    physics.getLinkState = lambda body, link: (
        None, None, None, None, positions[link], [0, 0, 0, 1]
    )
    controller = RobotController(
        physics, 1, 2, SafetyController(physics), [0.0] * 7, [0.0] * 8
    )
    assert controller.grasp_position() == [0.5, 1.0, 2.0]


def test_grasp_is_valid_only_when_the_fingers_are_on_the_object():
    physics = RecordingPhysics()
    controller = RobotController(
        physics, 1, 2, SafetyController(physics), [0.0] * 7, [0.0] * 8
    )
    physics.tool = [0.75, -0.45, 0.97]          # object sits at (0.75, -0.45)
    assert controller.grasp_is_valid(99)
    physics.tool = [0.95, -0.45, 0.97]          # 200 mm away
    assert not controller.grasp_is_valid(99)


def test_pick_and_place_fails_instead_of_faking_a_grasp_it_did_not_make():
    """A constraint attaches from any distance, so a miss must be caught first."""
    physics = MissingPhysics()
    controller = RobotController(
        physics, 1, 2, SafetyController(physics), [0.0] * 7, [0.0] * 8
    )
    assert controller.pick_and_place(99, [0.95, -0.45, 0.625]) is False
    assert physics.constraints == []


def test_undo_also_refuses_to_fake_a_grasp():
    physics = MissingPhysics()
    controller = RobotController(
        physics, 1, 2, SafetyController(physics), [0.0] * 7, [0.0] * 8
    )
    assert controller.reverse_pick_and_place(
        99, [0.75, -0.45, 0.625], [0, 0, 0, 1]
    ) is False
    assert physics.constraints == []
