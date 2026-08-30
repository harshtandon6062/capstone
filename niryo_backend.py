"""Driving a physical Niryo Ned 2 through the same interface as the simulation.

NOT YET VERIFIED AGAINST HARDWARE. Written against pyniryo 1.2.5, whose API was
read directly rather than recalled, but no part of this has been run with an arm
attached. Treat every distance in DEFAULT_WORKSPACE as a starting guess that has
to be calibrated before the arm is switched on.

Two facts about the real arm shaped this file.

First, every pyniryo motion call blocks until the arm arrives. There is no
stepped API, so the generator contract the application depends on is provided by
running each call on a worker thread and yielding while it is in flight. The
application loop therefore keeps reading the camera during hardware motion,
exactly as it does in simulation.

Second, and more importantly: pyniryo has no stop, halt or abort command. The
entire command set was searched; the only software way to interrupt a Ned 2 is
set_learning_mode(True), which cuts motor torque - the arm goes limp and drops
whatever it is carrying, which is the precise failure this project already fixed
in simulation. So on hardware:

    a gesture stop cannot halt a motion that is already in progress.

What it can do is refuse to issue the next waypoint. Long moves are therefore cut
into short segments, which bounds how long the arm keeps going after the operator
asks it to stop - roughly one segment. That is a real limitation and the physical
emergency-stop button remains the actual safety device. The panel should say so.
"""

import math
import threading
from dataclasses import dataclass

DEFAULT_IP = "169.254.200.200"

# Metres of travel per waypoint. Smaller means the arm notices a stop request
# sooner and moves less jerkily; larger means fewer round trips over TCP.
SEGMENT_LENGTH = 0.04

# How far above an object the arm travels before descending onto it.
APPROACH_CLEARANCE = 0.08

# Tool orientation for a vertical grasp, in the robot's own frame: roll, pitch,
# yaw with the gripper pointing straight down.
GRASP_RPY = (0.0, math.pi / 2, 0.0)

# Roll applied to tip a held tube. Same idea as the simulation, and the same
# caveat: which way the mouth swings depends on the sign.
POUR_ROLL = 1.6


class NiryoUnavailable(RuntimeError):
    """Raised when the arm cannot be reached or pyniryo is not installed."""


@dataclass
class WorkspaceTransform:
    """Map table coordinates onto the robot's own frame.

    The simulation places its table around x=0.75, y=-0.45..0.03, z=0.625, which
    is nowhere near a Ned 2's reachable envelope (roughly x 0.15-0.35 m from its
    own base). Nothing here is measured - `origin` and `scale` are exactly the
    numbers that have to be calibrated against the real table, and getting them
    wrong is the difference between reaching a tube and driving into it.
    """

    origin: tuple = (0.75, -0.21, 0.625)
    robot_origin: tuple = (0.25, 0.0, 0.10)
    scale: float = 0.45

    def to_robot(self, position):
        return [
            self.robot_origin[axis] + (position[axis] - self.origin[axis]) * self.scale
            for axis in range(3)
        ]

    def to_table(self, position):
        return [
            self.origin[axis] + (position[axis] - self.robot_origin[axis]) / self.scale
            for axis in range(3)
        ]


class BlockingCall:
    """Runs one blocking arm call on a worker thread so the caller can yield.

    The application loop must not stall for the seconds a real move takes, or the
    camera stops and the operator becomes invisible to the system.
    """

    def __init__(self, function, *args, **kwargs):
        self.error = None
        self.result = None
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(function, args, kwargs), daemon=True
        )
        self._thread.start()

    def _run(self, function, args, kwargs):
        try:
            self.result = function(*args, **kwargs)
        except Exception as error:  # surfaced to the caller, not swallowed
            self.error = error
        finally:
            self._done.set()

    @property
    def finished(self):
        return self._done.is_set()


def connect(ip_address=DEFAULT_IP, client=None):
    """Open a connection to the arm, or raise NiryoUnavailable saying why."""
    if client is not None:
        return client
    try:
        from pyniryo import NiryoRobot
    except ImportError as error:
        raise NiryoUnavailable(
            "pyniryo is not installed; run 'pip install pyniryo'"
        ) from error
    try:
        robot = NiryoRobot(ip_address)
        robot.calibrate_auto()
        robot.update_tool()
        return robot
    except Exception as error:
        raise NiryoUnavailable(f"could not reach the arm at {ip_address}: {error}") from error


class NiryoArmController:
    """The RobotBackend contract, implemented against a real Ned 2.

    Object poses come from perception rather than from the arm, because unlike
    the simulator the robot has no idea what is on the table.
    """

    def __init__(self, client, registry, safety, transform=None):
        self.client = client
        self.registry = registry
        self.safety = safety
        self.transform = transform or WorkspaceTransform()
        self._held_object = None

    # ── contract ────────────────────────────────────────────────────────────
    def get_object_state(self, object_id):
        obj = self.registry.by_handle(object_id)
        if obj is None:
            raise KeyError(f"no object {object_id} in the registry")
        return list(obj.position), [0.0, 0.0, 0.0, 1.0]

    @property
    def is_holding(self):
        return self._held_object is not None

    # ── motion ──────────────────────────────────────────────────────────────
    def _call_steps(self, function, *args, **kwargs):
        """Run one blocking arm call, yielding until it finishes."""
        call = BlockingCall(function, *args, **kwargs)
        # Yield once unconditionally. Without this the loop only gets control back
        # when a call happens to be slow, so how often the operator is looked at
        # would depend on how fast the arm answers.
        yield
        while not call.finished:
            yield
        if call.error is not None:
            raise NiryoUnavailable(str(call.error)) from call.error
        return call.result

    def _may_continue(self):
        """Whether the next waypoint should be sent at all.

        This is the whole of the software stop on hardware: not sending the next
        segment. It cannot retract a segment already in flight.
        """
        return self.safety.motion_allowed

    def move_to_steps(self, table_position, roll=None):
        """Travel to a table coordinate in short segments.

        Segmenting is what bounds the stop latency: the operator's stop takes
        effect at the next segment boundary rather than at the end of the whole
        move.
        """
        target = self.transform.to_robot(table_position)
        current = yield from self._call_steps(self.client.get_pose)
        start = [current.x, current.y, current.z]
        distance = math.dist(start, target)
        segments = max(1, int(math.ceil(distance / SEGMENT_LENGTH)))

        orientation = GRASP_RPY if roll is None else (roll, GRASP_RPY[1], GRASP_RPY[2])
        for segment in range(1, segments + 1):
            if not self._may_continue():
                return False
            fraction = segment / segments
            waypoint = [
                start[axis] + (target[axis] - start[axis]) * fraction for axis in range(3)
            ]
            yield from self._call_steps(
                self.client.move_pose,
                waypoint[0], waypoint[1], waypoint[2],
                orientation[0], orientation[1], orientation[2],
            )
        return True

    def approach_from_above_steps(self, table_position, roll=None):
        above = [table_position[0], table_position[1],
                 table_position[2] + APPROACH_CLEARANCE / self.transform.scale]
        if not (yield from self.move_to_steps(above, roll)):
            return False
        return (yield from self.move_to_steps(table_position, roll))

    def _grasp_steps(self, object_id):
        yield from self._call_steps(self.client.close_gripper)
        self._held_object = object_id
        return True

    def _release_steps(self):
        yield from self._call_steps(self.client.open_gripper)
        self._held_object = None
        return True

    def pick_and_place_steps(self, source_object, destination):
        source, _ = self.get_object_state(source_object)
        yield from self._call_steps(self.client.open_gripper)
        if not (yield from self.approach_from_above_steps(source)):
            return False
        yield from self._grasp_steps(source_object)
        lifted = [source[0], source[1], source[2] + APPROACH_CLEARANCE / self.transform.scale]
        if not (yield from self.move_to_steps(lifted)):
            return False
        if not (yield from self.approach_from_above_steps(list(destination))):
            return False
        yield from self._release_steps()
        return (yield from self.move_to_steps(
            [destination[0], destination[1],
             destination[2] + APPROACH_CLEARANCE / self.transform.scale]
        ))

    def reverse_pick_and_place_steps(self, source_object, saved_position,
                                     saved_orientation):
        return (yield from self.pick_and_place_steps(source_object, list(saved_position)))

    def pour_steps(self, source_object, target_object, return_position):
        source, _ = self.get_object_state(source_object)
        target, _ = self.get_object_state(target_object)
        if not (yield from self.approach_from_above_steps(source)):
            return False
        yield from self._grasp_steps(source_object)

        over_target = [target[0], target[1],
                       target[2] + 2 * APPROACH_CLEARANCE / self.transform.scale]
        if not (yield from self.move_to_steps(over_target)):
            return False
        # Sign chosen so the mouth swings toward the robot, as in simulation.
        roll = POUR_ROLL if target[1] < self.transform.origin[1] else -POUR_ROLL
        if not (yield from self.move_to_steps(over_target, roll=roll)):
            return False
        if not (yield from self.move_to_steps(over_target)):
            return False
        if not (yield from self.approach_from_above_steps(list(return_position))):
            return False
        yield from self._release_steps()
        return True

    def abort_safely_steps(self):
        """Set down what is held rather than opening the gripper in mid-air.

        Note this only runs once the operator has cleared the stop; while the stop
        is latched no waypoint is sent at all.
        """
        if self._held_object is None:
            return True
        current = yield from self._call_steps(self.client.get_pose)
        table = self.transform.to_table([current.x, current.y, current.z])
        table[2] = self.transform.origin[2]
        if not (yield from self.move_to_steps(table)):
            return False
        yield from self._release_steps()
        return True

    def close(self):
        try:
            self.client.close_connection()
        except Exception:
            pass
