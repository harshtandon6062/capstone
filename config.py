"""Configuration for the current PyBullet pick-and-place application."""

TABLE_BASE_POSITION = [1.0, -0.2, 0.0]
TABLE_BASE_ORIENTATION = [0, 0, 0.7071, 0.7071]

ROBOT_BASE_POSITION = [1.4, -0.2, 0.6]
ROBOT_BASE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]

GRIPPER_BASE_POSITION = [0.923103, -0.2, 1.250036]
GRIPPER_BASE_ORIENTATION = [-0.0, 0.964531, -0.000002, -0.263970]

INITIAL_KUKA_JOINT_POSITIONS = [-0.0, -0.0, 0.0, 1.570793, 0.0, -1.036725, 0.000001]
INITIAL_GRIPPER_JOINT_POSITIONS = [
    0.0, -0.011130, -0.206421, 0.205143,
    -0.009999, 0.0, -0.010055, 0.0,
]

CUBE_COLORS_RGBA = [
    [1.0, 0.0, 0.0, 1],
    [0.0, 0.7, 0.0, 1],
    [0.0, 0.0, 1.0, 1],
    [1.0, 0.9, 0.0, 1],
    [1.0, 0.0, 1.0, 1],
]
DESTINATION_SPOT_COLORS_RGBA = [
    [1.0, 0.0, 0.0, 0.9],
    [0.0, 0.8, 0.0, 0.9],
    [1.0, 0.55, 0.0, 0.9],
    [0.7, 0.15, 1.0, 0.9],
    [0.0, 0.8, 0.8, 0.9],
]

# Names are paired positionally with the RGBA values above. The UI derives its
# panel colours from those same RGBA values, so the panel and the simulation
# cannot drift apart.
CUBE_COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA"]
DESTINATION_SPOT_COLOR_NAMES = ["RED", "GREEN", "ORANGE", "PURPLE", "CYAN"]

# How close a tube has to be to a spot to count as standing on it. Occupancy is
# derived from live positions rather than remembered, so this is the one number
# that decides it. Spots are 0.12 m apart, so this must stay under half of that.
SPOT_OCCUPANCY_RADIUS = 0.06

OBJECT_COUNT = 5
CUBE_HALF_EXTENTS = [0.025, 0.025, 0.025]
DESTINATION_SPOT_HALF_EXTENTS = [0.03, 0.03, 0.002]
TEST_TUBE_RADIUS = 0.022
TEST_TUBE_HEIGHT = 0.10
TEST_TUBE_RIM_RADIUS = 0.025
TEST_TUBE_RIM_THICKNESS = 0.003
# The cap. Big enough to read across the table, because it is what tells the
# operator which tube this is once the contents no longer do.
TEST_TUBE_CAP_RADIUS = 0.027
TEST_TUBE_CAP_HEIGHT = 0.020
TEST_TUBE_LIQUID_HEIGHT = 0.055
DESTINATION_SPOT_RADIUS = 0.045
DESTINATION_SPOT_HEIGHT = 0.002
OBJECT_Y_START = -0.45
OBJECT_Y_STEP = 0.12
OBJECT_X = 0.75
OBJECT_Z = 0.65
DESTINATION_X = 0.95
DESTINATION_Z_OFFSET = -0.025

# Gesture timing configuration (seconds).
#
# Hold and cooldown are deliberately proportional to how costly the action is to
# get wrong. Navigating between tubes is free to undo, so it should feel
# immediate. Committing a destination starts the robot moving, so it stays
# slow enough to be a decision.
NAVIGATION_HOLD_DURATION = 0.25
NAVIGATION_COOLDOWN = 0.30
GESTURE_HOLD_DURATION = 0.9
GESTURE_COOLDOWN = 0.8
UNDO_GESTURE_COOLDOWN = 10.0

# The dynamic (LSTM) model costs ~80 ms per call and on the pick-and-place
# screen its output is only displayed - the state machine runs on static
# gestures. Running it every frame halved the frame rate for a label.
DYNAMIC_PREDICT_EVERY = 5

# Frames of agreement required before a static gesture is reported. This is
# pure latency, so keep it just long enough to smooth detector flicker.
GESTURE_STABILISER_WINDOW = 3

# Simulation steps advanced per camera frame while the robot is moving. The
# application loop drives a motion a slice at a time instead of blocking inside
# it, which is what keeps gestures live for the whole of a move. Roughly 16
# steps at ~30 fps reproduces the pace the blocking version ran at.
MOTION_STEPS_PER_FRAME = 16

# While the operator is choosing, the arm hovers over whatever is highlighted, so
# they can see which real object the panel means before they confirm anything.
# Set False to keep the arm still.
HOVER_OVER_SELECTION = True
HOVER_STEPS = 110

# How long a transient panel message stays on screen (seconds). Without this a
# message drawn on one frame vanishes in about 30 ms and is never read.
STATUS_MESSAGE_DURATION = 2.5

# How far the grasp point may be from an object and still count as a grasp.
# Attaching is a constraint, so it succeeds from any distance; without this check
# a move reports success even when the object was never between the fingers.
# Measured good grasps land under 7 mm, and a tube is 22 mm in radius.
GRASP_TOLERANCE = 0.03

TARGET_EULER = [0, 1.01 * 3.141592653589793, 0]

# Pouring. This is deliberately not a fluid simulation - the point of the action
# is that it cannot be undone, not that it models a liquid. The arm tips the tube
# over the target, holds, and the contents transfer.
# Applied as roll. The wrist's pitch range is limited by the URDF and saturates
# around 60 degrees of actual tube tilt; rolling reaches 94, which is past
# horizontal and is what pouring actually looks like.
POUR_TILT_RADIANS = 1.6
POUR_Z = 1.09
POUR_HOLD_STEPS = 150
# Height of the tilted tube's mouth above the rim of the tube being poured into.
POUR_CLEARANCE = 0.09
# How far sideways the mouth swings when the tube is tipped over. The tube hangs
# well below the wrist, so tipping it throws the open end a long way. The wrist is
# parked this far off the target before tipping, so that tipping brings the mouth
# onto the target rather than having to carry it there afterwards.
POUR_SWING = 0.26

# An irreversible action gets a longer, more deliberate confirmation than one
# that can be undone. Same interface, different cost of being wrong.
IRREVERSIBLE_HOLD_DURATION = 2.0

# What the operator can ask the robot to do. "reversible" drives both the undo
# offer and how long the confirming hold has to be held.
ACTIONS = [
    {"key": "move", "label": "MOVE", "hint": "place on a spot", "reversible": True},
    {"key": "pour", "label": "POUR", "hint": "into another tube", "reversible": False},
]
HOVER_Z = 0.97
GRAB_Z = 0.97
LIFT_Z = 1.15
