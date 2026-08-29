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

OBJECT_COUNT = 5
CUBE_HALF_EXTENTS = [0.025, 0.025, 0.025]
DESTINATION_SPOT_HALF_EXTENTS = [0.03, 0.03, 0.002]
TEST_TUBE_RADIUS = 0.022
TEST_TUBE_HEIGHT = 0.10
TEST_TUBE_RIM_RADIUS = 0.025
TEST_TUBE_RIM_THICKNESS = 0.003
TEST_TUBE_LIQUID_HEIGHT = 0.055
DESTINATION_SPOT_RADIUS = 0.045
DESTINATION_SPOT_HEIGHT = 0.002
OBJECT_Y_START = -0.45
OBJECT_Y_STEP = 0.12
OBJECT_X = 0.75
OBJECT_Z = 0.65
DESTINATION_X = 0.95
DESTINATION_Z_OFFSET = -0.025

# Gesture timing configuration (seconds)
GESTURE_COOLDOWN = 1.0
UNDO_GESTURE_COOLDOWN = 10.0
GESTURE_HOLD_DURATION = 1.0

# How long a transient panel message stays on screen (seconds). Without this a
# message drawn on one frame vanishes in about 30 ms and is never read.
STATUS_MESSAGE_DURATION = 2.5

TARGET_EULER = [0, 1.01 * 3.141592653589793, 0]
HOVER_Z = 0.97
GRAB_Z = 0.97
LIFT_Z = 1.15
