"""
Pick and Place Module
=====================
Callable from launcher.py. Wraps the entire pick-and-place flow into
run_pick_and_place() which sets up PyBullet, runs the gesture control loop,
and cleans up when done.
"""
import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import math
import sys
import os

# Add integrated dir to path
sys.path.insert(0, os.path.dirname(__file__))

# gesture_module is imported lazily inside run_pick_and_place()
# because it creates a HandLandmarker at import time and needs CWD set first

from ui_module import draw_ui
from safety_controller import EmergencyStopError, SafetyController, SafetyState
from config import (
    CUBE_COLORS_RGBA,
    CUBE_HALF_EXTENTS,
    DESTINATION_SPOT_COLORS_RGBA,
    DESTINATION_SPOT_HALF_EXTENTS,
    DESTINATION_X,
    DESTINATION_Z_OFFSET,
    GRAB_Z,
    GRIPPER_BASE_ORIENTATION,
    GRIPPER_BASE_POSITION,
    HOVER_Z,
    INITIAL_GRIPPER_JOINT_POSITIONS,
    INITIAL_KUKA_JOINT_POSITIONS,
    LIFT_Z,
    OBJECT_COUNT,
    OBJECT_X,
    OBJECT_Y_START,
    OBJECT_Y_STEP,
    OBJECT_Z,
    ROBOT_BASE_ORIENTATION,
    ROBOT_BASE_POSITION,
    TABLE_BASE_ORIENTATION,
    TABLE_BASE_POSITION,
    DESTINATION_SPOT_HEIGHT,
    DESTINATION_SPOT_RADIUS,
    TEST_TUBE_HEIGHT,
    TEST_TUBE_LIQUID_HEIGHT,
    TEST_TUBE_RADIUS,
    TEST_TUBE_RIM_RADIUS,
    TEST_TUBE_RIM_THICKNESS,
    TARGET_EULER,
)


def create_table(table_id):
    """Make every table visual link a clean, solid white."""
    for link_index in range(-1, p.getNumJoints(table_id)):
        p.changeVisualShape(table_id, link_index, rgbaColor=[1.0, 1.0, 1.0, 1.0])


def create_test_tube(position, color):
    """Create a grabbable tube with simple collision and layered visuals."""
    collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=TEST_TUBE_RADIUS,
        height=TEST_TUBE_HEIGHT,
        collisionFramePosition=[0, 0, TEST_TUBE_HEIGHT / 2],
    )
    body_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=TEST_TUBE_RADIUS,
        length=TEST_TUBE_HEIGHT,
        rgbaColor=color,
        visualFramePosition=[0, 0, TEST_TUBE_HEIGHT / 2],
    )
    liquid_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=TEST_TUBE_RADIUS * 0.82,
        length=TEST_TUBE_LIQUID_HEIGHT,
        rgbaColor=[color[0] * 0.75, color[1] * 0.75, color[2] * 0.75, 1.0],
    )
    rim_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=TEST_TUBE_RIM_RADIUS,
        length=TEST_TUBE_RIM_THICKNESS * 2,
        rgbaColor=[0.9, 0.9, 0.9, 1.0],
    )
    tube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=body_visual,
        basePosition=position,
        linkMasses=[0, 0],
        linkCollisionShapeIndices=[-1, -1],
        linkVisualShapeIndices=[liquid_visual, rim_visual],
        linkPositions=[
            [0, 0, TEST_TUBE_LIQUID_HEIGHT / 2 + 0.003],
            [0, 0, TEST_TUBE_HEIGHT + TEST_TUBE_RIM_THICKNESS],
        ],
        linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkParentIndices=[0, 0],
        linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
        linkJointAxis=[[0, 0, 0], [0, 0, 0]],
    )
    return tube_id


def create_destination_spot(position, color):
    """Create a circular placement marker with no collision geometry."""
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=DESTINATION_SPOT_RADIUS,
        length=DESTINATION_SPOT_HEIGHT,
        rgbaColor=color,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=[position[0], position[1], position[2] - 0.025],
    )


def run_pick_and_place():
    """Run the full pick-and-place simulation. Returns when user presses 'q'."""

    # Save and set CWD so gesture_module finds hand_landmarker.task locally
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Import here (not at top level) because gesture_module creates a
    # HandLandmarker at import time and needs CWD set
    from gesture_module import detect_gesture

    # ──────────────────────────────────────────────
    # PYBULLET SETUP
    # ──────────────────────────────────────────────
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)

    plane_id = p.loadURDF("plane.urdf")
    table_id = p.loadURDF(
        "table/table.urdf",
        basePosition=TABLE_BASE_POSITION,
        baseOrientation=TABLE_BASE_ORIENTATION
    )

    kuka_id = p.loadURDF(
        "kuka_iiwa/model_vr_limits.urdf",
        *ROBOT_BASE_POSITION,
        *ROBOT_BASE_ORIENTATION,
    )

    gripper_id = p.loadSDF(
        "gripper/wsg50_one_motor_gripper_new_free_base.sdf"
    )[0]
    safety = SafetyController(p, kuka_id, gripper_id)

    # ── Attach gripper ──
    p.createConstraint(
        kuka_id, 6, gripper_id, 0,
        p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0]
    )
    cid2 = p.createConstraint(
        gripper_id, 4, gripper_id, 6,
        jointType=p.JOINT_GEAR, jointAxis=[1, 1, 1],
        parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]
    )
    p.changeConstraint(cid2, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)

    # ── Initial poses ──
    init_joint_pos = INITIAL_KUKA_JOINT_POSITIONS
    for j in range(p.getNumJoints(kuka_id)):
        p.resetJointState(kuka_id, j, init_joint_pos[j])
        p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, init_joint_pos[j], 0)

    p.resetBasePositionAndOrientation(
        gripper_id,
        GRIPPER_BASE_POSITION,
        GRIPPER_BASE_ORIENTATION
    )
    init_gripper_pos = INITIAL_GRIPPER_JOINT_POSITIONS
    for j in range(p.getNumJoints(gripper_id)):
        p.resetJointState(gripper_id, j, init_gripper_pos[j])
        p.setJointMotorControl2(gripper_id, j, p.POSITION_CONTROL, init_gripper_pos[j], 0)

    num_kuka_joints = p.getNumJoints(kuka_id)

    # ── Spawn cubes ──
    source_positions = []
    cubes = []

    for i in range(OBJECT_COUNT):
        y = OBJECT_Y_START + i * OBJECT_Y_STEP
        pos = [OBJECT_X, y, OBJECT_Z]
        source_positions.append(pos)
        cubes.append(create_test_tube(pos, CUBE_COLORS_RGBA[i]))

    # ── White table and destination spots ──
    create_table(table_id)
    dest_positions = []
    for i in range(OBJECT_COUNT):
        y = OBJECT_Y_START + i * OBJECT_Y_STEP
        pos = [DESTINATION_X, y, OBJECT_Z]
        dest_positions.append(pos)
        create_destination_spot(pos, DESTINATION_SPOT_COLORS_RGBA[i])

    # ── Camera ──
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=-40, cameraPitch=-35,
                                  cameraTargetPosition=[0.85, -0.15, 0.5])
    for _ in range(480):
        p.stepSimulation()

    # ──────────────────────────────────────────────
    # ROBOT CONTROL FUNCTIONS (closures over local state)
    # ──────────────────────────────────────────────
    TARGET_ORN = p.getQuaternionFromEuler(TARGET_EULER)
    safety_poll = lambda: None

    def reset_robot():
        safety.require_motion()
        for j in range(p.getNumJoints(kuka_id)):
            p.resetJointState(kuka_id, j, init_joint_pos[j])
            p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, init_joint_pos[j], 0)
        p.resetBasePositionAndOrientation(gripper_id, [0.923103, -0.2, 1.250036],
                                           [-0.0, 0.964531, -0.000002, -0.263970])
        for j in range(p.getNumJoints(gripper_id)):
            p.resetJointState(gripper_id, j, init_gripper_pos[j])
            p.setJointMotorControl2(gripper_id, j, p.POSITION_CONTROL, init_gripper_pos[j], 0)
        for _ in range(120):
            if not safety.step_simulation(safety_poll, raise_on_stop=True):
                return False
        return True

    def move_to(target_pos, gripper_open, steps=150):
        if not safety.wait_until_running(safety_poll):
            return False
        gv = 0 if gripper_open else 1
        jp = p.calculateInverseKinematics(kuka_id, 6, target_pos, TARGET_ORN)
        for j in range(num_kuka_joints):
            p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, jp[j])
        p.setJointMotorControl2(gripper_id, 4, p.POSITION_CONTROL, gv * 0.05, force=100)
        p.setJointMotorControl2(gripper_id, 6, p.POSITION_CONTROL, gv * 0.05, force=100)
        for _ in range(steps):
            if not safety.step_simulation(safety_poll, raise_on_stop=True):
                return False
            time.sleep(1 / 480)
        return True

    def do_pick_and_place(src, dst, cube_id):
        if not move_to([src[0], src[1], HOVER_Z], True, 200):
            return False
        if not move_to([src[0], src[1], GRAB_Z], False, 150):
            return False
        if not move_to([src[0], src[1], LIFT_Z], False, 200):
            return False
        nw = max(5, int(abs(dst[1] - src[1]) / 0.05))
        for w in range(nw + 1):
            t = w / nw
            if not move_to([src[0] + t*(dst[0]-src[0]), src[1] + t*(dst[1]-src[1]), LIFT_Z], False, 80):
                return False
        if not move_to([dst[0], dst[1], GRAB_Z], False, 200):
            return False
        if not move_to([dst[0], dst[1], GRAB_Z], True, 100):
            return False
        if not move_to([dst[0], dst[1], HOVER_Z], True, 100):
            return False
        return reset_robot()

    # ──────────────────────────────────────────────
    # STATE MACHINE + MAIN LOOP
    # ──────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        p.disconnect()
        os.chdir(old_cwd)
        return

    system_state = "SELECT_SOURCE"
    selected_idx = 0
    source_idx = None
    dest_idx = None
    block_placed = [False] * 5
    last_gesture_time = 0
    gesture_cooldown = 0.4
    blue_glove_mode = True

    cv2.namedWindow("Pick and Place", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pick and Place", 640, 700)

    def poll_safety_input():
        key = cv2.waitKey(1) & 0xFF
        safety.handle_key(key)

    safety_poll = poll_safety_input
    previous_safety_gesture = None

    print("=" * 50)
    print("PICK AND PLACE MODE")
    print("Point L/R to navigate, Pinch to select")
    print("Arrow keys + Enter for keyboard | R=reset | B=glove | Q=back")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Blue glove: swap B and R channels so MediaPipe detects blue-gloved hands
        if blue_glove_mode:
            detect_frame = np.ascontiguousarray(frame[:, :, [2, 1, 0]])
        else:
            detect_frame = frame

        gesture = detect_gesture(detect_frame)
        now = time.time()

        safety.handle_gesture(gesture, previous_safety_gesture)
        previous_safety_gesture = gesture

        # ── Gesture Navigation ──
        if (safety.state is SafetyState.RUNNING
            and now - last_gesture_time > gesture_cooldown
            and system_state not in ("EXECUTING",)):
            if system_state in ("SELECT_SOURCE", "SELECT_DEST"):
                if gesture == "point_right":
                    selected_idx = min(selected_idx + 1, 4)
                    last_gesture_time = now
                elif gesture == "point_left":
                    selected_idx = max(selected_idx - 1, 0)
                    last_gesture_time = now
                elif gesture == "pinch":
                    if system_state == "SELECT_SOURCE" and not block_placed[selected_idx]:
                        source_idx = selected_idx
                        system_state = "CONFIRM_SOURCE"
                        print(f"Source highlighted: Block {source_idx + 1} — Thumbs up to confirm, Thumb left to cancel")
                    elif system_state == "SELECT_DEST":
                        dest_idx = selected_idx
                        system_state = "CONFIRM_DEST"
                        print(f"Dest highlighted: Spot {dest_idx + 1} — Thumbs up to confirm, Thumb left to cancel")
                    last_gesture_time = now

            elif system_state == "CONFIRM_SOURCE":
                if gesture == "thumbs_up":
                    system_state = "SELECT_DEST"
                    selected_idx = 0
                    print(f"Source CONFIRMED: Block {source_idx + 1}")
                    last_gesture_time = now
                elif gesture == "thumb_left":
                    system_state = "SELECT_SOURCE"
                    source_idx = None
                    print("Source cancelled — re-select")
                    last_gesture_time = now

            elif system_state == "CONFIRM_DEST":
                if gesture == "thumbs_up":
                    system_state = "EXECUTING"
                    print(f"Dest CONFIRMED: Spot {dest_idx + 1}")
                    last_gesture_time = now
                elif gesture == "thumb_left":
                    system_state = "SELECT_DEST"
                    dest_idx = None
                    print("Dest cancelled — re-select")
                    last_gesture_time = now

        # ── Execute ──
        if system_state == "EXECUTING" and safety.state is SafetyState.RUNNING:
            try:
                completed = do_pick_and_place(
                    source_positions[source_idx],
                    dest_positions[dest_idx],
                    cubes[source_idx],
                )
            except EmergencyStopError as error:
                print(error)
                completed = False
            if completed:
                block_placed[source_idx] = True
            system_state = "SELECT_SOURCE"
            source_idx = None
            dest_idx = None
            selected_idx = 0

        if safety.state is not SafetyState.EMERGENCY_STOPPED:
            safety.step_simulation(safety_poll)

        # ── Display ──
        frame = cv2.resize(frame, (640, 480))
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show confirm hint
        if system_state == "CONFIRM_SOURCE":
            cv2.putText(frame, f"Confirm Block {source_idx+1}? Thumbs Up / Thumb Left", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        elif system_state == "CONFIRM_DEST":
            cv2.putText(frame, f"Confirm Spot {dest_idx+1}? Thumbs Up / Thumb Left", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        glove_text = "GLOVE: ON" if blue_glove_mode else "GLOVE: OFF"
        cv2.putText(frame, glove_text, (520, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        display_state = (safety.state_name
                 if safety.state is not SafetyState.RUNNING
                 else system_state)
        ui = draw_ui(display_state, gesture, selected_idx, source_idx, dest_idx, block_placed)
        cv2.imshow("Pick and Place", np.vstack((frame, ui)))

        key = cv2.waitKey(1) & 0xFF
        safety.handle_key(key)
        if key == ord('q'):
            break
        elif key == ord('b'):
            blue_glove_mode = not blue_glove_mode
            print(f"  Blue glove mode: {'ON' if blue_glove_mode else 'OFF'}")
        elif key == ord('r'):
            if safety.state is SafetyState.EMERGENCY_STOPPED:
                print("Emergency stop remains active; press E to reset safety first.")
                continue
            # Reset simulation
            for i, c in enumerate(cubes):
                p.resetBasePositionAndOrientation(c, source_positions[i], [0, 0, 0, 1])
                p.resetBaseVelocity(c, [0, 0, 0], [0, 0, 0])
            reset_robot()
            system_state = "SELECT_SOURCE"
            selected_idx = 0
            source_idx = None
            dest_idx = None
            block_placed = [False] * 5
            for _ in range(240):
                safety.step_simulation(safety_poll)
        elif key == 8:  # backspace = go back (like thumb_left)
            if system_state == "CONFIRM_SOURCE":
                system_state = "SELECT_SOURCE"
                source_idx = None
            elif system_state == "CONFIRM_DEST":
                system_state = "SELECT_DEST"
                dest_idx = None
        elif system_state in ("SELECT_SOURCE", "SELECT_DEST"):
            if key == 83 or key == ord('d'):
                selected_idx = min(selected_idx + 1, 4)
            elif key == 81 or key == ord('a'):
                selected_idx = max(selected_idx - 1, 0)
            elif key == 13 or key == ord(' '):
                if system_state == "SELECT_SOURCE" and not block_placed[selected_idx]:
                    source_idx = selected_idx
                    system_state = "CONFIRM_SOURCE"
                elif system_state == "SELECT_DEST":
                    dest_idx = selected_idx
                    system_state = "CONFIRM_DEST"
        elif system_state in ("CONFIRM_SOURCE", "CONFIRM_DEST"):
            if key == 13 or key == ord(' '):  # enter = confirm (like thumbs_up)
                if system_state == "CONFIRM_SOURCE":
                    system_state = "SELECT_DEST"
                    selected_idx = 0
                elif system_state == "CONFIRM_DEST":
                    system_state = "EXECUTING"

        if safety.quit_requested:
            break

    # Cleanup
    cap.release()
    cv2.destroyWindow("Pick and Place")
    p.disconnect()
    os.chdir(old_cwd)
    print("Pick-and-place module closed.")


if __name__ == "__main__":
    run_pick_and_place()
