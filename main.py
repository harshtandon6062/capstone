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
        basePosition=[1.0, -0.2, 0.0],
        baseOrientation=[0, 0, 0.7071, 0.7071]
    )

    kuka_id = p.loadURDF(
        "kuka_iiwa/model_vr_limits.urdf",
        1.400000, -0.200000, 0.600000,
        0.000000, 0.000000, 0.000000, 1.000000
    )

    gripper_id = p.loadSDF(
        "gripper/wsg50_one_motor_gripper_new_free_base.sdf"
    )[0]

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
    init_joint_pos = [-0.0, -0.0, 0.0, 1.570793, 0.0, -1.036725, 0.000001]
    for j in range(p.getNumJoints(kuka_id)):
        p.resetJointState(kuka_id, j, init_joint_pos[j])
        p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, init_joint_pos[j], 0)

    p.resetBasePositionAndOrientation(
        gripper_id,
        [0.923103, -0.200000, 1.250036],
        [-0.000000, 0.964531, -0.000002, -0.263970]
    )
    init_gripper_pos = [0.0, -0.011130, -0.206421, 0.205143, -0.009999, 0.0, -0.010055, 0.0]
    for j in range(p.getNumJoints(gripper_id)):
        p.resetJointState(gripper_id, j, init_gripper_pos[j])
        p.setJointMotorControl2(gripper_id, j, p.POSITION_CONTROL, init_gripper_pos[j], 0)

    num_kuka_joints = p.getNumJoints(kuka_id)

    # ── Spawn cubes ──
    CUBE_COLORS = [
        [1.0, 0.0, 0.0, 1], [0.0, 0.7, 0.0, 1], [0.0, 0.0, 1.0, 1],
        [1.0, 0.9, 0.0, 1], [1.0, 0.0, 1.0, 1],
    ]
    source_positions = []
    cubes = []
    CUBE_Y_START, CUBE_Y_STEP, CUBE_X, CUBE_Z = -0.45, 0.12, 0.75, 0.65

    for i in range(5):
        y = CUBE_Y_START + i * CUBE_Y_STEP
        pos = [CUBE_X, y, CUBE_Z]
        source_positions.append(pos)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025],
                                   rgbaColor=CUBE_COLORS[i])
        cubes.append(p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col,
                                        baseVisualShapeIndex=vis, basePosition=pos))

    # ── Destination spots ──
    dest_positions = []
    DEST_X = 0.95
    for i in range(5):
        y = CUBE_Y_START + i * CUBE_Y_STEP
        pos = [DEST_X, y, CUBE_Z]
        dest_positions.append(pos)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.002],
                                   rgbaColor=[0.5, 0.5, 0.5, 0.5])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                          basePosition=[pos[0], pos[1], 0.625])

    # ── Camera ──
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=-40, cameraPitch=-35,
                                  cameraTargetPosition=[0.85, -0.15, 0.5])
    for _ in range(480):
        p.stepSimulation()

    # ──────────────────────────────────────────────
    # ROBOT CONTROL FUNCTIONS (closures over local state)
    # ──────────────────────────────────────────────
    TARGET_ORN = p.getQuaternionFromEuler([0, 1.01 * math.pi, 0])
    HOVER_Z, GRAB_Z, LIFT_Z = 0.97, 0.97, 1.15

    def reset_robot():
        for j in range(p.getNumJoints(kuka_id)):
            p.resetJointState(kuka_id, j, init_joint_pos[j])
            p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, init_joint_pos[j], 0)
        p.resetBasePositionAndOrientation(gripper_id, [0.923103, -0.2, 1.250036],
                                           [-0.0, 0.964531, -0.000002, -0.263970])
        for j in range(p.getNumJoints(gripper_id)):
            p.resetJointState(gripper_id, j, init_gripper_pos[j])
            p.setJointMotorControl2(gripper_id, j, p.POSITION_CONTROL, init_gripper_pos[j], 0)
        for _ in range(120):
            p.stepSimulation()

    def move_to(target_pos, gripper_open, steps=150):
        gv = 0 if gripper_open else 1
        jp = p.calculateInverseKinematics(kuka_id, 6, target_pos, TARGET_ORN)
        for j in range(num_kuka_joints):
            p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, jp[j])
        p.setJointMotorControl2(gripper_id, 4, p.POSITION_CONTROL, gv * 0.05, force=100)
        p.setJointMotorControl2(gripper_id, 6, p.POSITION_CONTROL, gv * 0.05, force=100)
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1 / 480)

    def do_pick_and_place(src, dst, cube_id):
        move_to([src[0], src[1], HOVER_Z], True, 200)
        move_to([src[0], src[1], GRAB_Z], False, 150)
        move_to([src[0], src[1], LIFT_Z], False, 200)
        nw = max(5, int(abs(dst[1] - src[1]) / 0.05))
        for w in range(nw + 1):
            t = w / nw
            move_to([src[0] + t*(dst[0]-src[0]), src[1] + t*(dst[1]-src[1]), LIFT_Z], False, 80)
        move_to([dst[0], dst[1], GRAB_Z], False, 200)
        move_to([dst[0], dst[1], GRAB_Z], True, 100)
        move_to([dst[0], dst[1], HOVER_Z], True, 100)
        reset_robot()

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

        # ── Gesture Navigation ──
        if now - last_gesture_time > gesture_cooldown and system_state not in ("EXECUTING",):
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
        if system_state == "EXECUTING":
            do_pick_and_place(source_positions[source_idx], dest_positions[dest_idx], cubes[source_idx])
            block_placed[source_idx] = True
            system_state = "SELECT_SOURCE"
            source_idx = None
            dest_idx = None
            selected_idx = 0

        p.stepSimulation()

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
        ui = draw_ui(system_state, gesture, selected_idx, source_idx, dest_idx, block_placed)
        cv2.imshow("Pick and Place", np.vstack((frame, ui)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            blue_glove_mode = not blue_glove_mode
            print(f"  Blue glove mode: {'ON' if blue_glove_mode else 'OFF'}")
        elif key == ord('r'):
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
                p.stepSimulation()
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

    # Cleanup
    cap.release()
    cv2.destroyWindow("Pick and Place")
    p.disconnect()
    os.chdir(old_cwd)
    print("Pick-and-place module closed.")


if __name__ == "__main__":
    run_pick_and_place()
