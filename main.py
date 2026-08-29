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
import math
import sys
import os
import time

# Add integrated dir to path
sys.path.insert(0, os.path.dirname(__file__))

# gesture_module is imported lazily inside run_pick_and_place()
# because it creates a HandLandmarker at import time and needs CWD set first

from ui_module import draw_ui
from safety_controller import EmergencyStopError, SafetyController, SafetyState
from robot_controller import RobotController
from commands import CommandInvoker, CommandMapper
from object_registry import ObjectRegistry
from perception import SimulatedPerception
from config import (
    CUBE_COLOR_NAMES,
    DESTINATION_SPOT_COLOR_NAMES,
    STATUS_MESSAGE_DURATION,
    CUBE_COLORS_RGBA,
    CUBE_HALF_EXTENTS,
    DESTINATION_SPOT_COLORS_RGBA,
    DESTINATION_SPOT_HALF_EXTENTS,
    DESTINATION_X,
    DESTINATION_Z_OFFSET,
    GESTURE_COOLDOWN,
    GESTURE_HOLD_DURATION,
    NAVIGATION_COOLDOWN,
    NAVIGATION_HOLD_DURATION,
    DYNAMIC_PREDICT_EVERY,
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
    UNDO_GESTURE_COOLDOWN,
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
    """Make every table visual link a clean, solid white without its dark mesh texture."""
    for link_index in range(-1, p.getNumJoints(table_id)):
        p.changeVisualShape(
            table_id,
            link_index,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            textureUniqueId=-1,
        )


def _camera_backends():
    """Backends to try, in order, for whichever platform we are on.

    cv2.CAP_DSHOW is DirectShow and exists only on Windows; requesting it on
    Linux or macOS fails to open the device at all. cv2.CAP_ANY lets OpenCV pick
    the right backend for the host, so it goes first and the rest are fallbacks.
    """
    backends = [("auto", cv2.CAP_ANY)]
    if sys.platform.startswith("win"):
        backends.append(("dshow", cv2.CAP_DSHOW))
    elif sys.platform.startswith("linux"):
        backends.append(("v4l2", getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)))
    return backends


def open_camera():
    """Open the webcam using a backend the host actually supports."""
    print("[START] camera index=0", flush=True)

    candidate = None
    for name, backend in _camera_backends():
        probe = cv2.VideoCapture(0, backend)
        if probe.isOpened():
            print(f"[CAMERA] opened with backend={name}", flush=True)
            candidate = probe
            break
        probe.release()

    if candidate is None:
        print("[WARN] camera unavailable - gesture control disabled", flush=True)
        return None

    candidate.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    candidate.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = candidate.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = candidate.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame = None
    ret = False
    for _ in range(20):
        ret, frame = candidate.read()

    if ret and frame is not None:
        frame_min = int(frame.min())
        frame_max = int(frame.max())
        frame_mean = float(frame.mean())
        print(
            f"[CAMERA] ret={ret} frame_shape={frame.shape} dtype={frame.dtype} "
            f"min={frame_min} max={frame_max} mean={frame_mean:.3f} "
            f"size={width}x{height} fourcc={candidate.get(cv2.CAP_PROP_FOURCC)}",
            flush=True,
        )
        if frame.ndim == 3 and frame.shape[2] == 3 and frame_max - frame_min > 5 and frame_mean > 1.0:
            print("[OK] camera", flush=True)
            return candidate

    print("[WARN] camera unavailable - gesture control disabled", flush=True)
    candidate.release()
    return None


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

    print("[START] application", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    old_cwd = os.getcwd()
    os.chdir(script_dir)

    print("[START] camera", flush=True)
    cap = open_camera()

    try:
        print("[START] gesture modules", flush=True)
        from gesture_controller import GestureController
        from gesture_module import detect_gesture
        from hand_landmark_provider import HandLandmarkProvider
        print("[OK] gesture modules", flush=True)

        print("[START] landmark provider", flush=True)
        landmark_provider = HandLandmarkProvider(
            model_asset_path=os.path.join(script_dir, "hand_landmarker.task")
        )
        print("[OK] landmark provider", flush=True)

        print("[START] gesture model", flush=True)
        from model_loader import load_gesture_model

        dynamic_model = load_gesture_model(os.path.join(script_dir, "gesture_landmark_model.h5"))
        dynamic_classes = np.load(os.path.join(script_dir, "classes.npy"))
        dynamic_gesture_controller = GestureController(
            dynamic_classes=list(dynamic_classes), cooldown=0.35
        )
        print(f"[OK] gesture model: {list(dynamic_classes)}", flush=True)
    except Exception as error:
        print(f"[ERROR] gesture initialization: {error!r}", flush=True)
        if cap is not None:
            cap.release()
        os.chdir(old_cwd)
        return False

    # ──────────────────────────────────────────────
    # PYBULLET SETUP
    # ──────────────────────────────────────────────
    print("[START] PyBullet", flush=True)
    try:
        physics_client = p.connect(p.GUI, options="--opengl2")
        if physics_client < 0:
            raise RuntimeError("p.connect(p.GUI) returned an invalid client id")
    except Exception as error:
        print(f"[ERROR] PyBullet initialization: {error!r}", flush=True)
        landmark_provider.close()
        cap.release()
        os.chdir(old_cwd)
        return False
    print("[OK] PyBullet", flush=True)
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

    print("[START] scene", flush=True)
    # The perception source is what the rest of the application asks about the
    # workspace. Today it reads PyBullet directly; swapping in a camera-backed
    # source later changes nothing below this point.
    perception = SimulatedPerception(p)

    source_positions = []
    for i in range(OBJECT_COUNT):
        y = OBJECT_Y_START + i * OBJECT_Y_STEP
        pos = [OBJECT_X, y, OBJECT_Z]
        source_positions.append(pos)
        color = CUBE_COLORS_RGBA[i]
        name = CUBE_COLOR_NAMES[i]
        perception.add_source(
            handle=create_test_tube(pos, color),
            label=f"{name.title()} tube",
            color_name=name,
            color_rgba=color,
        )

    # ── White table and destination spots ──
    create_table(table_id)
    for i in range(OBJECT_COUNT):
        y = OBJECT_Y_START + i * OBJECT_Y_STEP
        pos = [DESTINATION_X, y, OBJECT_Z]
        color = DESTINATION_SPOT_COLORS_RGBA[i]
        name = DESTINATION_SPOT_COLOR_NAMES[i]
        perception.add_destination(
            handle=create_destination_spot(pos, color),
            label=f"{name.title()} spot",
            color_name=name,
            color_rgba=color,
            position=pos,
        )

    registry = ObjectRegistry(perception)
    source_handles = [obj.handle for obj in registry.sources]
    print(f"[OK] scene: {registry.count('source')} tubes, "
          f"{registry.count('destination')} spots", flush=True)

    # ── Camera ──
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=-40, cameraPitch=-35,
                                  cameraTargetPosition=[0.85, -0.15, 0.5])
    for _ in range(480):
        p.stepSimulation()

    # RobotController is the only PyBullet-facing object used by commands.
    safety_poll = lambda: None
    robot_controller = RobotController(
        p,
        kuka_id,
        gripper_id,
        safety,
        init_joint_pos,
        init_gripper_pos,
        safety_poll,
    )
    command_mapper = CommandMapper(robot_controller)
    command_invoker = CommandInvoker()

    # ──────────────────────────────────────────────
    # STATE MACHINE + MAIN LOOP
    # ──────────────────────────────────────────────
    system_state = "SELECT_SOURCE"
    selected_handle = registry.first_available("source")
    source_handle = None
    dest_handle = None
    last_gesture_time = 0
    gesture_cooldown = GESTURE_COOLDOWN
    undo_gesture_cooldown = UNDO_GESTURE_COOLDOWN
    gesture_hold_duration = GESTURE_HOLD_DURATION
    held_gesture = None
    hold_start_time = None
    blue_glove_mode = True

    cv2.namedWindow("Pick and Place", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pick and Place", 640, 700)

    def poll_safety_input():
        key = cv2.waitKey(1) & 0xFF
        safety.handle_key(key)

    safety_poll = poll_safety_input
    robot_controller.set_safety_poll(safety_poll)
    previous_safety_gesture = None
    sequence_data = []
    dynamic_sequence_length = 20
    dynamic_confidence = 0.0
    current_dynamic_gesture = "none"
    status_message = ""
    status_message_time = 0.0
    frame_index = 0
    hold_progress = 0.0

    def set_status(message):
        """Show a panel message long enough for a human to actually read it."""
        nonlocal status_message, status_message_time
        status_message = message
        status_message_time = time.time()

    def selection_kind():
        return "source" if system_state in ("SELECT_SOURCE", "CONFIRM_SOURCE") else "destination"

    def release_placement(command):
        """Return the tube and spot a command consumed back to the available pool."""
        registry.release(command.source_object)
        for spot in registry.destinations:
            if list(spot.position) == list(command.destination):
                registry.release(spot.handle)
                break

    print("[START] application loop", flush=True)
    print("=" * 50)
    print("PICK AND PLACE MODE")
    print("Point L/R to navigate, Pinch to select")
    print("Arrow keys + Enter for keyboard | R=reset | B=glove | Q=back")
    print("=" * 50)

    camera_diagnostic_count = 0
    last_camera_diagnostic = 0.0
    camera_connected = cap is not None

    while True:
        raw_frame = None
        ret = False
        if cap is not None:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                print(f"[CAMERA] read failure ret={ret} frame_is_none={raw_frame is None}", flush=True)
                cap.release()
                cap = None
                camera_connected = False

        now = time.time()
        if raw_frame is not None:
            camera_diagnostic_count += 1
        if raw_frame is not None and (camera_diagnostic_count <= 5 or now - last_camera_diagnostic >= 5.0):
            frame_min = int(raw_frame.min())
            frame_max = int(raw_frame.max())
            frame_mean = float(raw_frame.mean())
            print(
                f"[CAMERA] ret={ret} frame_shape={raw_frame.shape} dtype={raw_frame.dtype} "
                f"min={frame_min} max={frame_max} mean={frame_mean:.3f}",
                flush=True,
            )
            if frame_max == 0:
                print("[CAMERA] warning: camera returned a completely black frame", flush=True)
            elif frame_max - frame_min <= 5:
                print("[CAMERA] warning: camera returned a nearly uniform frame", flush=True)
            last_camera_diagnostic = now

        frame = cv2.flip(raw_frame, 1) if raw_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        # Blue glove: swap B and R channels so MediaPipe detects blue-gloved hands
        if blue_glove_mode:
            detect_frame = np.ascontiguousarray(frame[:, :, [2, 1, 0]])
        else:
            detect_frame = frame

        gesture = "unknown"
        dynamic_landmarks = np.zeros(63, dtype=np.float32)
        if raw_frame is not None:
            landmark_provider.update_from_frame(detect_frame)
            gesture = detect_gesture(detect_frame, provider=landmark_provider)
            dynamic_landmarks = landmark_provider.latest_landmarks.copy()
        sequence_data.append(dynamic_landmarks)
        if len(sequence_data) > dynamic_sequence_length:
            sequence_data.pop(0)

        # The LSTM costs ~80 ms, which was half the frame budget, and on this
        # screen its result is only displayed - the state machine below runs on
        # static gestures. Sampling it periodically keeps the label alive without
        # paying that cost on every frame.
        frame_index += 1
        if (len(sequence_data) == dynamic_sequence_length
                and frame_index % DYNAMIC_PREDICT_EVERY == 0):
            input_data = np.expand_dims(np.array(sequence_data), axis=0)
            preds = dynamic_model.predict(input_data, verbose=0)[0]
            idx = np.argmax(preds)
            dynamic_confidence = float(preds[idx])
            current_dynamic_gesture = str(dynamic_classes[idx]) if preds[idx] > 0.60 else "none"

        static_event = dynamic_gesture_controller.handle_static_gesture(gesture)
        dynamic_event = dynamic_gesture_controller.handle_dynamic_gesture(current_dynamic_gesture, dynamic_confidence)
        command = dynamic_gesture_controller.resolve_command(static_event, dynamic_event)

        if command is not None:
            print(f"Gesture command -> {command['source']}:{command['command']} ({command['confidence']:.2f})")

        if gesture == "unknown":
            held_gesture = None
            hold_start_time = None
        elif held_gesture != gesture:
            held_gesture = gesture
            hold_start_time = now

        # Navigating is free to undo, so it fires quickly. Confirming a
        # destination starts the robot, so it stays a deliberate hold.
        committing = system_state == "CONFIRM_DEST" and gesture == "thumbs_up"
        required_hold = gesture_hold_duration if committing else NAVIGATION_HOLD_DURATION
        required_cooldown = gesture_cooldown if committing else NAVIGATION_COOLDOWN

        hold_elapsed = 0.0 if hold_start_time is None else now - hold_start_time
        hold_progress = min(1.0, hold_elapsed / required_hold) if required_hold else 1.0
        gesture_held = (
            held_gesture == gesture
            and hold_start_time is not None
            and hold_elapsed >= required_hold
        )

        safety.handle_gesture(gesture, previous_safety_gesture)
        previous_safety_gesture = gesture

        # An emergency stop must not leave a confirmed action queued. Clearing
        # the stop should hand control back to the operator, not silently launch
        # the move that was interrupted.
        if safety.consume_estop_interrupt():
            if system_state == "EXECUTING":
                set_status("STOPPED - RE-SELECT")
            system_state = "SELECT_SOURCE"
            source_handle = None
            dest_handle = None
            selected_handle = registry.first_available("source")

        if status_message and now - status_message_time > STATUS_MESSAGE_DURATION:
            status_message = ""

        # ── Gesture Navigation ──
        if (safety.state is SafetyState.RUNNING
            and gesture_held
            and now - last_gesture_time > required_cooldown
            and system_state not in ("EXECUTING",)):
            if gesture == "point_down":
                if command_invoker.can_undo:
                    if now - last_gesture_time > undo_gesture_cooldown:
                        if command_invoker.undo():
                            print("Last pick-and-place undone via gesture.")
                            release_placement(command_invoker.last_undone_command)
                            set_status("UNDO EXECUTED")
                        else:
                            set_status("UNDO FAILED")
                        last_gesture_time = now
                else:
                    # No-op: do not block future gestures or leave the menu stuck.
                    last_gesture_time = now

            if system_state in ("SELECT_SOURCE", "SELECT_DEST"):
                kind = selection_kind()
                if gesture in ("point_right", "point_left"):
                    step = 1 if gesture == "point_right" else -1
                    # Returns None when nothing is left; it never spins.
                    next_handle = registry.next_available(kind, selected_handle, step)
                    if next_handle is None:
                        set_status(f"NO {kind.upper()}S LEFT")
                    else:
                        selected_handle = next_handle
                    last_gesture_time = now
                elif gesture == "pinch":
                    chosen = registry.by_handle(selected_handle)
                    if chosen is not None and chosen.available:
                        if system_state == "SELECT_SOURCE":
                            source_handle = selected_handle
                            system_state = "CONFIRM_SOURCE"
                        else:
                            dest_handle = selected_handle
                            system_state = "CONFIRM_DEST"
                        print(f"Highlighted: {chosen.label} — Thumbs up to confirm, Thumb left to cancel")
                    last_gesture_time = now

            elif system_state == "CONFIRM_SOURCE":
                if gesture == "thumbs_up":
                    system_state = "SELECT_DEST"
                    selected_handle = registry.first_available("destination")
                    chosen = registry.by_handle(source_handle)
                    print(f"Source CONFIRMED: {chosen.label if chosen else source_handle}")
                    last_gesture_time = now
                elif gesture == "thumb_left":
                    system_state = "SELECT_SOURCE"
                    source_handle = None
                    print("Source cancelled — re-select")
                    last_gesture_time = now

            elif system_state == "CONFIRM_DEST":
                if gesture == "thumbs_up":
                    system_state = "EXECUTING"
                    chosen = registry.by_handle(dest_handle)
                    print(f"Destination CONFIRMED: {chosen.label if chosen else dest_handle}")
                    last_gesture_time = now
                elif gesture == "thumb_left":
                    system_state = "SELECT_DEST"
                    dest_handle = None
                    print("Destination cancelled — re-select")
                    last_gesture_time = now

        # ── Execute ──
        if system_state == "EXECUTING" and safety.state is SafetyState.RUNNING:
            destination = registry.by_handle(dest_handle)
            completed = False
            try:
                command = command_mapper.pick_and_place(
                    source_handle,
                    list(destination.position),
                )
                completed = command_invoker.execute(command)
            except EmergencyStopError as error:
                print(error)
            if completed:
                registry.consume(source_handle)
                registry.consume(dest_handle)
                set_status("PLACED")
            else:
                set_status("MOVE FAILED")
            system_state = "SELECT_SOURCE"
            source_handle = None
            dest_handle = None
            selected_handle = registry.first_available("source")

        # Never block here: the camera and gesture pipeline must keep running
        # while paused, otherwise the webcam freezes and a gesture-triggered
        # pause cannot be released by gesture.
        safety.step_if_running()

        # ── Display ──
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Dynamic: {current_dynamic_gesture} ({dynamic_confidence:.2f})",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Show confirm hint, naming the object rather than an index
        pending = None
        if system_state == "CONFIRM_SOURCE":
            pending = registry.by_handle(source_handle)
        elif system_state == "CONFIRM_DEST":
            pending = registry.by_handle(dest_handle)
        if pending is not None:
            cv2.putText(frame, f"Confirm {pending.label}? Thumbs Up / Thumb Left", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if status_message:
            color = (0, 0, 255) if "FAIL" in status_message or "STOPPED" in status_message else (0, 255, 255)
            cv2.putText(frame, status_message, (10, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        glove_text = "GLOVE: ON" if blue_glove_mode else "GLOVE: OFF"
        cv2.putText(frame, glove_text, (520, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        display_state = (safety.state_name
                 if safety.state is not SafetyState.RUNNING
                 else system_state)
        registry.refresh()
        ui = draw_ui(
            display_state,
            gesture,
            registry,
            selected_handle,
            source_handle,
            dest_handle,
            command_invoker.can_undo,
            status_message,
            hold_progress,
        )
        if camera_connected:
            cv2.imshow("Webcam", frame)
        cv2.imshow("Pick and Place", ui)

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
            for handle, home in zip(source_handles, source_positions):
                p.resetBasePositionAndOrientation(handle, home, [0, 0, 0, 1])
                p.resetBaseVelocity(handle, [0, 0, 0], [0, 0, 0])
            robot_controller.reset_robot()
            command_invoker.clear_history()
            registry.reset().refresh()
            system_state = "SELECT_SOURCE"
            source_handle = None
            dest_handle = None
            selected_handle = registry.first_available("source")
            set_status("SCENE RESET")
            for _ in range(240):
                safety.step_if_running()
        elif key == ord('u'):
            if safety.state is SafetyState.EMERGENCY_STOPPED:
                print("Emergency stop remains active; press E to reset safety first.")
            elif command_invoker.undo():
                print("Last pick-and-place undone.")
                release_placement(command_invoker.last_undone_command)
                set_status("UNDO EXECUTED")
            else:
                print("Nothing to undo.")
                set_status("NOTHING TO UNDO")
        elif key == 8:  # backspace = go back (like thumb_left)
            if system_state == "CONFIRM_SOURCE":
                system_state = "SELECT_SOURCE"
                source_handle = None
            elif system_state == "CONFIRM_DEST":
                system_state = "SELECT_DEST"
                dest_handle = None
        elif system_state in ("SELECT_SOURCE", "SELECT_DEST"):
            kind = selection_kind()
            if key == 83 or key == ord('d'):
                nxt = registry.next_available(kind, selected_handle, 1)
                if nxt is None:
                    set_status(f"NO {kind.upper()}S LEFT")
                else:
                    selected_handle = nxt
            elif key == 81 or key == ord('a'):
                nxt = registry.next_available(kind, selected_handle, -1)
                if nxt is None:
                    set_status(f"NO {kind.upper()}S LEFT")
                else:
                    selected_handle = nxt
            elif key == 13 or key == ord(' '):
                chosen = registry.by_handle(selected_handle)
                if chosen is not None and chosen.available:
                    if system_state == "SELECT_SOURCE":
                        source_handle = selected_handle
                        system_state = "CONFIRM_SOURCE"
                    else:
                        dest_handle = selected_handle
                        system_state = "CONFIRM_DEST"
        elif system_state in ("CONFIRM_SOURCE", "CONFIRM_DEST"):
            if key == 13 or key == ord(' '):  # enter = confirm (like thumbs_up)
                if system_state == "CONFIRM_SOURCE":
                    system_state = "SELECT_DEST"
                    selected_handle = registry.first_available("destination")
                elif system_state == "CONFIRM_DEST":
                    system_state = "EXECUTING"

        if safety.quit_requested:
            break

    # Cleanup. cap is None whenever the camera never opened or dropped out
    # mid-run, so every teardown step has to tolerate a missing resource.
    if cap is not None:
        cap.release()
    if camera_connected:
        cv2.destroyWindow("Webcam")
    cv2.destroyWindow("Pick and Place")
    landmark_provider.close()
    p.disconnect()
    os.chdir(old_cwd)
    print("Pick-and-place module closed.")


if __name__ == "__main__":
    run_pick_and_place()
