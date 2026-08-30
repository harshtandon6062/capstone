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
import sys
import os
import time

# Add integrated dir to path
sys.path.insert(0, os.path.dirname(__file__))

# gesture_module is imported lazily inside run_pick_and_place()
# because it creates a HandLandmarker at import time and needs CWD set first

from ui_module import (ACCENT, DANGER, OK, PANEL_HEIGHT, PANEL_WIDTH,
                       TEXT_BRIGHT, draw_ui)
from ui_text import text as draw_text
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
    DESTINATION_SPOT_COLORS_RGBA,
    DESTINATION_X,
    GESTURE_COOLDOWN,
    GESTURE_HOLD_DURATION,
    NAVIGATION_COOLDOWN,
    NAVIGATION_HOLD_DURATION,
    DYNAMIC_PREDICT_EVERY,
    MOTION_STEPS_PER_FRAME,
    ACTIONS, action_is_irreversible, action_needs_target,
    HOVER_OVER_SELECTION,
    IRREVERSIBLE_HOLD_DURATION,
    GRIPPER_BASE_ORIENTATION,
    GRIPPER_BASE_POSITION,
    INITIAL_GRIPPER_JOINT_POSITIONS,
    INITIAL_KUKA_JOINT_POSITIONS,
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
    TEST_TUBE_CAP_HEIGHT,
    TEST_TUBE_CAP_RADIUS,
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


def open_simulation_window():
    """Open the PyBullet window with a renderer that can actually repaint.

    --opengl2 is the compatibility fallback for machines whose drivers cannot run
    the default renderer. It also silently ignores changeVisualShape: a tube that
    has been emptied or mixed keeps its old colour on screen while the panel shows
    the new one, which reads as the colour change simply not working. Measured on
    this machine: recolouring a tube red to blue changed 0.00 of the rendered
    pixels under --opengl2, and 3.14 under the default renderer.

    So try the default first and only fall back when it will not start.
    """
    for options in ("", "--opengl2"):
        try:
            client = p.connect(p.GUI, options=options)
        except Exception:
            continue
        if client >= 0:
            if options:
                print("[WARN] falling back to --opengl2; tube colour changes will "
                      "not show in the simulation window", flush=True)
            return client
    return -1


def create_test_tube(position, color):
    """Create a grabbable tube whose cap names it and whose liquid is its contents.

    The body shows what the tube holds and is repainted as that changes. The cap
    shows which tube it is and is never repainted. Colour used to do both jobs at
    once, so pouring a tube out erased its identity and every empty tube looked
    the same as every other.

    A translucent glass body was tried first, with the liquid visible inside it.
    The renderer draws the body over the liquid, so full and empty tubes came out
    identical - the contents have to be on the outside to be seen at all.
    """
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
    cap_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=TEST_TUBE_CAP_RADIUS,
        length=TEST_TUBE_CAP_HEIGHT,
        rgbaColor=color,
    )
    tube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=body_visual,
        basePosition=position,
        linkMasses=[0, 0],
        linkCollisionShapeIndices=[-1, -1],
        linkVisualShapeIndices=[liquid_visual, cap_visual],
        linkPositions=[
            [0, 0, TEST_TUBE_LIQUID_HEIGHT / 2 + 0.003],
            [0, 0, TEST_TUBE_HEIGHT + TEST_TUBE_CAP_HEIGHT / 2 - 0.004],
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


def run_pick_and_place(initial_action="move"):
    """Run the full pick-and-place simulation. Returns when user presses 'q'.

    `initial_action` is the action the launcher picked, and it preselects that
    choice rather than skipping past it. Pouring cannot be undone, so choosing it
    on the launcher screen should not also count as confirming it here.
    """

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
        physics_client = open_simulation_window()
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

    p.loadURDF("plane.urdf")
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
    selected_handle = registry.first_selectable("move_source")
    source_handle = None
    dest_handle = None
    last_gesture_time = 0
    gesture_cooldown = GESTURE_COOLDOWN
    undo_gesture_cooldown = UNDO_GESTURE_COOLDOWN
    gesture_hold_duration = GESTURE_HOLD_DURATION
    held_gesture = None
    hold_start_time = None
    blue_glove_mode = True

    # GUI_NORMAL is the same resizable window without the HighGUI toolbar strip,
    # which is chrome the operator has no use for.
    cv2.namedWindow("Pick and Place", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Pick and Place", PANEL_WIDTH, 480 + PANEL_HEIGHT)

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
    # The motion currently in flight, advanced a slice at a time by this loop.
    active_motion = None
    active_kind = None
    # A motion an emergency stop interrupted. It stays suspended, still gripping,
    # until the operator clears the stop and the arm can set the sample down.
    interrupted_motion = None
    # The arm points at whatever is highlighted while the operator is choosing,
    # so they can see which real tube the panel means before confirming.
    hover_motion = None
    hover_handle = None
    # Which action the operator picked for this tube, and where they are in the
    # action list while choosing.
    pending_action = None
    # Which action is highlighted when the operator reaches the action step.
    default_action_index = next(
        (i for i, action in enumerate(ACTIONS) if action["key"] == initial_action), 0
    )
    action_index = default_action_index
    frame_index = 0
    hold_progress = 0.0

    def set_status(message):
        """Show a panel message long enough for a human to actually read it."""
        nonlocal status_message, status_message_time
        status_message = message
        status_message_time = time.time()

    def advance_motion(runner, budget):
        """Run part of the active motion. Returns (finished, result)."""
        for _ in range(budget):
            try:
                next(runner)
            except StopIteration as stop:
                return True, bool(stop.value)
        return False, None

    def selection_purpose():
        """What is being chosen right now, which decides what may be chosen.

        The action is not known yet while a source is being picked, so any tube
        is fair game there; the action only restricts the target.
        """
        if system_state in ("SELECT_SOURCE", "CONFIRM_SOURCE"):
            return "move_source"
        return "pour_target" if pending_action == "pour" else "move_target"

    def selection_exclude():
        """A tube cannot be poured into itself."""
        return source_handle if pending_action == "pour" else None

    def is_selectable(handle):
        options = registry.selectable(selection_purpose(), selection_exclude())
        return any(obj.handle == handle for obj in options)

    def action_blocks():
        """Why each action is unavailable for the tube in hand, if it is.

        Offering an action that cannot work and then silently doing nothing is
        the same failure the launcher had; the panel says why instead.
        """
        blocked = {}
        source = registry.by_handle(source_handle)
        if not registry.can_move():
            blocked["move"] = "spots full"
        if source is None or source.empty:
            blocked["pour"] = "tube is empty"
            blocked["mix"] = "tube is empty"
        elif not registry.can_pour(source_handle):
            blocked["pour"] = "no other tube"
        return blocked

    def usable_action_indices():
        blocked = action_blocks()
        return [i for i, action in enumerate(ACTIONS) if action["key"] not in blocked]

    def step_action(current, step):
        usable = usable_action_indices()
        if not usable:
            return None
        if current in usable:
            return usable[(usable.index(current) + step) % len(usable)]
        return usable[0]

    def choose_action():
        """Commit to the highlighted action and go wherever it needs next.

        Mix acts on the tube already chosen, so there is no destination to pick
        and it starts immediately. Move and pour both need a target first.

        Both the gesture path and the key path call this. When each carried its
        own copy they drifted, and the key path went on asking for a destination
        that mix never used.
        """
        nonlocal pending_action, system_state, selected_handle
        pending_action = ACTIONS[action_index]["key"]
        label = ACTIONS[action_index]["label"]

        if not ACTIONS[action_index]["needs_target"]:
            # No destination to pick does not mean nothing to confirm. Mix cannot
            # be undone, so it stops here for the same deliberate hold pour gets.
            system_state = "CONFIRM_ACTION"
            selected_handle = source_handle
            print(f"Action chosen: {label} - confirm to run")
            return

        # Set the state first: which objects count as selectable is read from it.
        system_state = "SELECT_DEST"
        selected_handle = registry.first_selectable(
            selection_purpose(), exclude=selection_exclude()
        )
        if selected_handle is None:
            set_status("NO TARGETS LEFT")
            system_state = "SELECT_ACTION"
            pending_action = None
        else:
            print(f"Action chosen: {label}")

    def hover_target():
        """Which object the arm should be pointing at, or None to stay still.

        Only while the operator is choosing - during an action the arm is busy
        doing the thing, and pointing at the same time would be a second motion
        fighting the first.
        """
        if not HOVER_OVER_SELECTION or active_motion is not None:
            return None
        if system_state not in ("SELECT_SOURCE", "CONFIRM_SOURCE", "SELECT_ACTION",
                                "CONFIRM_ACTION", "SELECT_DEST", "CONFIRM_DEST"):
            return None
        # The action step is not a choice of object, so keep pointing at the tube
        # the action will be performed on.
        if system_state in ("SELECT_ACTION", "CONFIRM_ACTION"):
            return source_handle
        return selected_handle

    def stop_hovering():
        nonlocal hover_motion, hover_handle
        if hover_motion is not None:
            hover_motion.close()
        hover_motion = None
        hover_handle = None

    def apply_appearance(obj):
        """Make the simulation show what the registry says the tube contains.

        The body and the liquid. Never the cap - that is the tube's name, and
        leaving it alone is what keeps two emptied tubes telling apart.
        """
        if obj.kind != "source":
            return
        p.changeVisualShape(obj.handle, -1, rgbaColor=obj.color_rgba)
        if obj.empty:
            p.changeVisualShape(obj.handle, 0, rgbaColor=[1.0, 1.0, 1.0, 0.0])
        else:
            p.changeVisualShape(
                obj.handle, 0,
                rgbaColor=[c * 0.8 for c in obj.color_rgba[:3]] + [1.0],
            )
        perception.set_source_contents(obj.handle, obj.color_rgba, obj.contents_name)

    def transfer_contents(from_handle, into_handle):
        """Apply a completed pour to the workspace.

        The registry decides what pouring means; this only makes the simulation
        and the perception source agree with it, so the panel and the scene keep
        showing the same thing.
        """
        changed = registry.transfer_contents(from_handle, into_handle)
        if changed is None:
            return
        for obj in changed:
            apply_appearance(obj)

    def mix_contents(source_handle):
        """Apply a completed mix to the workspace."""
        source = registry.by_handle(source_handle)
        if source is None or source.empty:
            return
        # Deliberately no colour change. Stirring one tube does not alter what
        # is in it; the previous mix_colors(c, c) averaged a colour with itself,
        # which is the identity, and only read as though it did something.
        source.contents_name = "MIXED"
        apply_appearance(source)

    print("[START] application loop", flush=True)
    print("=" * 50)
    print("PICK AND PLACE MODE")
    print("Point L/R to navigate, Pinch to select")
    print("Actions: MOVE (undoable) and POUR (cannot be undone)")
    print("Arrow keys + Enter for keyboard | R=reset | B=glove | Q=back")
    print("=" * 50)

    camera_diagnostic_count = 0
    last_camera_diagnostic = 0.0

    while True:
        raw_frame = None
        ret = False
        if cap is not None:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                print(f"[CAMERA] read failure ret={ret} frame_is_none={raw_frame is None}", flush=True)
                cap.release()
                cap = None

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

        direct_dynamic_mix = (
            source_handle is not None
            and active_motion is None
            and safety.state is SafetyState.RUNNING
            and current_dynamic_gesture == "wrist_rotation"
            and dynamic_confidence >= 0.75
            and now - last_gesture_time > gesture_cooldown
        )
        if direct_dynamic_mix:
            # A classifier reporting 0.75 is not consent. The shortcut skips the
            # action menu, not the confirmation that mix cannot be undone.
            pending_action = "mix"
            system_state = "CONFIRM_ACTION"
            selected_handle = source_handle
            last_gesture_time = now
            print("Dynamic wrist rotation detected: MIX selected - confirm to run.")

        if gesture == "unknown":
            held_gesture = None
            hold_start_time = None
        elif held_gesture != gesture:
            held_gesture = gesture
            hold_start_time = now

        # Navigating is free to undo, so it fires quickly. Confirming a
        # destination starts the robot, so it stays a deliberate hold.
        committing = (system_state in ("CONFIRM_DEST", "CONFIRM_ACTION")
                      and gesture == "thumbs_up")
        irreversible = action_is_irreversible(pending_action)
        if committing:
            required_hold = (IRREVERSIBLE_HOLD_DURATION if irreversible
                             else gesture_hold_duration)
            required_cooldown = gesture_cooldown
        else:
            required_hold = NAVIGATION_HOLD_DURATION
            required_cooldown = NAVIGATION_COOLDOWN

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
            if active_motion is not None:
                # Hold on to it rather than closing it here: closing runs the
                # release path, which would drop a tube from mid-air.
                interrupted_motion = active_motion
                active_motion = None
                active_kind = None
            if system_state in ("EXECUTING", "ABORTING"):
                set_status("STOPPED - RE-SELECT")
            system_state = "SELECT_SOURCE"
            source_handle = None
            dest_handle = None
            pending_action = None
            selected_handle = registry.first_selectable("move_source")

        # Once the stop is cleared, put down whatever the interrupted motion was
        # carrying before abandoning it.
        if (interrupted_motion is not None
                and active_motion is None
                and safety.state is SafetyState.RUNNING):
            if robot_controller.is_holding:
                active_motion = robot_controller.abort_safely_steps()
                active_kind = "abort"
                system_state = "ABORTING"
            else:
                interrupted_motion.close()
                interrupted_motion = None

        if status_message and now - status_message_time > STATUS_MESSAGE_DURATION:
            status_message = ""

        # ── Gesture Navigation ──
        if (safety.state is SafetyState.RUNNING
            and gesture_held
            and active_motion is None
            and now - last_gesture_time > required_cooldown
            and system_state not in ("EXECUTING", "ABORTING")):
            if gesture == "point_down":
                if command_invoker.can_undo:
                    if now - last_gesture_time > undo_gesture_cooldown:
                        active_motion = command_invoker.undo_steps()
                        active_kind = "undo"
                        system_state = "EXECUTING"
                        print("Undoing last pick-and-place via gesture.")
                        last_gesture_time = now
                else:
                    # No-op: do not block future gestures or leave the menu stuck.
                    last_gesture_time = now

            if system_state in ("SELECT_SOURCE", "SELECT_DEST"):
                if gesture in ("point_right", "point_left"):
                    step = 1 if gesture == "point_right" else -1
                    # Returns None when nothing is left; it never spins.
                    next_handle = registry.next_selectable(
                        selection_purpose(), selected_handle, step,
                        exclude=selection_exclude()
                    )
                    if next_handle is None:
                        set_status("NOTHING LEFT TO CHOOSE")
                    else:
                        selected_handle = next_handle
                    last_gesture_time = now
                elif gesture == "pinch":
                    chosen = registry.by_handle(selected_handle)
                    if chosen is not None and is_selectable(chosen.handle):
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
                    system_state = "SELECT_ACTION"
                    action_index = default_action_index
                    if action_index not in usable_action_indices():
                        action_index = step_action(action_index, 1) or 0
                    pending_action = None
                    chosen = registry.by_handle(source_handle)
                    print(f"Source CONFIRMED: {chosen.label if chosen else source_handle}")
                    last_gesture_time = now
                elif gesture == "thumb_left":
                    system_state = "SELECT_SOURCE"
                    source_handle = None
                    print("Source cancelled — re-select")
                    last_gesture_time = now

            elif system_state == "SELECT_ACTION":
                if gesture in ("point_right", "point_left"):
                    step = 1 if gesture == "point_right" else -1
                    nxt = step_action(action_index, step)
                    if nxt is None:
                        set_status("NO ACTION POSSIBLE")
                    else:
                        action_index = nxt
                    last_gesture_time = now
                elif gesture == "pinch" and ACTIONS[action_index]["key"] in action_blocks():
                    set_status(action_blocks()[ACTIONS[action_index]["key"]].upper())
                    last_gesture_time = now
                elif gesture == "pinch":
                    choose_action()
                    last_gesture_time = now
                elif gesture == "thumb_left":
                    system_state = "SELECT_SOURCE"
                    source_handle = None
                    pending_action = None
                    print("Cancelled — re-select a tube")
                    last_gesture_time = now

            elif system_state == "CONFIRM_ACTION":
                if gesture == "thumbs_up":
                    system_state = "EXECUTING"
                    chosen = registry.by_handle(source_handle)
                    print(f"Action CONFIRMED on {chosen.label if chosen else source_handle}")
                    last_gesture_time = now
                elif gesture == "thumb_left":
                    system_state = "SELECT_ACTION"
                    pending_action = None
                    print("Action cancelled — re-select")
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

        # ── Point at whatever is highlighted ──
        wanted = hover_target()
        if wanted != hover_handle:
            stop_hovering()
            hover_handle = wanted
            highlighted = registry.by_handle(wanted) if wanted is not None else None
            if highlighted is not None:
                hover_motion = robot_controller.hover_over_steps(
                    highlighted.position[0], highlighted.position[1]
                )

        # ── Execute ──
        # Motions are driven from here a slice at a time rather than run inside
        # one blocking call. That is the whole point: the camera keeps being read
        # and gestures keep being classified while the arm moves, so the stop
        # gesture is reachable during the motion it exists to interrupt.
        if (system_state == "EXECUTING"
                and active_motion is None
                and (not action_needs_target(pending_action)
                     or dest_handle is not None)):
            stop_hovering()
            if pending_action == "mix":
                command = command_mapper.mix(source_handle)
                command.transfer = mix_contents
                active_kind = "mix"
            elif pending_action == "pour":
                origin = registry.by_handle(source_handle)
                command = command_mapper.pour(
                    source_handle,
                    dest_handle,
                    list(origin.position),
                    transfer_contents,
                )
                active_kind = "pour"
            else:
                destination = registry.by_handle(dest_handle)
                command = command_mapper.pick_and_place(
                    source_handle,
                    list(destination.position),
                )
                active_kind = "move"
            active_motion = command_invoker.execute_steps(command)

        if active_motion is not None:
            if safety.state is SafetyState.RUNNING:
                try:
                    finished, result = advance_motion(
                        active_motion, MOTION_STEPS_PER_FRAME
                    )
                except EmergencyStopError as error:
                    print(error)
                    finished, result = True, False
                if finished:
                    if active_kind == "move":
                        # Nothing is used up by moving. The tube can be picked up
                        # again from wherever it now is, and the spot it left is
                        # free again because occupancy is read from positions.
                        set_status("PLACED" if result else "MOVE FAILED")
                    elif active_kind == "pour":
                        # transfer_contents already emptied the source and mixed
                        # the target; the target tube stays usable.
                        set_status("POURED" if result else "POUR FAILED")
                    elif active_kind == "mix":
                        set_status("MIXED" if result else "MIX FAILED")
                    elif active_kind == "undo":
                        if result:
                            set_status("UNDO EXECUTED")
                        else:
                            set_status("UNDO FAILED")
                    else:
                        set_status("SAMPLE SET DOWN")
                        if interrupted_motion is not None:
                            interrupted_motion.close()
                            interrupted_motion = None
                    active_motion = None
                    active_kind = None
                    hover_handle = None
                    system_state = "SELECT_SOURCE"
                    source_handle = None
                    dest_handle = None
                    pending_action = None
                    selected_handle = registry.first_selectable("move_source")
        elif hover_motion is not None and safety.state is SafetyState.RUNNING:
            try:
                finished, _ = advance_motion(hover_motion, MOTION_STEPS_PER_FRAME)
            except EmergencyStopError:
                finished = True
            if finished:
                hover_motion = None
        else:
            # Never block here: the camera and gesture pipeline must keep running
            # while paused, otherwise the webcam freezes and a gesture-triggered
            # pause cannot be released by gesture.
            safety.step_if_running()

        # ── Display ──
        draw_text(frame, f"Gesture: {gesture}", (10, 40), 21, ACCENT, bold=True)
        draw_text(frame, f"Dynamic: {current_dynamic_gesture} ({dynamic_confidence:.2f})",
                  (10, 68), 12, TEXT_BRIGHT)

        # Show confirm hint, naming the object rather than an index
        pending = None
        if system_state == "CONFIRM_SOURCE":
            pending = registry.by_handle(source_handle)
        elif system_state == "CONFIRM_DEST":
            pending = registry.by_handle(dest_handle)
        if pending is not None:
            draw_text(frame, f"Confirm {pending.label}? Thumbs Up / Thumb Left",
                      (10, 470), 13, ACCENT, bold=True)

        if status_message:
            color = DANGER if "FAIL" in status_message or "STOPPED" in status_message else ACCENT
            draw_text(frame, status_message, (10, 445), 14, color, bold=True)

        glove_text = "GLOVE: ON" if blue_glove_mode else "GLOVE: OFF"
        draw_text(frame, glove_text, (520, 30), 12, OK if blue_glove_mode else TEXT_BRIGHT)
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
            ACTIONS,
            action_index,
            pending_action,
            action_blocks(),
        )
        # One window: the camera view sits directly above the panel, so the
        # operator never has to look in two places or hunt for a second window.
        camera_view = cv2.resize(frame, (PANEL_WIDTH, 480))
        cv2.imshow("Pick and Place", np.vstack((camera_view, ui)))

        key = cv2.waitKey(1) & 0xFF
        safety.handle_key(key)
        if key == ord('q'):
            break
        elif key == ord('b'):
            blue_glove_mode = not blue_glove_mode
            print(f"  Blue glove mode: {'ON' if blue_glove_mode else 'OFF'}")
        elif key == ord('r'):
            if active_motion is not None:
                print("Robot is moving; stop it before resetting the scene.")
                continue
            if safety.state is SafetyState.EMERGENCY_STOPPED:
                print("Emergency stop remains active; press E to reset safety first.")
                continue
            # Reset simulation
            for handle, home in zip(source_handles, source_positions):
                p.resetBasePositionAndOrientation(handle, home, [0, 0, 0, 1])
                p.resetBaseVelocity(handle, [0, 0, 0], [0, 0, 0])
            robot_controller.reset_robot()
            command_invoker.clear_history()
            stop_hovering()
            registry.reset().refresh()
            # Put the liquids back too, not just the positions.
            for obj in registry.sources:
                apply_appearance(obj)
            system_state = "SELECT_SOURCE"
            source_handle = None
            dest_handle = None
            pending_action = None
            selected_handle = registry.first_selectable("move_source")
            set_status("SCENE RESET")
            for _ in range(240):
                safety.step_if_running()
        elif key == ord('u'):
            if safety.state is SafetyState.EMERGENCY_STOPPED:
                print("Emergency stop remains active; press E to reset safety first.")
            elif active_motion is not None:
                print("Robot is already moving.")
            elif command_invoker.can_undo:
                active_motion = command_invoker.undo_steps()
                active_kind = "undo"
                system_state = "EXECUTING"
                print("Undoing last pick-and-place.")
            else:
                print("Nothing to undo.")
                set_status("NOTHING TO UNDO")
        elif key == 8:  # backspace = go back (like thumb_left)
            if system_state == "CONFIRM_SOURCE":
                system_state = "SELECT_SOURCE"
                source_handle = None
            elif system_state == "SELECT_ACTION":
                system_state = "SELECT_SOURCE"
                source_handle = None
                pending_action = None
            elif system_state == "CONFIRM_ACTION":
                system_state = "SELECT_ACTION"
                pending_action = None
            elif system_state == "CONFIRM_DEST":
                system_state = "SELECT_DEST"
                dest_handle = None
        elif system_state == "CONFIRM_ACTION" and active_motion is None:
            if key == 13 or key == ord(' '):
                system_state = "EXECUTING"
        elif system_state == "SELECT_ACTION" and active_motion is None:
            if key in (83, ord('d'), 81, ord('a')):
                nxt = step_action(action_index, 1 if key in (83, ord('d')) else -1)
                if nxt is None:
                    set_status("NO ACTION POSSIBLE")
                else:
                    action_index = nxt
            elif key == 13 or key == ord(' '):
                if ACTIONS[action_index]["key"] in action_blocks():
                    set_status(action_blocks()[ACTIONS[action_index]["key"]].upper())
                    continue
                choose_action()
        elif system_state in ("SELECT_SOURCE", "SELECT_DEST") and active_motion is None:
            if key in (83, ord('d'), 81, ord('a')):
                nxt = registry.next_selectable(
                    selection_purpose(), selected_handle,
                    1 if key in (83, ord('d')) else -1,
                    exclude=selection_exclude(),
                )
                if nxt is None:
                    set_status("NOTHING LEFT TO CHOOSE")
                else:
                    selected_handle = nxt
            elif key == 13 or key == ord(' '):
                chosen = registry.by_handle(selected_handle)
                if chosen is not None and is_selectable(chosen.handle):
                    if system_state == "SELECT_SOURCE":
                        source_handle = selected_handle
                        system_state = "CONFIRM_SOURCE"
                    else:
                        dest_handle = selected_handle
                        system_state = "CONFIRM_DEST"
        elif system_state in ("CONFIRM_SOURCE", "CONFIRM_DEST") and active_motion is None:
            if key == 13 or key == ord(' '):  # enter = confirm (like thumbs_up)
                if system_state == "CONFIRM_SOURCE":
                    system_state = "SELECT_ACTION"
                    action_index = default_action_index
                    if action_index not in usable_action_indices():
                        action_index = step_action(action_index, 1) or 0
                    pending_action = None
                elif system_state == "CONFIRM_DEST":
                    system_state = "EXECUTING"

        if safety.quit_requested:
            break

    # Cleanup. cap is None whenever the camera never opened or dropped out
    # mid-run, so every teardown step has to tolerate a missing resource.
    if cap is not None:
        cap.release()
    cv2.destroyWindow("Pick and Place")
    landmark_provider.close()
    p.disconnect()
    os.chdir(old_cwd)
    print("Pick-and-place module closed.")


if __name__ == "__main__":
    run_pick_and_place()
