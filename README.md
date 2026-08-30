# Gesture-Controlled Robotic Pick & Place

A gesture-controlled robotic simulation that integrates **dynamic gesture recognition** (LSTM + MediaPipe) with a **PyBullet-based Kuka robot** pick-and-place task.

## Features

- **Dynamic Gesture Launcher** — Uses a trained LSTM model to recognize hand gestures from webcam. Hold a gesture for 3 seconds to trigger a task.
- **Pick & Place Module** — Kuka IIWA robot with WSG50 gripper picks up colored blocks and places them at destination spots.
- **Static Gesture Control** — Navigate and select blocks/spots using:
  - **Point Left/Right** — Navigate between items
  - **Pinch** — Select an item
  - **Thumbs Up** — Confirm selection
  - **Thumb Left** — Cancel and re-select
- **Blue Glove Mode** — Channel swap trick for better hand detection with blue lab gloves.
- **Safety Controller** — Implements pause/resume and emergency stop behavior in the simulation loop.
- **Self-Contained** — All models and modules are included in this directory.

## Gesture → Task Mapping

| Dynamic Gesture | Task | Status |
|---|---|---|
| Grasp | Pick & Place | ✅ Ready |
| Tilt | Pour | ✅ Ready — **cannot be undone** |
| Wrist Rotation | Mix | 🔜 Coming Soon |

Once a source tube is confirmed, the operator picks an action before choosing a
target, because a **move** goes to a spot while a **pour** goes into another tube.

## Files

| File | Description |
|---|---|
| `launcher.py` | Entry point — dynamic gesture recognition + task selection UI |
| `main.py` | Pick-and-place module (Kuka robot + static gesture control) |
| `ui_module.py` | UI panel rendering for block/spot selection |
| `robot_backend.py` | The contract any arm must satisfy — simulated or real |
| `niryo_backend.py` | Niryo Ned 2 driver (⚠️ untested against hardware) |
| `object_registry.py` | What is in the workspace, and the single source of colour |
| `perception.py` | Where workspace contents come from (simulation today, camera later) |
| `gesture_module.py` | Static gesture detection (MediaPipe hand landmarks) |
| `gesture_landmark_model.h5` | Trained LSTM model for dynamic gesture classification |
| `classes.npy` | Gesture class labels |
| `hand_landmarker.task` | MediaPipe hand landmarker model |

## Setup

### Prerequisites

```bash
# Python 3.12 virtual environment with:
pip install pybullet mediapipe opencv-python numpy tensorflow-cpu
```

### Run

```bash
python launcher.py
```

### Controls

**Launcher Screen:**
- Hold **Grasp** gesture for 3s → opens Pick & Place
- Press **1** → quick-launch Pick & Place (testing shortcut)
- **B** → toggle blue glove mode
- **Q** → quit

**Pick & Place Screen:**
- **Point Left/Right** or **Arrow Keys** → navigate blocks/spots
- **Pinch** or **Enter/Space** → select
- **Thumbs Up** or **Enter** → confirm selection
- **Thumb Left** or **Backspace** → cancel selection
- **R** → reset simulation
- **P** → pause / resume motion
- **X** → emergency stop
- **E** → clear emergency stop
- **U** → undo the most recent successful pick-and-place
- **B** → toggle blue glove mode
- **Q** → return to launcher

### Timing Configuration

Gesture timing is centralized in [config.py](config.py):
- `GESTURE_COOLDOWN = 1.0`
- `UNDO_GESTURE_COOLDOWN = 10.0`
- `GESTURE_HOLD_DURATION = 1.0`

Adjust these values in one place to tune gesture responsiveness.

## Architecture

```
Launcher (Dynamic Gesture Recognition)
    │
    ├── Grasp held 3s ──► Pick & Place Module
    │                        │
    │                        ├── PerceptionSource ──► ObjectRegistry
    │                        │     (what is in the workspace, and where)
    │                        │            ├──► UI Panel  (what may be selected)
    │                        │            └──► Commands  (where to move)
    │                        │
    │                        ├── Gestures / Keyboard
    │                        │       └── CommandMapper (confirmed destination)
    │                        ├── CommandInvoker → CommandHistory
    │                        ├── PickPlaceCommand
    │                        ├── SafetyController (gates every motion step)
    │                        └── RobotController
    │                                └── PyBullet (Kuka + WSG50 + tubes)
    │
    └── Q pressed ──► Exit
```

### The object registry

`ObjectRegistry` is the single description of what is in the workspace. A
`PerceptionSource` fills it, the panel reads it to decide what can be selected,
and commands read it to decide where to move. Objects are referred to by
identity (`Red tube`) rather than by list index, so nothing downstream depends on
how many objects exist or what order they are in.

Colour lives in the registry once. The panel derives its swatch from the same
RGBA value the simulation renders, so the two cannot disagree.

`SimulatedPerception` currently reads poses straight from PyBullet. Replacing it
with a camera-backed source returning the same observation shape requires no
change anywhere else.

### Known limitation: no motion planning

`RobotController.move_to()` sets inverse-kinematics joint targets directly. There
is no path planning and no collision checking, so the arm can sweep sideways
through an object on the way to a target. `approach_from_above()` mitigates this
by travelling at clearance height and descending vertically, but it is not a
substitute for a planner. Targets close to the robot's forward axis remain the
worst case, because IK can flip configuration there and swing the arm wide.

### Known limitation: objects still drift slowly

The gripper hangs off the wrist on a constraint, so its centre trails the
inverse-kinematics target by a few centimetres and that offset changes with
pose. Closing without correcting for it made the fingers meet the tube
off-centre and shove it roughly 8 mm sideways on every grasp, always the same
direction, so repeated pick and undo cycles walked tubes across the table.

`center_gripper_over()` measures where the fingers actually are and re-aims the
wrist before closing. Measured over five place-and-undo cycles on one tube, total
drift fell from 93 mm to 31 mm, and the per-cycle error stopped accumulating in a
single direction. It is reduced, not eliminated: expect a few millimetres of
movement per grasp. Press `R` to reset the scene if tubes wander far enough to
matter.

`PickPlaceCommand` stores the source object's position and orientation before
execution. Undo visibly moves the robot to the current object, closes and
attaches the WSG50 gripper, carries the object back to its saved pose, releases
it, and returns the robot home. A failed reverse operation remains undoable and
does not enter a success state. Undo does not reset the simulation or enter the
emergency-stop history. Emergency stop remains latched until the explicit `E`
recovery action is used.

The current PyBullet backend is isolated behind `RobotController`, leaving the
command API suitable for a future ROS-backed controller.


## What can be selected, and when

Nothing in the workspace is ever "used up". A tube is a tube: it can be moved as
many times as you like, wherever it currently is, and emptying it does not take
it off the table.

| Step | What may be chosen |
|---|---|
| Source | every tube, however many times it has already been moved |
| Action → Move | only spots with no tube standing on them |
| Action → Pour | any other tube; offered only if this one has something in it |

**Spot occupancy is derived, not remembered.** A spot is taken when a tube is
actually standing within `SPOT_OCCUPANCY_RADIUS` of it, worked out from live
positions each frame. Flags for this went stale every time the world changed by
another route — an undo, a scene reset, a tube moved somewhere else — and left
spots looking used with nothing on them. Nothing has to be released or cleaned
up now; moving a tube off a spot frees it because the tube is no longer there.

An action that cannot work is dimmed and says why (`tube is empty`, `spots
full`), rather than being offered and then silently doing nothing.

Pouring an empty tube is refused. Pouring into an empty tube gives it the
contents as they are — "empty" is not a liquid, and averaging with it tinted
every subsequent pour toward the empty-tube colour.

`R` resets contents as well as positions, so tubes do not stay permanently empty.

## Reversibility, and why it is the point

Every action except pouring can be undone. That asymmetry is deliberate, and the
interface treats the two differently in proportion to what they cost:

| | hold to commit | undo | history |
|---|---|---|---|
| Move | 0.9 s | yes | kept |
| Pour | 2.0 s | refused | cleared |

Pouring clears the undo history rather than just declining its own undo, because
you cannot undo an earlier move back into a world that no longer exists. The
operator is told before choosing, not after: the POUR action is outlined in red
and both the target and confirmation screens say it cannot be undone.

This matters more than it looks. If every action is reversible, a confirmation
step has nothing to protect and is just friction. One irreversible action is what
makes the confirmation gate mean something.

## A renderer trap worth knowing about

The simulation window used to open with `--opengl2`. That flag is a compatibility
fallback for machines whose drivers cannot run the default renderer, and it
**silently ignores `changeVisualShape`** — so a tube that had been emptied or
mixed kept its old colour on screen while the panel showed the new one. Measured
here: recolouring a tube red to blue changed 0.00 of the rendered pixels under
`--opengl2` and 3.14 under the default renderer.

`open_simulation_window()` now tries the default first and only falls back to
`--opengl2` if it will not start, warning on the console when it does.

## Running on a real Niryo Ned 2

`niryo_backend.py` implements the same interface as the simulated controller, so
commands, the state machine and the UI do not change. It is **written but not yet
verified against hardware** — it was built against the pyniryo 1.2.5 API read
from source, with no arm attached. Before switching an arm on, calibrate
`WorkspaceTransform`: its `origin`, `robot_origin` and `scale` are guesses, and
getting them wrong is the difference between reaching a tube and driving into it.

### The safety difference you must know about

In simulation the application drives the physics one step at a time, so an
emergency stop lands within a frame. **A real Ned 2 cannot do this.** Every
pyniryo motion call blocks until the arm arrives, and the API has no stop, halt
or abort command at all — the entire command set was searched. The only software
way to interrupt a Ned 2 is `set_learning_mode(True)`, which cuts motor torque:
the arm goes limp and drops what it is carrying, which is exactly the failure
this project already fixed in simulation. The backend therefore never uses it.

What the software stop actually does on hardware is **refuse to send the next
waypoint**. Moves are cut into 4 cm segments so that the arm stops soon after
being asked, but it will finish the segment already in flight.

> On real hardware the physical emergency-stop button is the safety device. The
> gesture stop is "hold at the next waypoint", not an instantaneous halt, and it
> should be described that way to anyone operating it.

## Deferred: YOLO perception

`perception.py` already isolates where workspace contents come from, so a
detector drops in without touching anything above it. It has been left out on
purpose for now:

- Running it in simulation would be circular — it would detect objects whose
  exact poses are already known from `getBasePositionAndOrientation`. It would
  demo well and prove nothing.
- For five coloured tubes on a white table, HSV thresholding beats a learned
  detector on both accuracy and speed.
- The real work is the camera-to-robot calibration, not the detector.

The version worth building runs on the real arm and uses **detection confidence**
to drive the confirmation policy: a low-confidence detection is exactly when the
human should be made to confirm. That is the same idea as the pour asymmetry
above, which is a better reason to use a detector than using one for its own sake.
