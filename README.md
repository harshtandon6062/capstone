# Gesture-Controlled Robotic Pick & Place

A gesture-controlled laboratory robot. A webcam reads hand gestures, an operator
panel shows what is selected, and a Kuka IIWA arm with a WSG50 gripper moves,
pours and mixes colour-coded test tubes in a PyBullet simulation.

The same command interface drives a real **Niryo Ned 2** through
[niryo_backend.py](niryo_backend.py).

The project is about the interaction, not the arm: how an operator selects a real
physical object without ambiguity, how the system asks for confirmation in
proportion to what an action costs, and how an emergency stop stays reachable
*during* the motion it exists to interrupt.

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Controls](#controls)
- [Repository layout](#repository-layout)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Configuration](#configuration)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Running on a real Niryo Ned 2](#running-on-a-real-niryo-ned-2)
- [Deferred: YOLO perception](#deferred-yolo-perception)
- [Repository history](#repository-history)

---

## What it does

Three tasks, chosen at the launcher by holding a dynamic gesture or pressing a key:

| Key | Dynamic gesture | Task | Reversible |
|---|---|---|---|
| `1` | Grasp | **Pick & Place** — carry a tube to an empty spot | yes |
| `2` | Tilt | **Pour** — tip one tube's contents into another | **no** |
| `3` | Wrist rotation | **Mix** — lift, rotate and set a tube back down | no |

Inside a task the operator works through a small state machine: pick a source
tube, pick an action, pick a target if the action needs one, confirm, watch it
run. A move goes to a **spot**; a pour goes into another **tube**; a mix needs no
target at all because it acts on the tube already chosen.

---

## Quick start

### Prerequisites

Python 3.12, a webcam, and a machine that can open an OpenGL window.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions this was developed and measured against.
`pyniryo` is only needed to drive a physical arm — the simulation runs without it.

### Run

```bash
python launcher.py
```

Or skip the launcher and go straight into a task:

```bash
python main.py                    # starts on Move
python -c "from main import run_pick_and_place; run_pick_and_place(initial_action='pour')"
```

---

## Controls

### Launcher

| Input | Effect |
|---|---|
| Hold **Grasp** / **Tilt** / **Wrist rotation** 4 s | start Pick & Place / Pour / Mix |
| `1` `2` `3` | start the same three directly |
| `B` | toggle blue glove mode |
| `Q` | quit |

### Inside a task

| Gesture | Key | Effect |
|---|---|---|
| Point left / right | `←` `→` or `A` `D` | move the selection |
| Pinch | `Enter` / `Space` | select |
| Thumbs up | `Enter` | confirm |
| Thumb left | `Backspace` | cancel, go back one step |
| Open palm | `P` | pause / resume |
| **Thumbs down** | `X` | **emergency stop** |
| — | `E` | clear the emergency stop |
| — | `U` | undo the last successful move |
| — | `R` | reset the scene |
| — | `B` | toggle blue glove mode |
| — | `Q` | quit |

Clearing an emergency stop is **deliberately not a gesture**. A gesture that
resumes is a gesture that can resume by accident, and the whole point of a
latched stop is that a human decides when it ends.

**Blue glove mode** swaps colour channels before hand detection, which makes
MediaPipe far more reliable on blue nitrile lab gloves.

---

## Repository layout

### Application

| File | What it is |
|---|---|
| [launcher.py](launcher.py) | Entry point — dynamic gesture recognition and task selection |
| [main.py](main.py) | The task loop: perception, state machine, motion driving, rendering |
| [config.py](config.py) | Every tunable number and the action table, in one place |

### Robot

| File | What it is |
|---|---|
| [robot_backend.py](robot_backend.py) | The contract any arm must satisfy, simulated or real |
| [robot_controller.py](robot_controller.py) | PyBullet Kuka IIWA + WSG50 implementation |
| [niryo_backend.py](niryo_backend.py) | Niryo Ned 2 driver — ⚠️ not yet verified against hardware |
| [safety_controller.py](safety_controller.py) | Pause, emergency stop, and the gate every motion step passes |

### Workspace model

| File | What it is |
|---|---|
| [object_registry.py](object_registry.py) | What is in the workspace; the single source of colour and identity |
| [perception.py](perception.py) | Where workspace contents come from — simulation today, a camera later |

### Commands

| File | What it is |
|---|---|
| [commands/base_command.py](commands/base_command.py) | Command interface, including the stepped `execute_steps()` form |
| [commands/command_mapper.py](commands/command_mapper.py) | Builds the right command for a confirmed choice |
| [commands/command_invoker.py](commands/command_invoker.py) | Runs commands and records them |
| [commands/command_history.py](commands/command_history.py) | Undo stack |
| [commands/pick_place_command.py](commands/pick_place_command.py) | Move, with a real reverse motion for undo |
| [commands/pour_command.py](commands/pour_command.py) | Pour — refuses undo and clears the history |
| [commands/mix_command.py](commands/mix_command.py) | Mix — in-place rotation |

### Interface

| File | What it is |
|---|---|
| [ui_module.py](ui_module.py) | The operator panel |
| [ui_text.py](ui_text.py) | Cached TrueType text rendering |
| [assets/fonts/](assets/fonts/) | Noto Sans, bundled so the panel looks the same everywhere |

### Gesture recognition

| File | What it is |
|---|---|
| [gesture_module.py](gesture_module.py) | Static gestures from MediaPipe hand landmarks |
| [gesture_controller.py](gesture_controller.py) | Turns raw classifications into debounced events |
| [hand_landmark_provider.py](hand_landmark_provider.py) | MediaPipe HandLandmarker wrapper |
| [model_loader.py](model_loader.py) | Loads the LSTM and its labels |
| `gesture_landmark_model.h5`, `classes.npy`, `hand_landmarker.task` | Trained models |

---

## Architecture

```
Launcher (dynamic gesture recognition)
    │
    └── task chosen ──► Task loop (main.py)
                            │
                            ├── PerceptionSource ──► ObjectRegistry
                            │     (what is in the workspace, and where)
                            │            ├──► UI panel   (what may be selected)
                            │            └──► Commands   (where to move)
                            │
                            ├── Gestures / keyboard
                            │       └── CommandMapper (confirmed choice)
                            ├── CommandInvoker → CommandHistory
                            ├── SafetyController (gates every motion step)
                            └── RobotController → PyBullet (Kuka + WSG50 + tubes)
```

### The object registry

`ObjectRegistry` is the single description of what is in the workspace. A
`PerceptionSource` fills it, the panel reads it to decide what can be selected,
and commands read it to decide where to move. Objects are referred to by identity
(`Red tube`) rather than by list index, so nothing downstream depends on how many
objects exist or what order they are in.

Colour lives in the registry once. The panel derives its swatch from the same
RGBA value the simulation renders, so the two cannot disagree.

`SimulatedPerception` currently reads poses straight from PyBullet. Replacing it
with a camera-backed source returning the same observation shape requires no
change anywhere else.

### Motion is stepped, not blocking

Every motion exists twice: a generator that yields once per simulation step, and
a blocking wrapper that drains it. The task loop drives the generator a slice at
a time, so the camera keeps being read and gestures keep being classified **while
the arm is moving**.

This is not a performance detail. It is the reason the stop gesture works during
the motion it exists to interrupt. When motions ran inside one blocking call, the
main loop could not read input until the arm finished, and an emergency stop was
reachable only when the robot was already idle.

---

## Design decisions

### Which tube is which

Every tube has a **coloured cap** that never changes, and a **body** that shows
what it currently holds. Those are two different questions, and colour used to
answer both at once — so pouring a tube out erased which tube it was. Empty two
of them and they became impossible to tell apart, on the table and in the panel.

| | Says | Changes? |
|---|---|---|
| Cap | which tube this is | never |
| Body / liquid | what is in it | yes — mixed, emptied, refilled |

The panel swatch mirrors it: the **frame** is the cap colour, the **middle** is
the contents, and it goes dark when the tube is empty. One square answers both
questions. Names follow the same rule — the red-capped tube is "Red tube" whether
it is full, "Red tube (mixed)" or "Red tube (empty)". It is never renamed.

This is how real labware works: you label a tube, and pouring it out does not
remove the label. It is also what makes camera perception possible later — if two
empty tubes are indistinguishable to a technician, they are indistinguishable to
a detector, and the registry cannot keep a stable handle for either.

### The robot points at what you have selected

While you are choosing, the arm parks above the highlighted tube or spot. A grey
menu highlight tells you *a* square is selected; the arm tells you *which real
object* that is, before you confirm anything.

It stays at clearance height, never descends and never closes the gripper — it is
a way of pointing, not a way of picking. Measured: it arrives within 38 mm worst
case in about 7 frames, well inside the 60 mm that would make two neighbouring
tubes ambiguous.

This matters more than a UI nicety. The confirmation step is only worth having if
the operator is sure *which object* they are confirming; a perfect confirm on the
wrong referent is still the wrong action.

On a real bench there is no rendered scene to draw a marker into, so the arm
itself does the pointing — `hover_over_steps()` is part of the
[robot_backend.py](robot_backend.py) contract and both controllers implement it.
Set `HOVER_OVER_SELECTION = False` to turn it off.

### What can be selected, and when

Nothing in the workspace is ever "used up". A tube is a tube: it can be moved as
many times as you like, wherever it currently is, and emptying it does not take
it off the table.

| Step | What may be chosen |
|---|---|
| Source | every tube, however many times it has already been handled |
| Action → Move | only spots with no tube standing on them |
| Action → Pour | any other tube; offered only if this one has something in it |
| Action → Mix | no target — it acts on the tube already chosen |

**Spot occupancy is derived, not remembered.** A spot is taken when a tube is
actually standing within `SPOT_OCCUPANCY_RADIUS` of it, worked out from live
positions each frame. Flags for this went stale every time the world changed by
another route — an undo, a scene reset, a tube moved elsewhere — and left spots
looking used with nothing on them. Nothing has to be released or cleaned up now;
moving a tube off a spot frees it because the tube is no longer there.

An action that cannot work is dimmed and says why (`tube is empty`, `spots
full`), rather than being offered and then silently doing nothing.

Pouring an empty tube is refused. Pouring into an empty tube gives it the
contents as they are — "empty" is not a liquid, and averaging with it tinted
every subsequent pour toward the empty-tube colour.

`R` resets contents as well as positions, so tubes do not stay permanently empty.

### Whether an action needs a target is declared, not inferred

Each entry in `ACTIONS` states `needs_target`. `action_needs_target()` answers
`True` for anything unrecognised, because refusing to start without a target is
the safe way to be wrong.

This replaced an `== "mix"` test written separately on the gesture path and the
key path. The two drifted: mix skipped target selection when chosen by pinch and
demanded it when chosen by Enter. One declaration, read by one `choose_action()`,
removes the class of bug rather than the instance.

### Pour height is what keeps the arm off the other tubes

Lining the tube's mouth up with the target is done by measurement, so how high the
pour happens decides how low the wrist goes. Pouring 0.09 m above the rim put the
wrist at 0.757 — below the tops of the tubes — and the arm's own links dragged
neighbours up to 87 mm across the bench. Not the gripper, and nothing to do with
the tube being carried: the arm itself.

Raising the pour to 0.16 m fixes it outright, and costs nothing in accuracy.
Measured over all 20 ordered pairs, during the pour itself rather than at the best
instant in passing:

| pour height | mouth inside the 22 mm rim | worst bystander shove |
|---|---|---|
| 0.09 m | 19/20 | 86.8 mm |
| **0.16 m** | **20/20** | **0.0 mm** |

The wrist now stays above 0.822 throughout, against tube tops at 0.725.

### Reversibility, and why it is the point

| | hold to commit | undo | history |
|---|---|---|---|
| Move | 1.5 s | yes | kept |
| Pour | 2.0 s | refused | cleared |
| Mix | none — see below | refused | kept |

Pouring clears the undo history rather than just declining its own undo, because
you cannot undo an earlier move back into a world that no longer exists. The
operator is told before choosing, not after: the POUR action is outlined in red,
and both the target and confirmation screens say it cannot be undone.

This matters more than it looks. If every action is reversible, a confirmation
step has nothing to protect and is just friction. One irreversible action is what
makes the confirmation gate mean something.

### The operator panel

The panel is a NumPy array drawn with OpenCV shapes and blitted TrueType text,
composited above the camera view in one window.

Text is the interesting part. OpenCV can only draw Hershey stroke fonts, which
were designed for pen plotters. Pillow renders real TrueType but costs ~8 ms a
frame — twenty times the whole panel and more than half of hand detection. So each
string is rendered **once** into an alpha mask and cached; the panel's vocabulary
is small and fixed, so the cache fills within the first second and after that
essentially never misses.

| approach | ms/frame |
|---|---|
| Hershey stroke fonts (before) | 0.32 |
| **cached TrueType (now)** | **1.31** |
| Pillow re-rendered every frame | 7.82 |
| Pillow at 2× supersampling | 18.49 |

Against a 33 ms frame in which hand detection alone is 14.7 ms.

Saturation is reserved for meaning — a tube's identity, its contents, and the red
on the action that cannot be undone. Everything else is muted, so the colours that
carry information are the ones that stand out.

### A renderer trap worth knowing about

The simulation window used to open with `--opengl2`. That flag is a compatibility
fallback for machines whose drivers cannot run the default renderer, and it
**silently ignores `changeVisualShape`** — so a tube that had been emptied or
mixed kept its old colour on screen while the panel showed the new one. Measured
here: recolouring a tube red to blue changed 0.00 of the rendered pixels under
`--opengl2` and 3.14 under the default renderer.

`open_simulation_window()` now tries the default first and only falls back to
`--opengl2` if it will not start, warning on the console when it does.

---

## Configuration

Every tunable lives in [config.py](config.py).

| Setting | Value | What it controls |
|---|---|---|
| `NAVIGATION_HOLD_DURATION` | 0.45 | hold to move the selection |
| `NAVIGATION_COOLDOWN` | 0.60 | gap between navigation steps |
| `GESTURE_HOLD_DURATION` | 1.5 | hold to select or confirm |
| `GESTURE_COOLDOWN` | 1.2 | gap between committing gestures |
| `IRREVERSIBLE_HOLD_DURATION` | 2.0 | hold to commit a pour |
| `UNDO_GESTURE_COOLDOWN` | 10.0 | gap between undos |
| `HOVER_OVER_SELECTION` | `True` | arm points at the highlighted object |
| `SPOT_OCCUPANCY_RADIUS` | 0.06 | how near a tube must be to occupy a spot |
| `POUR_CLEARANCE` | 0.16 | pour height above the target rim |
| `GRASP_TOLERANCE` | 0.03 | how close the fingers must be to count as holding |
| `MOTION_STEPS_PER_FRAME` | 16 | simulation steps per rendered frame |

Navigating is free to undo, so it is quick. Committing starts the robot, so it
stays slow enough to be a decision.

---

## Tests

```bash
pip install pytest
python -m pytest -q          # 120 passing
```

| File | Covers |
|---|---|
| `test_object_registry.py` | identity vs contents, occupancy, selectability |
| `test_robot_controller.py` | grasp validation, centring, motion generators |
| `test_safety_controller.py` | pause, latched stop, holding pose |
| `test_motion_interrupt.py` | that a stop actually interrupts a running motion |
| `test_pour.py` | aim, clearance, contents transfer |
| `test_commands.py` | undo, history clearing, failure states |
| `test_gesture_integration.py` | gesture events, action target declarations |
| `test_gesture_direction.py` | left/right mapping |
| `test_launcher_flow.py` | task dispatch |
| `test_ui_module.py` | every panel state renders, clips and anti-aliases |
| `test_niryo_backend.py` | waypoint segmentation, transform, stop behaviour |

---

## Known limitations

### No motion planning

`RobotController.move_to()` sets inverse-kinematics joint targets directly. There
is no path planning and no collision checking, so the arm can sweep sideways
through an object on the way to a target. `approach_from_above()` mitigates this
by travelling at clearance height and descending vertically, but it is not a
substitute for a planner. Targets close to the robot's forward axis remain the
worst case, because IK can flip configuration there and swing the arm wide.

### Objects still drift slowly

The gripper hangs off the wrist on a constraint, so its centre trails the
inverse-kinematics target by a few centimetres, and that offset changes with pose.
Closing without correcting for it made the fingers meet the tube off-centre and
shove it roughly 8 mm sideways on every grasp, always the same direction, so
repeated pick-and-undo cycles walked tubes across the table.

`center_gripper_over()` measures where the fingers actually are and re-aims the
wrist before closing. Over five place-and-undo cycles on one tube, total drift
fell from 93 mm to 31 mm and stopped accumulating in one direction. Reduced, not
eliminated: expect a few millimetres per grasp. Press `R` to reset if tubes wander
far enough to matter.

### Mix commits without a confirmation

`MixCommand` is irreversible — `undo()` returns `False` — but it does not get the
confirmation hold that pour does, and `direct_dynamic_mix` in [main.py](main.py)
starts it the moment the LSTM reports `wrist_rotation` at ≥ 0.75 confidence, with
no pinch and no thumbs-up.

In practice a mix is harmless: it stirs one tube and puts it back. But that is a
judgement about this particular action, and the shortcut is the kind of thing that
gets copied to the next one. Either give mix a confirm step or record in the code
why it does not need one.

### Mixing does not change the rendered colour

`mix_contents()` calls `mix_colors(c, c)` — a colour averaged with itself, which
is the identity. Stirring one tube arguably *should not* change what is in it, so
the behaviour is defensible; but the call reads as though it does something, and
the only visible effect is the panel label gaining `(mixed)`.

---

## Running on a real Niryo Ned 2

[niryo_backend.py](niryo_backend.py) implements the same interface as the
simulated controller, so commands, the state machine and the UI do not change.
It is **written but not yet verified against hardware** — built against the
pyniryo 1.2.5 API read from source, with no arm attached.

Before switching an arm on, calibrate `WorkspaceTransform`: its `origin`,
`robot_origin` and `scale` are guesses, and getting them wrong is the difference
between reaching a tube and driving into it.

### The safety difference you must know about

In simulation the application drives the physics one step at a time, so an
emergency stop lands within a frame. **A real Ned 2 cannot do this.** Every
pyniryo motion call blocks until the arm arrives, and the API has no stop, halt or
abort command at all — the entire command set was searched. The only software way
to interrupt a Ned 2 is `set_learning_mode(True)`, which cuts motor torque: the arm
goes limp and drops what it is carrying, which is exactly the failure this project
already fixed in simulation. The backend therefore never uses it.

What the software stop actually does on hardware is **refuse to send the next
waypoint**. Moves are cut into 4 cm segments so the arm stops soon after being
asked, but it will finish the segment already in flight.

> On real hardware the physical emergency-stop button is the safety device. The
> gesture stop is "hold at the next waypoint", not an instantaneous halt, and it
> should be described that way to anyone operating it.

---

## Deferred: YOLO perception

[perception.py](perception.py) already isolates where workspace contents come
from, so a detector drops in without touching anything above it. It has been left
out on purpose:

- Running it in simulation would be circular — it would detect objects whose exact
  poses are already known from `getBasePositionAndOrientation`. It would demo well
  and prove nothing.
- For five coloured tubes on a white table, HSV thresholding beats a learned
  detector on both accuracy and speed.
- The real work is camera-to-robot calibration, not the detector.

The version worth building runs on the real arm and uses **detection confidence**
to drive the confirmation policy: a low-confidence detection is exactly when the
human should be made to confirm. That is the same idea as the pour asymmetry
above, and a better reason to use a detector than using one for its own sake.

---

## Repository history

This repository began with two **unrelated** root commits. `master` held the
original Jupyter exploration; every line of the application grew on a separate
lineage starting at `b1ef224`, sharing no ancestor with it. Anyone opening the
repository landed on the prototype and could not see the system at all.

The two were joined with `git merge --allow-unrelated-histories`, with the
application tree taking precedence. The original notebooks — `hand_landmarker.ipynb`,
`hand_landmarker_live.ipynb`, `test.ipynb` — are preserved unchanged on the
[`archive/notebook-prototype`](../../tree/archive/notebook-prototype) branch.
