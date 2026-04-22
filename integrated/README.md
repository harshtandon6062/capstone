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
- **Self-Contained** — All models and modules are included in this directory.

## Gesture → Task Mapping

| Dynamic Gesture | Task | Status |
|---|---|---|
| Grasp | Pick & Place | ✅ Ready |
| Tilt | Pour | 🔜 Coming Soon |
| Wrist Rotation | Mix | 🔜 Coming Soon |

## Files

| File | Description |
|---|---|
| `launcher.py` | Entry point — dynamic gesture recognition + task selection UI |
| `main.py` | Pick-and-place module (Kuka robot + static gesture control) |
| `ui_module.py` | UI panel rendering for block/spot selection |
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
- **B** → toggle blue glove mode
- **Q** → return to launcher

## Architecture

```
Launcher (Dynamic Gesture Recognition)
    │
    ├── Grasp held 3s ──► Pick & Place Module
    │                        ├── PyBullet (Kuka + WSG50 + Cubes)
    │                        ├── Static Gesture Control (MediaPipe)
    │                        └── UI Panel (OpenCV)
    │
    └── Q pressed ──► Exit
```
