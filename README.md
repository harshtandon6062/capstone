# Capstone Project

This repository contains three Jupyter notebooks for hand landmark detection using MediaPipe and robot simulation using PyBullet.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Common Setup](#common-setup)
- [Notebooks](#notebooks)
  - [1. Hand Landmarker (Static Image)](#1-hand-landmarker-static-image)
  - [2. Hand Landmarker Live (Webcam)](#2-hand-landmarker-live-webcam)
  - [3. Robot Simulation (PyBullet)](#3-robot-simulation-pybullet)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.12** (tested with Python 3.12.3)
- **Git** installed
- **Webcam** (for live hand detection notebook)
- **Linux/macOS** recommended (Windows may require additional setup for PyBullet)

---

## Common Setup

### 1. Clone the Repository

```bash
git clone https://github.com/harshtandon6062/capstone.git
cd capstone
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Select the Virtual Environment in VS Code

1. Open VS Code
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
3. Type "Python: Select Interpreter"
4. Choose `./venv/bin/python`

---

## Notebooks

### 1. Hand Landmarker (Static Image)

**File:** `hand_landmarker.ipynb`

**Description:**  
Detects hand landmarks from a static image using MediaPipe. Downloads a sample image and visualizes the 21 hand landmarks with connections.

**What it does:**
- Downloads the MediaPipe hand landmarker model
- Loads a test image of hands
- Detects hand landmarks (21 points per hand)
- Draws landmarks and connections on the image
- Labels each hand as "Left" or "Right"

**How to run:**

1. Open `hand_landmarker.ipynb` in VS Code or Jupyter
2. Select the `venv` kernel
3. Run all cells in order:
   - Cell 1: Installs mediapipe (skip if already installed via requirements.txt)
   - Cell 2: Downloads the hand landmarker model
   - Cell 3: Defines visualization functions
   - Cell 4: Downloads and displays test image
   - Cell 5: Runs hand detection and shows results

**Expected Output:**  
An image with hand landmarks drawn as colored dots connected by lines.

---

### 2. Hand Landmarker Live (Webcam)

**File:** `hand_landmarker_live.ipynb`

**Description:**  
Real-time hand landmark detection using your webcam. Tracks up to 2 hands simultaneously with live FPS display.

**What it does:**
- Opens your webcam for live video capture
- Detects hands in real-time using VIDEO mode (optimized for video streams)
- Draws 21 landmarks per hand with skeleton connections
- Displays FPS and number of hands detected
- Mirrors the display for intuitive interaction

**How to run:**

1. Open `hand_landmarker_live.ipynb` in VS Code or Jupyter
2. Select the `venv` kernel
3. Run cells in order:
   - Cells 1-2: Install dependencies and download model
   - Cell 3: Import libraries
   - Cell 4: Define visualization function
   - Cell 5: Initialize the hand detector
   - Cell 6: **Run live detection** (opens webcam window)

4. **Press 'q'** to quit the webcam window

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit and close window |

**Camera Issues:**  
If your camera doesn't open, try changing `VideoCapture(0)` to `VideoCapture(1)` or `VideoCapture(2)` in the code.

---

### 3. Robot Simulation (PyBullet)

**File:** `test.ipynb`

**Description:**  
Simulates a KUKA KR210 industrial robot arm using PyBullet physics engine. Demonstrates robot loading, joint control, and physics simulation.

**What it does:**
- Connects to PyBullet physics simulator (opens GUI window)
- Loads a ground plane and KUKA robot arm
- Clones the KUKA robot URDF from ROS Industrial
- Controls robot joints using position control
- Runs physics simulation steps

**How to run:**

1. Open `test.ipynb` in VS Code or Jupyter
2. Select the `venv` kernel
3. Run cells in order:
   - Cell 1: Connect to PyBullet (opens simulation window)
   - Cell 2: Set search path for URDF files
   - Cell 3: Load ground plane
   - Cell 4: Clone KUKA robot repository (takes a few seconds)
   - Cell 5: Load robot URDF
   - Cell 6-11: Control and simulate the robot
   - Cell 12: Disconnect from PyBullet

**Note:** The first run will clone the `kuka_experimental` repository (~may take a minute). This is automatically ignored by `.gitignore`.

**Expected Output:**  
A PyBullet GUI window showing the KUKA robot arm on a ground plane. The robot joints will move based on position control commands.

---

## Troubleshooting

### MediaPipe Import Errors

If you get `ImportError: cannot import name 'solutions' from 'mediapipe'`:
- Make sure you're using `mediapipe==0.10.14` (already in requirements.txt)
- Restart the Jupyter kernel after installation

### Camera Not Found

```
ERROR: Could not open camera!
```
- Try changing `VideoCapture(0)` to `VideoCapture(1)` or `VideoCapture(2)`
- Check if another application is using the camera

### PyBullet Window Doesn't Open

- Make sure you have a display connected
- On headless servers, use `pybullet.DIRECT` instead of `pybullet.GUI`

### Kernel Not Using venv

- Restart VS Code
- Re-select the Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `./venv/bin/python`

### Hand Detection is Slow

- Reduce camera resolution (see optional cell in `hand_landmarker_live.ipynb`)
- Ensure good lighting conditions
- Close other CPU-intensive applications

---

## Project Structure

```
capstone/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── hand_landmarker.ipynb        # Static image hand detection
├── hand_landmarker_live.ipynb   # Live webcam hand detection
├── test.ipynb                   # PyBullet robot simulation
├── .gitignore                   # Git ignore rules
└── venv/                        # Virtual environment (not in repo)
```

---

## License

Hand landmarker notebooks are based on MediaPipe examples (Apache 2.0 License).
