"""
Gesture-Controlled Robotics Launcher
=====================================
Uses dynamic gesture recognition (LSTM + MediaPipe) to select robotic tasks.
Hold a gesture for 3 seconds to trigger the corresponding action.

Currently supported:
  - GRASP → Pick and Place

Run: ~/shared_ml_env/bin/python ~/capstone_project/integrated/launcher.py
"""
import cv2
import numpy as np
import time
import os
import sys

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# LOAD DYNAMIC GESTURE MODEL
# ──────────────────────────────────────────────
import tensorflow as tf
import mediapipe as mp

model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, 'gesture_landmark_model.h5'))
classes = np.load(os.path.join(SCRIPT_DIR, 'classes.npy'))
print(f"Loaded dynamic gesture model. Classes: {list(classes)}")

# MediaPipe hand landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_landmarks = np.zeros(63)
raw_lms_for_drawing = []

def handle_result(result, output_image, timestamp_ms):
    global latest_landmarks, raw_lms_for_drawing
    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        raw_lms_for_drawing = lms
        base_x, base_y, base_z = lms[0].x, lms[0].y, lms[0].z
        latest_landmarks = np.array(
            [[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in lms]
        ).flatten()
    else:
        latest_landmarks = np.zeros(63)
        raw_lms_for_drawing = []

lm_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(SCRIPT_DIR, 'hand_landmarker.task')),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result,
    num_hands=1
)

# ──────────────────────────────────────────────
# TASK DEFINITIONS
# ──────────────────────────────────────────────
TASKS = [
    {"gesture": "grasp",          "name": "Pick & Place",  "status": "READY",       "color": (0, 255, 0)},
    {"gesture": "tilt",           "name": "Pour",          "status": "Coming Soon", "color": (0, 200, 255)},
    {"gesture": "wrist_rotation", "name": "Mix",           "status": "Coming Soon", "color": (255, 0, 255)},
]

HOLD_DURATION = 3.0  # seconds to hold gesture before triggering

# ──────────────────────────────────────────────
# WELCOME UI
# ──────────────────────────────────────────────
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_welcome_ui(current_gesture, confidence, hold_progress):
    """Draw the welcome/task selection panel."""
    panel = np.zeros((280, 640, 3), dtype=np.uint8)

    # Title
    cv2.putText(panel, "GESTURE-CONTROLLED ROBOTICS", (60, 35),
                font, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, "Hold a gesture for 3 seconds to select a task",
                (80, 60), font, 0.4, (180, 180, 180), 1)

    # Task cards
    for i, task in enumerate(TASKS):
        x = 20 + i * 210
        y = 80

        is_active = (current_gesture == task["gesture"] and confidence > 0.6)
        border_color = task["color"] if is_active else (80, 80, 80)
        thickness = 2 if is_active else 1

        # Card background
        cv2.rectangle(panel, (x, y), (x + 190, y + 100), border_color, thickness)

        # Gesture name
        cv2.putText(panel, task["gesture"].upper(), (x + 10, y + 25),
                    font, 0.5, task["color"], 2)

        # Task name
        cv2.putText(panel, task["name"], (x + 10, y + 55),
                    font, 0.6, (255, 255, 255), 1)

        # Status
        status_color = (0, 255, 0) if task["status"] == "READY" else (120, 120, 120)
        cv2.putText(panel, task["status"], (x + 10, y + 80),
                    font, 0.4, status_color, 1)

        # Progress bar if this gesture is held
        if is_active and task["status"] == "READY" and hold_progress > 0:
            bar_width = int(170 * min(hold_progress, 1.0))
            cv2.rectangle(panel, (x + 10, y + 88), (x + 10 + bar_width, y + 95),
                          task["color"], -1)
            cv2.rectangle(panel, (x + 10, y + 88), (x + 180, y + 95), border_color, 1)

    # Current detection
    det_color = (0, 255, 0) if current_gesture not in ("none", "None", "") else (100, 100, 100)
    cv2.putText(panel, f"Detected: {current_gesture.upper()}  ({confidence*100:.0f}%)",
                (20, 210), font, 0.6, det_color, 2)

    # Instructions
    cv2.putText(panel, "B: Toggle Blue Glove | 1: Quick Pick&Place | Q: Quit",
                (50, 250), font, 0.4, (120, 120, 120), 1)

    return panel

# ──────────────────────────────────────────────
# MAIN LAUNCHER LOOP
# ──────────────────────────────────────────────
def run_launcher():
    global latest_landmarks, raw_lms_for_drawing

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    sequence_data = []
    SEQUENCE_LENGTH = 20
    BLUE_GLOVE_MODE = True

    current_gesture = "none"
    current_confidence = 0.0

    # Gesture hold tracking
    hold_gesture = None
    hold_start_time = 0.0

    cv2.namedWindow("Robotics Launcher", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robotics Launcher", 640, 760)

    print("=" * 50)
    print("GESTURE-CONTROLLED ROBOTICS LAUNCHER")
    print("=" * 50)
    print("Hold GRASP for 3 seconds to start Pick & Place")
    print("Press 1 for quick start, Q to quit")
    print("=" * 50)

    with HandLandmarker.create_from_options(lm_options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # ── Hand detection ──
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if BLUE_GLOVE_MODE:
                frame_detect = np.ascontiguousarray(frame_rgb[:, :, [2, 1, 0]])
            else:
                frame_detect = frame_rgb

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_detect)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            # ── Draw landmarks ──
            if raw_lms_for_drawing:
                for lm in raw_lms_for_drawing:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # ── Prediction ──
            sequence_data.append(latest_landmarks.copy())
            if len(sequence_data) > SEQUENCE_LENGTH:
                sequence_data.pop(0)

            if len(sequence_data) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence_data), axis=0)
                preds = model.predict(input_data, verbose=0)[0]
                idx = np.argmax(preds)
                if preds[idx] > 0.60:
                    current_gesture = str(classes[idx])
                    current_confidence = float(preds[idx])
                else:
                    current_gesture = "none"
                    current_confidence = 0.0

            # ── Hold tracking ──
            hold_progress = 0.0
            if current_gesture != "none" and current_confidence > 0.6:
                if current_gesture == hold_gesture:
                    elapsed = time.time() - hold_start_time
                    hold_progress = elapsed / HOLD_DURATION
                else:
                    hold_gesture = current_gesture
                    hold_start_time = time.time()
                    hold_progress = 0.0
            else:
                hold_gesture = None
                hold_start_time = 0.0

            # ── Check if gesture held long enough ──
            trigger_task = None
            if hold_progress >= 1.0:
                for task in TASKS:
                    if task["gesture"] == hold_gesture and task["status"] == "READY":
                        trigger_task = task
                        break
                # Reset hold
                hold_gesture = None
                hold_start_time = 0.0

            # ── Display ──
            frame_display = cv2.resize(frame, (640, 480))

            glove_text = "GLOVE: ON" if BLUE_GLOVE_MODE else "GLOVE: OFF"
            cv2.putText(frame_display, glove_text, (520, 30), font, 0.5, (255, 165, 0), 1)

            ui = draw_welcome_ui(current_gesture, current_confidence, hold_progress)
            combined = np.vstack((frame_display, ui))
            cv2.imshow("Robotics Launcher", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                BLUE_GLOVE_MODE = not BLUE_GLOVE_MODE
            elif key == ord('1'):
                trigger_task = TASKS[0]  # quick trigger pick & place

            # ── Launch task if triggered ──
            if trigger_task:
                print(f"\n>>> Launching: {trigger_task['name']} <<<\n")

                # Close launcher windows temporarily
                cap.release()
                cv2.destroyWindow("Robotics Launcher")

                if trigger_task["gesture"] == "grasp":
                    from main import run_pick_and_place
                    run_pick_and_place()

                # Reopen launcher
                print("\n>>> Returning to launcher <<<\n")
                cap = cv2.VideoCapture(0)
                cv2.namedWindow("Robotics Launcher", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Robotics Launcher", 640, 760)

                # Reset state
                sequence_data.clear()
                current_gesture = "none"
                current_confidence = 0.0
                hold_gesture = None
                hold_start_time = 0.0
                trigger_task = None

    cap.release()
    cv2.destroyAllWindows()
    print("Launcher closed.")


if __name__ == "__main__":
    run_launcher()
