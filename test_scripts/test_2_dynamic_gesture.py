"""
TEST 2: Dynamic Gesture Recognition
=====================================
Tests the LSTM-based dynamic gesture model from dynamic_gestures/.
Requires TensorFlow. Install with:
    ~/shared_ml_env/bin/pip install tensorflow

Run: ~/shared_ml_env/bin/python test_scripts/test_2_dynamic_gesture.py
"""
import sys
import os

# Check TensorFlow before anything else
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} OK")
except ImportError:
    print("=" * 50)
    print("ERROR: TensorFlow is not installed.")
    print("Install it with:")
    print("  ~/shared_ml_env/bin/pip install tensorflow")
    print("(~600MB download)")
    print("=" * 50)
    sys.exit(1)

dynamic_dir = os.path.join(os.path.dirname(__file__), '..', 'dynamic_gestures')
os.chdir(dynamic_dir)

# Check if trained model exists
model_path = os.path.join(dynamic_dir, 'gesture_landmark_model.h5')
classes_path = os.path.join(dynamic_dir, 'classes.npy')

if not os.path.exists(model_path) or not os.path.exists(classes_path):
    print("=" * 50)
    print("WARNING: No trained model found!")
    print(f"  Missing: {model_path}")
    print("You need to first:")
    print("  1. Record gesture videos into dynamic_gestures/dynamic gestures/")
    print("  2. Run: ~/shared_ml_env/bin/python dynamic_gestures/dynamic_landmarks.py")
    print("  3. Run: ~/shared_ml_env/bin/python dynamic_gestures/dynamic_gesture_train.py")
    print("=" * 50)
    
    # Fall back to a simulated demo
    print("\nRunning SIMULATED dynamic gesture demo instead...\n")
    
    import cv2
    import numpy as np
    import mediapipe as mp
    import time
    
    mp_path = 'hand_landmarker.task'
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Shared variables
    latest_landmarks = np.zeros(63)
    raw_lms = []
    prev_wrist_positions = []
    
    def handle_result(result, output_image, timestamp_ms):
        global latest_landmarks, raw_lms
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            raw_lms = lms
            base_x, base_y, base_z = lms[0].x, lms[0].y, lms[0].z
            latest_landmarks = np.array(
                [[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in lms]
            ).flatten()
        else:
            latest_landmarks = np.zeros(63)
            raw_lms = []
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=handle_result,
        num_hands=1
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        sys.exit(1)
    
    WINDOW = 20
    sequence = []
    
    print("=" * 50)
    print("SIMULATED DYNAMIC GESTURE TEST")
    print("=" * 50)
    print("This shows landmark tracking + motion heuristics.")
    print("(Without a trained model, using simple wrist motion detection)")
    print("Move your hand to see motion tracked.")
    print("Press 'q' to quit")
    print("=" * 50)
    
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mp_image, int(time.time() * 1000))
            
            # Draw landmarks
            if raw_lms:
                for lm in raw_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                
                # Track wrist motion for simple heuristic
                wrist_x, wrist_y = raw_lms[0].x, raw_lms[0].y
                prev_wrist_positions.append((wrist_x, wrist_y))
                if len(prev_wrist_positions) > WINDOW:
                    prev_wrist_positions.pop(0)
                
                if len(prev_wrist_positions) >= WINDOW:
                    dx = prev_wrist_positions[-1][0] - prev_wrist_positions[0][0]
                    dy = prev_wrist_positions[-1][1] - prev_wrist_positions[0][1]
                    motion = np.sqrt(dx**2 + dy**2)
                    
                    if motion > 0.15:
                        if abs(dx) > abs(dy):
                            gesture = "SWEEP (horizontal motion)"
                        else:
                            gesture = "TILT (vertical motion)"
                        color = (0, 255, 0)
                    elif motion > 0.05:
                        gesture = "SMALL MOTION"
                        color = (0, 255, 255)
                    else:
                        gesture = "STATIC"
                        color = (0, 0, 255)
                    
                    cv2.putText(frame, f"Motion: {gesture}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Magnitude: {motion:.3f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(frame, "[No trained model - heuristic mode]", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.imshow("Test 2: Dynamic Gesture (Simulated)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Dynamic gesture test (simulated) complete!")
    sys.exit(0)

# --- If model exists, run real inference ---
import cv2
import numpy as np
import mediapipe as mp
import time

model = tf.keras.models.load_model('gesture_landmark_model.h5')
classes = np.load('classes.npy')

print("=" * 50)
print("DYNAMIC GESTURE TEST (REAL MODEL)")
print("=" * 50)
print(f"Model loaded! Classes: {list(classes)}")
print("Perform dynamic gestures in front of your webcam.")
print("Press 'q' to quit, 'b' to toggle blue glove mode")
print("=" * 50)

mp_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_landmarks = np.zeros(63)
raw_lms_for_drawing = []
sequence_data = []
SEQUENCE_LENGTH = 20
BLUE_GLOVE_MODE = True

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

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mp_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result,
    num_hands=1
)

cap = cv2.VideoCapture(0)
current_prediction, current_confidence = "None", 0.0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if BLUE_GLOVE_MODE:
            frame_detect = np.ascontiguousarray(frame_rgb[:, :, [2, 1, 0]])
        else:
            frame_detect = frame_rgb

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_detect)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if raw_lms_for_drawing:
            for lm in raw_lms_for_drawing:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        sequence_data.append(latest_landmarks)
        if len(sequence_data) > SEQUENCE_LENGTH:
            sequence_data.pop(0)

        if len(sequence_data) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(sequence_data), axis=0)
            preds = model.predict(input_data, verbose=0)[0]
            idx = np.argmax(preds)
            if preds[idx] > 0.60:
                current_prediction = classes[idx]
                current_confidence = preds[idx]
            else:
                current_prediction = "None"
                current_confidence = 0.0

        color = (0, 255, 0) if current_prediction.lower() != "none" else (0, 0, 255)
        cv2.putText(frame, f"GESTURE: {current_prediction.upper()}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"CONFIDENCE: {current_confidence*100:.1f}%", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        status = "GLOVE MODE: ON" if BLUE_GLOVE_MODE else "GLOVE MODE: OFF"
        cv2.putText(frame, status, (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(frame, "B: Toggle Glove | Q: Quit", (20, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Test 2: Dynamic Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('b'):
            BLUE_GLOVE_MODE = not BLUE_GLOVE_MODE

cap.release()
cv2.destroyAllWindows()
print("Dynamic gesture test complete!")
