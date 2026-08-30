import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# 1. Load Model and Classes
try:
    model = tf.keras.models.load_model('gesture_landmark_model.h5')
    classes = np.load('classes.npy')
except Exception as e:
    print(f"Error loading model or classes: {e}")
    exit()

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Shared variables
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
        # WRIST-RELATIVE NORMALIZATION (Match Training)
        base_x, base_y, base_z = lms[0].x, lms[0].y, lms[0].z
        latest_landmarks = np.array([[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in lms]).flatten()
    else:
        latest_landmarks = np.zeros(63)
        raw_lms_for_drawing = []

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result,
    num_hands=1
)

def run_live_test():
    global BLUE_GLOVE_MODE, raw_lms_for_drawing
    cap = cv2.VideoCapture(0)
    current_prediction, current_confidence = "None", 0.0

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # --- BLUE GLOVE TRICK ---
            # 1. Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Swap R and B channels if Glove Mode is ON
            if BLUE_GLOVE_MODE:
                # We use the [2, 1, 0] trick on the RGB frame
                frame_detect = np.ascontiguousarray(frame_rgb[:,:, [2,1,0]])
            else:
                frame_detect = frame_rgb

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_detect)
            
            # 3. Detect
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            # --- DRAW LANDMARKS ---
            if raw_lms_for_drawing:
                for lm in raw_lms_for_drawing:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # --- PREDICTION ---
            sequence_data.append(latest_landmarks)
            if len(sequence_data) > SEQUENCE_LENGTH:
                sequence_data.pop(0)

            if len(sequence_data) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence_data), axis=0)
                preds = model.predict(input_data, verbose=0)[0]
                idx = np.argmax(preds)

                # Set threshold (0.6 is often better for small datasets)
                if preds[idx] > 0.60:
                    current_prediction = classes[idx]
                    current_confidence = preds[idx]
                else:
                    current_prediction = "None"
                    current_confidence = 0.0

            # --- UI ---
            color = (0, 255, 0) if current_prediction.lower() != "none" else (0, 0, 255)
            cv2.putText(frame, f"GESTURE: {current_prediction.upper()}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(frame, f"CONFIDENCE: {current_confidence*100:.1f}%", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            status = "GLOVE MODE: ON" if BLUE_GLOVE_MODE else "GLOVE MODE: OFF"
            cv2.putText(frame, status, (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            cv2.putText(frame, "B: Toggle Glove | Q: Quit", (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Capstone Final Demo', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('b'): BLUE_GLOVE_MODE = not BLUE_GLOVE_MODE

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_test()