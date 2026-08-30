import cv2
import numpy as np
import os
import mediapipe as mp

# Configuration
model_path = 'hand_landmarker.task' 
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

DATA_DIR = 'dynamic gestures'
SEQUENCE_LENGTH = 20

def process_single_video(video_full_path):
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_full_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        all_landmarks = []
        timestamp_ms = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- BLUE GLOVE TRICK ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_detect = np.ascontiguousarray(frame_rgb[:,:, [2,1,0]])
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_detect)
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += int(1000 / fps)
            
            if detection_result.hand_landmarks:
                lms = detection_result.hand_landmarks[0]
                # WRIST-RELATIVE NORMALIZATION
                base_x, base_y, base_z = lms[0].x, lms[0].y, lms[0].z
                res = np.array([[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in lms]).flatten()
                all_landmarks.append(res)
        
        cap.release()
        
        total_found = len(all_landmarks)
        if total_found == 0:
            return np.zeros((SEQUENCE_LENGTH, 63))
            
        # Resample to 20 frames spread across the WHOLE video duration
        indices = np.linspace(0, total_found - 1, SEQUENCE_LENGTH, dtype=int)
        return np.array([all_landmarks[i] for i in indices])

if __name__ == "__main__":
    X, y = [], []
    video_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mp4')]
    print(f"Extracting landmarks from {len(video_files)} videos...")
    
    for filename in video_files:
        path = os.path.join(DATA_DIR, filename)
        landmarks = process_single_video(path)
        
        raw_name = filename.split('(')[0].split('.')[0].strip().lower()
        # This checks if the word exists anywhere in the filename
        if "grasp" in raw_name:
            label = "grasp"
        elif "sweep" in raw_name:
            label = "sweep"
        elif "tilt" in raw_name:
            label = "tilt"
        elif "wrist" in raw_name:
            label = "wrist_rotation"
        else:
            label = "none"
        
        X.append(landmarks)
        y.append(label)
        print(f"Processed: {filename} -> {label}")

    np.save('X_landmarks.npy', np.array(X))
    np.save('y_labels.npy', np.array(y))
    print("Dataset saved!")