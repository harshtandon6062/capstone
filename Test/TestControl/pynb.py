import os
import urllib.request

# Only download if the model doesn't exist
if not os.path.exists('hand_landmarker.task'):
    print("Downloading model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'hand_landmarker.task'
    )
    print("Model downloaded successfully!")
else:
    print("Model already exists!")


import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

print("All libraries imported successfully!")

# Constants for drawing
MARGIN = 10  # pixels above hand to place label
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # Vibrant green (BGR format)

def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draws hand landmarks and handedness labels on an image.
    
    Args:
        rgb_image: Input image as NumPy array (RGB format)
        detection_result: MediaPipe HandLandmarkerResult object
    
    Returns:
        Image with landmarks drawn on it
    """
    # Get lists of landmarks and handedness from detection result
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    # Create a copy so we don't modify the original
    annotated_image = np.copy(rgb_image)

    # Loop through each detected hand
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]  # 21 landmarks for this hand
        handedness = handedness_list[idx]  # Left or Right classification

        # Convert landmarks to protobuf format (required by MediaPipe drawing utils)
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, 
                y=landmark.y, 
                z=landmark.z
            ) for landmark in hand_landmarks
        ])
        
        # Draw the hand skeleton (dots + connecting lines)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,  # Defines which landmarks to connect
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Calculate position for the handedness label (top-left of hand)
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        
        # Convert normalized coordinates (0-1) to pixel coordinates
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw "Left" or "Right" label
        # Since we flip the image horizontally for mirror effect, we need to 
        # swap the handedness labels (Left becomes Right, Right becomes Left)
        original_label = handedness[0].category_name
        corrected_label = "Right" if original_label == "Left" else "Left"
        
        cv2.putText(
            annotated_image,
            f"{corrected_label}",  # Corrected label for mirrored display
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA  # Anti-aliased (smooth) text
        )

    return annotated_image

print("Visualization function defined!")

# Configure the detector
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # Optimized for video frames
    num_hands=2,  # Detect up to 2 hands
    min_hand_detection_confidence=0.5,  # Minimum confidence to detect a hand
    min_hand_presence_confidence=0.5,   # Minimum confidence hand is still present
    min_tracking_confidence=0.5         # Minimum confidence for tracking between frames
)

# Create the detector
detector = vision.HandLandmarker.create_from_options(options)

print("Hand detector initialized!")
print(f"- Running mode: VIDEO")
print(f"- Max hands: 2")
print(f"- Detection confidence: 0.5")

# Open the webcam (0 = default camera, try 1 or 2 if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("ERROR: Could not open camera!")
    print("Try changing VideoCapture(0) to VideoCapture(1)")
else:
    print("Camera opened successfully!")
    print("Press 'q' to quit, 'b' to toggle Blue Glove Mode")
    print("-" * 40)

# Variables for FPS calculation
prev_time = 0
fps = 0

# Toggle for blue glove mode (swap R and B channels)
BLUE_GLOVE_MODE = True  # Set to True when wearing blue gloves

# Main loop - runs until 'q' is pressed
while cap.isOpened():
    # Step 1: Read a frame from the webcam
    success, frame = cap.read()
    
    if not success:
        print("Failed to read frame from camera")
        break
    
    # Step 2: Flip the frame horizontally (mirror effect - more intuitive)
    frame = cv2.flip(frame, 1)
    
    # Step 3: Convert BGR (OpenCV format) to RGB (MediaPipe format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Step 3.5: BLUE GLOVE MODE - Swap R and B channels for detection
    # This makes blue gloves appear orange/red, which MediaPipe can detect better
    if BLUE_GLOVE_MODE:
        # Swap R and B channels and make contiguous for MediaPipe
        frame_for_detection = np.ascontiguousarray(frame_rgb[:, :, [2, 1, 0]])
    else:
        frame_for_detection = frame_rgb
    
    # Step 4: Create MediaPipe Image object (use color-swapped frame for detection)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_for_detection)
    
    # Step 5: Get current timestamp in milliseconds (required for VIDEO mode)
    timestamp_ms = int(time.time() * 1000)
    
    # Step 6: Run hand detection
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    # Step 7: Draw landmarks on the ORIGINAL frame (not color-swapped)
    # This way the display looks normal but detection uses swapped colors
    annotated_frame = draw_landmarks_on_image(frame_rgb, detection_result)
    
    # Step 8: Convert back to BGR for OpenCV display
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    # Step 9: Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
    
    # Draw FPS on the frame
    cv2.putText(
        annotated_frame_bgr,
        f"FPS: {fps:.1f}",
        (10, 30),  # Position: top-left corner
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # Font size
        (0, 255, 0),  # Green color (BGR)
        2,  # Thickness
        cv2.LINE_AA
    )
    
    # Draw number of hands detected
    num_hands = len(detection_result.hand_landmarks)
    cv2.putText(
        annotated_frame_bgr,
        f"Hands: {num_hands}",
        (10, 70),  # Position: below FPS
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    
    # Show blue glove mode status
    mode_text = "Blue Glove Mode: ON (press 'b' to toggle)" if BLUE_GLOVE_MODE else "Blue Glove Mode: OFF (press 'b' to toggle)"
    cv2.putText(
        annotated_frame_bgr,
        mode_text,
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 165, 0) if BLUE_GLOVE_MODE else (128, 128, 128),
        2,
        cv2.LINE_AA
    )
    
    # Step 10: Display the frame in a window
    cv2.imshow('Hand Landmark Detection - Press Q to Quit', annotated_frame_bgr)
    
    # Step 11: Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('b'):
        BLUE_GLOVE_MODE = not BLUE_GLOVE_MODE
        status = "ON" if BLUE_GLOVE_MODE else "OFF"
        print(f"Blue Glove Mode: {status}")

# Cleanup: Release the camera and close windows
cap.release()
cv2.destroyAllWindows()



print("Camera released and windows closed!")


# Run this cell if you want to use a lower resolution for better performance
# Then run the main detection cell again

cap = cv2.VideoCapture(0)

# Set resolution to 640x480 (lower = faster)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Resolution set to: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

cap.release()