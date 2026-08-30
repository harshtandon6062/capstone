"""
TEST 1: Static Gesture Recognition
===================================
Tests the gesture_module from WithUI/ using your webcam.
Press 'q' to quit.

Run: ~/shared_ml_env/bin/python test_scripts/test_1_static_gesture.py
"""
import sys
import os

# Add WithUI to path so we can import gesture_module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'WithUI'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'WithUI'))

import cv2
import time
from gesture_module import detect_gesture

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

print("=" * 50)
print("STATIC GESTURE TEST")
print("=" * 50)
print("Show these gestures to your webcam:")
print("  - Open palm  -> 'open_palm'")
print("  - Thumbs down -> 'thumbs_down'")
print("  - Point (index finger) -> 'point_up/down/left/right'")
print("  - Pinch (thumb+index) -> 'pinch'")
print("  - Thumbs up -> 'thumbs_up'")
print("Press 'q' to quit")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gesture = detect_gesture(frame)

    # Color based on detection
    color = (0, 255, 0) if gesture != "unknown" else (0, 0, 255)

    cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Test 1: Static Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Static gesture test complete!")
