"""
TEST 4: UI Panel
================
Tests the UI module by cycling through different states/gestures.
No webcam needed — uses simulated state changes.

Run: ~/shared_ml_env/bin/python test_scripts/test_4_ui.py
"""
import sys
import os
import cv2
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'WithUI'))

from ui_module import draw_ui

print("=" * 50)
print("UI PANEL TEST")
print("=" * 50)
print("Cycling through states and gestures.")
print("Press 'q' to quit, any other key to advance.")
print("=" * 50)

states = ["RUNNING", "PAUSED", "EMERGENCY STOP", "RUNNING"]
gestures = ["unknown", "open_palm", "thumbs_down", "point_right", "pinch", "thumbs_up"]

cv2.namedWindow("Test 4: UI Panel", cv2.WINDOW_NORMAL)

idx = 0
for state in states:
    for gesture in gestures:
        # Draw the UI panel
        panel = draw_ui(state, gesture)
        panel = cv2.resize(panel, (640, 480))

        # Add test info overlay
        info = np.zeros((60, 640, 3), dtype=np.uint8)
        cv2.putText(info, f"Test {idx+1}: state='{state}' gesture='{gesture}'",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        combined = np.vstack((info, panel))
        cv2.imshow("Test 4: UI Panel", combined)

        idx += 1
        key = cv2.waitKey(800)  # auto-advance after 800ms
        if key == ord('q'):
            cv2.destroyAllWindows()
            print("UI test complete!")
            sys.exit(0)

cv2.destroyAllWindows()
print("UI test complete! All states/gestures rendered successfully.")
