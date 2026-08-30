import cv2
import numpy as np

font=cv2.FONT_HERSHEY_SIMPLEX

def draw_ui(state,gesture):

    panel=np.zeros((480,640,3),dtype=np.uint8)

    cv2.putText(
        panel,
        "SYSTEM STATUS",
        (160,80),
        font,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        panel,
        f"{state}",
        (200,180),
        font,
        1.2,
        (0,255,255),
        3
    )

    cv2.putText(
        panel,
        "GESTURE",
        (230,280),
        font,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        panel,
        gesture,
        (200,360),
        font,
        1,
        (0,255,0),
        2
    )

    return panel



'''
# UI on pybullet side
import cv2
import numpy as np

font=cv2.FONT_HERSHEY_SIMPLEX

def draw_ui(state,gesture):

    panel=np.zeros((120,640,3),dtype=np.uint8)

    cv2.putText(
        panel,
        f"STATE: {state}",
        (20,40),
        font,
        0.9,
        (0,255,255),
        2
    )

    cv2.putText(
        panel,
        f"GESTURE: {gesture}",
        (20,90),
        font,
        0.9,
        (0,255,0),
        2
    )

    return panel

'''