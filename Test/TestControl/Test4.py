import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time

from gesture_module import detect_gesture
from ui_module import draw_ui
from robot_controller import execute_pick_place


# Physics Simulation

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)

plane = p.loadURDF("plane.urdf")
table = p.loadURDF("table/table.urdf",[0.5,0,0])

robot = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0.2,0,0],
    useFixedBase=True
)

# -------------------------------
# Spawn Objects
# -------------------------------

objects=[]

for i in range(5):
    obj = p.loadURDF(
        "cube_small.urdf",
        basePosition=[0.4,-0.2+i*0.1,0.75]
    )
    objects.append(obj)

num_objects=len(objects)

cap=cv2.VideoCapture(0)

# -------------------------------
# UI STATE MACHINE
# -------------------------------

selected_object=0
source_object=None
destination_object=None

system_state="SELECT_SOURCE"

last_gesture_time=0
gesture_cooldown=0.35

# -------------------------------
# WINDOW (single combined window)
# -------------------------------

cv2.namedWindow("Gesture Control",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control",640,680)

# -------------------------------
# MAIN LOOP
# -------------------------------

while True:

    success,webcam_frame=cap.read()
    if not success:
        break

    webcam_frame=cv2.flip(webcam_frame,1)

    gesture=detect_gesture(webcam_frame)

    now=time.time()

    # -------------------------------
    # Gesture Navigation
    # -------------------------------

    if now-last_gesture_time>gesture_cooldown:

        if gesture=="point_right":
            selected_object=min(selected_object+1,num_objects-1)
            last_gesture_time=now

        elif gesture=="point_left":
            selected_object=max(selected_object-1,0)
            last_gesture_time=now

        elif gesture=="pinch":
            if system_state=="SELECT_SOURCE":
                source_object=selected_object
                system_state="SELECT_DEST"
                print("SOURCE:",source_object)

            elif system_state=="SELECT_DEST":
                destination_object=selected_object
                system_state="EXECUTE"
                print("DEST:",destination_object)

            last_gesture_time=now

    # -------------------------------
    # Execute Robot Task
    # -------------------------------

    if system_state=="EXECUTE":
        execute_pick_place(
            robot,
            objects[source_object],
            objects[destination_object]
        )
        system_state="SELECT_SOURCE"
        source_object=None
        destination_object=None

    # -------------------------------
    # Simulation Step
    # -------------------------------

    p.stepSimulation()

    # -------------------------------
    # Display: webcam + UI in one window
    # -------------------------------

    webcam_frame=cv2.resize(webcam_frame,(640,480))

    cv2.putText(
        webcam_frame,
        f"Gesture: {gesture}",
        (10,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    ui_panel=draw_ui(
        num_objects,
        selected_object,
        source_object,
        destination_object,
        system_state,
        gesture
    )

    combined=np.vstack((webcam_frame,ui_panel))
    cv2.imshow("Gesture Control",combined)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()