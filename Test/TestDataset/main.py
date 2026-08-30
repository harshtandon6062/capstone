import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
from ultralytics import YOLO

from gesture_module import detect_gesture
from ui_module import draw_ui

# -----------------------------
# PyBullet setup
# -----------------------------

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

# -----------------------------
# Spawn objects
# -----------------------------

# -----------------------------
# Spawn test tubes (cylinders)
# -----------------------------

objects = []

for i in range(5):

    x = 0.4
    y = -0.2 + i * 0.1
    z = 0.8

    radius = 0.015
    height = 0.12

    collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=[0.8, 0.8, 1, 1]  # light blue glass look
    )

    tube = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[x, y, z]
    )

    objects.append(tube)

# -----------------------------
# YOLO
# -----------------------------

model = YOLO("yolov8n.pt")

# -----------------------------
# Camera parameters
# -----------------------------

width=640
height=480

view_matrix=p.computeViewMatrix(
    [0.7,-0.7,1.2], # camera eye
    [0.3,0,0.6],    # camera target
    [0,0,1]         # camera_up
)

projection_matrix=p.computeProjectionMatrixFOV(
    75,
    width/height,
    0.1,
    3.1
)

cap=cv2.VideoCapture(0)

# -----------------------------
# SYSTEM STATE
# -----------------------------

system_state="RUNNING"

gesture_cooldown=0.4
last_gesture_time=0

# -----------------------------
# WINDOWS
# -----------------------------


#cv2.namedWindow("Gesture Camera",cv2.WINDOW_NORMAL)
cv2.namedWindow("Cabinet Scan Camera",cv2.WINDOW_NORMAL)

cv2.namedWindow("Gesture Control Interface", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control Interface",1280,480)

# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    success,webcam_frame=cap.read()
    if not success:
        break

    webcam_frame=cv2.flip(webcam_frame,1)

    gesture=detect_gesture(webcam_frame)

    now=time.time()

    # -----------------------------
    # Gesture controls
    # -----------------------------

    if now-last_gesture_time>gesture_cooldown:

        if gesture=="open_palm":

            if system_state=="RUNNING":
                system_state="PAUSED"
            elif system_state=="PAUSED":
                system_state="RUNNING"

            print("STATE:",system_state)
            last_gesture_time=now

        elif gesture=="thumbs_down":

            system_state="EMERGENCY STOP"
            print("EMERGENCY STOP ACTIVATED")

            for j in range(p.getNumJoints(robot)):
                p.setJointMotorControl2(
                    robot,
                    j,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0
                )

            last_gesture_time=now

    # -----------------------------
    # Simulation
    # -----------------------------

    if system_state=="RUNNING":
        p.stepSimulation()

    # -----------------------------
    # Gesture camera display
    # -----------------------------

    '''cv2.putText(
        webcam_frame,
        f"Gesture: {gesture}",
        (10,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Gesture Camera",webcam_frame)'''

    # -----------------------------
    # PyBullet camera
    # -----------------------------

    img=p.getCameraImage(
        width,
        height,
        view_matrix,
        projection_matrix
    )

    rgb=img[2]

    frame=np.reshape(rgb,(height,width,4))
    frame=frame[:,:,:3]
    frame=frame.astype(np.uint8)
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    # -----------------------------
    # YOLO detection
    # -----------------------------

    results=model(frame,verbose=False)

    annotated=frame.copy()

    for r in results:

        for box in r.boxes:

            x1,y1,x2,y2=map(int,box.xyxy[0])

            cv2.rectangle(
                annotated,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

    # -----------------------------
    # Draw UI
    # -----------------------------

    
    # UI on pybullet sim side
    annotated=cv2.resize(annotated,(640,480))
    cv2.imshow("Cabinet Scan Camera", annotated)
    

    # -----------------------------
    # Resize camera frame
    # -----------------------------

    webcam_frame = cv2.resize(webcam_frame,(640,480))

    # draw gesture text on camera
    cv2.putText(
        webcam_frame,
        f"Gesture: {gesture}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # -----------------------------
    #  UI panel
    # -----------------------------

    ui_panel = draw_ui(system_state,gesture)

    ui_panel = cv2.resize(ui_panel,(640,480))

    combined_view = np.hstack((webcam_frame, ui_panel))

    cv2.imshow("Gesture Control Interface", combined_view)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()