import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
from ultralytics import YOLO
from gesture_module import detect_gesture

# --------------------------------
# Start physics simulation
# --------------------------------
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.8)
# Load YOLO model
model = YOLO("yolov8n.pt")   # small fast model
cap=cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera error")

    exit()

# WINDOW SETTINGS (ADD HERE)

cv2.namedWindow("Gesture Camera", cv2.WINDOW_NORMAL)

cv2.namedWindow("Cabinet Scan Camera", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Gesture Camera",640,480)

cv2.resizeWindow("Cabinet Scan Camera",640,480)
# --------------------------------
# Load environment
# --------------------------------
plane = p.loadURDF("plane.urdf")

table = p.loadURDF(
    "table/table.urdf",
    basePosition=[0.5, 0, 0]
)

# --------------------------------
# Spawn objects (test tube placeholders)
# --------------------------------
objects = []

for i in range(5):

    x = 0.4
    y = -0.2 + i * 0.1
    z = 0.75

    obj = p.loadURDF(
        "cube_small.urdf",
        basePosition=[x, y, z]
    )

    objects.append(obj)

# --------------------------------
# Load robot
# --------------------------------
robot = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0.2, 0, 0],
    useFixedBase=True
)
paused=False

# --------------------------------
# Camera parameters
# --------------------------------

width = 640
height = 480


camera_eye = [0.7, -0.7, 1.2]     # camera position
camera_target = [0.3, 0, 0.6]   # looking at table center
camera_up = [0, 0, 1]


view_matrix = p.computeViewMatrix(
    camera_eye,
    camera_target,
    camera_up
)

projection_matrix = p.computeProjectionMatrixFOV(
    fov=75,
    aspect=width/height,
    nearVal=0.1,
    farVal=3.1
)
print(p.getBasePositionAndOrientation(robot))
# --------------------------------
# Simulation loop
# --------------------------------

while True:
    success,webcam_frame=cap.read()

    if not success:
        break

    gesture=detect_gesture(webcam_frame)

    if gesture!="unknown":

        print("Detected:",gesture)
    

    if gesture=="open_palm":

         paused = not paused

         print("PAUSE TOGGLED:",paused)

         time.sleep(0.4)   # prevent rapid toggling
        # paused=False

        # print("ROBOT RESUMED")

        # time.sleep(0.3)


    elif gesture=="thumbs_down":

        print("EMERGENCY STOP")

        paused=True

        # stop all robot joints (realistic emergency stop)

        for j in range(p.getNumJoints(robot)):

            p.setJointMotorControl2(

                robot,

                j,

                p.VELOCITY_CONTROL,

                targetVelocity=0,

                force=0
            )


    if not paused:

        p.stepSimulation()

        time.sleep(1/240)



    
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

    cv2.imshow("Gesture Camera",webcam_frame)

    img = p.getCameraImage(width, height, view_matrix, projection_matrix)
    rgb = img[2]

    frame = np.reshape(rgb, (height, width, 4))
    frame = frame[:, :, :3]
    frame = frame.astype(np.uint8)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- YOLO detection ---
    results = model(frame, verbose=False)

    annotated = frame.copy()

    for r in results:
        boxes = r.boxes

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            label = f"Obj {i+1} ({conf:.2f})"

            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )
    annotated=cv2.resize(annotated,(640,480))
    cv2.imshow("Cabinet Scan Camera", annotated)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()

cv2.destroyAllWindows()   