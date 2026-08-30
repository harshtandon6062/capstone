import pybullet as p
import pybullet_data
import cv2
import numpy as np
import os

save_path = "dataset/images/train"
os.makedirs(save_path, exist_ok=True)

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf",[0.5,0,0])

# spawn tubes
for i in range(5):
    p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.12),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.12),
        basePosition=[0.4,-0.2+i*0.1,0.8]
    )

width, height = 640, 480

view = p.computeViewMatrix([0.7,-0.7,1.2],[0.3,0,0.6],[0,0,1])
proj = p.computeProjectionMatrixFOV(75,width/height,0.1,3.1)

count = 0

while True:

    img = p.getCameraImage(width,height,view,proj)
    frame = np.reshape(img[2],(height,width,4))[:,:,:3]
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Capture",frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = f"{save_path}/img_{count}.jpg"
        cv2.imwrite(filename,frame)
        print("Saved:",filename)
        count += 1

    elif key == ord('q'):
        break

cv2.destroyAllWindows()