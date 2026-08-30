import cv2
import numpy as np

font=cv2.FONT_HERSHEY_SIMPLEX

def draw_ui(num_objects,selected,source,dest,state,gesture):

    panel=np.zeros((200,640,3),dtype=np.uint8)

    cv2.putText(panel,f"STATE: {state}",(10,30),font,0.7,(0,255,255),2)
    cv2.putText(panel,f"GESTURE: {gesture}",(10,60),font,0.7,(0,255,0),2)

    for i in range(num_objects):

        x=20+i*110
        y=120

        color=(200,200,200)

        if i==selected:
            color=(0,255,255)

        if source==i:
            color=(255,0,0)

        if dest==i:
            color=(0,0,255)

        cv2.rectangle(panel,(x,y),(x+80,y+50),color,2)

        cv2.putText(
            panel,
            f"Obj {i+1}",
            (x+10,y+30),
            font,
            0.5,
            color,
            2
        )

    return panel