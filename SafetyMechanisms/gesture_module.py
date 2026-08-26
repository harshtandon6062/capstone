import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

import numpy as np
import math
from collections import deque

class GestureDetector:

    def __init__(self):

        self.history=deque(maxlen=6)

        self.pinch_threshold=0.045

        self.direction_threshold=1.25

    def vector(self,p1,p2):

        return np.array([p2.x-p1.x,p2.y-p1.y])

    def magnitude(self,v):

        return np.linalg.norm(v)

    def angle(self,v1,v2):

        cos=np.dot(v1,v2)/(self.magnitude(v1)*self.magnitude(v2)+1e-6)

        return math.degrees(
            math.acos(
                np.clip(cos,-1,1)
            )
        )

    def finger_extended(self,mcp,pip,tip):

        base=self.vector(mcp,pip)

        top=self.vector(pip,tip)

        ang=self.angle(base,top)

        return ang<35

    def thumb_extended(self,l):

        base=self.vector(l[2],l[3])

        top=self.vector(l[3],l[4])

        return self.angle(base,top)<40

    def distance(self,p1,p2):

        return math.sqrt(
            (p1.x-p2.x)**2+
            (p1.y-p2.y)**2
        )

    def finger_direction(self,mcp,tip):

        dx=tip.x-mcp.x
        dy=tip.y-mcp.y

        if abs(dx)>abs(dy)*self.direction_threshold:

            if dx>0:
                return "point_left"

            else:
                return "point_right"

        else:

            if dy<0:
                return "point_up"

            else:
                return "point_down"

    def thumb_direction(self,l):

        wrist=l[0]

        tip=l[4]

        dx=tip.x-wrist.x
        dy=tip.y-wrist.y

        if abs(dx)>abs(dy)*self.direction_threshold:

            if dx>0:
                return "thumb_right"

            else:
                return "thumb_left"

        else:

            if dy<0:
                return "thumbs_up"

            else:
                return "thumbs_down"

    def stable(self,g):

        self.history.append(g)

        return max(
            set(self.history),
            key=self.history.count
        )

    def detect(self,l):

        thumb=self.thumb_extended(l)

        index=self.finger_extended(l[5],l[6],l[8])

        middle=self.finger_extended(l[9],l[10],l[12])

        ring=self.finger_extended(l[13],l[14],l[16])

        pinky=self.finger_extended(l[17],l[18],l[20])

        extended=sum([thumb,index,middle,ring,pinky])

        gesture="unknown"

        if extended>=4:

            gesture="open_palm"

        elif self.distance(l[4],l[8])<self.pinch_threshold:

            if index:

                gesture="pinch"

        elif thumb and not index and not middle and not ring and not pinky:

            gesture=self.thumb_direction(l)

        elif index and not middle and not ring and not pinky:

            gesture=self.finger_direction(
                l[5],
                l[8]
            )

        return self.stable(gesture)
    
options = vision.HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),

    running_mode=vision.RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.7,

    min_tracking_confidence=0.7
)

detector_mp = vision.HandLandmarker.create_from_options(options)

gesture_detector=GestureDetector()


def detect_gesture(frame):

    frame=cv2.flip(frame,1)

    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image=mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=frame_rgb
    )

    timestamp=int(time.time()*1000)

    result=detector_mp.detect_for_video(
        mp_image,
        timestamp
    )

    gesture="unknown"

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            gesture=gesture_detector.detect(hand)
            break

    return gesture