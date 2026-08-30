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
