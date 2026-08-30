"""Shared MediaPipe hand landmark capture used by both static and dynamic gesture paths."""

import cv2
import numpy as np
import time

import mediapipe as mp

from config import HAND_LANDMARKER_TASK


class HandLandmarkProvider:
    """Provide a single MediaPipe landmark provider and keep a shared latest frame state."""

    def __init__(self, model_asset_path=None, num_hands=1):
        self.model_asset_path = model_asset_path or HAND_LANDMARKER_TASK
        self.num_hands = num_hands

        self.latest_landmarks = np.zeros(63, dtype=np.float32)
        self.latest_hand_landmarks = []
        self.raw_lms_for_drawing = []

        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.model_asset_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        )

    def update_from_frame(self, frame_bgr):
        """Synchronously update the shared landmark state from a single camera frame."""
        if frame_bgr is None:
            self.latest_landmarks = np.zeros(63, dtype=np.float32)
            self.latest_hand_landmarks = []
            self.raw_lms_for_drawing = []
            return self.latest_landmarks.copy()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, int(time.time() * 1000))

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            self.raw_lms_for_drawing = hand
            self.latest_hand_landmarks = [hand]
            base_x, base_y, base_z = hand[0].x, hand[0].y, hand[0].z
            self.latest_landmarks = np.array(
                [[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in hand],
                dtype=np.float32,
            ).flatten()
        else:
            self.latest_hand_landmarks = []
            self.raw_lms_for_drawing = []
            self.latest_landmarks = np.zeros(63, dtype=np.float32)

        return self.latest_landmarks.copy()

    def close(self):
        if hasattr(self, "_landmarker"):
            self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
