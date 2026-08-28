from gesture_controller import GestureController
from hand_landmark_provider import HandLandmarkProvider


def test_gesture_controller_emits_commands():
    controller = GestureController(dynamic_classes=["grasp", "none", "sweep", "tilt", "wrist_rotation"])
    event = controller.handle_static_gesture("pinch")
    assert event is not None
    assert event["source"] == "static"

    dynamic_event = controller.handle_dynamic_gesture("grasp", 0.95)
    assert dynamic_event is not None
    assert dynamic_event["source"] == "dynamic"


def test_landmark_provider_has_expected_shape():
    provider = HandLandmarkProvider(num_hands=1)
    assert provider.latest_landmarks.shape == (63,)
