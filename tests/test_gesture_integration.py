from gestures.controller import GestureController
from gestures.landmarks import HandLandmarkProvider
from config import (ACTIONS, GESTURE_COOLDOWN, GESTURE_HOLD_DURATION,
                    action_is_irreversible, action_needs_target)


def test_gesture_controller_emits_commands():
    controller = GestureController(dynamic_classes=["grasp", "none", "sweep", "tilt", "wrist_rotation"])
    event = controller.handle_static_gesture("pinch")
    assert event is not None
    assert event["source"] == "static"

    dynamic_event = controller.handle_dynamic_gesture("grasp", 0.95)
    assert dynamic_event is not None
    assert dynamic_event["source"] == "dynamic"


def test_mix_action_and_hold_timing_are_available():
    assert any(action["key"] == "mix" for action in ACTIONS)
    assert GESTURE_HOLD_DURATION >= 1.2
    assert GESTURE_COOLDOWN >= 1.0


def test_landmark_provider_has_expected_shape():
    provider = HandLandmarkProvider(num_hands=1)
    assert provider.latest_landmarks.shape == (63,)


def test_every_action_declares_whether_it_needs_a_target():
    """A missing declaration would silently fall back to demanding a target."""
    for action in ACTIONS:
        assert "needs_target" in action, f"{action['key']} does not say"
        assert isinstance(action["needs_target"], bool)


def test_mix_runs_on_the_tube_already_chosen():
    """Mix lifts, rotates and sets down one tube, so there is no destination.

    Asking for one is what the operator reported: a selection step whose answer
    the motion never reads.
    """
    assert action_needs_target("mix") is False
    assert action_needs_target("move") is True
    assert action_needs_target("pour") is True


def test_unknown_action_is_assumed_to_need_a_target():
    """Refusing to start without a target is the safe way to be wrong."""
    assert action_needs_target("something_new") is True
    assert action_needs_target(None) is True


def test_every_irreversible_action_is_treated_as_irreversible():
    """The longer commit hold must follow the flag, not a hardcoded action name.

    Mix had no confirmation at all because the check named pour specifically,
    even though MixCommand.undo() also refuses.
    """
    for action in ACTIONS:
        assert action_is_irreversible(action["key"]) is (not action["reversible"])

    assert action_is_irreversible("pour") is True
    assert action_is_irreversible("mix") is True
    assert action_is_irreversible("move") is False


def test_unknown_action_is_assumed_irreversible():
    """Demanding a deliberate hold for something unrecognised is the safe error."""
    assert action_is_irreversible("something_new") is True
    assert action_is_irreversible(None) is True


def test_a_target_less_action_is_still_confirmed():
    """No destination to pick is not the same as nothing to confirm.

    Mix needs no target and cannot be undone, so it is exactly the case where
    skipping confirmation would be easiest and worst.
    """
    assert action_needs_target("mix") is False
    assert action_is_irreversible("mix") is True
