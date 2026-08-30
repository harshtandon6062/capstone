"""Direction naming must match what the operator actually did.

The camera frame is mirrored once before detection, so increasing image x is the
operator's right. These tests pin that convention down, because it silently
inverted when the second flip was removed during the refactor.
"""

from gestures.static import GestureDetector


class Landmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def detector():
    return GestureDetector()


def test_pointing_right_is_named_right():
    """In a mirrored frame the operator's right is increasing image x."""
    mcp = Landmark(0.50, 0.50)
    tip = Landmark(0.75, 0.50)
    assert detector().finger_direction(mcp, tip) == "point_right"


def test_pointing_left_is_named_left():
    mcp = Landmark(0.50, 0.50)
    tip = Landmark(0.25, 0.50)
    assert detector().finger_direction(mcp, tip) == "point_left"


def test_pointing_up_and_down_are_unaffected():
    mcp = Landmark(0.50, 0.50)
    assert detector().finger_direction(mcp, Landmark(0.50, 0.20)) == "point_up"
    assert detector().finger_direction(mcp, Landmark(0.50, 0.80)) == "point_down"


def _hand_with_thumb(tip_x, tip_y):
    """21 landmarks; only the wrist (0) and thumb tip (4) matter here."""
    hand = [Landmark(0.5, 0.5) for _ in range(21)]
    hand[0] = Landmark(0.50, 0.50)
    hand[4] = Landmark(tip_x, tip_y)
    return hand


def test_thumb_right_is_named_right():
    assert detector().thumb_direction(_hand_with_thumb(0.75, 0.50)) == "thumb_right"


def test_thumb_left_is_named_left():
    assert detector().thumb_direction(_hand_with_thumb(0.25, 0.50)) == "thumb_left"


def test_thumbs_up_and_down_are_unaffected():
    assert detector().thumb_direction(_hand_with_thumb(0.50, 0.20)) == "thumbs_up"
    assert detector().thumb_direction(_hand_with_thumb(0.50, 0.80)) == "thumbs_down"


def test_horizontal_needs_to_clearly_beat_vertical():
    """A mostly-vertical gesture should not be reported as a sideways one."""
    mcp = Landmark(0.50, 0.50)
    barely_sideways = Landmark(0.56, 0.20)
    assert detector().finger_direction(mcp, barely_sideways) == "point_up"
