"""Rendering smoke tests for the operator panel.

These do not check that the panel looks good - they check that it draws at all.
Nothing else in the suite imports ui_module, so without this a change to the
drawing code could blank the panel and every other test would still pass.
"""

import numpy as np
import pytest

import ui_module
import ui_text
from ui_module import PANEL_HEIGHT, PANEL_WIDTH, draw_ui


class FakeObject:
    def __init__(self, handle, name, bgr, empty=False):
        self.handle = handle
        self.label = f"{name} tube"
        self.color_name = name
        self.color_bgr = bgr
        self.identity_bgr = bgr
        self.empty = empty

    @property
    def description(self):
        return f"{self.label} (empty)" if self.empty else self.label


class FakeRegistry:
    def __init__(self, tubes=5, spots=5, occupied=()):
        colors = [("Red", (60, 60, 220)), ("Green", (80, 200, 90)),
                  ("Blue", (220, 120, 60)), ("Yellow", (70, 210, 230)),
                  ("Purple", (200, 80, 190))]
        self.sources = [FakeObject(i, colors[i % 5][0], colors[i % 5][1],
                                   empty=(i == 2)) for i in range(tubes)]
        self.destinations = [FakeObject(100 + i, f"Spot{i}", (110, 110, 110))
                             for i in range(spots)]
        self._occupied = dict.fromkeys(occupied, 0)

    def by_handle(self, handle):
        for obj in self.sources + self.destinations:
            if obj.handle == handle:
                return obj
        return None

    def occupancy(self):
        return self._occupied


ACTIONS = [
    {"key": "move", "label": "MOVE", "hint": "pick & place", "reversible": True},
    {"key": "pour", "label": "POUR", "hint": "cannot undo", "reversible": False},
]

STATES = list(ui_module.INSTRUCTIONS) + ["SOME_UNKNOWN_STATE"]


def render(**overrides):
    kwargs = dict(
        state="SELECT_SOURCE",
        gesture="pinch",
        registry=FakeRegistry(occupied=(100, 101)),
        selected_handle=1,
        source_handle=0,
        dest_handle=103,
        undo_available=True,
        status_message="",
        hold_progress=0.0,
        actions=ACTIONS,
        action_index=0,
        pending_action=None,
        blocked_actions=None,
    )
    kwargs.update(overrides)
    return draw_ui(**kwargs)


def ink(panel):
    """Pixels that differ from the background - i.e. something was drawn."""
    return int(np.any(panel != np.array(ui_module.BG, dtype=np.uint8), axis=2).sum())


@pytest.mark.parametrize("state", STATES)
def test_every_state_renders_something(state):
    panel = render(state=state)
    assert panel.shape == (PANEL_HEIGHT, PANEL_WIDTH, 3)
    assert panel.dtype == np.uint8
    # A blank or near-blank panel means the operator is flying blind.
    assert ink(panel) > 2000, f"{state} rendered almost nothing"


def test_pour_confirmation_is_visibly_different():
    """The irreversible warning has to actually change what is on screen."""
    plain = render(state="CONFIRM_DEST", pending_action="move")
    pour = render(state="CONFIRM_DEST", pending_action="pour")
    assert not np.array_equal(plain, pour)


def test_selection_and_hold_bar_change_the_panel():
    base = render(state="SELECT_SOURCE", selected_handle=1, hold_progress=0.0)
    moved = render(state="SELECT_SOURCE", selected_handle=3, hold_progress=0.0)
    holding = render(state="SELECT_SOURCE", selected_handle=1, hold_progress=0.5)
    assert not np.array_equal(base, moved)
    assert not np.array_equal(base, holding)


def test_empty_registry_does_not_crash():
    panel = render(registry=FakeRegistry(tubes=0, spots=0), source_handle=None,
                   dest_handle=None, selected_handle=None, actions=())
    assert panel.shape == (PANEL_HEIGHT, PANEL_WIDTH, 3)


def test_long_status_message_is_clipped_not_fatal():
    panel = render(status_message="X" * 400)
    assert panel.shape == (PANEL_HEIGHT, PANEL_WIDTH, 3)


def test_blocked_action_renders_its_reason():
    without = render(state="SELECT_ACTION", blocked_actions=None)
    with_block = render(state="SELECT_ACTION",
                        blocked_actions={"pour": "source is empty"})
    assert not np.array_equal(without, with_block)


def test_many_tubes_still_fit_the_panel():
    panel = render(registry=FakeRegistry(tubes=12, spots=12))
    assert panel.shape == (PANEL_HEIGHT, PANEL_WIDTH, 3)


def test_text_stays_inside_the_panel():
    """Nothing may be drawn hard against the edges, which means it overflowed."""
    panel = render(state="CONFIRM_DEST", pending_action="pour",
                   status_message="Pour complete")
    assert ink(panel[:, -2:]) == 0
    assert ink(panel[-1:, :]) == 0


def test_text_is_antialiased():
    """Real type produces intermediate greys; stroke fonts drawn flat do not."""
    panel = np.zeros((40, 300, 3), dtype=np.uint8)
    ui_text.text(panel, "Antialiasing", (5, 28), 15, (255, 255, 255))
    values = np.unique(panel[:, :, 0])
    assert len(values) > 8, "expected a range of intensities, got a hard mask"


def test_text_off_panel_is_survivable():
    panel = np.zeros((40, 100, 3), dtype=np.uint8)
    for org in ((-500, 20), (500, 20), (20, -500), (20, 500)):
        ui_text.text(panel, "offscreen", org, 12, (255, 255, 255))
    assert panel.shape == (40, 100, 3)


def test_sprite_cache_is_reused_and_bounded():
    ui_text._sprites.clear()
    for _ in range(5):
        ui_text.text_width("repeated string", 12)
        ui_text.text(np.zeros((30, 200, 3), np.uint8), "repeated string", (5, 20), 12)
    assert len(ui_text._sprites) == 1

    for i in range(ui_text._MAX_ENTRIES + 10):
        ui_text.text(np.zeros((30, 200, 3), np.uint8), f"unique {i}", (5, 20), 12)
    assert len(ui_text._sprites) <= ui_text._MAX_ENTRIES
