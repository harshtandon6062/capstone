"""The registry decides what the operator is allowed to pick, and when.

Two rules matter most here and both were bugs before:

  * a tube is never used up. Moving it, or emptying it, does not stop it being a
    thing you can pick up and move again.
  * a spot is occupied when a tube is standing on it, worked out from where the
    tubes actually are. Remembering it with a flag went stale every time the
    world changed by another route.
"""

import pytest

from workspace.registry import EMPTY_LIQUID_RGBA, ObjectRegistry, mix_colors, rgba_to_bgr
from workspace.perception import StaticPerception


def build_registry(source_count=3, destination_count=3, source_positions=None):
    observations = []
    for i in range(source_count):
        position = (source_positions[i] if source_positions
                    else [0.75, -0.4 + 0.12 * i, 0.65])
        observations.append({
            "handle": 100 + i,
            "label": f"Tube {i}",
            "color_name": ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA"][i % 5],
            "color_rgba": [[1.0, 0, 0, 1], [0, 1.0, 0, 1], [0, 0, 1.0, 1],
                           [1.0, 1.0, 0, 1], [1.0, 0, 1.0, 1]][i % 5],
            "position": list(position),
            "kind": "source",
        })
    for i in range(destination_count):
        observations.append({
            "handle": 200 + i,
            "label": f"Spot {i}",
            "color_name": "CYAN",
            "color_rgba": [0.0, 0.8, 0.8, 0.9],
            "position": [0.95, -0.4 + 0.12 * i, 0.65],
            "kind": "destination",
        })
    return ObjectRegistry(StaticPerception(observations))


def handles(objects):
    return [obj.handle for obj in objects]


def test_registry_splits_sources_and_destinations():
    registry = build_registry(3, 2)
    assert registry.count("source") == 3
    assert registry.count("destination") == 2
    assert len(registry) == 5


def test_panel_colour_comes_from_the_simulation_colour():
    assert rgba_to_bgr([1.0, 0.0, 0.0, 1.0]) == (0, 0, 255)
    assert rgba_to_bgr([0.0, 0.0, 1.0, 1.0]) == (255, 0, 0)


# ── a tube is never used up ─────────────────────────────────────────────────

def test_a_moved_tube_can_be_moved_again():
    """Regression: moving a tube used to take it out of the list for good."""
    registry = build_registry(3, 3)
    moved = registry.by_handle(100)
    moved.position = [0.95, -0.4, 0.65]          # now standing on spot 0
    registry.refresh()

    assert 100 in handles(registry.selectable("move_source"))
    assert registry.first_selectable("move_source") == 100


def test_an_emptied_tube_can_still_be_moved_but_not_poured():
    registry = build_registry(3, 3)
    registry.transfer_contents(100, 101)

    assert 100 in handles(registry.selectable("move_source")), "empty tubes still move"
    assert 100 not in handles(registry.selectable("pour_source")), "nothing to pour"
    assert not registry.can_pour(100)
    assert registry.can_pour(101), "the tube it was poured into has contents now"


def test_navigation_wraps_without_spinning():
    registry = build_registry(3, 0)
    assert registry.next_selectable("move_source", 100, 1) == 101
    assert registry.next_selectable("move_source", 102, 1) == 100
    assert registry.next_selectable("move_source", 100, -1) == 102


def test_navigation_returns_none_when_there_is_nothing_to_choose():
    registry = build_registry(1, 0)
    assert registry.next_selectable("pour_target", 100, 1, exclude=100) is None
    assert registry.first_selectable("pour_target", exclude=100) is None


def test_a_tube_is_never_offered_as_its_own_pour_target():
    registry = build_registry(3, 0)
    assert 100 not in handles(registry.selectable("pour_target", exclude=100))
    assert registry.next_selectable("pour_target", 101, 1, exclude=100) == 102


def test_unknown_purpose_is_rejected_rather_than_guessed():
    registry = build_registry(2, 2)
    with pytest.raises(ValueError):
        registry.selectable("teleport")


# ── occupancy is derived, not remembered ────────────────────────────────────

def test_a_spot_is_occupied_when_a_tube_is_standing_on_it():
    registry = build_registry(3, 3)
    assert registry.occupancy() == {}, "tubes start on the tube row, not on spots"

    registry.by_handle(100).position = [0.95, -0.4, 0.65]
    assert registry.occupancy() == {200: 100}
    assert registry.occupant_of(200) == 100
    assert registry.spot_under(100) == 200


def test_an_occupied_spot_cannot_be_chosen_as_a_destination():
    registry = build_registry(3, 3)
    registry.by_handle(100).position = [0.95, -0.4, 0.65]

    free = handles(registry.selectable("move_target"))
    assert 200 not in free
    assert free == [201, 202]


def test_moving_a_tube_off_a_spot_frees_it_with_no_bookkeeping():
    """This is the whole reason occupancy is derived rather than flagged."""
    registry = build_registry(3, 3)
    registry.by_handle(100).position = [0.95, -0.4, 0.65]
    assert 200 not in handles(registry.selectable("move_target"))

    registry.by_handle(100).position = [0.95, -0.16, 0.65]   # moved to spot 2
    free = handles(registry.selectable("move_target"))
    assert 200 in free, "the spot it left is free again"
    assert 202 not in free, "the spot it moved to is taken"


def test_a_tube_between_two_spots_only_claims_the_nearer_one():
    registry = build_registry(1, 3)
    registry.by_handle(100).position = [0.95, -0.35, 0.65]   # between spot 0 and 1
    occupancy = registry.occupancy()
    assert len(occupancy) == 1


def test_can_move_reports_whether_any_spot_is_left():
    registry = build_registry(2, 2)
    assert registry.can_move()
    registry.by_handle(100).position = [0.95, -0.4, 0.65]
    registry.by_handle(101).position = [0.95, -0.28, 0.65]
    assert not registry.can_move()


# ── contents ────────────────────────────────────────────────────────────────

def test_pouring_into_an_empty_tube_does_not_blend_with_the_empty_colour():
    """Regression: an emptied tube is white, and averaging tinted every pour."""
    registry = build_registry(3, 0)
    registry.transfer_contents(100, 101)        # 100 is now empty
    original_green = list(registry.by_handle(102).color_rgba)

    registry.transfer_contents(102, 100)        # green into the empty tube

    assert registry.by_handle(100).color_rgba == original_green
    assert registry.by_handle(100).color_rgba != EMPTY_LIQUID_RGBA
    assert not registry.by_handle(100).empty


def test_pouring_two_full_tubes_together_mixes_them():
    registry = build_registry(2, 0)
    red = list(registry.by_handle(100).color_rgba)
    green = list(registry.by_handle(101).color_rgba)

    registry.transfer_contents(100, 101)

    assert registry.by_handle(101).color_rgba == mix_colors(red, green)
    assert registry.by_handle(101).contents_name == "MIXED"


def test_an_empty_tube_has_nothing_to_pour():
    registry = build_registry(3, 0)
    registry.transfer_contents(100, 101)
    before = list(registry.by_handle(102).color_rgba)

    assert registry.transfer_contents(100, 102) is None
    assert registry.by_handle(102).color_rgba == before, "the target was changed anyway"


def test_reset_puts_the_contents_back():
    """Regression: a reset restored positions but left tubes permanently empty."""
    registry = build_registry(2, 0)
    red = list(registry.by_handle(100).color_rgba)

    registry.transfer_contents(100, 101)
    assert registry.by_handle(100).empty

    registry.reset()

    assert not registry.by_handle(100).empty
    assert registry.by_handle(100).color_rgba == red
    assert registry.by_handle(100).color_name == "RED"
    assert registry.by_handle(101).color_name == "GREEN"


def test_refresh_updates_position_without_disturbing_contents():
    registry = build_registry(2, 0)
    registry.transfer_contents(100, 101)
    registry.perception._observations[0]["position"] = [0.9, -0.1, 0.65]

    registry.refresh()

    assert registry.by_handle(100).position == [0.9, -0.1, 0.65]
    assert registry.by_handle(100).empty, "a position update wiped the contents"


# ── identity survives whatever happens to the contents ──────────────────────

def test_a_tube_keeps_its_name_after_being_emptied():
    """Regression: emptying a tube used to erase which tube it was, so two empty
    tubes were impossible to tell apart."""
    registry = build_registry(2, 0)
    registry.transfer_contents(100, 101)

    emptied = registry.by_handle(100)
    assert emptied.empty
    assert emptied.label == "Tube 0", "the tube was renamed"
    assert emptied.color_name == "RED", "the tube lost its marking"
    assert emptied.identity_rgba == [1.0, 0, 0, 1], "the marking colour changed"
    assert emptied.description == "Tube 0 (empty)"


def test_two_empty_tubes_are_still_distinguishable():
    registry = build_registry(3, 0)
    registry.transfer_contents(100, 102)
    registry.transfer_contents(101, 102)

    first, second = registry.by_handle(100), registry.by_handle(101)
    assert first.empty and second.empty
    assert first.identity_bgr != second.identity_bgr, "both empty tubes look alike"
    assert first.color_name != second.color_name


def test_a_mixed_tube_keeps_its_own_name():
    registry = build_registry(2, 0)
    registry.transfer_contents(100, 101)

    mixed = registry.by_handle(101)
    assert mixed.color_name == "GREEN", "the tube was renamed after being mixed"
    assert mixed.contents_name == "MIXED"
    assert mixed.description == "Tube 1 (mixed)"
    assert mixed.identity_rgba == [0, 1.0, 0, 1]


def test_reset_restores_contents_without_touching_identity():
    registry = build_registry(2, 0)
    registry.transfer_contents(100, 101)
    registry.reset()

    assert registry.by_handle(101).contents_name == "GREEN"
    assert registry.by_handle(101).identity_rgba == [0, 1.0, 0, 1]
    assert registry.by_handle(100).description == "Tube 0"
