from object_registry import ObjectRegistry, rgba_to_bgr
from perception import StaticPerception

import config


def build_registry(source_count=3, destination_count=3):
    observations = []
    for i in range(source_count):
        observations.append({
            "handle": 100 + i,
            "label": f"Tube {i}",
            "color_name": "RED",
            "color_rgba": [1.0, 0.0, 0.0, 1.0],
            "position": [0.75, -0.4 + 0.1 * i, 0.65],
            "kind": "source",
        })
    for i in range(destination_count):
        observations.append({
            "handle": 200 + i,
            "label": f"Spot {i}",
            "color_name": "CYAN",
            "color_rgba": [0.0, 0.8, 0.8, 0.9],
            "position": [0.95, -0.4 + 0.1 * i, 0.65],
            "kind": "destination",
        })
    return ObjectRegistry(StaticPerception(observations))


def test_registry_splits_sources_and_destinations():
    registry = build_registry(3, 2)
    assert registry.count("source") == 3
    assert registry.count("destination") == 2
    assert len(registry) == 5


def test_navigation_wraps_and_skips_consumed():
    registry = build_registry(3, 0)
    registry.consume(101)

    assert registry.next_available("source", 100, 1) == 102
    assert registry.next_available("source", 102, 1) == 100
    assert registry.next_available("source", 100, -1) == 102


def test_navigation_returns_none_when_everything_is_used():
    """The old index loop spun forever here and froze the whole application."""
    registry = build_registry(3, 0)
    for handle in (100, 101, 102):
        registry.consume(handle)

    assert registry.next_available("source", 100, 1) is None
    assert registry.next_available("source", 100, -1) is None
    assert registry.first_available("source") is None


def test_navigation_terminates_with_a_single_object():
    registry = build_registry(1, 0)
    assert registry.next_available("source", 100, 1) == 100
    registry.consume(100)
    assert registry.next_available("source", 100, 1) is None


def test_release_and_reset_restore_availability():
    registry = build_registry(2, 0)
    registry.consume(100)
    assert registry.first_available("source") == 101

    registry.release(100)
    assert registry.first_available("source") == 100

    registry.consume(100)
    registry.consume(101)
    registry.reset()
    assert len(registry.available("source")) == 2


def test_refresh_updates_position_but_keeps_consumed_state():
    observations = [{
        "handle": 1, "label": "Tube", "color_name": "RED",
        "color_rgba": [1, 0, 0, 1], "position": [0.0, 0.0, 0.0], "kind": "source",
    }]
    perception = StaticPerception(observations)
    registry = ObjectRegistry(perception)
    registry.consume(1)

    perception._observations[0]["position"] = [9.0, 9.0, 9.0]
    registry.refresh()

    assert registry.by_handle(1).position == [9.0, 9.0, 9.0]
    assert registry.by_handle(1).consumed is True


def test_panel_colour_is_derived_from_simulation_colour():
    """The panel and the simulation must read from one colour, not two tables."""
    registry = build_registry(1, 1)
    tube = registry.sources[0]
    spot = registry.destinations[0]

    assert tube.color_bgr == rgba_to_bgr(tube.color_rgba)
    assert spot.color_bgr == rgba_to_bgr(spot.color_rgba)
    assert spot.color_bgr == (204, 204, 0)  # cyan, not the yellow the old table used


def test_config_colour_names_line_up_with_colour_values():
    assert len(config.CUBE_COLOR_NAMES) == len(config.CUBE_COLORS_RGBA)
    assert len(config.DESTINATION_SPOT_COLOR_NAMES) == len(config.DESTINATION_SPOT_COLORS_RGBA)
    assert len(config.CUBE_COLORS_RGBA) >= config.OBJECT_COUNT
