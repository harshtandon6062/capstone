import numpy as np

import launcher


def test_launcher_has_task_menu():
    assert isinstance(launcher.TASKS, list)
    assert any(task["gesture"] == "grasp" for task in launcher.TASKS)
    assert launcher.HOLD_DURATION > 0


def test_draw_welcome_ui_renders_panel():
    panel = launcher.draw_welcome_ui("grasp", 0.9, 0.5)
    assert panel.shape == (280, 640, 3)
    assert panel.dtype == np.uint8


def test_every_ready_task_actually_launches_something():
    """A task that reads READY and then does nothing is worse than one that says
    Coming Soon - the operator sees 100% recognition and no result."""
    for task in launcher.TASKS:
        if task["status"] == "READY":
            assert task["action"], f"{task['name']} is READY but launches nothing"
        else:
            assert not task["action"], f"{task['name']} is not READY but has an action"


def test_task_actions_are_actions_the_module_understands():
    from config import ACTIONS

    known = {action["key"] for action in ACTIONS}
    for task in launcher.TASKS:
        if task["action"]:
            assert task["action"] in known, f"{task['action']} is not a known action"


def test_pour_is_reachable_from_the_launcher():
    pour = [task for task in launcher.TASKS if task["action"] == "pour"]
    assert pour, "there is no way to start a pour from the launcher"
    assert pour[0]["status"] == "READY"


def test_number_keys_cover_every_task():
    """1, 2 and 3 index the task list directly, so the list must be that long."""
    assert len(launcher.TASKS) == 3


def test_pick_and_place_accepts_the_action_the_launcher_passes():
    import inspect

    import main

    signature = inspect.signature(main.run_pick_and_place)
    assert "initial_action" in signature.parameters
    assert signature.parameters["initial_action"].default == "move"
