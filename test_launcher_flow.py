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
