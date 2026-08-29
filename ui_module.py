"""Operator panel.

Everything drawn here comes from the object registry, including colour. There is
no second colour table to keep in step with the simulation.
"""

import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

PANEL_WIDTH = 640
PANEL_HEIGHT = 235

INSTRUCTIONS = {
    "RUNNING": ("Robot ready", (0, 255, 0)),
    "PAUSED": ("PAUSED - press P or open palm to resume", (0, 200, 255)),
    "EMERGENCY_STOPPED": ("EMERGENCY STOP - press E to re-enable", (0, 0, 255)),
    "SELECT_SOURCE": ("Point L/R to select a tube, PINCH to choose", (200, 200, 200)),
    "CONFIRM_SOURCE": ("THUMBS UP to confirm | THUMB LEFT to cancel", (0, 255, 255)),
    "SELECT_DEST": ("Point L/R to select a spot, PINCH to choose", (200, 200, 200)),
    "CONFIRM_DEST": ("THUMBS UP to confirm | THUMB LEFT to cancel", (0, 255, 255)),
    "EXECUTING": ("Robot is moving...", (0, 200, 255)),
}


def _readable_text_color(bgr):
    """Pick black or white lettering so the label stays legible on any swatch."""
    blue, green, red = bgr
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _draw_row(panel, objects, y, selected_handle, chosen_handle, is_selecting,
              is_confirming, used_caption):
    """Draw one row of swatches, spaced to fit however many objects there are."""
    if not objects:
        return

    count = len(objects)
    left = 110
    span = PANEL_WIDTH - left - 15
    cell = min(80, max(30, span // count - 8))
    gap = (span - cell * count) // max(1, count - 1) if count > 1 else 0

    for position, obj in enumerate(objects):
        x = left + position * (cell + gap)

        if obj.consumed:
            cv2.rectangle(panel, (x, y), (x + cell, y + 40), (60, 60, 60), 1)
            cv2.putText(panel, used_caption, (x + 6, y + 27), font, 0.4, (80, 80, 80), 1)
            continue

        color = obj.color_bgr
        cv2.rectangle(panel, (x, y), (x + cell, y + 40), color, -1)
        cv2.rectangle(panel, (x, y), (x + cell, y + 40), (255, 255, 255), 1)

        caption = obj.color_name[:3]
        text_size = cv2.getTextSize(caption, font, 0.5, 2)[0]
        cv2.putText(
            panel,
            caption,
            (x + max(2, (cell - text_size[0]) // 2), y + 27),
            font,
            0.5,
            _readable_text_color(color),
            2,
        )

        if is_selecting and obj.handle == selected_handle:
            cv2.rectangle(panel, (x - 3, y - 3), (x + cell + 3, y + 43), (180, 180, 180), 2)

        if obj.handle == chosen_handle:
            thickness = 3 if is_confirming else 2
            offset = 4 if is_confirming else 3
            cv2.rectangle(
                panel,
                (x - offset, y - offset),
                (x + cell + offset, y + 40 + offset),
                (0, 255, 255),
                thickness,
            )


def draw_ui(state, gesture, registry, selected_handle=None, source_handle=None,
            dest_handle=None, undo_available=False, status_message=""):
    """Render the operator panel from the live registry."""
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

    cv2.putText(panel, f"STATE: {state}", (10, 25), font, 0.65, (0, 255, 255), 2)
    cv2.putText(panel, f"GESTURE: {gesture}", (350, 25), font, 0.65, (0, 255, 0), 2)

    source = registry.by_handle(source_handle) if source_handle is not None else None
    destination = registry.by_handle(dest_handle) if dest_handle is not None else None

    source_text = source.label if source is not None else "-"
    destination_text = destination.label if destination is not None else "-"
    undo_text = "available" if undo_available else "unavailable"

    cv2.putText(panel, f"SOURCE: {source_text}", (10, 50), font, 0.45, (220, 220, 220), 1)
    cv2.putText(panel, f"DEST: {destination_text}", (210, 50), font, 0.45, (220, 220, 220), 1)
    cv2.putText(panel, f"UNDO: {undo_text}", (420, 50), font, 0.45,
                (0, 255, 0) if undo_available else (140, 140, 140), 1)

    instruction, instruction_color = INSTRUCTIONS.get(state, (state, (200, 200, 200)))
    cv2.putText(panel, instruction, (10, 75), font, 0.45, instruction_color, 1)

    cv2.putText(panel, "TUBES:", (10, 110), font, 0.5, (180, 180, 180), 1)
    _draw_row(
        panel,
        registry.sources,
        90,
        selected_handle,
        source_handle,
        state == "SELECT_SOURCE",
        state == "CONFIRM_SOURCE",
        "done",
    )

    cv2.putText(panel, "SPOTS:", (10, 175), font, 0.5, (180, 180, 180), 1)
    _draw_row(
        panel,
        registry.destinations,
        155,
        selected_handle,
        dest_handle,
        state == "SELECT_DEST",
        state == "CONFIRM_DEST",
        "used",
    )

    if status_message:
        color = (0, 255, 255) if "FAIL" not in status_message.upper() else (0, 0, 255)
        text_size = cv2.getTextSize(status_message, font, 0.55, 2)[0]
        cv2.putText(panel, status_message,
                    (PANEL_WIDTH - text_size[0] - 12, 205), font, 0.55, color, 2)

    cv2.putText(panel, "Pinch=Select | ThumbsUp=Confirm | ThumbLeft=Cancel | U=Undo | Q=Quit",
                (15, 228), font, 0.36, (120, 120, 120), 1)

    return panel
