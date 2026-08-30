"""Operator panel.

Everything drawn here comes from the object registry, including colour. There is
no second colour table to keep in step with the simulation.
"""

import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

PANEL_WIDTH = 640
PANEL_HEIGHT = 305

ACTION_ROW_Y = 90
TUBE_ROW_Y = 140
SPOT_ROW_Y = 205
STATUS_Y = 272
HINT_Y = 296

INSTRUCTIONS = {
    "RUNNING": ("Robot ready", (0, 255, 0)),
    "PAUSED": ("PAUSED - press P or open palm to resume", (0, 200, 255)),
    "EMERGENCY_STOPPED": ("EMERGENCY STOP - press E to re-enable", (0, 0, 255)),
    "SELECT_SOURCE": ("Point L/R to select a tube, PINCH to choose", (200, 200, 200)),
    "CONFIRM_SOURCE": ("THUMBS UP to confirm | THUMB LEFT to cancel", (0, 255, 255)),
    "SELECT_ACTION": ("Point L/R to pick an action, PINCH to choose", (200, 200, 200)),
    "SELECT_DEST": ("Point L/R to select a target, PINCH to choose", (200, 200, 200)),
    "CONFIRM_DEST": ("THUMBS UP to confirm | THUMB LEFT to cancel", (0, 255, 255)),
    "EXECUTING": ("Robot is moving - THUMBS DOWN or X to stop", (0, 200, 255)),
    "ABORTING": ("Setting the sample down after the stop...", (0, 200, 255)),
}


def _readable_text_color(bgr):
    """Pick black or white lettering so the label stays legible on any swatch."""
    blue, green, red = bgr
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _draw_row(panel, objects, y, selected_handle, chosen_handles, is_selecting,
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
            cv2.putText(panel, obj.consumed_caption or used_caption,
                        (x + 6, y + 27), font, 0.4, (80, 80, 80), 1)
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

        if obj.handle in chosen_handles:
            thickness = 3 if is_confirming else 2
            offset = 4 if is_confirming else 3
            cv2.rectangle(
                panel,
                (x - offset, y - offset),
                (x + cell + offset, y + 40 + offset),
                (0, 255, 255),
                thickness,
            )


def _draw_actions(panel, actions, action_index, pending_action, is_selecting):
    """Show what the robot is being asked to do, and which choice cannot be undone."""
    if not actions:
        return

    left = 110
    cell = 172
    gap = 16
    for position, action in enumerate(actions):
        x = left + position * (cell + gap)
        chosen = action["key"] == pending_action
        # Red is reserved for the choice that cannot be taken back.
        edge = (0, 0, 255) if not action["reversible"] else (150, 150, 150)
        fill = (35, 35, 55) if not action["reversible"] else (45, 45, 45)

        cv2.rectangle(panel, (x, ACTION_ROW_Y), (x + cell, ACTION_ROW_Y + 30), fill, -1)
        cv2.rectangle(panel, (x, ACTION_ROW_Y), (x + cell, ACTION_ROW_Y + 30), edge, 1)
        cv2.putText(panel, action["label"], (x + 8, ACTION_ROW_Y + 21), font, 0.5,
                    (255, 255, 255), 2)
        cv2.putText(panel, action["hint"], (x + 62, ACTION_ROW_Y + 21), font, 0.34,
                    (150, 150, 150), 1)

        if is_selecting and position == action_index:
            cv2.rectangle(panel, (x - 3, ACTION_ROW_Y - 3),
                          (x + cell + 3, ACTION_ROW_Y + 33), (180, 180, 180), 2)
        if chosen:
            cv2.rectangle(panel, (x - 4, ACTION_ROW_Y - 4),
                          (x + cell + 4, ACTION_ROW_Y + 34), (0, 255, 255), 2)


def draw_ui(state, gesture, registry, selected_handle=None, source_handle=None,
            dest_handle=None, undo_available=False, status_message="",
            hold_progress=0.0, actions=(), action_index=0, pending_action=None):
    """Render the operator panel from the live registry."""
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

    cv2.putText(panel, f"STATE: {state}", (10, 25), font, 0.65, (0, 255, 255), 2)
    cv2.putText(panel, f"GESTURE: {gesture}", (350, 25), font, 0.65, (0, 255, 0), 2)

    # A charging bar under the gesture name. Without it a hold that has not yet
    # completed is indistinguishable from the system ignoring you.
    if gesture not in ("unknown", "", None) and 0.0 < hold_progress < 1.0:
        bar_left, bar_top, bar_width = 350, 31, 200
        cv2.rectangle(panel, (bar_left, bar_top), (bar_left + bar_width, bar_top + 5),
                      (60, 60, 60), -1)
        cv2.rectangle(panel, (bar_left, bar_top),
                      (bar_left + int(bar_width * hold_progress), bar_top + 5),
                      (0, 220, 255), -1)

    source = registry.by_handle(source_handle) if source_handle is not None else None
    destination = registry.by_handle(dest_handle) if dest_handle is not None else None

    source_text = source.label if source is not None else "-"
    destination_text = destination.label if destination is not None else "-"
    undo_text = "available" if undo_available else "unavailable"

    cv2.putText(panel, f"SOURCE: {source_text}", (10, 50), font, 0.45, (220, 220, 220), 1)
    cv2.putText(panel, f"TARGET: {destination_text}", (210, 50), font, 0.45, (220, 220, 220), 1)
    cv2.putText(panel, f"UNDO: {undo_text}", (420, 50), font, 0.45,
                (0, 255, 0) if undo_available else (140, 140, 140), 1)

    instruction, instruction_color = INSTRUCTIONS.get(state, (state, (200, 200, 200)))
    # The operator has to learn this before committing, not after.
    pouring = pending_action == "pour"
    if pouring and state == "CONFIRM_DEST":
        instruction = "POUR CANNOT BE UNDONE - hold THUMBS UP to commit"
        instruction_color = (0, 0, 255)
    elif pouring and state == "SELECT_DEST":
        instruction = "Choose the tube to pour INTO - this cannot be undone"
        instruction_color = (0, 165, 255)
    cv2.putText(panel, instruction, (10, 75), font, 0.45, instruction_color, 1)

    cv2.putText(panel, "ACTION:", (10, ACTION_ROW_Y + 21), font, 0.5, (180, 180, 180), 1)
    _draw_actions(panel, actions, action_index, pending_action,
                  state == "SELECT_ACTION")

    # For a pour the target is another tube, so the tube row is where both the
    # source and the target are shown.
    selecting_tubes = state == "SELECT_SOURCE" or (state == "SELECT_DEST" and pouring)
    confirming_tubes = state == "CONFIRM_SOURCE" or (state == "CONFIRM_DEST" and pouring)
    chosen_tubes = {source_handle}
    if pouring:
        chosen_tubes.add(dest_handle)

    cv2.putText(panel, "TUBES:", (10, TUBE_ROW_Y + 20), font, 0.5, (180, 180, 180), 1)
    _draw_row(
        panel,
        registry.sources,
        TUBE_ROW_Y,
        selected_handle,
        chosen_tubes,
        selecting_tubes,
        confirming_tubes,
        "done",
    )

    cv2.putText(panel, "SPOTS:", (10, SPOT_ROW_Y + 20), font, 0.5, (180, 180, 180), 1)
    _draw_row(
        panel,
        registry.destinations,
        SPOT_ROW_Y,
        selected_handle,
        set() if pouring else {dest_handle},
        state == "SELECT_DEST" and not pouring,
        state == "CONFIRM_DEST" and not pouring,
        "used",
    )

    # Sits on its own line below the spot row so it never overlaps a swatch.
    if status_message:
        upper = status_message.upper()
        color = (0, 0, 255) if ("FAIL" in upper or "STOPPED" in upper) else (0, 255, 255)
        text_size = cv2.getTextSize(status_message, font, 0.55, 2)[0]
        cv2.putText(panel, status_message,
                    (PANEL_WIDTH - text_size[0] - 12, STATUS_Y), font, 0.55, color, 2)

    cv2.putText(panel, "Pinch=Select | ThumbsUp=Confirm | ThumbLeft=Cancel | U=Undo | Q=Quit",
                (15, HINT_Y), font, 0.36, (120, 120, 120), 1)

    return panel
