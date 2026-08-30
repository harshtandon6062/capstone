"""Operator panel.

Everything drawn here comes from the object registry, including colour. There is
no second colour table to keep in step with the simulation.

Chrome colours are muted on purpose: the only saturated colours on the panel
should be the ones that mean something - a tube's identity, its contents, and
the warning attached to an action that cannot be undone.
"""

import cv2
import numpy as np

from ui_text import text, text_width

PANEL_WIDTH = 640
PANEL_HEIGHT = 305

ACTION_ROW_Y = 90
TUBE_ROW_Y = 140
SPOT_ROW_Y = 205
STATUS_Y = 272
HINT_Y = 296

# Type sizes in pixels. Four steps, so hierarchy comes from size and weight
# rather than from colour, which is reserved for meaning.
SIZE_TITLE = 15
SIZE_STATUS = 13
SIZE_BODY = 12
SIZE_LABEL = 11
SIZE_SMALL = 9

# Palette, BGR. A near-black neutral ground rather than pure #000000, which
# makes every foreground colour look harsher than it is.
BG = (26, 22, 20)
SURFACE = (42, 37, 33)
SURFACE_SUNK = (33, 29, 26)
SURFACE_DEAD = (34, 31, 29)
BORDER = (72, 65, 58)
BORDER_DEAD = (58, 53, 48)

TEXT_BRIGHT = (232, 230, 226)
TEXT_BODY = (196, 190, 182)
TEXT_MUTED = (160, 152, 144)
TEXT_DIM = (136, 129, 121)

ACCENT = (196, 178, 90)
OK = (120, 200, 130)
WARN = (90, 170, 235)
DANGER = (95, 90, 225)
SELECTION = (176, 170, 162)

INSTRUCTIONS = {
    "RUNNING": ("Robot ready", OK),
    "PAUSED": ("PAUSED - press P or open palm to resume", WARN),
    "EMERGENCY_STOPPED": ("EMERGENCY STOP - press E to re-enable", DANGER),
    "SELECT_SOURCE": ("Point L/R to select a tube, PINCH to choose", TEXT_MUTED),
    "CONFIRM_SOURCE": ("THUMBS UP to confirm | THUMB LEFT to cancel", ACCENT),
    "SELECT_ACTION": ("Point L/R to pick an action, PINCH to choose", TEXT_MUTED),
    "SELECT_DEST": ("Point L/R to select a target, PINCH to choose", TEXT_MUTED),
    "CONFIRM_DEST": ("THUMBS UP to confirm | THUMB LEFT to cancel", ACCENT),
    "EXECUTING": ("Robot is moving - THUMBS DOWN or X to stop", WARN),
    "ABORTING": ("Setting the sample down after the stop...", WARN),
}


def _readable_text_color(bgr):
    """Pick black or white lettering so the label stays legible on any swatch."""
    blue, green, red = bgr
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (20, 18, 16) if luminance > 140 else TEXT_BRIGHT


def _box(panel, top_left, bottom_right, fill=None, edge=None, thickness=1):
    """Filled and/or outlined rectangle.

    Deliberately not LINE_AA: every rectangle here is axis-aligned on integer
    coordinates, so there is no partial pixel coverage to smooth and the flag
    only buys a 2x slower fill. The panel's smooth edges come from the type.
    """
    if fill is not None:
        cv2.rectangle(panel, top_left, bottom_right, fill, -1)
    if edge is not None:
        cv2.rectangle(panel, top_left, bottom_right, edge, thickness)


def _draw_row(panel, objects, y, selected_handle, chosen_handles, is_selecting,
              is_confirming, unavailable=(), unavailable_caption="used"):
    """Draw one row of swatches, spaced to fit however many objects there are.

    `unavailable` is passed in rather than read off the object, because whether
    something can be chosen depends on what is being chosen for. A spot is
    unavailable when a tube is standing on it; a tube is never unavailable as a
    source, however many times it has already been handled.
    """
    if not objects:
        return

    count = len(objects)
    left = 110
    span = PANEL_WIDTH - left - 15
    cell = min(80, max(30, span // count - 8))
    gap = (span - cell * count) // max(1, count - 1) if count > 1 else 0

    for position, obj in enumerate(objects):
        x = left + position * (cell + gap)

        if obj.handle in unavailable:
            _box(panel, (x, y), (x + cell, y + 40), SURFACE_DEAD, BORDER_DEAD)
            caption_x = x + max(3, (cell - text_width(unavailable_caption, SIZE_SMALL)) // 2)
            text(panel, unavailable_caption, (caption_x, y + 25), SIZE_SMALL, TEXT_MUTED)
            continue

        # The frame is the cap: it names the tube and never changes. The middle
        # is what the tube currently holds, and goes dark when there is nothing
        # in it. One square therefore answers both "which tube" and "what is in
        # it", which used to be a single colour trying to say both.
        _box(panel, (x, y), (x + cell, y + 40), obj.identity_bgr, BORDER)

        inset = 7
        contents = SURFACE_SUNK if obj.empty else obj.color_bgr
        _box(panel, (x + inset, y + inset),
             (x + cell - inset, y + 40 - inset), contents)

        caption = obj.color_name[:3]
        caption_x = x + max(2, (cell - text_width(caption, SIZE_LABEL, True)) // 2)
        text(panel, caption, (caption_x, y + 25), SIZE_LABEL,
             TEXT_BODY if obj.empty else _readable_text_color(contents), bold=True)

        if is_selecting and obj.handle == selected_handle:
            _box(panel, (x - 3, y - 3), (x + cell + 3, y + 43), None, SELECTION, 2)

        if obj.handle in chosen_handles:
            thickness = 3 if is_confirming else 2
            offset = 4 if is_confirming else 3
            _box(panel, (x - offset, y - offset),
                 (x + cell + offset, y + 40 + offset), None, ACCENT, thickness)


def _draw_actions(panel, actions, action_index, pending_action, is_selecting,
                  blocked=None):
    """Show what the robot is being asked to do, and which choice cannot be undone.

    An action that cannot work right now is dimmed and says why. Offering it and
    then doing nothing is the failure mode this panel exists to avoid.
    """
    blocked = blocked or {}
    if not actions:
        return

    left = 110
    cell = 172
    gap = 16
    for position, action in enumerate(actions):
        x = left + position * (cell + gap)
        chosen = action["key"] == pending_action
        reason = blocked.get(action["key"])
        # Red is reserved for the choice that cannot be taken back.
        if reason:
            edge, fill = BORDER_DEAD, SURFACE_DEAD
            label_color, hint_color = TEXT_DIM, TEXT_DIM
        elif not action["reversible"]:
            edge, fill = DANGER, (38, 32, 44)
            label_color, hint_color = TEXT_BRIGHT, TEXT_MUTED
        else:
            edge, fill = BORDER, SURFACE
            label_color, hint_color = TEXT_BRIGHT, TEXT_MUTED

        _box(panel, (x, ACTION_ROW_Y), (x + cell, ACTION_ROW_Y + 30), fill, edge)
        text(panel, action["label"], (x + 9, ACTION_ROW_Y + 20), SIZE_BODY,
             label_color, bold=True)
        text(panel, reason or action["hint"], (x + 62, ACTION_ROW_Y + 20),
             SIZE_SMALL, hint_color)
        if reason:
            continue

        if is_selecting and position == action_index:
            _box(panel, (x - 3, ACTION_ROW_Y - 3),
                 (x + cell + 3, ACTION_ROW_Y + 33), None, SELECTION, 2)
        if chosen:
            _box(panel, (x - 4, ACTION_ROW_Y - 4),
                 (x + cell + 4, ACTION_ROW_Y + 34), None, ACCENT, 2)


def draw_ui(state, gesture, registry, selected_handle=None, source_handle=None,
            dest_handle=None, undo_available=False, status_message="",
            hold_progress=0.0, actions=(), action_index=0, pending_action=None,
            blocked_actions=None):
    """Render the operator panel from the live registry."""
    panel = np.empty((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = BG

    text(panel, f"STATE: {state}", (10, 25), SIZE_TITLE, TEXT_BRIGHT, bold=True)
    text(panel, f"GESTURE: {gesture}", (350, 25), SIZE_TITLE, ACCENT, bold=True)

    # A charging bar under the gesture name. Without it a hold that has not yet
    # completed is indistinguishable from the system ignoring you.
    if gesture not in ("unknown", "", None) and 0.0 < hold_progress < 1.0:
        bar_left, bar_top, bar_width = 350, 31, 200
        _box(panel, (bar_left, bar_top), (bar_left + bar_width, bar_top + 5), SURFACE)
        _box(panel, (bar_left, bar_top),
             (bar_left + int(bar_width * hold_progress), bar_top + 5), ACCENT)

    source = registry.by_handle(source_handle) if source_handle is not None else None
    destination = registry.by_handle(dest_handle) if dest_handle is not None else None

    source_text = source.description if source is not None else "-"
    destination_text = destination.description if destination is not None else "-"
    undo_text = "available" if undo_available else "unavailable"

    # Names carry their contents now ("Red tube (mixed)"), so these lines are
    # longer than they used to be and need the room.
    text(panel, f"SOURCE: {source_text}", (10, 50), SIZE_LABEL, TEXT_BODY)
    text(panel, f"TARGET: {destination_text}", (240, 50), SIZE_LABEL, TEXT_BODY)
    text(panel, f"UNDO: {undo_text}", (470, 50), SIZE_LABEL,
         OK if undo_available else TEXT_DIM)

    instruction, instruction_color = INSTRUCTIONS.get(state, (state, TEXT_MUTED))
    # The operator has to learn this before committing, not after.
    pouring = pending_action == "pour"
    if pouring and state == "CONFIRM_DEST":
        instruction = "POUR CANNOT BE UNDONE - hold THUMBS UP to commit"
        instruction_color = DANGER
    elif pouring and state == "SELECT_DEST":
        instruction = "Choose the tube to pour INTO - this cannot be undone"
        instruction_color = WARN
    text(panel, instruction, (10, 75), SIZE_BODY, instruction_color)

    text(panel, "ACTION:", (10, ACTION_ROW_Y + 20), SIZE_LABEL, TEXT_MUTED, bold=True)
    _draw_actions(panel, actions, action_index, pending_action,
                  state == "SELECT_ACTION", blocked_actions)

    # For a pour the target is another tube, so the tube row is where both the
    # source and the target are shown.
    selecting_tubes = state == "SELECT_SOURCE" or (state == "SELECT_DEST" and pouring)
    confirming_tubes = state == "CONFIRM_SOURCE" or (state == "CONFIRM_DEST" and pouring)
    chosen_tubes = {source_handle}
    if pouring:
        chosen_tubes.add(dest_handle)

    text(panel, "TUBES:", (10, TUBE_ROW_Y + 25), SIZE_LABEL, TEXT_MUTED, bold=True)
    _draw_row(
        panel,
        registry.sources,
        TUBE_ROW_Y,
        selected_handle,
        chosen_tubes,
        selecting_tubes,
        confirming_tubes,
    )

    # A spot is taken when a tube is standing on it, which is read from where the
    # tubes are rather than from a flag someone had to remember to clear.
    occupied = set(registry.occupancy())
    text(panel, "SPOTS:", (10, SPOT_ROW_Y + 25), SIZE_LABEL, TEXT_MUTED, bold=True)
    _draw_row(
        panel,
        registry.destinations,
        SPOT_ROW_Y,
        selected_handle,
        set() if pouring else {dest_handle},
        state == "SELECT_DEST" and not pouring,
        state == "CONFIRM_DEST" and not pouring,
        occupied,
        "taken",
    )

    # Sits on its own line below the spot row so it never overlaps a swatch.
    if status_message:
        upper = status_message.upper()
        color = DANGER if ("FAIL" in upper or "STOPPED" in upper) else ACCENT
        width = text_width(status_message, SIZE_STATUS, True)
        text(panel, status_message, (PANEL_WIDTH - width - 12, STATUS_Y),
             SIZE_STATUS, color, bold=True)

    text(panel, "Pinch=Select | ThumbsUp=Confirm | ThumbLeft=Cancel | U=Undo | Q=Quit",
         (15, HINT_Y), SIZE_SMALL, TEXT_MUTED)

    return panel
