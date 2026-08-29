import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

BLOCK_COLORS_BGR = [
    (0, 0, 255),     # red
    (0, 180, 0),     # green
    (255, 0, 0),     # blue
    (0, 200, 255),   # yellow
    (255, 0, 255),   # magenta
]

BLOCK_NAMES = ["RED", "GRN", "BLU", "YLW", "MAG"]
DESTINATION_SPOT_COLORS_BGR = [
    (0, 0, 255),
    (0, 180, 0),
    (0, 140, 255),
    (255, 0, 255),
    (0, 255, 255),
]


def draw_ui(state, gesture, selected_idx, source_idx, dest_idx, block_placed,
            dest_placed, undo_available=False):
    """
    Draw UI panel with two rows:
      Row 1: Source blocks (colored squares)
      Row 2: Destination spots (empty squares)

    block_placed: list of bools, True if block i has been placed already
    dest_placed: list of bools, True if spot i is already occupied
    """
    panel = np.zeros((235, 640, 3), dtype=np.uint8)

    # State and gesture
    cv2.putText(panel, f"STATE: {state}", (10, 25), font, 0.65, (0, 255, 255), 2)
    cv2.putText(panel, f"GESTURE: {gesture}", (350, 25), font, 0.65, (0, 255, 0), 2)
    source_text = "-" if source_idx is None else f"Block {source_idx + 1}"
    destination_text = "-" if dest_idx is None else f"Spot {dest_idx + 1}"
    undo_text = "available" if undo_available else "unavailable"
    cv2.putText(panel, f"SOURCE: {source_text}", (10, 50), font, 0.45, (220, 220, 220), 1)
    cv2.putText(panel, f"DEST: {destination_text}", (190, 50), font, 0.45, (220, 220, 220), 1)
    cv2.putText(panel, f"UNDO: {undo_text}", (390, 50), font, 0.45,
                (0, 255, 0) if undo_available else (140, 140, 140), 1)

    # Instructions
    if state == "RUNNING":
        cv2.putText(panel, "Robot ready",
                    (10, 75), font, 0.45, (0, 255, 0), 1)
    elif state == "PAUSED":
        cv2.putText(panel, "PAUSED - press P or open palm to resume",
                    (10, 75), font, 0.45, (0, 200, 255), 1)
    elif state == "EMERGENCY_STOPPED":
        cv2.putText(panel, "EMERGENCY STOP - press E to re-enable",
                    (10, 75), font, 0.45, (0, 0, 255), 1)
    elif state == "SELECT_SOURCE":
        cv2.putText(panel, "Point L/R to select block, PINCH to choose",
                    (10, 75), font, 0.45, (200, 200, 200), 1)
    elif state == "CONFIRM_SOURCE":
        cv2.putText(panel, "THUMBS UP to confirm | THUMB LEFT to cancel",
                    (10, 75), font, 0.45, (0, 255, 255), 1)
    elif state == "SELECT_DEST":
        cv2.putText(panel, "Point L/R to select spot, PINCH to choose",
                    (10, 75), font, 0.45, (200, 200, 200), 1)
    elif state == "CONFIRM_DEST":
        cv2.putText(panel, "THUMBS UP to confirm | THUMB LEFT to cancel",
                    (10, 75), font, 0.45, (0, 255, 255), 1)
    elif state == "EXECUTING":
        cv2.putText(panel, "Robot is moving...",
                    (10, 75), font, 0.45, (0, 200, 255), 1)

    # --- Row 1: Source Blocks ---
    cv2.putText(panel, "BLOCKS:", (10, 110), font, 0.5, (180, 180, 180), 1)
    for i in range(5):
        x = 110 + i * 105
        y = 90

        color = BLOCK_COLORS_BGR[i]
        thickness = -1  # filled

        if block_placed[i]:
            cv2.rectangle(panel, (x, y), (x + 80, y + 40), (60, 60, 60), 1)
            cv2.putText(panel, "done", (x + 20, y + 27), font, 0.4, (80, 80, 80), 1)
            continue

        cv2.rectangle(panel, (x, y), (x + 80, y + 40), color, thickness)
        cv2.putText(panel, BLOCK_NAMES[i], (x + 15, y + 27), font, 0.5, (255, 255, 255), 2)

        # Selection highlight (grey border visible over the white focus border)
        if state == "SELECT_SOURCE" and i == selected_idx:
            cv2.rectangle(panel, (x - 3, y - 3), (x + 83, y + 43), (180, 180, 180), 2)

        # Confirmed source marker (bright yellow thick border)
        if source_idx == i and state in ("CONFIRM_SOURCE",):
            cv2.rectangle(panel, (x - 4, y - 4), (x + 84, y + 44), (0, 255, 255), 3)
        elif source_idx == i:
            cv2.rectangle(panel, (x - 3, y - 3), (x + 83, y + 43), (0, 255, 255), 2)

    # --- Row 2: Destination Spots ---
    cv2.putText(panel, "SPOTS:", (10, 175), font, 0.5, (180, 180, 180), 1)
    for i in range(5):
        x = 110 + i * 105
        y = 155
        spot_color = DESTINATION_SPOT_COLORS_BGR[i]

        if dest_placed[i]:
            cv2.rectangle(panel, (x, y), (x + 80, y + 40), (60, 60, 60), 1)
            cv2.putText(panel, "used", (x + 20, y + 27), font, 0.4, (80, 80, 80), 1)
            continue

        cv2.rectangle(panel, (x, y), (x + 80, y + 40), spot_color, -1)
        cv2.rectangle(panel, (x, y), (x + 80, y + 40), (255, 255, 255), 2)
        cv2.putText(panel, f"Spot {i+1}", (x + 8, y + 27), font, 0.4, (255, 255, 255), 1)

        # Selection highlight (grey border visible over the white focus border)
        if state == "SELECT_DEST" and i == selected_idx:
            cv2.rectangle(panel, (x - 3, y - 3), (x + 83, y + 43), (180, 180, 180), 2)

        # Confirmed dest marker (bright yellow thick border)
        if dest_idx == i and state in ("CONFIRM_DEST",):
            cv2.rectangle(panel, (x - 4, y - 4), (x + 84, y + 44), (0, 255, 255), 3)
        elif dest_idx == i:
            cv2.rectangle(panel, (x - 3, y - 3), (x + 83, y + 43), (0, 255, 255), 2)

    # Navigation hints
    cv2.putText(panel, "Pinch=Select | ThumbsUp=Confirm | ThumbLeft=Cancel | U=Undo | Q=Quit",
                (15, 225), font, 0.36, (120, 120, 120), 1)

    return panel
