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


def draw_ui(state, gesture, selected_idx, source_idx, dest_idx, block_placed):
    """
    Draw UI panel with two rows:
      Row 1: Source blocks (colored squares)
      Row 2: Destination spots (empty squares)

    block_placed: list of bools, True if block i has been placed already
    """
    panel = np.zeros((220, 640, 3), dtype=np.uint8)

    # State and gesture
    cv2.putText(panel, f"STATE: {state}", (10, 25), font, 0.65, (0, 255, 255), 2)
    cv2.putText(panel, f"GESTURE: {gesture}", (350, 25), font, 0.65, (0, 255, 0), 2)

    # Instructions
    if state == "SELECT_SOURCE":
        cv2.putText(panel, "Point L/R to select block, PINCH to choose",
                    (10, 50), font, 0.45, (200, 200, 200), 1)
    elif state == "CONFIRM_SOURCE":
        cv2.putText(panel, "THUMBS UP to confirm | THUMB LEFT to cancel",
                    (10, 50), font, 0.45, (0, 255, 255), 1)
    elif state == "SELECT_DEST":
        cv2.putText(panel, "Point L/R to select spot, PINCH to choose",
                    (10, 50), font, 0.45, (200, 200, 200), 1)
    elif state == "CONFIRM_DEST":
        cv2.putText(panel, "THUMBS UP to confirm | THUMB LEFT to cancel",
                    (10, 50), font, 0.45, (0, 255, 255), 1)
    elif state == "EXECUTING":
        cv2.putText(panel, "Robot is moving...",
                    (10, 50), font, 0.45, (0, 200, 255), 1)

    # --- Row 1: Source Blocks ---
    cv2.putText(panel, "BLOCKS:", (10, 85), font, 0.5, (180, 180, 180), 1)
    for i in range(5):
        x = 110 + i * 105
        y = 65

        color = BLOCK_COLORS_BGR[i]
        thickness = -1  # filled

        if block_placed[i]:
            cv2.rectangle(panel, (x, y), (x + 80, y + 40), (60, 60, 60), 1)
            cv2.putText(panel, "done", (x + 20, y + 27), font, 0.4, (80, 80, 80), 1)
            continue

        cv2.rectangle(panel, (x, y), (x + 80, y + 40), color, thickness)
        cv2.putText(panel, BLOCK_NAMES[i], (x + 15, y + 27), font, 0.5, (255, 255, 255), 2)

        # Selection highlight (white border)
        if state == "SELECT_SOURCE" and i == selected_idx:
            cv2.rectangle(panel, (x - 3, y - 3), (x + 83, y + 43), (255, 255, 255), 2)

        # Confirmed source marker (bright yellow thick border)
        if source_idx == i and state in ("CONFIRM_SOURCE",):
            cv2.rectangle(panel, (x - 4, y - 4), (x + 84, y + 44), (0, 255, 255), 3)
        elif source_idx == i:
            cv2.rectangle(panel, (x - 3, y - 3), (x + 83, y + 43), (0, 255, 255), 2)

    # --- Row 2: Destination Spots ---
    cv2.putText(panel, "SPOTS:", (10, 150), font, 0.5, (180, 180, 180), 1)
    for i in range(5):
        x = 110 + i * 105
        y = 130

        cv2.rectangle(panel, (x, y), (x + 80, y + 40), (100, 100, 100), 2)
        cv2.putText(panel, f"Spot {i+1}", (x + 8, y + 27), font, 0.4, (150, 150, 150), 1)

        # Selection highlight
        if state == "SELECT_DEST" and i == selected_idx:
            cv2.rectangle(panel, (x - 3, y - 3), (x + 83, y + 43), (255, 255, 255), 2)

        # Confirmed dest marker (bright yellow thick border)
        if dest_idx == i and state in ("CONFIRM_DEST",):
            cv2.rectangle(panel, (x - 4, y - 4), (x + 84, y + 44), (0, 255, 255), 3)
        elif dest_idx == i:
            cv2.rectangle(panel, (x - 3, y - 3), (x + 83, y + 43), (0, 255, 255), 2)

    # Navigation hints
    cv2.putText(panel, "Pinch=Select | ThumbsUp=Confirm | ThumbLeft=Cancel | Q=Quit",
                (25, 200), font, 0.38, (120, 120, 120), 1)

    return panel
