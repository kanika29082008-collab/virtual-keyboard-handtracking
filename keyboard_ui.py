import cv2
import numpy as np

KEY_GAP = 6

LAYERS = {
    "letters": [
        ["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","<-"],
        ["SPACE"]
    ]
}

# Transparent theme colors
BORDER_NORMAL = (180, 180, 180)
BORDER_HOVER  = (255, 200, 80)
BORDER_PRESS  = (100, 255, 180)
TEXT_NORMAL   = (255, 255, 255)
TEXT_HOVER    = (255, 220, 100)
TEXT_PRESS    = (100, 255, 180)

def build_keys(start_x, start_y, key_w, key_h, gap, layer="letters"):
    layout = LAYERS[layer]
    keys = []
    y = start_y
    for row in layout:
        x = start_x
        for label in row:
            w = key_w
            if label == "SPACE":
                w = key_w * 6 + gap * 5
            keys.append({
                "label": label,
                "rect": (int(x), int(y), int(x + w), int(y + key_h)),
            })
            x += w + gap
        y += key_h + gap
    return keys

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

def draw_key(frame, key, hovered=False, pressed=False):
    x1, y1, x2, y2 = key["rect"]
    label = key["label"]

    # Border only (transparent body)
    border_color = BORDER_PRESS if pressed else (BORDER_HOVER if hovered else BORDER_NORMAL)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    # Text
    text_color = TEXT_PRESS if pressed else (TEXT_HOVER if hovered else TEXT_NORMAL)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42 if len(label) > 2 else 0.75
    thickness = 1 if len(label) > 2 else 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    tx = x1 + ((x2 - x1) - text_size[0]) // 2
    ty = y1 + ((y2 - y1) + text_size[1]) // 2
    cv2.putText(frame, label, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

def detect_key(keys, px, py, hover_radius=35, press_radius=20):
    """Return hovered and pressed key based on proximity."""
    hovered = None
    pressed = None
    for key in keys:
        x1, y1, x2, y2 = key["rect"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = np.hypot(px - cx, py - cy)
        if dist < hover_radius:
            hovered = key
        if dist < press_radius:
            pressed = key
    return hovered, pressed
