"""
Virtual Keyboard — control your keyboard using hand gestures over a webcam.

Updated for newer MediaPipe versions (0.10.30+) which removed the old
`mp.solutions.hands` API in favor of the Tasks API (`mp.tasks.vision`).

How it works:
- MediaPipe's HandLandmarker tracks your hand (21 landmarks) from the webcam feed.
- An on-screen QWERTY keyboard is drawn over the video.
- Move your INDEX FINGER over a key to highlight it.
- Pinch your THUMB and INDEX FINGER together (like a tiny click) to "press" the
  highlighted key. The key is actually typed into whatever window has focus
  (Notepad, browser, etc.) using pynput.

Controls:
- 'q' : quit
- SPACE key on the virtual keyboard types a space
- "<-" key on the virtual keyboard deletes the last character (Backspace)

Run:
    pip install -r requirements.txt
    python virtual_keyboard.py

First run will auto-download the hand landmark model (~10MB) to
./models/hand_landmarker.task
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pynput.keyboard import Controller, Key

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
PINCH_THRESHOLD = 35
CLICK_COOLDOWN = 0.6
HOVER_COLOR = (60, 60, 60)
HOVER_HIGHLIGHT = (0, 200, 255)
PRESS_FLASH_COLOR = (0, 255, 0)
KEY_GAP = 8

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

LAYOUT = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "<-"],
    ["SPACE"],
]

# MediaPipe HandLandmarker landmark indices we care about
INDEX_TIP = 8
THUMB_TIP = 4

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

keyboard_ctrl = Controller()


def ensure_model():
    """Download the hand_landmarker.task model if it isn't present yet."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading hand landmark model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def build_keys(start_x, start_y, key_w, key_h, gap):
    keys = []
    y = start_y
    for row in LAYOUT:
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


def send_key(label):
    if label == "SPACE":
        keyboard_ctrl.press(Key.space)
        keyboard_ctrl.release(Key.space)
    elif label == "<-":
        keyboard_ctrl.press(Key.backspace)
        keyboard_ctrl.release(Key.backspace)
    else:
        keyboard_ctrl.press(label.lower())
        keyboard_ctrl.release(label.lower())


def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print("Could not open webcam. Try changing CAM_INDEX in the script.")
        return

    keys = build_keys(start_x=40, start_y=320, key_w=95, key_h=70, gap=KEY_GAP)

    last_press_time = 0
    pressed_label = None
    flash_until = 0
    typed_text = ""
    start_time = time.time()

    print("Virtual Keyboard running. Press 'q' in the video window to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hovered_key = None
        pinching = False

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]  # first detected hand
            h, w, _ = frame.shape

            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (200, 200, 200), 2)
            for p in points:
                cv2.circle(frame, p, 4, (0, 140, 255), cv2.FILLED)

            index_tip = points[INDEX_TIP]
            thumb_tip = points[THUMB_TIP]

            dist = np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            pinching = dist < PINCH_THRESHOLD

            cv2.circle(frame, index_tip, 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, thumb_tip, 10, (255, 0, 255), cv2.FILLED)
            line_color = PRESS_FLASH_COLOR if pinching else (255, 0, 255)
            cv2.line(frame, index_tip, thumb_tip, line_color, 3)

            for key in keys:
                if point_in_rect(index_tip[0], index_tip[1], key["rect"]):
                    hovered_key = key
                    break

        now = time.time()
        if hovered_key and pinching and (now - last_press_time) > CLICK_COOLDOWN:
            send_key(hovered_key["label"])
            last_press_time = now
            pressed_label = hovered_key["label"]
            flash_until = now + 0.25

            if hovered_key["label"] == "SPACE":
                typed_text += " "
            elif hovered_key["label"] == "<-":
                typed_text = typed_text[:-1]
            else:
                typed_text += hovered_key["label"].lower()

        for key in keys:
            x1, y1, x2, y2 = key["rect"]
            color = HOVER_COLOR
            if key is hovered_key:
                color = HOVER_HIGHLIGHT
            if pressed_label == key["label"] and now < flash_until:
                color = PRESS_FLASH_COLOR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            label = key["label"]
            font_scale = 0.5 if len(label) > 2 else 0.9
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            tx = x1 + ((x2 - x1) - text_size[0]) // 2
            ty = y1 + ((y2 - y1) + text_size[1]) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), 2)

        cv2.rectangle(frame, (40, 40), (FRAME_W - 40, 100), (30, 30, 30), cv2.FILLED)
        preview = typed_text[-40:]
        cv2.putText(frame, preview, (55, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        cv2.imshow("Virtual Keyboard - press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()