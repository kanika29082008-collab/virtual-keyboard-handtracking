import cv2
import numpy as np
from keyboard_ui import build_keys, draw_key, detect_key
from hand_tracking import create_landmarker

HOVER_RADIUS = 35
PRESS_RADIUS = 20

def send_key(label):
    """Simulate sending a key (for now just print)."""
    print(f"Key pressed: {label}")

def get_fingertip_position(hand_landmarks, frame_w, frame_h):
    """Return fingertip coordinates (index finger tip)."""
    x = int(hand_landmarks[8].x * frame_w)
    y = int(hand_landmarks[8].y * frame_h)
    return x, y

def run_controller():
    cap = cv2.VideoCapture(0)
    landmarker = create_landmarker()

    keys = build_keys(start_x=50, start_y=250, key_w=60, key_h=60, gap=6, layer="letters")
    typed_text = ""

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Detect hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(rgb)

        hovered_key = None
        pressed_key = None

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                px, py = get_fingertip_position(hand_landmarks, w, h)
                hovered_key, pressed_key = detect_key(keys, px, py, hover_radius=HOVER_RADIUS, press_radius=PRESS_RADIUS)

                cv2.circle(frame, (px, py), 8, (0, 255, 255), -1)

                if pressed_key:
                    if pressed_key["label"] == "<-":
                        typed_text = typed_text[:-1]
                    elif pressed_key["label"] == "SPACE":
                        typed_text += " "
                    else:
                        typed_text += pressed_key["label"].lower()
                    send_key(pressed_key["label"])

        for key in keys:
            draw_key(frame, key, hovered=(key == hovered_key), pressed=(key == pressed_key))

        cv2.putText(frame, typed_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Gesture Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
