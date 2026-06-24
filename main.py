import cv2, time, numpy as np
import mediapipe as mp
from hand_tracking import create_landmarker
import keyboard_ui
from input_controller import send_key

CAM_INDEX = 0
PINCH_THRESHOLD = 35
CLICK_COOLDOWN = 0.6
INDEX_TIP, THUMB_TIP = 8, 4

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, result, frame_w, frame_h):
    if not result.hand_landmarks:
        return
    for hand_landmarks in result.hand_landmarks:
        points = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 200, 100), 2, cv2.LINE_AA)
        for px, py in points:
            cv2.circle(frame, (px, py), 5, (0, 180, 255), cv2.FILLED)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

def is_pinch(points, tip1, tip2, threshold=PINCH_THRESHOLD):
    dist = np.hypot(points[tip1][0]-points[tip2][0], points[tip1][1]-points[tip2][1])
    return dist < threshold

def main():
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(CAM_INDEX)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"Camera resolution: {frame_w}x{frame_h}")

    layer = "letters"
    keys = keyboard_ui.build_keys(
        start_x=int(frame_w*0.05),
        start_y=int(frame_h*0.45),
        key_w=int(frame_w*0.07),
        key_h=int(frame_h*0.1),
        gap=keyboard_ui.KEY_GAP,
        layer=layer
    )

    last_press_time, pressed_label, flash_until = 0, None, 0
    typed_text = ""
    start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        draw_landmarks(frame, result, frame_w, frame_h)

        hovered_key, pinching = None, False
        if result.hand_landmarks and len(result.hand_landmarks) >= 1:
            right_hand = result.hand_landmarks[0]
            rh_points = [(int(lm.x*frame_w), int(lm.y*frame_h)) for lm in right_hand]
            index_tip = rh_points[INDEX_TIP]

            if is_pinch(rh_points, THUMB_TIP, INDEX_TIP):
                pinching = True

            for key in keys:
                if keyboard_ui.point_in_rect(index_tip[0], index_tip[1], key["rect"]):
                    hovered_key = key
                    break

        # Rebuild keys each frame
        keys = keyboard_ui.build_keys(
            start_x=int(frame_w*0.05),
            start_y=int(frame_h*0.45),
            key_w=int(frame_w*0.07),
            key_h=int(frame_h*0.1),
            gap=keyboard_ui.KEY_GAP,
            layer=layer
        )

        now = time.time()
        if hovered_key and pinching and (now - last_press_time) > CLICK_COOLDOWN:
            label = hovered_key["label"]
            send_key(label)
            last_press_time = now
            pressed_label = label
            flash_until = now + 0.25
            if label == "SPACE":
                typed_text += " "
            elif label == "<-":
                typed_text = typed_text[:-1]
            else:
                typed_text += label.lower()

        for key in keys:
            keyboard_ui.draw_key(frame, key,
                hovered=(key is hovered_key),
                pressed=(pressed_label == key["label"] and now < flash_until))

        # --- HUD ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (30, 20), (frame_w-30, 75), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (30, 20), (frame_w-30, 75), (70, 70, 70), 1)

        preview = typed_text[-50:] if typed_text else "Start typing..."
        preview_color = (180, 180, 180) if not typed_text else (100, 255, 180)
        cv2.putText(frame, preview, (45, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, preview_color, 2, cv2.LINE_AA)

        cv2.imshow("Virtual Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
