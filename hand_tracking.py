import os
import urllib.request
from mediapipe.tasks.python import vision
import mediapipe as mp

def create_landmarker():
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "hand_landmarker.task")

    if not os.path.exists(model_path):
        print("[INFO] Downloading hand_landmarker.task model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("[INFO] Model downloaded successfully.")

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO
    )

    landmarker = vision.HandLandmarker.create_from_options(options)
    return landmarker
