<div align="center">

# 🖐️ Virtual Keyboard (Gesture Controlled)

### Control your keyboard using nothing but hand gestures over a webcam

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00A98F?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## ✨ Features

- 🖐️ Tracks your hand with **MediaPipe HandLandmarker** (21 landmarks, live)
- ⌨️ On-screen QWERTY keyboard rendered directly over the webcam feed
- 👁️ Hover with your **index finger** to highlight a key
- 🤏 Pinch **thumb + index finger** together to press it
- 🔁 Keys are typed into *any* active window (Notepad, browser, etc.) via `pynput`
- 💫 Visual feedback: hover glow, press flash, live typed-text preview

---

## 🎮 Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `SPACE` (on-screen key) | Types a space |
| `<-` (on-screen key) | Backspace — deletes last character |

---

## 📂 Project Structure
VIRTUAL_KEYBOARD/

│

├── main.py                # Entry point: webcam loop, orchestrates everything

├── hand_tracking.py        # MediaPipe setup & hand landmark detection

├── keyboard_ui.py           # Keyboard layout, drawing, and visual effects

├── input_controller.py      # Sending keypresses via pynput

│

├── requirements.txt         # Minimal dependencies (for quick install)

├── requirements-full.txt    # Full frozen environment (optional, reproducibility)

├── LICENSE                  # MIT License

├── README.md                 # You're reading it

│

├── models/                  # Auto-downloaded hand_landmarker.task model

│   └── hand_landmarker.task

│

└── venv/                    # Local virtual environment (not pushed to GitHub)
---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/virtual_keyboard.git
cd virtual_keyboard

# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main.py
```

The first run will auto-download the hand landmark model (~10MB) into `models/`.
Open any text field (Notepad, browser, etc.) to give it focus, then hover + pinch to type.

---

## 🛠️ Tech Stack

- **Python** — core logic
- **OpenCV** — webcam capture & rendering
- **MediaPipe Tasks API** — real-time hand landmark detection
- **pynput** — sends real OS-level keystrokes
- **NumPy** — geometry & gesture math

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

⭐ **If you found this interesting, consider starring the repo!**

</div>