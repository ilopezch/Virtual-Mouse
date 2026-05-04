
# 🖱️ Virtual Mouse Using Hand Gestures
[![License](https://img.shields.io/github/license/whitehatboy005/Virtual-Mouse)](LICENSE.md)

This Python application enables control of the mouse cursor through hand gestures captured via webcam. It leverages Mediapipe for hand tracking and OpenCV for video processing, allowing users to perform actions like moving the cursor, left-clicking, right-clicking, dragging, and scrolling using intuitive gestures.

## 📌 Features
- **Hand Tracking**: Utilizes Mediapipe to detect and track landmarks of the user's hand in real-time.
- **Cursor Control**: Moves the mouse cursor based on the position of the index finger.
- **Gesture Actions**: Supports gestures for left-clicking, right-clicking, dragging, and scrolling using specific hand configurations.
- **User Instructions**: Provides on-screen instructions for gesture controls and actions.

## Gestures
|ACTION|GESTURE||DESCRIPTION|
| :--- | :---: | :---: | ---:|
| MOVE | Index finger up, others down | → | cursor follows index tip |
| LEFT CLICK | Fist (all 5 fingers closed) | → | left click (hold > 0.6s = drag) |
| RIGHT CLICK | Pinky only up | → | right click (1.5s cooldown) |
| SCROLL UP | Index + Middle up | →  | hold still to keep scrolling up |
| SCROLL DOWN | Ring + Index + Middle up | →  | hold still to keep scrolling down |

## Result
![Virtual-Mouse](https://github.com/user-attachments/assets/95b3c0bc-c22a-4cb8-984e-ffa0eda5d55e)


## ⚙️ Installation (Linux):

## Clone the Repository
```bash
git clone https://github.com/ilopezch/Virtual-Mouse
cd Virtual-Mouse
```
## Install Dependencies
```bash
pip install -r requirements.txt
```
## Setup environment
```bash
echo 'KERNEL=="uinput", MODE="0660", GROUP="input"' | sudo tee /etc/udev/rules.d/99-uinput.rules

# Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger
```
```bash
# To make uinput load automatically on boot:
echo 'uinput' | sudo tee /etc/modules-load.d/uinput.conf
```

## Run the Program
```bash
python mouse.py
```

## ⚙️ Installation (Windows):
```bash
pip install mediapipe opencv-python numpy pyautogui 
```
## Run the Program
```bash
python mouse_windows.py
```

## 📝 License
This project is licensed under the terms of the [MIT license](LICENSE.md).
