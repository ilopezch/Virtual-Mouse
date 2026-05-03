import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import os
import urllib.request
import numpy as np
import screeninfo
import evdev
from evdev import UInput, ecodes as e

# ── Drawing constants ──────────────────────────────────────────────────────────
CIRCLE_RADIUS    = 5
CIRCLE_COLOR     = (0, 255, 0)
CIRCLE_THICKNESS = -1
LINE_COLOR       = (0, 255, 0)
LINE_THICKNESS   = 2

# ── Active zone ────────────────────────────────────────────────────────────────
# Only hand movements inside this inner rectangle map to the full screen.
# Raise to 0.20 if cursor still won't reach edges; lower to 0.10 for less gain.
FRAME_MARGIN = 0.15

HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
])

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
              'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')


# ── Screen / device helpers ────────────────────────────────────────────────────

def get_screen_size():
    """Primary screen resolution via screeninfo — no display server required."""
    try:
        monitors = screeninfo.get_monitors()
        primary  = next((m for m in monitors if getattr(m, 'is_primary', False)), monitors[0])
        return primary.width, primary.height
    except Exception:
        return 1920, 1080          # safe fallback


def create_mouse_device(screen_w, screen_h):
    """
    Virtual mouse via Linux uinput (kernel level).
    Works on X11, Wayland, and bare TTY — no display server needed.

    Pre-requisites (run once as root, then re-login):
        sudo modprobe uinput
        sudo usermod -aG input $USER
    """
    caps = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
        e.EV_ABS: [
            (e.ABS_X, evdev.AbsInfo(value=0, min=0, max=screen_w - 1,
                                    fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, evdev.AbsInfo(value=0, min=0, max=screen_h - 1,
                                    fuzz=0, flat=0, resolution=0)),
        ],
        e.EV_REL: [e.REL_WHEEL],
    }
    return UInput(caps, name='virtual-hand-mouse')


# ── Model ──────────────────────────────────────────────────────────────────────

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~28 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# ── Webcam / frame helpers ─────────────────────────────────────────────────────

def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    return cap


def process_frame(frame):
    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame


def draw_landmarks(frame, landmarks):
    for lm in landmarks:
        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
    for s, d in HAND_CONNECTIONS:
        sx = int(landmarks[s].x * frame.shape[1]); sy = int(landmarks[s].y * frame.shape[0])
        dx = int(landmarks[d].x * frame.shape[1]); dy = int(landmarks[d].y * frame.shape[0])
        cv2.line(frame, (sx, sy), (dx, dy), LINE_COLOR, LINE_THICKNESS)


def draw_active_region(frame):
    h, w   = frame.shape[:2]
    x1, y1 = int(w * FRAME_MARGIN),       int(h * FRAME_MARGIN)
    x2, y2 = int(w * (1 - FRAME_MARGIN)), int(h * (1 - FRAME_MARGIN))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, "Active Zone", (x1 + 4, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


# ── Coordinate mapping ─────────────────────────────────────────────────────────

def get_landmark_coords(landmarks, fw, fh):
    return {i: (int(lm.x * fw), int(lm.y * fh)) for i, lm in enumerate(landmarks)}


def map_to_screen(coords, sw, sh, fw, fh):
    """Map from the active inner rectangle to the full screen using np.interp."""
    x_min, x_max = fw * FRAME_MARGIN, fw * (1 - FRAME_MARGIN)
    y_min, y_max = fh * FRAME_MARGIN, fh * (1 - FRAME_MARGIN)
    return {
        i: (int(np.interp(x, [x_min, x_max], [0, sw])),
            int(np.interp(y, [y_min, y_max], [0, sh])))
        for i, (x, y) in coords.items()
    }


# ── Mouse actions via uinput ───────────────────────────────────────────────────

def move_cursor(ui, index_coords, plocx, plocy, sw, sh, smoothening):
    ix, iy = index_coords
    cx = int(plocx + (ix - plocx) / smoothening)
    cy = int(plocy + (iy - plocy) / smoothening)
    cx = max(0, min(cx, sw - 1))
    cy = max(0, min(cy, sh - 1))
    ui.write(e.EV_ABS, e.ABS_X, cx)
    ui.write(e.EV_ABS, e.ABS_Y, cy)
    ui.syn()
    return cx, cy


def mouse_click(ui, btn):
    ui.write(e.EV_KEY, btn, 1); ui.syn()
    ui.write(e.EV_KEY, btn, 0); ui.syn()


def mouse_double_click(ui, btn):
    mouse_click(ui, btn)
    mouse_click(ui, btn)


def mouse_press(ui, btn):
    ui.write(e.EV_KEY, btn, 1); ui.syn()


def mouse_release(ui, btn):
    ui.write(e.EV_KEY, btn, 0); ui.syn()


def mouse_scroll(ui, direction):
    ui.write(e.EV_REL, e.REL_WHEEL, direction); ui.syn()


# ── Gesture detection ──────────────────────────────────────────────────────────

def detect_gestures(ui, coords, thumb_y,
                    click_time, click_threshold, single_click_flag,
                    left_dragging, right_click_time, right_click_cooldown):
    now = time.time()

    # Left click — thumb near index tip (landmark 8)
    if abs(coords[8][1] - thumb_y) < 70:
        if now - click_time < click_threshold:
            mouse_double_click(ui, e.BTN_LEFT)
            click_time = 0
        elif not single_click_flag:
            mouse_click(ui, e.BTN_LEFT)
            single_click_flag = True
            click_time = now
    else:
        single_click_flag = False

    # Drag — thumb near ring finger tip (landmark 16)
    if abs(coords[16][1] - thumb_y) < 70:
        if not left_dragging:
            mouse_press(ui, e.BTN_LEFT)
            left_dragging = True
    else:
        if left_dragging:
            mouse_release(ui, e.BTN_LEFT)
            left_dragging = False

    # Right click — thumb near middle finger tip (landmark 12), 1s cooldown
    if abs(coords[12][1] - thumb_y) < 40:
        if now - right_click_time > right_click_cooldown:
            mouse_click(ui, e.BTN_RIGHT)
            right_click_time = now

    # Scroll — fist with thumb up/down (no other extended fingers)
    extended = [i for i in [8, 12, 16, 20] if coords[i][1] < coords[i - 2][1]]
    if not extended:
        if coords[4][1] < coords[0][1] - 40:
            mouse_scroll(ui, 3)
        elif coords[4][1] > coords[0][1] + 40:
            mouse_scroll(ui, -3)

    return click_time, single_click_flag, left_dragging, right_click_time


# ── HUD ────────────────────────────────────────────────────────────────────────

def add_hud(frame):
    lines = [
        "Virtual Mouse (uinput - no X11 needed)",
        "Move  : Index finger inside yellow box",
        "LClick: Thumb + Index",
        "RClick: Thumb + Middle  (1s cooldown)",
        "Drag  : Thumb + Ring",
        "Scroll: Fist + Thumb Up / Down",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 20 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_model()

    sw, sh = get_screen_size()
    print(f"Screen: {sw}x{sh}")

    try:
        ui = create_mouse_device(sw, sh)
    except PermissionError:
        print("\nERROR: Permission denied opening /dev/uinput.")
        print("Run these commands, then log out and back in:\n")
        print("  sudo modprobe uinput")
        print("  sudo usermod -aG input $USER\n")
        raise

    cap = init_webcam()

    base_options  = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options       = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    hand_detector = mp_vision.HandLandmarker.create_from_options(options)

    smoothening         = 7
    plocx = plocy       = 0
    click_time          = 0
    click_threshold     = 0.3
    single_click_flag   = False
    left_dragging       = False
    right_click_time    = 0
    right_click_cooldown = 1.0

    print("Running — press Esc in the camera window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam read failed.")
            break

        frame, rgb_frame = process_frame(frame)
        fh, fw, _        = frame.shape

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = hand_detector.detect(mp_img)

        draw_active_region(frame)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks)

            coords   = get_landmark_coords(landmarks, fw, fh)
            mapped   = map_to_screen(coords, sw, sh, fw, fh)
            thumb_y  = mapped[4][1]

            plocx, plocy = move_cursor(ui, mapped[8], plocx, plocy, sw, sh, smoothening)

            (click_time, single_click_flag,
             left_dragging, right_click_time) = detect_gestures(
                ui, mapped, thumb_y,
                click_time, click_threshold, single_click_flag,
                left_dragging, right_click_time, right_click_cooldown
            )

        add_hud(frame)
        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ui.close()


if __name__ == "__main__":
    main()
