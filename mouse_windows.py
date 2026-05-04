"""
Virtual Mouse (Windows) — simple gesture design
======================================
Gestures
─────────────────────────────────────────
  MOVE        Index finger up, others down  → cursor follows index tip
  LEFT CLICK  Fist (all 5 fingers closed)   → left click (hold > 0.6s = drag)
  RIGHT CLICK Pinky only up                 → right click (1.5s cooldown)
  SCROLL DOWN Index + Middle up (peace ✌)   → scroll down at fixed rate
  SCROLL UP   Index + Middle + Ring up      → scroll up at fixed rate

Requires: pip install mediapipe opencv-python numpy pyautogui

Tuning constants are at the top — adjust to taste.
"""

import cv2, time, os, urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pyautogui

pyautogui.PAUSE = 0        # disable per-call delay (we control timing ourselves)
pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort

BTN_LEFT  = 'left'
BTN_RIGHT = 'right'

# ── Tuning ─────────────────────────────────────────────────────────────────────
FRAME_MARGIN        = 0.15
SMOOTHENING         = 7
CONFIRM_FRAMES      = 4
DRAG_HOLD_SECS      = 0.6
RCLICK_COOLDOWN     = 1.5
SCROLL_INTERVAL     = 5

HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
])

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
              'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')


# ══ Setup helpers ══════════════════════════════════════════════════════════════

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~28 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def get_screen_size():
    s = pyautogui.size()
    return s.width, s.height


def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    return cap


def process_frame(frame):
    frame = cv2.flip(frame, 1)
    return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ══ Coordinate helpers ══════════════════════════════════════════════════════════

def lm_px(landmarks, fw, fh):
    return {i: (int(lm.x * fw), int(lm.y * fh)) for i, lm in enumerate(landmarks)}


def map_screen(coords, sw, sh, fw, fh):
    xlo, xhi = fw * FRAME_MARGIN, fw * (1 - FRAME_MARGIN)
    ylo, yhi = fh * FRAME_MARGIN, fh * (1 - FRAME_MARGIN)
    return {
        i: (int(np.interp(x, [xlo, xhi], [0, sw])),
            int(np.interp(y, [ylo, yhi], [0, sh])))
        for i, (x, y) in coords.items()
    }


# ══ Finger state detection ═════════════════════════════════════════════════════

def finger_up(c, tip, mcp):
    return c[tip][1] < c[mcp][1] - 15


def get_finger_states(c):
    idx  = finger_up(c, 8,  5)
    mid  = finger_up(c, 12, 9)
    rng  = finger_up(c, 16, 13)
    pnk  = finger_up(c, 20, 17)
    fist = not idx and not mid and not rng and not pnk
    return idx, mid, rng, pnk, fist


# ══ Mouse primitives (pyautogui) ══════════════════════════════════════════════

def ui_move(x, y):
    pyautogui.moveTo(x, y)

def ui_click(btn):
    pyautogui.click(button=btn)

def ui_dblclick(btn):
    pyautogui.doubleClick(button=btn)

def ui_press(btn):
    pyautogui.mouseDown(button=btn)

def ui_release(btn):
    pyautogui.mouseUp(button=btn)

def ui_scroll(d):
    # positive d = scroll up, negative = scroll down — matches pyautogui convention
    pyautogui.scroll(d)


# ══ State ══════════════════════════════════════════════════════════════════════

class State:
    def __init__(self):
        self.cx = self.cy       = 0

        self.fist_frames        = 0
        self.fist_fired         = False
        self.fist_start         = 0.0
        self.drag_active        = False
        self.lclick_at          = 0.0

        self.rclick_frames      = 0
        self.rclick_fired       = False
        self.last_rclick        = 0.0

        self.scroll_counter     = 0
        self.scroll_dir         = 0

        self.label              = "MOVE"
        self.label_until        = 0.0


# ══ Main update ════════════════════════════════════════════════════════════════

def update(raw, mapped, sw, sh, fh, st):
    now = time.time()
    idx, mid, rng, pnk, fist = get_finger_states(raw)

    scroll_down = idx and mid and not rng and not pnk
    scroll_up   = idx and mid and rng and not pnk

    if scroll_down or scroll_up:
        _set_label(st, "SCROLL", now)
        st.scroll_dir = -1 if scroll_down else 1
        st.scroll_counter += 1
        if st.scroll_counter >= SCROLL_INTERVAL:
            st.scroll_counter = 0
            ui_scroll(st.scroll_dir)
        return

    st.scroll_counter = 0
    st.scroll_dir     = 0

    if idx and not fist:
        ix, iy = mapped[8]
        st.cx = int(st.cx + (ix - st.cx) / SMOOTHENING)
        st.cy = int(st.cy + (iy - st.cy) / SMOOTHENING)
        st.cx = max(0, min(st.cx, sw - 1))
        st.cy = max(0, min(st.cy, sh - 1))
        ui_move(st.cx, st.cy)
        _set_label(st, "MOVE", now)

    if fist:
        st.fist_frames += 1
        if st.fist_start == 0:
            st.fist_start = now

        held = now - st.fist_start

        if not st.drag_active and not st.fist_fired and held > DRAG_HOLD_SECS:
            ui_press(BTN_LEFT)
            st.drag_active = True
            _set_label(st, "DRAG", now)

        if st.fist_frames >= CONFIRM_FRAMES and not st.fist_fired:
            if not st.drag_active:
                if now - st.lclick_at < 0.4:
                    ui_dblclick(BTN_LEFT)
                    _set_label(st, "DBL-CLICK", now)
                    st.lclick_at = 0
                else:
                    ui_click(BTN_LEFT)
                    _set_label(st, "L-CLICK", now)
                    st.lclick_at = now
                st.fist_fired = True
    else:
        if st.drag_active:
            ui_release(BTN_LEFT)
            st.drag_active = False
            _set_label(st, "MOVE", now)
        st.fist_frames = 0
        st.fist_fired  = False
        st.fist_start  = 0.0

    pinky_only = pnk and not idx and not fist
    if pinky_only:
        st.rclick_frames += 1
        if (st.rclick_frames >= CONFIRM_FRAMES
                and not st.rclick_fired
                and now - st.last_rclick > RCLICK_COOLDOWN):
            ui_click(BTN_RIGHT)
            st.last_rclick = now
            st.rclick_fired = True
            _set_label(st, "R-CLICK", now)
    else:
        st.rclick_frames = 0
        st.rclick_fired  = False


def _set_label(st, label, now):
    if label == "MOVE" and st.label not in ("MOVE", "SCROLL") and now < st.label_until:
        return
    st.label = label
    if label not in ("MOVE", "SCROLL"):
        st.label_until = now + 0.9


# ══ Drawing ════════════════════════════════════════════════════════════════════

def draw_landmarks(frame, landmarks):
    fw, fh = frame.shape[1], frame.shape[0]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x*fw), int(lm.y*fh)), 5, (0,255,0), -1)
    for s, d in HAND_CONNECTIONS:
        cv2.line(frame,
                 (int(landmarks[s].x*fw), int(landmarks[s].y*fh)),
                 (int(landmarks[d].x*fw), int(landmarks[d].y*fh)),
                 (0,255,0), 2)


def draw_hud(frame, st, now):
    lines = [
        "MOVE  : index finger up",
        "SCROLL DOWN: index+middle up (peace)",
        "SCROLL UP  : index+middle+ring up",
        "LClick: fist",
        "Drag  : hold fist > 0.6s",
        "RClick: pinky only up",
    ]
    for i, l in enumerate(lines):
        cv2.putText(frame, l, (10, 20 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,220,0), 1, cv2.LINE_AA)

    h = frame.shape[0]
    y0 = h - 70
    cv2.rectangle(frame, (0, y0-10), (280, h), (0,0,0), -1)

    label_col = {
        "MOVE":      (160,160,160),
        "SCROLL":    (0,200,255),
        "L-CLICK":   (0,255,100),
        "DBL-CLICK": (0,255,100),
        "R-CLICK":   (0,100,255),
        "DRAG":      (0,140,255),
    }.get(st.label, (200,200,200))

    cv2.putText(frame, f"  {st.label}", (8, y0+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_col, 2)

    if st.scroll_dir:
        arrow = "UP" if st.scroll_dir > 0 else "DOWN"
        cv2.putText(frame, f"Scroll {arrow}", (8, y0+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)


# ══ Main ═══════════════════════════════════════════════════════════════════════

def main():
    ensure_model()
    sw, sh = get_screen_size()
    print(f"Screen: {sw}x{sh}")

    cap = init_webcam()
    det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1))

    st = State()
    print("Press Esc in the camera window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, rgb = process_frame(frame)
        fh, fw, _  = frame.shape

        result = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if result.hand_landmarks:
            lms    = result.hand_landmarks[0]
            draw_landmarks(frame, lms)
            raw    = lm_px(lms, fw, fh)
            mapped = map_screen(raw, sw, sh, fw, fh)
            update(raw, mapped, sw, sh, fh, st)

        now = time.time()
        if st.label not in ("MOVE", "SCROLL") and now > st.label_until:
            st.label = "MOVE"

        draw_hud(frame, st, now)
        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
