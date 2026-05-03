"""
Virtual Mouse — simple gesture design
======================================
Gestures
─────────────────────────────────────────
  MOVE        Index finger up, others down  → cursor follows index tip
  LEFT CLICK  Fist (all 5 fingers closed)   → left click (hold > 0.6s = drag)
  RIGHT CLICK Pinky only up                 → right click (1.5s cooldown)
  SCROLL      Index + Middle up             → move hand into top/bottom zone
                                               to latch scroll direction; hold still to keep scrolling

Tuning constants are at the top — adjust to taste.
"""

import cv2, math, time, os, urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import screeninfo, evdev
from evdev import UInput, ecodes as e

# ── Tuning ─────────────────────────────────────────────────────────────────────
FRAME_MARGIN        = 0.15   # active-zone margin; raise if can't reach screen edges
SMOOTHENING         = 7      # cursor lag vs smoothness (higher = smoother/laggier)
CONFIRM_FRAMES      = 4      # frames gesture must be stable before firing
DRAG_HOLD_SECS      = 0.6    # seconds to hold fist before drag starts
RCLICK_COOLDOWN     = 1.5    # min seconds between right clicks
SCROLL_INTERVAL     = 10     # frames between scroll ticks (higher = slower)
FINGER_CLOSED_RATIO = 0.7    # tip/mcp ratio threshold for "finger closed" detection

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
    try:
        ms = screeninfo.get_monitors()
        p  = next((m for m in ms if getattr(m, 'is_primary', False)), ms[0])
        return p.width, p.height
    except Exception:
        return 1920, 1080


def create_mouse_device(sw, sh):
    caps = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
        e.EV_ABS: [
            (e.ABS_X, evdev.AbsInfo(0, 0, sw-1, 0, 0, 0)),
            (e.ABS_Y, evdev.AbsInfo(0, 0, sh-1, 0, 0, 0)),
        ],
        e.EV_REL: [e.REL_WHEEL],
    }
    return UInput(caps, name='virtual-hand-mouse')


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
    """
    True when fingertip is clearly above its MCP knuckle.
    Using MCP (not PIP) gives a much larger gap — reliable across
    all hand heights and angles in the frame.
      index:  tip=8,  mcp=5
      middle: tip=12, mcp=9
      ring:   tip=16, mcp=13
      pinky:  tip=20, mcp=17
    """
    return c[tip][1] < c[mcp][1] - 15   # 15px margin avoids borderline reads


def get_finger_states(c):
    """
    Returns (index_up, middle_up, ring_up, pinky_up, fist).
    fist = all four fingers closed.
    """
    idx  = finger_up(c, 8,  5)
    mid  = finger_up(c, 12, 9)
    rng  = finger_up(c, 16, 13)
    pnk  = finger_up(c, 20, 17)
    fist = not idx and not mid and not rng and not pnk
    return idx, mid, rng, pnk, fist


# ══ uinput primitives ══════════════════════════════════════════════════════════

def ui_move(ui, x, y):
    ui.write(e.EV_ABS, e.ABS_X, x)
    ui.write(e.EV_ABS, e.ABS_Y, y)
    ui.syn()

def ui_click(ui, btn):
    ui.write(e.EV_KEY, btn, 1); ui.syn()
    ui.write(e.EV_KEY, btn, 0); ui.syn()

def ui_dblclick(ui, btn):
    ui_click(ui, btn); ui_click(ui, btn)

def ui_press(ui, btn):
    ui.write(e.EV_KEY, btn, 1); ui.syn()

def ui_release(ui, btn):
    ui.write(e.EV_KEY, btn, 0); ui.syn()

def ui_scroll(ui, d):
    ui.write(e.EV_REL, e.REL_WHEEL, d); ui.syn()


# ══ State ══════════════════════════════════════════════════════════════════════

class State:
    def __init__(self):
        self.cx = self.cy       = 0      # current smoothed cursor position

        # fist / left-click / drag
        self.fist_frames        = 0
        self.fist_fired         = False  # click already sent this fist
        self.fist_start         = 0.0   # when fist began (for drag timer)
        self.drag_active        = False
        self.lclick_at          = 0.0   # timestamp of last left click (double-click)

        # right click
        self.rclick_frames      = 0
        self.rclick_fired       = False
        self.last_rclick        = 0.0

        # scroll
        self.scroll_counter     = 0
        self.scroll_dir         = 0

        # hud
        self.label              = "MOVE"
        self.label_until        = 0.0


# ══ Main update ════════════════════════════════════════════════════════════════

def update(ui, raw, mapped, sw, sh, fh, st):
    now = time.time()
    idx, mid, rng, pnk, fist = get_finger_states(raw)

    # ── SCROLL DOWN: index + middle up, ring + pinky down (peace sign ✌️)
    scroll_down = idx and mid and not rng and not pnk
    # ── SCROLL UP: index + middle + ring up, pinky down (three fingers)
    scroll_up   = idx and mid and rng and not pnk

    if scroll_down or scroll_up:
        _set_label(st, "SCROLL", now)

        # Latch direction from gesture — overwrite each frame so switching works
        st.scroll_dir = -1 if scroll_down else 1

        # Fire at fixed rate
        st.scroll_counter += 1
        if st.scroll_counter >= SCROLL_INTERVAL:
            st.scroll_counter = 0
            ui_scroll(ui, st.scroll_dir)

        return   # don't move cursor or process clicks in scroll mode

    # Leaving scroll mode — reset
    st.scroll_counter = 0
    st.scroll_dir     = 0

    # ── MOVE: only index finger up ─────────────────────────────────────────
    if idx and not fist:  # move: index up, not a fist
        ix, iy = mapped[8]
        st.cx = int(st.cx + (ix - st.cx) / SMOOTHENING)
        st.cy = int(st.cy + (iy - st.cy) / SMOOTHENING)
        st.cx = max(0, min(st.cx, sw - 1))
        st.cy = max(0, min(st.cy, sh - 1))
        ui_move(ui, st.cx, st.cy)
        _set_label(st, "MOVE", now)

    # ── LEFT CLICK / DRAG: fist ────────────────────────────────────────────
    if fist:
        st.fist_frames += 1
        if st.fist_start == 0:
            st.fist_start = now

        held = now - st.fist_start

        # Drag: fist held long enough without having clicked
        if not st.drag_active and not st.fist_fired and held > DRAG_HOLD_SECS:
            ui_press(ui, e.BTN_LEFT)
            st.drag_active = True
            _set_label(st, "DRAG", now)

        # Click: confirmed frames, fire once
        if st.fist_frames >= CONFIRM_FRAMES and not st.fist_fired:
            if not st.drag_active:
                if now - st.lclick_at < 0.4:
                    ui_dblclick(ui, e.BTN_LEFT)
                    _set_label(st, "DBL-CLICK", now)
                    st.lclick_at = 0
                else:
                    ui_click(ui, e.BTN_LEFT)
                    _set_label(st, "L-CLICK", now)
                    st.lclick_at = now
                st.fist_fired = True
    else:
        # Fist released
        if st.drag_active:
            ui_release(ui, e.BTN_LEFT)
            st.drag_active = False
            _set_label(st, "MOVE", now)
        st.fist_frames = 0
        st.fist_fired  = False
        st.fist_start  = 0.0

    # ── RIGHT CLICK: pinky only up ─────────────────────────────────────────
    pinky_only = pnk and not idx and not fist  # pinky up, index down
    if pinky_only:
        st.rclick_frames += 1
        if (st.rclick_frames >= CONFIRM_FRAMES
                and not st.rclick_fired
                and now - st.last_rclick > RCLICK_COOLDOWN):
            ui_click(ui, e.BTN_RIGHT)
            st.last_rclick = now
            st.rclick_fired = True
            _set_label(st, "R-CLICK", now)
    else:
        st.rclick_frames = 0
        st.rclick_fired  = False


def _set_label(st, label, now):
    # Don't overwrite an action label with MOVE until it expires
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


def draw_active_zone(frame, scroll_mode=False):
    h, w = frame.shape[:2]
    x1, y1 = int(w*FRAME_MARGIN), int(h*FRAME_MARGIN)
    x2, y2 = int(w*(1-FRAME_MARGIN)), int(h*(1-FRAME_MARGIN))
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
    cv2.putText(frame, "Active Zone", (x1+4, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    if scroll_mode:
        # Draw the three scroll zones on the right edge of the active area
        zone_h  = y2 - y1
        up_y    = y1 + int(zone_h * 0.30)
        down_y  = y1 + int(zone_h * 0.60)
        mid_x   = x2 + 8

        # Up zone (top 30%) — blue
        cv2.rectangle(frame, (x2+2, y1), (x2+18, up_y), (255,120,0), -1)
        cv2.putText(frame, "▲", (x2+4, y1+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Dead zone (middle 40%) — grey
        cv2.rectangle(frame, (x2+2, up_y), (x2+18, down_y), (80,80,80), -1)

        # Down zone (bottom 30%) — blue
        cv2.rectangle(frame, (x2+2, down_y), (x2+18, y2), (255,120,0), -1)
        cv2.putText(frame, "▼", (x2+4, y2-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


def draw_hud(frame, st, now):
    # ── top-left instructions ──────────────────────────────────────────────
    lines = [
        "MOVE  : index finger up",
        "SCROLL DOWN: index+middle up (peace ✌)",
        "SCROLL UP  : index+middle+ring up",
        "LClick: fist",
        "Drag  : hold fist > 0.6s",
        "RClick: pinky only up",
    ]
    for i, l in enumerate(lines):
        cv2.putText(frame, l, (10, 20 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,220,0), 1, cv2.LINE_AA)

    # ── bottom-left status ─────────────────────────────────────────────────
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
        arrow = "▲ UP" if st.scroll_dir > 0 else "▼ DOWN"
        cv2.putText(frame, f"Scroll {arrow}", (8, y0+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)


# ══ Main ═══════════════════════════════════════════════════════════════════════

def main():
    ensure_model()
    sw, sh = get_screen_size()
    print(f"Screen: {sw}×{sh}")

    try:
        ui = create_mouse_device(sw, sh)
    except PermissionError:
        print("\nERROR: /dev/uinput permission denied. Run:")
        print("  sudo modprobe uinput && sudo usermod -aG input $USER")
        print("Then: newgrp input  (or re-login)\n")
        raise

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
        draw_active_zone(frame)

        if result.hand_landmarks:
            lms    = result.hand_landmarks[0]
            draw_landmarks(frame, lms)
            raw    = lm_px(lms, fw, fh)
            mapped = map_screen(raw, sw, sh, fw, fh)
            update(ui, raw, mapped, sw, sh, fh, st)

        now = time.time()
        if st.label not in ("MOVE", "SCROLL") and now > st.label_until:
            st.label = "MOVE"

        draw_hud(frame, st, now)
        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ui.close()


if __name__ == "__main__":
    main()
