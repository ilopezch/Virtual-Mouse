"""
Virtual Mouse — mode-switching design
======================================
The root cause of click drift is that the cursor-control finger (index) is
also part of the click gesture, so the pinch motion itself moves the cursor.

Fix: as soon as ANY non-index finger starts closing toward the thumb the
cursor is FROZEN.  The click is evaluated on a still cursor, then movement
resumes only after the hand fully opens again.

Gesture map
───────────────────────────────────────────────────────────────────────────────
 MOVE mode   (only index finger up)
   • Index tip position  → cursor moves

 FREEZE + ACTION  (any other finger closes toward thumb while in move mode)
   • Cursor STOPS immediately
   • Pinch thumb ↔ index   → Left click  (hold 0.6 s → drag)
   • Pinch thumb ↔ middle  → Right click (1.5 s cooldown)
   • Cursor resumes only when hand fully opens (all 4 fingers extended)

 SCROLL mode  (index + middle both up, ring + pinky down)
   • Move whole hand up / down to scroll
   • Exit: open all fingers or close all fingers

Tuning constants at the top of the file — adjust to taste.
───────────────────────────────────────────────────────────────────────────────
"""

import cv2, math, time, os, urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import screeninfo, evdev
from evdev import UInput, ecodes as e

# ── Tuning ─────────────────────────────────────────────────────────────────────
FRAME_MARGIN         = 0.15   # active-zone margin (raise if can't reach edges)
PINCH_THRESHOLD      = 38     # px in camera space; lower = tighter pinch needed
PINCH_CONFIRM_FRAMES = 4      # frames pinch must be stable before firing
DRAG_HOLD_SECS       = 0.55   # seconds to hold left-pinch before drag starts
RCLICK_COOLDOWN      = 1.5    # min seconds between right clicks
SCROLL_INTERVAL      = 10     # frames between scroll ticks (higher = slower)
SMOOTHENING          = 7      # cursor smoothing (higher = smoother but laggier)

# ── Drawing ────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
])

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
              'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')


# ══ Helpers ════════════════════════════════════════════════════════════════════

def get_screen_size():
    try:
        ms = screeninfo.get_monitors()
        p  = next((m for m in ms if getattr(m,'is_primary',False)), ms[0])
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


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~28 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def init_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    return cap


def process_frame(frame):
    return cv2.flip(frame, 1), cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)


def lm_px(landmarks, fw, fh):
    """Landmark → pixel dict in camera space."""
    return {i: (int(lm.x*fw), int(lm.y*fh)) for i, lm in enumerate(landmarks)}


def map_screen(coords, sw, sh, fw, fh):
    xlo, xhi = fw*FRAME_MARGIN, fw*(1-FRAME_MARGIN)
    ylo, yhi = fh*FRAME_MARGIN, fh*(1-FRAME_MARGIN)
    return {i: (int(np.interp(x,[xlo,xhi],[0,sw])),
                int(np.interp(y,[ylo,yhi],[0,sh])))
            for i,(x,y) in coords.items()}


def pdist(c, a, b):
    return math.hypot(c[a][0]-c[b][0], c[a][1]-c[b][1])


def is_up(c, tip, pip):
    """Finger extended: tip above PIP joint."""
    return c[tip][1] < c[pip][1]


# ── uinput primitives ──────────────────────────────────────────────────────────

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


# ══ Mode machine ═══════════════════════════════════════════════════════════════

class Mode:
    MOVE   = "MOVE"
    FROZEN = "FROZEN"     # cursor locked, waiting for action
    SCROLL = "SCROLL"
    DRAG   = "DRAG"


class State:
    def __init__(self):
        self.mode            = Mode.MOVE

        # cursor
        self.cx = self.cy    = 0

        # freeze / click
        self.freeze_x        = 0
        self.freeze_y        = 0
        self.lpin_frames     = 0
        self.lpin_fired      = False
        self.lpin_start      = 0.0
        self.lclick_at       = 0.0     # for double-click detection
        self.drag_active     = False

        self.rpin_frames     = 0
        self.rpin_fired      = False
        self.last_rclick     = 0.0

        # scroll
        self.scroll_ref_y    = None
        self.scroll_counter  = 0
        self.scroll_dir      = 0

        # hud
        self.label           = Mode.MOVE
        self.label_until     = 0.0

        # pinch distances for visualiser
        self.d_left          = PINCH_THRESHOLD * 2
        self.d_right         = PINCH_THRESHOLD * 2


def update(ui, raw, mapped, sw, sh, state):
    now = time.time()

    # ── finger flags ──────────────────────────────────────────────────────────
    idx_up = is_up(raw, 8,  6)
    mid_up = is_up(raw, 12, 10)
    rng_up = is_up(raw, 16, 14)
    pnk_up = is_up(raw, 20, 18)

    d_left  = pdist(raw, 4, 8)
    d_right = pdist(raw, 4, 12)
    state.d_left  = d_left
    state.d_right = d_right

    pin_left  = d_left  < PINCH_THRESHOLD
    pin_right = d_right < PINCH_THRESHOLD

    # ── transition rules ───────────────────────────────────────────────────────

    # Enter SCROLL: index + middle up, ring + pinky down
    if idx_up and mid_up and not rng_up and not pnk_up:
        if state.mode != Mode.SCROLL:
            state.mode         = Mode.SCROLL
            state.scroll_ref_y = raw[8][1]
            state.scroll_counter = 0
        _do_scroll(ui, raw, state)
        _show_label(state, "SCROLL", now)
        return

    # Exit SCROLL when hand opens or closes fully
    if state.mode == Mode.SCROLL:
        state.mode        = Mode.MOVE
        state.scroll_ref_y = None

    # Enter FROZEN: any closing finger detected while in MOVE
    # (we detect "non-index finger approaching thumb" as the trigger)
    any_closing = (not mid_up or not rng_up or not pnk_up) and not pin_left and not pin_right
    # simpler: freeze as soon as a pinch starts OR a non-index finger curls down
    should_freeze = (pin_left or pin_right or
                     (not mid_up and idx_up) or
                     (not rng_up and idx_up) or
                     (not pnk_up and idx_up))

    if state.mode == Mode.MOVE:
        if should_freeze:
            # Lock cursor position right now
            state.freeze_x = state.cx
            state.freeze_y = state.cy
            state.mode     = Mode.FROZEN
            state.lpin_frames = state.rpin_frames = 0
            state.lpin_fired  = state.rpin_fired  = False
            state.lpin_start  = now

    # ── MOVE mode ─────────────────────────────────────────────────────────────
    if state.mode == Mode.MOVE:
        ix, iy = mapped[8]
        state.cx = int(state.cx + (ix - state.cx) / SMOOTHENING)
        state.cy = int(state.cy + (iy - state.cy) / SMOOTHENING)
        state.cx = max(0, min(state.cx, sw-1))
        state.cy = max(0, min(state.cy, sh-1))
        ui_move(ui, state.cx, state.cy)
        _show_label(state, "MOVE", now)

    # ── FROZEN / ACTION mode ───────────────────────────────────────────────────
    elif state.mode in (Mode.FROZEN, Mode.DRAG):
        # Keep cursor frozen at locked position
        ui_move(ui, state.freeze_x, state.freeze_y)

        # ── Left pinch → click or drag ─────────────────────────────────────
        if pin_left and not pin_right:
            state.lpin_frames += 1

            # Drag: held long enough without firing a click
            if (not state.drag_active and not state.lpin_fired
                    and now - state.lpin_start > DRAG_HOLD_SECS):
                ui_press(ui, e.BTN_LEFT)
                state.drag_active = True
                state.mode        = Mode.DRAG
                _show_label(state, "DRAG", now)

            # Click: confirmed frames, fires once per pinch
            if state.lpin_frames >= PINCH_CONFIRM_FRAMES and not state.lpin_fired:
                if not state.drag_active:
                    if now - state.lclick_at < 0.35:
                        ui_dblclick(ui, e.BTN_LEFT)
                        _show_label(state, "DBLCLICK", now)
                        state.lclick_at = 0
                    else:
                        ui_click(ui, e.BTN_LEFT)
                        _show_label(state, "L-CLICK", now)
                        state.lclick_at = now
                    state.lpin_fired = True
        else:
            # Pinch released
            if state.drag_active:
                ui_release(ui, e.BTN_LEFT)
                state.drag_active = False
                _show_label(state, "MOVE", now)
            state.lpin_frames = 0
            state.lpin_fired  = False
            state.lpin_start  = now   # reset drag timer

        # ── Right pinch ────────────────────────────────────────────────────
        if pin_right and not pin_left:
            state.rpin_frames += 1
            if (state.rpin_frames >= PINCH_CONFIRM_FRAMES
                    and not state.rpin_fired
                    and now - state.last_rclick > RCLICK_COOLDOWN):
                ui_click(ui, e.BTN_RIGHT)
                state.last_rclick = now
                state.rpin_fired  = True
                _show_label(state, "R-CLICK", now)
        else:
            state.rpin_frames = 0
            state.rpin_fired  = False

        # ── Exit FROZEN: all four fingers fully open → back to MOVE ────────
        all_open = idx_up and mid_up and rng_up and pnk_up
        if all_open and not state.drag_active:
            state.mode = Mode.MOVE
            # Snap stored cursor to current index position so there's no jump
            state.cx, state.cy = mapped[8]


def _do_scroll(ui, raw, state):
    if state.scroll_ref_y is None:
        return
    delta = state.scroll_ref_y - raw[8][1]   # positive = hand moved up
    state.scroll_counter += 1
    if state.scroll_counter >= SCROLL_INTERVAL:
        state.scroll_counter = 0
        if delta > 12:
            ui_scroll(ui, 1);  state.scroll_dir =  1
        elif delta < -12:
            ui_scroll(ui, -1); state.scroll_dir = -1
        else:
            state.scroll_dir = 0


def _show_label(state, label, now):
    if label == "MOVE" and state.label in ("L-CLICK","R-CLICK","DBLCLICK","DRAG"):
        return   # don't overwrite a recent action label immediately
    state.label = label
    if label != "MOVE":
        state.label_until = now + 0.9


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


def draw_active_zone(frame):
    h, w = frame.shape[:2]
    x1, y1 = int(w*FRAME_MARGIN), int(h*FRAME_MARGIN)
    x2, y2 = int(w*(1-FRAME_MARGIN)), int(h*(1-FRAME_MARGIN))
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
    cv2.putText(frame, "Active Zone", (x1+4, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)


def draw_hud(frame, state, now):
    # ── top-left: instructions ─────────────────────────────────────────────
    lines = [
        "MOVE  : only index finger up",
        "FREEZE: curl any other finger",
        "LClick: pinch thumb+index",
        "RClick: pinch thumb+middle",
        "Drag  : hold left-pinch 0.6s",
        "Scroll: index+middle up, move hand",
        "Resume: open all fingers",
    ]
    for i, l in enumerate(lines):
        cv2.putText(frame, l, (10, 20+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,220,0), 1, cv2.LINE_AA)

    # ── bottom-left: status panel ──────────────────────────────────────────
    h = frame.shape[0]
    y0 = h - 115
    cv2.rectangle(frame, (0, y0-10), (300, h), (0,0,0), -1)

    # Mode badge
    MODE_COLOR = {
        Mode.MOVE:   (180,180,180),
        Mode.FROZEN: (0,180,255),
        Mode.SCROLL: (0,200,255),
        Mode.DRAG:   (0,100,255),
    }
    badge_col = MODE_COLOR.get(state.mode, (200,200,200))
    cv2.putText(frame, f"Mode: {state.mode}", (8, y0+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_col, 2)

    # Gesture label (fades after label_until)
    if state.label != "MOVE" and now < state.label_until:
        cv2.putText(frame, state.label, (8, y0+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Pinch distance bars
    def bar(label, dist, row):
        ratio    = min(dist / PINCH_THRESHOLD, 2.0)
        bar_w    = int(100 * ratio)
        col      = (0,80,255) if dist < PINCH_THRESHOLD else (0,180,60)
        cv2.putText(frame, f"{label} {int(dist):3d}px", (8, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,200,200), 1)
        cv2.rectangle(frame, (135, row-11), (135+bar_w, row), col, -1)
        cv2.line(frame, (135+100, row-11), (135+100, row), (120,120,120), 1)

    bar("L-pinch(4↔8) ", state.d_left,  y0+68)
    bar("R-pinch(4↔12)", state.d_right, y0+90)

    if state.scroll_dir:
        cv2.putText(frame, "▲ SCROLL UP" if state.scroll_dir>0 else "▼ SCROLL DOWN",
                    (8, y0+112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)


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

    state = State()
    print("Press Esc in camera window to quit.")

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
            update(ui, raw, mapped, sw, sh, state)

        now = time.time()
        # expire gesture label
        if state.label != "MOVE" and now > state.label_until:
            state.label = "MOVE"

        draw_hud(frame, state, now)
        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ui.close()


if __name__ == "__main__":
    main()
