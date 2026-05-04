# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the application

```bash
# Set up virtual environment (first time)
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run
python mouse.py
```

The app auto-downloads `hand_landmarker.task` (~28 MB) on first run if it's missing.

## Linux permissions requirement

The app uses `evdev.UInput` to create a virtual input device. If `/dev/uinput` is not accessible:

```bash
sudo modprobe uinput && sudo usermod -aG input $USER
newgrp input  # or re-login
```

## Architecture

Everything lives in `mouse.py` — a single-file application with no tests or build step.

**Execution flow:**
1. `main()` initializes webcam, MediaPipe `HandLandmarker`, a `UInput` virtual mouse device, and a `State` object, then enters the frame loop.
2. Each frame: `process_frame()` flips and converts color → MediaPipe detects hand landmarks → `lm_px()` converts normalized landmarks to pixel coords → `map_screen()` maps the active zone to screen coordinates → `update()` applies gesture logic and fires uinput events → drawing functions render the HUD.

**Key design decisions:**
- `State` class holds all mutable gesture state (smoothed cursor position, fist/drag/rclick timers, scroll direction, HUD label).
- `update()` in `mouse.py:182` is the gesture state machine — scroll gestures take priority and return early, preventing cursor/click processing while scrolling.
- Cursor smoothing uses exponential approach: `cx += (target - cx) / SMOOTHENING` each frame.
- Finger "up" detection (`finger_up()`) compares fingertip Y vs MCP knuckle Y with a 15px margin — uses MCP (not PIP) for a larger, more reliable gap.
- `CONFIRM_FRAMES` debounces gesture recognition; `RCLICK_COOLDOWN` prevents right-click spam.

## Gesture map

| Gesture | Fingers up | Action |
|---|---|---|
| MOVE | Index only | Cursor follows index tip |
| SCROLL DOWN | Index + Middle (✌) | Scroll down at fixed rate |
| SCROLL UP | Index + Middle + Ring | Scroll up at fixed rate |
| LEFT CLICK | Fist (none) | Click; hold >0.6s = drag |
| DBL-CLICK | Fist twice <0.4s apart | Double-click |
| RIGHT CLICK | Pinky only | Right click (1.5s cooldown) |

## Tuning constants (`mouse.py:23-30`)

| Constant | Default | Effect |
|---|---|---|
| `FRAME_MARGIN` | 0.15 | Active-zone margin; increase if cursor can't reach screen edges |
| `SMOOTHENING` | 7 | Higher = smoother cursor but more lag |
| `CONFIRM_FRAMES` | 4 | Frames a gesture must be stable before firing |
| `DRAG_HOLD_SECS` | 0.6 | Hold time before fist becomes drag |
| `RCLICK_COOLDOWN` | 1.5 | Minimum seconds between right clicks |
| `SCROLL_INTERVAL` | 5 | Frames between scroll ticks; increase to slow scrolling |
