#!/usr/bin/env python3
# Visualize processed (64-frame) sequences with a toggle:
#  - PROC mode: normalized overlay at fixed anchor (no absolute placement)
#  - RAW mode : reconstruct to original pixel space using wrist (u,v) and scale
#
# Keys:
#   m : toggle RAW/PROC mode
#   q : quit

import numpy as np
import cv2

# ---------- Paths (edit as needed) ----------
RAW_FILE  = 'demo/gesture_4_3.npz'            # raw recording (variable length)
PROC_FILE = 'processed_demo/gesture_4_3.npz'  # preprocessed (fixed 64 frames)

# ---------- Drawing config ----------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]
ARM_CONNECTIONS = [(0, 1), (1, 2)]  # shoulder->elbow->wrist (RIGHT)

DRAW_SCALE_PROC = 220          # visual scale for PROC mode
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------- Load data ----------
raw = np.load(RAW_FILE, allow_pickle=True)
rgb_frames = raw['rgb']              # (T_raw, H, W, 3) BGR

proc = np.load(PROC_FILE, allow_pickle=True)
hand_seq  = proc['landmarks']        # (64, 21, 3) normalized about wrist
arm_seq   = proc['arm_landmarks'] if 'arm_landmarks' in proc.files else None
wrist_seq = proc['wrist_uvz']   if 'wrist_uvz'     in proc.files else None  # (64, 3): (u_px, v_px, z_m)
scale_seq = proc['scale']       if 'scale'         in proc.files else None  # (64,)
label     = proc['label']

# Valid mask compatibility
if 'valid_T' in proc.files:
    valid_seq = proc['valid_T'].astype(bool)
elif 'valid' in proc.files:
    valid_seq = proc['valid'].astype(bool)
elif 'valid_raw' in proc.files:
    vr = proc['valid_raw'].astype(bool)
    T_proc = hand_seq.shape[0]
    if len(vr) != T_proc:
        idx = np.round(np.linspace(0, len(vr)-1, T_proc)).astype(int)
        valid_seq = vr[idx]
    else:
        valid_seq = vr
else:
    valid_seq = np.ones((hand_seq.shape[0],), dtype=bool)

T_proc = hand_seq.shape[0]  # should be 64
T_raw  = len(rgb_frames)

# ---------- Modes ----------
MODE = 'RAW'  # 'PROC' (default) or 'RAW'

def map_proc_to_raw_idx(t, T_proc, T_raw):
    if T_raw < 2: return 0
    idx = int(round(t * (T_raw - 1) / max(T_proc - 1, 1)))
    return max(0, min(T_raw - 1, idx))

def draw_hand_and_arm_PROC(frame, hand_norm, arm_norm, valid, anchor, draw_scale):
    """Draw normalized landmarks at a fixed anchor (no absolute placement)."""
    h, w, _ = frame.shape
    cx, cy = anchor
    # Colors (dim if invalid)
    hand_line_color  = (0, 255, 0) if valid else (60, 120, 60)
    hand_point_color = (0, 0, 255) if valid else (60, 60, 120)
    arm_line_color   = (255, 255, 0) if valid else (120, 120, 60)
    arm_point_color  = (0, 165, 255) if valid else (90, 90, 60)
    text_color       = (255, 255, 255) if valid else (180, 180, 180)

    # Hand
    pts = []
    for (x_n, y_n, z_n) in hand_norm:
        px = int(cx + x_n * draw_scale)
        py = int(cy + y_n * draw_scale)
        pts.append((px, py, z_n))
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s][:2], pts[e][:2], hand_line_color, 2)
    for (px, py, z_n) in pts:
        cv2.circle(frame, (px, py), 4, hand_point_color, -1)
        # You can annotate z if desired:
        # cv2.putText(frame, f'{z_n:.2f}', (px+5, py-5), FONT, 0.4, text_color, 1)

    # Arm (optional)
    if arm_norm is not None:
        apts = []
        for (x_n, y_n, z_n) in arm_norm:
            px = int(cx + x_n * draw_scale)
            py = int(cy + y_n * draw_scale)
            apts.append((px, py, z_n))
        for s, e in ARM_CONNECTIONS:
            cv2.line(frame, apts[s][:2], apts[e][:2], arm_line_color, 2)
        for (px, py, z_n) in apts:
            cv2.circle(frame, (px, py), 5, arm_point_color, -1)

def draw_hand_and_arm_RAW(frame, hand_norm, arm_norm, valid, wrist_uvz, scale):
    """Reconstruct to original pixel space: px = wrist + norm * scale."""
    h, w, _ = frame.shape
    # Colors (dim if invalid)
    hand_line_color  = (0, 255, 0) if valid else (60, 120, 60)
    hand_point_color = (0, 0, 255) if valid else (60, 60, 120)
    arm_line_color   = (255, 255, 0) if valid else (120, 120, 60)
    arm_point_color  = (0, 165, 255) if valid else (90, 90, 60)

    # Wrist and scale
    if wrist_uvz is None or scale is None or not np.isfinite(scale):
        return
    u, v, _ = wrist_uvz  # (pixels, pixels, meters)
    if not (np.isfinite(u) and np.isfinite(v)):
        return
    cx, cy = int(round(u)), int(round(v))
    s = float(scale)

    # Hand in absolute pixels
    pts = []
    for (x_n, y_n, z_n) in hand_norm:
        px = int(round(cx + x_n * s))
        py = int(round(cy + y_n * s))
        pts.append((px, py, z_n))
    for sidx, eidx in HAND_CONNECTIONS:
        cv2.line(frame, pts[sidx][:2], pts[eidx][:2], hand_line_color, 2)
    for (px, py, _) in pts:
        cv2.circle(frame, (px, py), 4, hand_point_color, -1)

    # Arm in absolute pixels (if provided)
    if arm_norm is not None:
        apts = []
        for (x_n, y_n, z_n) in arm_norm:
            px = int(round(cx + x_n * s))
            py = int(round(cy + y_n * s))
            apts.append((px, py, z_n))
        for sidx, eidx in ARM_CONNECTIONS:
            cv2.line(frame, apts[sidx][:2], apts[eidx][:2], arm_line_color, 2)
        for (px, py, _) in apts:
            cv2.circle(frame, (px, py), 5, arm_point_color, -1)

# ---------- Playback ----------
t = 0
while True:
    # Map processed index to raw for the background
    raw_idx = map_proc_to_raw_idx(t, T_proc, T_raw)
    frame = rgb_frames[raw_idx].copy()
    H, W, _ = frame.shape

    # Per-frame data
    hand_norm = hand_seq[t]                    # (21,3)
    arm_norm  = arm_seq[t] if arm_seq is not None else None  # (3,3) or None
    valid     = bool(valid_seq[t])
    wrist_uvz = wrist_seq[t] if wrist_seq is not None else None
    scale     = scale_seq[t] if scale_seq is not None else None

    # Draw
    if MODE == 'PROC':
        anchor = (W // 2, int(H * 0.75))  # fixed location
        draw_hand_and_arm_PROC(frame, hand_norm, arm_norm, valid, anchor, DRAW_SCALE_PROC)
        mode_text = "PROC: normalized @ fixed anchor"
    else:  # RAW
        draw_hand_and_arm_RAW(frame, hand_norm, arm_norm, valid, wrist_uvz, scale)
        mode_text = "RAW : absolute @ wrist(u,v) with original scale"

    # HUD
    lbl = str(label.item() if hasattr(label, 'item') else label)
    cv2.putText(frame, f'Gesture: {lbl}  Frame: {t+1}/{T_proc}  Mode: {mode_text}',
                (10, 24), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'm' to toggle RAW/PROC, 'q' to quit",
                (10, 48), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.imshow('Processed overlay viewer', frame)
    key = cv2.waitKey(60) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        MODE = 'RAW' if MODE == 'PROC' else 'PROC'

    # Advance frame
    t = (t + 1) % T_proc

cv2.destroyAllWindows()
