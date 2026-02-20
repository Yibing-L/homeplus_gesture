#!/usr/bin/env python3
# live_holistic_right_only.py — RealSense + MediaPipe Holistic
# Right arm + Right hand only, per-landmark depth labels, wrist-anchored normalization.

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import time

# ---------------- User toggles ----------------
POSE_VIS_THRESH    = 0.40
BRIGHTEN_ALPHA     = 1.0   # 1.0–1.5 contrast
BRIGHTEN_BETA      = 0     # 0–50 brightness
LABEL_LANDMARKS    = [0, 4, 8, 12, 16, 20]  # hand landmark ids to label (reduce clutter)
FPS                = 30
W, H               = 640, 480

# ---------------- RealSense setup ------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# ---------------- MediaPipe Holistic ---------
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils

hol = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,       # 0/1/2 (1 is a good live tradeoff)
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,  # we don't need face refinement
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------- Utilities (same spirit as your scripts) ----------
def get_depth_m(depth_img, x, y):
    """Depth in meters with small median fill if zero (mm→m)."""
    h, w = depth_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    d = depth_img[y, x]
    if d == 0:
        y0, y1 = max(0, y-2), min(h, y+3)
        x0, x1 = max(0, x-2), min(w, x+3)
        nz = depth_img[y0:y1, x0:x1]
        nz = nz[nz > 0]
        if nz.size == 0:
            return None
        d = int(np.median(nz))
    return d * 0.001  # mm → m

def hand_scale_from_landmarks(hand21_px):
    """Stable scale: max(wrist→index-M distance, tight box size)."""
    wrist = hand21_px[0, :2]
    idxM  = hand21_px[5, :2]
    d     = np.linalg.norm(idxM - wrist)
    xmin, ymin = hand21_px[:,0].min(), hand21_px[:,1].min()
    xmax, ymax = hand21_px[:,0].max(), hand21_px[:,1].max()
    box   = max(xmax - xmin, ymax - ymin)
    return float(max(d, box, 1e-6))

# Pose landmark enum shortcuts
PL = mp_holistic.PoseLandmark
R_SHOULDER = PL.RIGHT_SHOULDER.value
R_ELBOW    = PL.RIGHT_ELBOW.value
R_WRIST    = PL.RIGHT_WRIST.value

# ---------------- Main loop ------------------
fps_t0, fps_cnt = time.time(), 0
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            continue

        color = np.asanyarray(cf.get_data())    # BGR, uint8
        depth = np.asanyarray(df.get_data())    # uint16 (mm)

        if BRIGHTEN_ALPHA != 1.0 or BRIGHTEN_BETA != 0:
            color = cv2.convertScaleAbs(color, alpha=BRIGHTEN_ALPHA, beta=BRIGHTEN_BETA)

        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        # ---- Holistic inference (pose + hands in one pass) ----
        res = hol.process(rgb)

        # ---- RIGHT ARM (draw only shoulder→elbow→wrist) ----
        right_wrist_px = None
        arm_xyz = None
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            def px(id_):
                return int(lm[id_].x * W), int(lm[id_].y * H), lm[id_].visibility

            sx, sy, svis = px(R_SHOULDER)
            ex, ey, evis = px(R_ELBOW)
            wx, wy, wvis = px(R_WRIST)
            right_wrist_px = (wx, wy)

            # draw the right arm only
            if svis > POSE_VIS_THRESH and evis > POSE_VIS_THRESH:
                cv2.line(color, (sx, sy), (ex, ey), (0, 255, 255), 2)
            if evis > POSE_VIS_THRESH and wvis > POSE_VIS_THRESH:
                cv2.line(color, (ex, ey), (wx, wy), (0, 255, 255), 2)

            # depth annotate these joints
            for tag, (x, y, vis) in zip(["Sh","El","Wr"], [(sx,sy,svis),(ex,ey,evis),(wx,wy,wvis)]):
                if vis > POSE_VIS_THRESH:
                    d_m = get_depth_m(depth, x, y)
                    if d_m is not None:
                        cv2.circle(color, (x, y), 4, (0, 165, 255), -1)
                        cv2.putText(color, f'R-{tag}:{d_m:.2f}m', (x+6, y-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1, cv2.LINE_AA)

            # arm XYZ for normalization (depth at joints)
            if svis > POSE_VIS_THRESH and evis > POSE_VIS_THRESH and wvis > POSE_VIS_THRESH:
                sz = get_depth_m(depth, sx, sy)
                ez = get_depth_m(depth, ex, ey)
                wz = get_depth_m(depth, wx, wy)
                if (sz is not None) and (ez is not None) and (wz is not None):
                    arm_xyz = np.array([[sx, sy, sz],
                                        [ex, ey, ez],
                                        [wx, wy, wz]], dtype=np.float32)

        # ---- RIGHT HAND (Holistic exposes right_hand_landmarks directly) ----
        if res.right_hand_landmarks:
            hand_lm = res.right_hand_landmarks  # 21 landmarks
            # draw skeleton
            mp_draw.draw_landmarks(color, hand_lm, mp_holistic.HAND_CONNECTIONS)

            # raw px (+z from MP)
            hand_px = np.array([[lm.x * W, lm.y * H, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
            wrist_px = hand_px[0, :2]
            scale    = hand_scale_from_landmarks(hand_px)

            # normalized (wrist-anchored, scale in px) — same logic you used
            hand_norm = np.zeros((21,3), dtype=np.float32)
            hand_norm[:, :2] = (hand_px[:, :2] - wrist_px[None, :]) / max(scale, 1e-6)
            hand_norm[:,  2] = hand_px[:, 2]

            arm_norm = None
            if arm_xyz is not None:
                arm_norm = np.zeros((3,3), dtype=np.float32)
                arm_norm[:, :2] = (arm_xyz[:, :2] - wrist_px[None, :]) / max(scale, 1e-6)
                arm_norm[:,  2] = arm_xyz[:, 2]

            # per-landmark depth labels (meters)
            for k in LABEL_LANDMARKS:
                x_px = int(round(hand_px[k,0]))
                y_px = int(round(hand_px[k,1]))
                d_m  = get_depth_m(depth, x_px, y_px)
                if d_m is None:
                    continue
                cv2.circle(color, (x_px, y_px), 2, (255, 0, 0), -1)
                cv2.putText(color, f'{k}:{d_m:.2f}m', (x_px+4, y_px-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1, cv2.LINE_AA)

            # wrist depth + scale summary
            wz = get_depth_m(depth, int(round(wrist_px[0])), int(round(wrist_px[1])))
            if wz is not None:
                cv2.putText(color, f'Wrist:{wz:.2f}m  Scale:{scale:.1f}px',
                            (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,220,50), 1, cv2.LINE_AA)

        # ---- FPS overlay ----
        fps_cnt += 1
        now = time.time()
        if now - fps_t0 > 0.5:
            fps = fps_cnt / (now - fps_t0)
            fps_t0, fps_cnt = now, 0
            cv2.putText(color, f'FPS: {fps:.1f}', (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow('Holistic: RIGHT arm + RIGHT hand (with depth)', color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    hol.close()
