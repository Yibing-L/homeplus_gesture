#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
landmark_with_xyz.py — Offline RGB-D -> XYZ gesture features (.npz) for Home+.

Key differences from landmark.py:
  - Looks up RealSense depth at ALL 21 hand joints (not just wrist)
  - Deprojects every joint to metric 3D (X,Y,Z in meters) via pyrealsense2
  - Normalizes in XYZ space: wrist-relative + metric bone-length scale
  - Saves hand_uvd (t, 21, 3) so you never lose the raw per-joint data again
  - Requires real camera intrinsics (--from-camera or --fx/fy/cx/cy)

Outputs per clip:
  X, valid_T, label, subject_id,
  hand_xyz_m, hand_xyz_norm, hand_uvd,
  arm_xyz_m, arm_xyz_norm, arm_uvd, wrist_xyz,
  scale_m, depth_scale, valid_raw, valid_joint, valid_arm_joint,
  intrinsics, intrinsics_model, intrinsics_model_str, intrinsics_coeffs,
  T_target, coord_system

Usage:
  python landmark_with_xyz.py --input /path/to/raw/ --output /path/to/processed/ --from-camera
  python landmark_with_xyz.py --input /path/to/raw/ --output /path/to/processed/ --fx 615 --fy 616 --cx 318 --cy 245
"""

import os, sys, re, argparse
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
from contextlib import contextmanager

# ── Config ──
POSE_VIS_THRESH = 0.40
T_TARGET = 64

INCLUDE_ARM = True
INCLUDE_VELOCITY = True
NORMALIZE_VELOCITY = True
INCLUDE_SCALE_CH = True

CLIP_FEATURES_ABS = 50.0
EPS = 1e-6
HAND_MIN_VALID = 6  # minimum joints with valid depth to treat frame as valid

STREAM_W = 640
STREAM_H = 480

# ── Args ──
def parse_args():
    ap = argparse.ArgumentParser(
        description="Preprocess raw RGB-D NPZ files to XYZ gesture features."
    )
    ap.add_argument("--input", required=True,
                    help="Root dir with integer-named participant subfolders of raw .npz files")
    ap.add_argument("--output", required=True,
                    help="Output root dir (mirrors input structure)")
    ap.add_argument("--fx", type=float, default=None, help="Focal length x")
    ap.add_argument("--fy", type=float, default=None, help="Focal length y")
    ap.add_argument("--cx", type=float, default=None, help="Principal point x")
    ap.add_argument("--cy", type=float, default=None, help="Principal point y")
    ap.add_argument("--from-camera", action="store_true",
                    help="Query intrinsics from connected RealSense — must be the EXACT device used for recording")
    ap.add_argument("--depth-scale", type=float, default=0.001,
                    help="Multiply raw depth values by this to get meters (default 0.001 for Z16/uint16 mm)")
    return ap.parse_args()


# ── Intrinsics ──
def get_intrinsics_from_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        print(f"[Camera] Queried intrinsics:")
        print(f"  fx={intr.fx:.4f}  fy={intr.fy:.4f}")
        print(f"  cx={intr.ppx:.4f}  cy={intr.ppy:.4f}")
        print(f"  model={intr.model}  coeffs={intr.coeffs}")
        return intr
    finally:
        pipeline.stop()


def make_intrinsics(fx, fy, cx, cy):
    intr = rs.intrinsics()
    intr.width = STREAM_W
    intr.height = STREAM_H
    intr.fx = float(fx)
    intr.fy = float(fy)
    intr.ppx = float(cx)
    intr.ppy = float(cy)
    intr.model = rs.distortion.none
    intr.coeffs = [0.0] * 5
    return intr


def deproject(intr, u, v, z):
    return rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z))


# ── MediaPipe Holistic ──
mp_holistic = mp.solutions.holistic

PL = mp_holistic.PoseLandmark
R_SHOULDER = PL.RIGHT_SHOULDER.value
R_ELBOW = PL.RIGHT_ELBOW.value
R_WRIST = PL.RIGHT_WRIST.value


# ── Utilities ──
def get_depth_m(depth_img, x, y, depth_scale):
    h, w = depth_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    d = depth_img[y, x]
    if d == 0:
        y0, y1 = max(0, y - 2), min(h, y + 3)
        x0, x1 = max(0, x - 2), min(w, x + 3)
        neigh = depth_img[y0:y1, x0:x1]
        nz = neigh[neigh > 0]
        if nz.size == 0:
            return None
        d = int(np.median(nz))
    return float(d) * depth_scale


def metric_hand_scale(hand_xyz_wrist_rel, joint_valid):
    """Stable scale in meters: average of available wrist→MCP distances (joints 5, 9, 17)."""
    dists = []
    for j in [5, 9, 17]:
        if joint_valid[j]:
            d = float(np.linalg.norm(hand_xyz_wrist_rel[j]))
            if np.isfinite(d) and d > EPS:
                dists.append(d)
    return float(max(np.mean(dists) if dists else EPS, EPS))


def resample_2d(arr, T):
    t = arr.shape[0]
    if t == T:
        return arr.astype(np.float32)
    if t < 2:
        return np.repeat(arr.astype(np.float32), T, axis=0)
    idx = np.linspace(0, t - 1, T)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, t - 1)
    w = (idx - lo)[:, None]
    return ((1 - w) * arr[lo] + w * arr[hi]).astype(np.float32)


def resample_mask(mask_bool, T):
    m = mask_bool.astype(np.float32)[:, None]
    return resample_2d(m, T).squeeze() > 0.5


def make_velocity(x):
    return np.diff(x, axis=0, prepend=x[0:1])


def normalize_velocity_per_clip(dx):
    mag = np.sqrt((dx * dx).sum(axis=1) + EPS)
    m = float(mag.mean())
    return dx if m < EPS else (dx / m)


class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)
    def flush(self):
        for st in self.streams:
            st.flush()


@contextmanager
def _tee_terminal_output(log_txt_path):
    os.makedirs(os.path.dirname(log_txt_path), exist_ok=True)
    with open(log_txt_path, 'w', encoding='utf-8') as f:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _TeeIO(orig_out, f)
        sys.stderr = _TeeIO(orig_err, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err


def _list_participant_dirs(base_dir):
    out = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        # Accept plain integers ("0", "1", ...) or "{N}_done" ("0_done", "2_done", ...)
        m = re.fullmatch(r"(\d+)(?:_done)?", name)
        if m:
            out.append((int(m.group(1)), name, full))
    out.sort(key=lambda x: x[0])
    return out


# ── Main ──
def main():
    args = parse_args()

    # Resolve intrinsics
    if args.from_camera:
        print("[WARN] --from-camera: intrinsics are only correct if the connected device")
        print("       is the EXACT same unit used during data collection. Using a different")
        print("       camera (even same model) will inject systematic XYZ errors.")
        intr = get_intrinsics_from_camera()
    elif all(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        intr = make_intrinsics(args.fx, args.fy, args.cx, args.cy)
        print(f"[Manual] Intrinsics: fx={intr.fx:.4f} fy={intr.fy:.4f} "
              f"cx={intr.ppx:.4f} cy={intr.ppy:.4f} (no distortion)")
    else:
        print("ERROR: You must either:")
        print("  --from-camera               (plug in the RealSense used for collection)")
        print("  --fx F --fy F --cx F --cy F  (provide all four intrinsic values)")
        sys.exit(1)

    print(f"[INFO] depth_scale={args.depth_scale} "
          f"({'default mm->m' if args.depth_scale == 0.001 else 'CUSTOM'})")

    depth_scale = args.depth_scale
    intr_array = np.array([intr.fx, intr.fy, intr.ppx, intr.ppy], dtype=np.float32)
    intr_model = int(intr.model)   # integer enum value (e.g. 4 = brown_conrady)
    intr_model_str = str(intr.model)  # human-readable string (e.g. "distortion.brown_conrady")
    intr_coeffs = np.array(intr.coeffs, dtype=np.float32)

    hol = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    input_root = args.input
    output_root = args.output

    participant_dirs = _list_participant_dirs(input_root)
    if not participant_dirs:
        print(f"[WARN] No integer-named participant folders found in {input_root}")
        hol.close()
        sys.exit(0)

    for participant_id, participant_name, DATA_DIR in participant_dirs:
        OUTPUT_DIR = os.path.join(output_root, participant_name)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        log_path = os.path.join(OUTPUT_DIR, "terminal_output.txt")
        with _tee_terminal_output(log_path):
            files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
            if not files:
                print(f"[WARN] No .npz in {DATA_DIR}")
                continue

            processed_files = 0
            skipped_files = 0
            per_file_stats = []
            label_counts = {}
            total_frames = 0
            total_valid_frames = 0

            for file in files:
                path = os.path.join(DATA_DIR, file)
                data = np.load(path, allow_pickle=True)
                if not all(k in data for k in ['rgb', 'depth', 'label']):
                    print(f"[WARN] Skipping {file}: missing one of ['rgb','depth','label']")
                    skipped_files += 1
                    continue

                rgb_frames = data['rgb']
                depth_frames = data['depth']
                label = int(data['label'])

                # Per-frame accumulators
                hand_uvd_seq = []      # (21, 3) per frame: u_px, v_px, depth_m
                hand_xyz_m_seq = []    # (21, 3) wrist-relative XYZ in meters
                hand_xyz_norm_seq = [] # (21, 3) wrist-relative, scale-normalized (unitless)
                arm_uvd_seq = []       # (3, 3) per frame: u_px, v_px, depth_m
                arm_xyz_m_seq = []     # (3, 3) wrist-relative XYZ in meters
                arm_xyz_norm_seq = []  # (3, 3) wrist-relative, scale-normalized (unitless)
                wrist_xyz_seq = []     # (3,) per frame: absolute wrist XYZ meters
                scale_m_seq = []       # metric scale per frame
                valid_seq = []            # frame-level validity
                valid_joint_seq = []      # (21,) per-joint depth validity (hand)
                valid_arm_joint_seq = []  # (3,)  per-joint depth validity (arm)

                for rgb, depth in zip(rgb_frames, depth_frames):
                    image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    H, W = rgb.shape[:2]

                    res = hol.process(image_rgb)

                    # ── Arm joints (shoulder, elbow, wrist) — per-joint ──
                    arm_uvd = np.full((3, 3), np.nan, dtype=np.float32)
                    arm_joint_valid = np.zeros(3, dtype=np.uint8)
                    if res.pose_landmarks:
                        lm = res.pose_landmarks.landmark

                        def px(idx):
                            x = int(np.clip(lm[idx].x * W, 0, W - 1))
                            y = int(np.clip(lm[idx].y * H, 0, H - 1))
                            return x, y, lm[idx].visibility

                        for ji, mp_idx in enumerate([R_SHOULDER, R_ELBOW, R_WRIST]):
                            jx, jy, jvis = px(mp_idx)
                            if jvis > POSE_VIS_THRESH:
                                dz = get_depth_m(depth, jx, jy, depth_scale)
                                if dz is not None:
                                    arm_uvd[ji] = [jx, jy, dz]
                                    arm_joint_valid[ji] = 1

                    # ── Hand landmarks ──
                    chosen = res.right_hand_landmarks if hasattr(res, 'right_hand_landmarks') else None

                    hand_uvd = np.full((21, 3), np.nan, dtype=np.float32)
                    hand_xyz_m = np.zeros((21, 3), dtype=np.float32)
                    hand_xyz_norm = np.zeros((21, 3), dtype=np.float32)
                    arm_xyz_m = np.zeros((3, 3), dtype=np.float32)
                    arm_xyz_norm = np.zeros((3, 3), dtype=np.float32)
                    wrist_xyz = np.full(3, np.nan, dtype=np.float32)
                    scale_m = np.nan
                    joint_valid = np.zeros(21, dtype=np.uint8)

                    if chosen is not None:
                        # Get pixel positions + depth for all 21 hand joints
                        for j, jlm in enumerate(chosen.landmark):
                            u_j = jlm.x * W
                            v_j = jlm.y * H
                            u_px = min(max(int(round(u_j)), 0), W - 1)
                            v_px = min(max(int(round(v_j)), 0), H - 1)
                            dz = get_depth_m(depth, u_px, v_px, depth_scale)
                            hand_uvd[j, 0] = u_j
                            hand_uvd[j, 1] = v_j
                            if dz is not None:
                                hand_uvd[j, 2] = dz
                                joint_valid[j] = 1

                        # Wrist must have depth to proceed
                        wrist_u, wrist_v, wrist_d = hand_uvd[0]
                        wrist_u_dep = float(np.clip(wrist_u, 0, W - 1))
                        wrist_v_dep = float(np.clip(wrist_v, 0, H - 1))
                        if np.isfinite(wrist_d) and wrist_d > 0 and joint_valid.sum() >= HAND_MIN_VALID:
                            wrist_xyz[:] = deproject(intr, wrist_u_dep, wrist_v_dep, wrist_d)

                            # Deproject all hand joints, make wrist-relative (meters)
                            for j in range(21):
                                if joint_valid[j]:
                                    u_dep = float(np.clip(hand_uvd[j, 0], 0, W - 1))
                                    v_dep = float(np.clip(hand_uvd[j, 1], 0, H - 1))
                                    j_xyz = np.array(deproject(
                                        intr, u_dep, v_dep, hand_uvd[j, 2]
                                    ), dtype=np.float32)
                                    hand_xyz_m[j] = j_xyz - wrist_xyz

                            # Metric scale: fallback chain (MCP avg → joint 9 → joint 5)
                            if any(joint_valid[j] for j in [5, 9, 17]):
                                scale_m = metric_hand_scale(hand_xyz_m, joint_valid)
                            elif joint_valid[9]:
                                scale_m = float(max(np.linalg.norm(hand_xyz_m[9]), EPS))
                            elif joint_valid[5]:
                                scale_m = float(max(np.linalg.norm(hand_xyz_m[5]), EPS))

                            if not (np.isfinite(scale_m) and scale_m > EPS):
                                # No usable scale — treat as invalid frame
                                pass
                            else:
                                # Scale-normalize and zero out missing joints
                                hand_xyz_norm = hand_xyz_m / scale_m
                                hand_xyz_norm[joint_valid == 0] = 0.0

                                # Deproject each arm joint independently (no all-or-nothing gate)
                                for j in range(3):
                                    if arm_joint_valid[j]:
                                        u_dep = float(np.clip(arm_uvd[j, 0], 0, W - 1))
                                        v_dep = float(np.clip(arm_uvd[j, 1], 0, H - 1))
                                        j_xyz = np.array(deproject(
                                            intr, u_dep, v_dep, arm_uvd[j, 2]
                                        ), dtype=np.float32)
                                        arm_xyz_m[j] = j_xyz - wrist_xyz
                                arm_xyz_norm = arm_xyz_m / scale_m
                                arm_xyz_norm[arm_joint_valid == 0] = 0.0

                                hand_uvd_seq.append(hand_uvd)
                                hand_xyz_m_seq.append(hand_xyz_m)
                                hand_xyz_norm_seq.append(hand_xyz_norm)
                                arm_uvd_seq.append(arm_uvd)
                                arm_xyz_m_seq.append(arm_xyz_m)
                                arm_xyz_norm_seq.append(arm_xyz_norm)
                                wrist_xyz_seq.append(wrist_xyz)
                                scale_m_seq.append(scale_m)
                                valid_seq.append(True)
                                valid_joint_seq.append(joint_valid)
                                valid_arm_joint_seq.append(arm_joint_valid)
                                continue

                    # Invalid frame (no hand or no wrist depth)
                    hand_uvd_seq.append(hand_uvd)
                    hand_xyz_m_seq.append(hand_xyz_m)
                    hand_xyz_norm_seq.append(hand_xyz_norm)
                    arm_uvd_seq.append(arm_uvd)
                    arm_xyz_m_seq.append(arm_xyz_m)
                    arm_xyz_norm_seq.append(arm_xyz_norm)
                    wrist_xyz_seq.append(wrist_xyz)
                    scale_m_seq.append(np.nan)
                    valid_seq.append(False)
                    valid_joint_seq.append(joint_valid)
                    valid_arm_joint_seq.append(arm_joint_valid)

                # ── Convert to arrays ──
                hand_uvd_arr = np.asarray(hand_uvd_seq, dtype=np.float32)         # (t, 21, 3)
                hand_xyz_m_arr = np.asarray(hand_xyz_m_seq, dtype=np.float32)     # (t, 21, 3) meters
                hand_xyz_norm_arr = np.asarray(hand_xyz_norm_seq, dtype=np.float32) # (t, 21, 3) unitless
                arm_uvd_arr = np.asarray(arm_uvd_seq, dtype=np.float32)           # (t, 3, 3)
                arm_xyz_m_arr = np.asarray(arm_xyz_m_seq, dtype=np.float32)       # (t, 3, 3) meters
                arm_xyz_norm_arr = np.asarray(arm_xyz_norm_seq, dtype=np.float32) # (t, 3, 3) unitless
                wrist_xyz_arr = np.asarray(wrist_xyz_seq, dtype=np.float32)       # (t, 3)
                scale_m_arr = np.asarray(scale_m_seq, dtype=np.float32)           # (t,)
                valid_arr = np.asarray(valid_seq, dtype=bool)                         # (t,)
                valid_joint_arr = np.asarray(valid_joint_seq, dtype=np.uint8)         # (t, 21)
                valid_arm_joint_arr = np.asarray(valid_arm_joint_seq, dtype=np.uint8) # (t, 3)

                # ── Build feature matrix X (uses scale-normalized coords) ──
                if INCLUDE_ARM:
                    pose_feat = np.concatenate([hand_xyz_norm_arr, arm_xyz_norm_arr], axis=1)
                else:
                    pose_feat = hand_xyz_norm_arr

                t_len, J, C = pose_feat.shape
                pose_2d = pose_feat.reshape(t_len, J * C).astype(np.float32)

                scale_col = scale_m_arr[:, None] if INCLUDE_SCALE_CH else None

                # Resample to T_TARGET
                pose_T = resample_2d(pose_2d, T_TARGET)
                wrist_xyz_T = resample_2d(
                    np.nan_to_num(wrist_xyz_arr, nan=0.0), T_TARGET
                )
                valid_T = resample_mask(valid_arr, T_TARGET)
                scale_T = None
                if scale_col is not None:
                    scale_T = resample_2d(
                        np.nan_to_num(scale_col, nan=0.0), T_TARGET
                    )

                # Validity masks resampled to T_TARGET then re-thresholded to binary {0,1}
                valid_joint_T = (resample_2d(
                    valid_joint_arr.astype(np.float32), T_TARGET
                ) > 0.5).astype(np.float32)  # (T, 21)
                valid_arm_joint_T = (resample_2d(
                    valid_arm_joint_arr.astype(np.float32), T_TARGET
                ) > 0.5).astype(np.float32)  # (T, 3)

                # Velocity (valid-aware)
                feat_chunks = [pose_T]
                if INCLUDE_VELOCITY:
                    d_pose = make_velocity(pose_T)
                    d_wrist = make_velocity(wrist_xyz_T)  # (m/frame, m/frame, m/frame)
                    bad_now = ~valid_T
                    bad_prev = np.concatenate([[bad_now[0]], bad_now[:-1]])
                    bad = (bad_now | bad_prev)[:, None]
                    d_pose[bad.repeat(d_pose.shape[1], axis=1)] = 0.0
                    d_wrist[bad.repeat(3, axis=1)] = 0.0
                    if NORMALIZE_VELOCITY:
                        d_pose = normalize_velocity_per_clip(d_pose)
                        d_wrist = normalize_velocity_per_clip(d_wrist)
                    feat_chunks.extend([d_pose, d_wrist])

                if scale_T is not None:
                    feat_chunks.append(scale_T)

                # Validity channels: frame-level, per-joint, coverage scalars
                feat_chunks.append(valid_T.astype(np.float32)[:, None])        # (T, 1)
                feat_chunks.append(valid_joint_T)                               # (T, 21)
                feat_chunks.append(valid_arm_joint_T)                          # (T, 3)
                feat_chunks.append(valid_joint_T.mean(axis=1, keepdims=True))  # (T, 1) hand coverage
                feat_chunks.append(valid_arm_joint_T.mean(axis=1, keepdims=True))  # (T, 1) arm coverage

                X = np.concatenate(feat_chunks, axis=1).astype(np.float32)
                # Clip only continuous channels (pose, velocity, scale); validity masks are already {0,1}
                # Validity cols appended last: valid_T(1) + valid_joint(21) + valid_arm(3) + cov_hand(1) + cov_arm(1) = 27
                n_validity_cols = 1 + 21 + 3 + 1 + 1
                continuous_dim = X.shape[1] - n_validity_cols
                X[:, :continuous_dim] = np.clip(X[:, :continuous_dim], -CLIP_FEATURES_ABS, CLIP_FEATURES_ABS)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                # ── Save ──
                out_name = re.sub(r"\.npz$", "", file)
                out_path = os.path.join(OUTPUT_DIR, f"{out_name}.npz")
                np.savez_compressed(
                    out_path,
                    X=X,
                    valid_T=valid_T.astype(np.float32),
                    label=label,
                    subject_id=participant_id,
                    hand_xyz_m=hand_xyz_m_arr,         # (t, 21, 3) wrist-rel, meters
                    hand_xyz_norm=hand_xyz_norm_arr,   # (t, 21, 3) wrist-rel, scale-norm (unitless)
                    hand_uvd=hand_uvd_arr,             # (t, 21, 3) raw pixels + depth_m
                    arm_xyz_m=arm_xyz_m_arr,           # (t, 3, 3)  wrist-rel, meters
                    arm_xyz_norm=arm_xyz_norm_arr,     # (t, 3, 3)  wrist-rel, scale-norm (unitless)
                    arm_uvd=arm_uvd_arr,               # (t, 3, 3)  raw pixels + depth_m
                    wrist_xyz=wrist_xyz_arr,           # (t, 3) absolute wrist in meters
                    scale_m=scale_m_arr,               # (t,) metric bone-length scale (meters)
                    valid_raw=valid_arr.astype(np.uint8),
                    valid_joint=valid_joint_arr,           # (t, 21) per-joint depth validity (hand)
                    valid_arm_joint=valid_arm_joint_arr,   # (t, 3)  per-joint depth validity (arm)
                    intrinsics=intr_array,                 # [fx, fy, cx, cy]
                    intrinsics_model=intr_model,           # int enum (e.g. 4 = brown_conrady)
                    intrinsics_model_str=intr_model_str,   # human-readable string
                    intrinsics_coeffs=intr_coeffs,         # distortion coefficients [k1..k3, p1, p2]
                    depth_scale=depth_scale,               # multiplier used: raw_depth * depth_scale = meters
                    T_target=T_TARGET,
                    coord_system='xyz',
                )
                arm_depth_rate = (valid_arm_joint_arr[valid_arr].mean()
                                  if valid_arr.any() else float('nan'))
                jd_all  = valid_joint_arr.mean()
                jd_good = valid_joint_arr[valid_arr].mean() if valid_arr.any() else float('nan')
                print(f"[OK] {file} -> {out_path} | X {X.shape} | "
                      f"frames {len(valid_arr)} | valid {valid_arr.sum()}/{len(valid_arr)} | "
                      f"hand_joint_depth all={jd_all:.2f} good={jd_good:.2f} | "
                      f"arm_joint_depth {arm_depth_rate:.2f}")

                # ── Stats ──
                processed_files += 1
                t_raw = int(valid_arr.shape[0])
                v_raw = int(valid_arr.sum())
                raw_ratio = float(v_raw) / float(max(t_raw, 1))
                res_ratio = float(valid_T.mean())

                scale_valid = scale_m_arr[valid_arr]
                scale_valid = scale_valid[np.isfinite(scale_valid)]
                scale_mean = float(np.mean(scale_valid)) if scale_valid.size else float("nan")
                scale_med = float(np.median(scale_valid)) if scale_valid.size else float("nan")

                per_file_stats.append({
                    "file": file, "t_raw": t_raw, "v_raw": v_raw,
                    "raw_ratio": raw_ratio, "res_ratio": res_ratio,
                    "scale_mean": scale_mean, "scale_med": scale_med,
                    "label": int(label),
                })
                total_frames += t_raw
                total_valid_frames += v_raw
                label_counts[label] = label_counts.get(label, 0) + 1

            # ── Folder summary ──
            if processed_files == 0:
                print("\n==================== FOLDER SUMMARY ====================")
                print(f"DATA_DIR: {DATA_DIR}")
                print(f"OUTPUT_DIR: {OUTPUT_DIR}")
                print(f"Processed: {processed_files} | Skipped: {skipped_files}")
                print("No files processed (all skipped or empty).")
                print("========================================================\n")
            else:
                raw_ratios = np.array([d["raw_ratio"] for d in per_file_stats], dtype=np.float32)
                res_ratios = np.array([d["res_ratio"] for d in per_file_stats], dtype=np.float32)
                lengths = np.array([d["t_raw"] for d in per_file_stats], dtype=np.int32)

                overall_raw_valid_ratio = float(total_valid_frames) / float(max(total_frames, 1))

                worst_idx = np.argsort(raw_ratios)[:min(5, raw_ratios.size)]
                shortest_idx = np.argsort(lengths)[:min(5, lengths.size)]
                longest_idx = np.argsort(-lengths)[:min(5, lengths.size)]

                p10 = float(np.percentile(lengths, 10))
                p50 = float(np.percentile(lengths, 50))
                p90 = float(np.percentile(lengths, 90))

                print("\n==================== FOLDER SUMMARY ====================")
                print(f"DATA_DIR: {DATA_DIR}")
                print(f"OUTPUT_DIR: {OUTPUT_DIR}")
                print(f"Processed: {processed_files} | Skipped: {skipped_files}")
                print(f"T_TARGET: {T_TARGET}")
                print(f"Flags: INCLUDE_ARM={INCLUDE_ARM}, INCLUDE_VELOCITY={INCLUDE_VELOCITY}, "
                      f"NORMALIZE_VELOCITY={NORMALIZE_VELOCITY}, INCLUDE_SCALE_CH={INCLUDE_SCALE_CH}")
                print(f"Coord system: XYZ (metric 3D, wrist-relative, bone-length normalized)")
                print("--------------------------------------------------------")
                print(f"Raw frames per clip: mean={lengths.mean():.2f}, std={lengths.std():.2f}, "
                      f"min={lengths.min()}, p10={p10:.1f}, median={p50:.1f}, p90={p90:.1f}, max={lengths.max()}")
                print(f"Total raw frames: {int(lengths.sum())}")
                print("--------------------------------------------------------")
                print(f"Raw validity (per-clip): mean={raw_ratios.mean():.4f}, std={raw_ratios.std():.4f}, "
                      f"min={raw_ratios.min():.4f}, max={raw_ratios.max():.4f}")
                print(f"Raw validity (overall frames): {overall_raw_valid_ratio:.4f} "
                      f"({total_valid_frames}/{total_frames})")
                print(f"Resampled validity @T_TARGET (per-clip): mean={res_ratios.mean():.4f}, "
                      f"std={res_ratios.std():.4f}, min={res_ratios.min():.4f}, max={res_ratios.max():.4f}")

                print("--------------------------------------------------------")
                print("Label counts:")
                for k in sorted(label_counts.keys()):
                    print(f"  label {k}: {label_counts[k]}")

                print("--------------------------------------------------------")
                print("Shortest clips by raw frame count:")
                for i in shortest_idx:
                    d = per_file_stats[int(i)]
                    print(f"  {d['file']}: frames={d['t_raw']}, raw_valid={d['raw_ratio']:.4f}, label={d['label']}")

                print("--------------------------------------------------------")
                print("Longest clips by raw frame count:")
                for i in longest_idx:
                    d = per_file_stats[int(i)]
                    print(f"  {d['file']}: frames={d['t_raw']}, raw_valid={d['raw_ratio']:.4f}, label={d['label']}")

                print("--------------------------------------------------------")
                print("Worst clips by raw valid ratio:")
                for i in worst_idx:
                    d = per_file_stats[int(i)]
                    print(f"  {d['file']}: raw_valid={d['raw_ratio']:.4f} ({d['v_raw']}/{d['t_raw']}), "
                          f"frames={d['t_raw']}, res_valid={d['res_ratio']:.4f}, label={d['label']}, "
                          f"scale_m_med={d['scale_med']:.4f}")

                print("========================================================\n")

            print("All gesture files processed.")

    hol.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
