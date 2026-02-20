#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
landmark.py — Offline RGB–D → gesture features (.npz) for Home+.

Switched MP extraction to Holistic (single-pass) and use ONLY RIGHT arm + RIGHT hand,
but keep all inputs/outputs identical to the previous preprocessing script.

Outputs (unchanged):
  X, valid_T, label, landmarks, arm_landmarks, wrist_uvz, scale, valid_raw,
  T_target, include_arm, include_velocity, normalize_velocity, include_scale_channel
"""

import os, sys, re
import numpy as np
import cv2
import mediapipe as mp
from contextlib import contextmanager

# ---------------- Config (unchanged IO) ----------------
# Process ALL integer-named participant folders in the same directory as this script.
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUTDIR = os.path.join(BASE_DIR, 'output')
os.makedirs(BASE_OUTPUTDIR, exist_ok=True)

POSE_VIS_THRESH    = 0.40
T_TARGET           = 64

INCLUDE_ARM           = True
INCLUDE_VELOCITY      = True
NORMALIZE_VELOCITY    = True
INCLUDE_SCALE_CH      = True

CLIP_FEATURES_ABS     = 50.0
EPS                   = 1e-6

# ---------------- MediaPipe Holistic (replaces hands+pose) ----------------
mp_holistic = mp.solutions.holistic

hol = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,        # same trade-off used in live script
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Pose landmark enums (RIGHT-side only)
PL = mp_holistic.PoseLandmark
R_SHOULDER = PL.RIGHT_SHOULDER.value
R_ELBOW    = PL.RIGHT_ELBOW.value
R_WRIST    = PL.RIGHT_WRIST.value

# ---------------- Utilities (unchanged) ----------------
def get_depth_m(depth_img, x, y):
    h, w = depth_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h): return None
    d = depth_img[y, x]
    if d == 0:
        y0, y1 = max(0, y-2), min(h, y+3)
        x0, x1 = max(0, x-2), min(w, x+3)
        neigh = depth_img[y0:y1, x0:x1]
        nz = neigh[neigh > 0]
        if nz.size == 0: return None
        d = int(np.median(nz))
    return d * 0.001  # mm → m

def resample_2d(arr, T):
    t = arr.shape[0]
    if t == T: return arr.astype(np.float32)
    if t < 2:  return np.repeat(arr.astype(np.float32), T, axis=0)
    idx = np.linspace(0, t-1, T)
    lo  = np.floor(idx).astype(int)
    hi  = np.clip(lo+1, 0, t-1)
    w   = (idx - lo)[:, None]
    return ((1-w)*arr[lo] + w*arr[hi]).astype(np.float32)

def resample_mask(mask_bool, T):
    m = mask_bool.astype(np.float32)[:, None]
    return (resample_2d(m, T).squeeze() > 0.5)

def make_velocity(x):
    return np.diff(x, axis=0, prepend=x[0:1])

def normalize_velocity_per_clip(dx):
    mag = np.sqrt((dx*dx).sum(axis=1) + EPS)
    m = float(mag.mean())
    return dx if m < EPS else (dx / m)

def hand_scale_from_landmarks(hand21_px):
    """Stable scale: max(wrist→index-M distance, tight box size)."""
    wrist = hand21_px[0, :2]
    idxM  = hand21_px[5, :2]
    d = np.linalg.norm(idxM - wrist)
    xmin, ymin = hand21_px[:,0].min(), hand21_px[:,1].min()
    xmax, ymax = hand21_px[:,0].max(), hand21_px[:,1].max()
    box = max(xmax - xmin, ymax - ymin)
    return float(max(d, box, EPS))

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
        if os.path.isdir(full) and re.fullmatch(r"\d+", name):
            out.append((int(name), name, full))
    out.sort(key=lambda x: x[0])
    return out

# ---------------- Main loop (inputs/outputs preserved) ----------------
participant_dirs = _list_participant_dirs(BASE_DIR)
if not participant_dirs:
    print(f"[WARN] No integer-named participant folders found in {BASE_DIR}")
    hol.close()
    sys.exit(0)

for _, participant_name, DATA_DIR in participant_dirs:
    OUTPUT_DIR = os.path.join(BASE_OUTPUTDIR, participant_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, "terminal_output.txt")
    with _tee_terminal_output(log_path):
        files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
        if not files:
            print(f"[WARN] No .npz in {DATA_DIR}")
            continue

        # ---------------- Folder-level summary accumulators ----------------
        processed_files = 0
        skipped_files   = 0

        per_file_stats = []   # list of dicts
        label_counts   = {}   # label -> count

        total_frames = 0
        total_valid_frames = 0

        for file in files:
            path = os.path.join(DATA_DIR, file)
            data = np.load(path, allow_pickle=True)
            if not all(k in data for k in ['rgb','depth','label']):
                print(f"[WARN] Skipping {file}: missing one of ['rgb','depth','label']")
                skipped_files += 1
                continue

            rgb_frames   = data['rgb']
            depth_frames = data['depth']
            label        = int(data['label'])

            hand_seq, arm_seq = [], []
            wrist_uvz_seq, scale_seq, valid_seq = [], [], []

            for rgb, depth in zip(rgb_frames, depth_frames):
                image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                H, W = rgb.shape[:2]

                # ---- Holistic inference (single pass: pose + hands) ----
                res = hol.process(image_rgb)

                # ---- RIGHT arm joints (shoulder, elbow, wrist) ----
                right_wrist_px = None
                arm_xyz = None
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark

                    def px(idx):
                        return int(lm[idx].x * W), int(lm[idx].y * H), lm[idx].visibility

                    sx, sy, svis = px(R_SHOULDER)
                    ex, ey, evis = px(R_ELBOW)
                    wx, wy, wvis = px(R_WRIST)
                    right_wrist_px = (wx, wy)

                    if (svis > POSE_VIS_THRESH) and (evis > POSE_VIS_THRESH) and (wvis > POSE_VIS_THRESH):
                        sz = get_depth_m(depth, sx, sy)
                        ez = get_depth_m(depth, ex, ey)
                        wz = get_depth_m(depth, wx, wy)
                        if (sz is not None) and (ez is not None) and (wz is not None):
                            arm_xyz = np.array([[sx, sy, sz],
                                                [ex, ey, ez],
                                                [wx, wy, wz]], dtype=np.float32)

                # ---- RIGHT hand landmarks (Holistic exposes right_hand_landmarks) ----
                hand_norm = np.zeros((21,3), dtype=np.float32)
                arm_norm  = np.zeros((3,3),  dtype=np.float32)

                chosen = res.right_hand_landmarks if hasattr(res, 'right_hand_landmarks') else None

                if chosen is not None:
                    # raw hand px (+z from MP)
                    hand_px = np.array([[lm.x * W, lm.y * H, lm.z] for lm in chosen.landmark], dtype=np.float32)
                    wrist_px = hand_px[0, :2]
                    scale = hand_scale_from_landmarks(hand_px)

                    hand_xy = (hand_px[:, :2] - wrist_px[None, :]) / max(scale, EPS)
                    wz = get_depth_m(depth, int(round(wrist_px[0])), int(round(wrist_px[1])))
                    wz = 0.0 if (wz is None or not np.isfinite(wz)) else float(wz)

                    hand_norm[:, :2] = hand_xy
                    hand_norm[:,  2] = hand_px[:, 2]  # keep MP z as-is

                    if arm_xyz is not None:
                        arm_xy = (arm_xyz[:, :2] - wrist_px[None, :]) / max(scale, EPS)
                        arm_norm[:, :2] = arm_xy
                        arm_norm[:,  2] = arm_xyz[:, 2]

                    wrist_uvz_seq.append([wrist_px[0], wrist_px[1], wz])
                    scale_seq.append(scale)
                    valid_seq.append(True)
                else:
                    wrist_uvz_seq.append([np.nan, np.nan, np.nan])
                    scale_seq.append(np.nan)
                    valid_seq.append(False)

                hand_seq.append(hand_norm)
                arm_seq.append(arm_norm)

            # ----- Arrays (unchanged) -----
            hand_seq      = np.asarray(hand_seq, dtype=np.float32)
            arm_seq       = np.asarray(arm_seq,  dtype=np.float32)
            wrist_uvz_seq = np.asarray(wrist_uvz_seq, dtype=np.float32)
            scale_seq     = np.asarray(scale_seq, dtype=np.float32)
            valid_seq     = np.asarray(valid_seq, dtype=bool)

            # ---------------- Build features (unchanged) ----------------
            pose_feat = np.concatenate([hand_seq, arm_seq], axis=1) if INCLUDE_ARM else hand_seq  # (t,J,3)
            t_len, J, C = pose_feat.shape
            pose_2d = pose_feat.reshape(t_len, J*C).astype(np.float32)  # (t, F_pose)

            scale_col = scale_seq[:, None] if INCLUDE_SCALE_CH else None
            wrist_uvz = wrist_uvz_seq

            # ---- Resample to T_TARGET ----
            pose_T      = resample_2d(pose_2d,   T_TARGET)
            wrist_uvz_T = resample_2d(wrist_uvz, T_TARGET)
            valid_T     = resample_mask(valid_seq, T_TARGET)
            scale_T     = resample_2d(scale_col, T_TARGET) if scale_col is not None else None

            # ---- Sanitize resampled streams ----
            wrist_uvz_T = np.nan_to_num(wrist_uvz_T, nan=0.0, posinf=0.0, neginf=0.0)
            if scale_T is not None:
                scale_T = np.nan_to_num(scale_T, nan=0.0, posinf=0.0, neginf=0.0)

            # ---- Velocity (valid-aware) ----
            feat_chunks = [pose_T]
            if INCLUDE_VELOCITY:
                d_pose  = make_velocity(pose_T)
                d_wrist = make_velocity(wrist_uvz_T)
                bad_now  = (~valid_T).astype(np.bool_)
                bad_prev = np.concatenate([[bad_now[0]], bad_now[:-1]])
                bad = (bad_now | bad_prev)[:, None]
                d_pose[bad.repeat(d_pose.shape[1], axis=1)] = 0.0
                d_wrist[bad.repeat(3, axis=1)] = 0.0
                if NORMALIZE_VELOCITY:
                    d_pose  = normalize_velocity_per_clip(d_pose)
                    d_wrist = normalize_velocity_per_clip(d_wrist)
                feat_chunks.extend([d_pose, d_wrist])

            if scale_T is not None:
                feat_chunks.append(scale_T)

            X = np.concatenate(feat_chunks, axis=1).astype(np.float32)
            if CLIP_FEATURES_ABS is not None:
                X = np.clip(X, -CLIP_FEATURES_ABS, CLIP_FEATURES_ABS)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            out_name = re.sub(r"\.npz$", "", file)
            out_path = os.path.join(OUTPUT_DIR, f"{out_name}.npz")
            np.savez_compressed(
                out_path,
                X=X,
                valid_T=valid_T.astype(np.float32),
                label=label,
                landmarks=hand_seq,
                arm_landmarks=arm_seq,
                wrist_uvz=wrist_uvz_seq,
                scale=scale_seq,
                valid_raw=valid_seq.astype(np.uint8),
                T_target=T_TARGET,
                include_arm=INCLUDE_ARM,
                include_velocity=INCLUDE_VELOCITY,
                normalize_velocity=NORMALIZE_VELOCITY,
                include_scale_channel=INCLUDE_SCALE_CH
            )
            print(f"[OK] {file} -> {out_path} | X {X.shape} | frames {len(valid_seq)} | valid {valid_seq.sum()}/{len(valid_seq)}")

            # ---------------- Record per-file summary stats ----------------
            processed_files += 1

            t_raw = int(valid_seq.shape[0])
            v_raw = int(valid_seq.sum())
            raw_ratio = float(v_raw) / float(max(t_raw, 1))
            res_ratio = float(valid_T.mean())

            scale_valid = scale_seq[valid_seq] if scale_seq.size else np.array([], dtype=np.float32)
            scale_mean_valid = float(np.nanmean(scale_valid)) if scale_valid.size else float("nan")
            scale_med_valid  = float(np.nanmedian(scale_valid)) if scale_valid.size else float("nan")

            per_file_stats.append({
                "file": file,
                "t_raw": t_raw,
                "v_raw": v_raw,
                "raw_ratio": raw_ratio,
                "res_ratio": res_ratio,
                "scale_mean_valid": scale_mean_valid,
                "scale_med_valid": scale_med_valid,
                "label": int(label),
            })

            total_frames += t_raw
            total_valid_frames += v_raw
            label_counts[label] = label_counts.get(label, 0) + 1

        # ---------------- Final folder summary print ----------------
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
            lengths    = np.array([d["t_raw"] for d in per_file_stats], dtype=np.int32)

            overall_raw_valid_ratio = float(total_valid_frames) / float(max(total_frames, 1))

            # Worst by validity, shortest/longest by raw frames
            worst_idx    = np.argsort(raw_ratios)[:min(5, raw_ratios.size)]
            shortest_idx = np.argsort(lengths)[:min(5, lengths.size)]
            longest_idx  = np.argsort(-lengths)[:min(5, lengths.size)]

            # Length percentiles (useful “at a glance”)
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
            print("--------------------------------------------------------")
            print(f"Raw frames per clip: mean={lengths.mean():.2f}, std={lengths.std():.2f}, "
                  f"min={lengths.min()}, p10={p10:.1f}, median={p50:.1f}, p90={p90:.1f}, max={lengths.max()}")
            print(f"Total raw frames: {int(lengths.sum())}")
            print("--------------------------------------------------------")
            print(f"Raw validity (per-clip): mean={raw_ratios.mean():.4f}, std={raw_ratios.std():.4f}, "
                  f"min={raw_ratios.min():.4f}, max={raw_ratios.max():.4f}")
            print(f"Raw validity (overall frames): {overall_raw_valid_ratio:.4f} "
                  f"({total_valid_frames}/{total_frames})")
            print(f"Resampled validity @T_TARGET (per-clip): mean={res_ratios.mean():.4f}, std={res_ratios.std():.4f}, "
                  f"min={res_ratios.min():.4f}, max={res_ratios.max():.4f}")

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
                      f"scale_med_valid={d['scale_med_valid']:.3f}")

            print("========================================================\n")

        print("All gesture files processed.")

# Cleanup
hol.close()
