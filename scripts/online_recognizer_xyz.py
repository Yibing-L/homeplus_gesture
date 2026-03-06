#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
online_recognizer_xyz.py — Live RGB-D gesture recognition using metric XYZ features.

Mirrors the offline pipeline of landmark_with_xyz.py + train_xyz.py so the model
sees exactly the same input distribution at inference time.

Key differences from online_recognizer.py:
  - MediaPipe Holistic (single pass) instead of separate Hands + Pose
  - Metric 3D XYZ features via RealSense deprojection (not pixel-space)
  - Per-joint depth validity channels (27 extra channels, total 175-dim)
  - Standardization applied only to continuous channels [0:n_continuous]
  - Supports attention_bilstm_xyz model type from train_xyz.py

Usage
------
python online_recognizer_xyz.py --checkpoint runs/xyz_7class/model_7.pt
python online_recognizer_xyz.py --checkpoint runs/xyz_7class/model_7.pt --label_map labels.json
Press 'q' to quit.
"""

import os
os.environ["MEDIAPIPE_DISABLE_TF"] = "1"

import sys, json, argparse, math, collections
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import importlib

# ------------------------------------------------------------------ Constants
EPS             = 1e-6
POSE_VIS_THRESH = 0.40
HAND_MIN_VALID  = 6       # min joints with valid depth to treat frame as valid
T_TARGET        = 64
DEPTH_SCALE     = 0.001   # Z16 uint16 mm → meters
CLIP_ABS        = 50.0

# ------------------------------------------------------------------ Angle computation
# Mirrors train_with_angles.py exactly so models trained there get the same features.
_HAND_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]
_ARM_TRIPLETS = [(0, 1, 2)]


def _insert_angles(X_raw):
    """Expand 175-dim feature array to 207-dim by inserting joint bend angles.

    Mirrors train_with_angles.py:insert_angles() so inference matches training.
    """
    T = X_raw.shape[0]
    hand_xyz = X_raw[:, 0:63].reshape(T, 21, 3)
    arm_xyz  = X_raw[:, 63:72].reshape(T, 3, 3)
    hand_jv  = X_raw[:, 149:170].astype(bool)
    arm_jv   = X_raw[:, 170:173].astype(bool)

    angles      = np.zeros((T, 16), dtype=np.float32)
    angle_valid = np.zeros((T, 16), dtype=np.float32)

    for ai, (p, j, c) in enumerate(_HAND_TRIPLETS):
        valid = hand_jv[:, p] & hand_jv[:, j] & hand_jv[:, c]
        if valid.any():
            v1 = hand_xyz[valid, p] - hand_xyz[valid, j]
            v2 = hand_xyz[valid, c] - hand_xyz[valid, j]
            n1 = np.linalg.norm(v1, axis=1, keepdims=True).clip(EPS)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True).clip(EPS)
            cos_a = ((v1 / n1) * (v2 / n2)).sum(axis=1).clip(-1.0, 1.0)
            angles[valid, ai]      = np.arccos(cos_a)
            angle_valid[valid, ai] = 1.0

    for ai, (p, j, c) in enumerate(_ARM_TRIPLETS):
        valid = arm_jv[:, p] & arm_jv[:, j] & arm_jv[:, c]
        if valid.any():
            v1 = arm_xyz[valid, p] - arm_xyz[valid, j]
            v2 = arm_xyz[valid, c] - arm_xyz[valid, j]
            n1 = np.linalg.norm(v1, axis=1, keepdims=True).clip(EPS)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True).clip(EPS)
            cos_a = ((v1 / n1) * (v2 / n2)).sum(axis=1).clip(-1.0, 1.0)
            angles[valid, 15 + ai]      = np.arccos(cos_a)
            angle_valid[valid, 15 + ai] = 1.0

    return np.concatenate([
        X_raw[:, :148],
        angles,
        X_raw[:, 148:],
        angle_valid,
    ], axis=1).astype(np.float32)

# MediaPipe Holistic pose landmark indices (right arm)
_PL         = mp.solutions.holistic.PoseLandmark
R_SHOULDER  = _PL.RIGHT_SHOULDER.value
R_ELBOW     = _PL.RIGHT_ELBOW.value
R_WRIST     = _PL.RIGHT_WRIST.value


# ------------------------------------------------------------------ CLI
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .pt checkpoint saved by train_xyz.py")
    ap.add_argument("--min_valid_ratio", type=float, default=0.25,
                    help="Min fraction of valid frames in the window to run inference")
    ap.add_argument("--t_target", type=int, default=64,
                    help="Sequence length the model expects")
    ap.add_argument("--stride_frames", type=int, default=2,
                    help="Process every Nth frame to reduce CPU load (>=1)")
    ap.add_argument("--ema_alpha", type=float, default=0.2,
                    help="EMA update rate for probability accumulation (higher=faster response)")
    ap.add_argument("--switch_margin", type=float, default=0.15,
                    help="Hysteresis: new class must exceed current by this margin to switch")
    ap.add_argument("--min_hold_frames", type=int, default=15,
                    help="Frames to hold a prediction before allowing switch (~0.5s at 30fps)")
    ap.add_argument("--inference_stride", type=int, default=12,
                    help="Run model every N MediaPipe frames (~5 updates per 2s gesture)")
    ap.add_argument("--confidence_thresh", type=float, default=0.45,
                    help="Min EMA max-prob to emit any prediction (else show idle)")
    ap.add_argument("--hand_motion_thresh", type=float, default=0.06,
                    help="Min hand config std (bone-norm units) to consider user active")
    ap.add_argument("--idle_reset_frames", type=int, default=20,
                    help="Reset EMA after this many consecutive idle frames")
    ap.add_argument("--label_map", type=str, default=None,
                    help="Optional JSON mapping class index → name")
    ap.add_argument("--show_landmarks", action="store_true",
                    help="Draw holistic landmarks on the display")
    return ap.parse_args()


# ------------------------------------------------------------------ Utilities
def resample_2d(arr, T):
    t = arr.shape[0]
    arr = arr.astype(np.float32)
    if t == T:
        return arr
    if t < 2:
        return np.repeat(arr, T, axis=0)
    idx = np.linspace(0, t - 1, T)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, t - 1)
    w = (idx - lo)[:, None]
    return ((1 - w) * arr[lo] + w * arr[hi]).astype(np.float32)


def resample_mask(mask_bool, T):
    m = mask_bool.astype(np.float32)[:, None]
    return (resample_2d(m, T).squeeze() > 0.5)


def make_velocity(x):
    return np.diff(x, axis=0, prepend=x[0:1])


def normalize_velocity_per_clip(dx):
    mag = np.sqrt((dx * dx).sum(axis=1) + EPS)
    m = float(mag.mean())
    return dx if m < EPS else (dx / m)


def get_depth_m(depth_img, x, y):
    """Return depth in meters at pixel (x,y) with 5×5 median fallback for zero pixels."""
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
    return float(d) * DEPTH_SCALE


def deproject(intr, u, v, z_m):
    """Deproject pixel (u, v) with depth z_m (metres) → metric XYZ point."""
    return rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z_m))


def metric_hand_scale(hand_xyz_wrist_rel, joint_valid):
    """Stable metric scale: average wrist→MCP distance for joints 5, 9, 17."""
    dists = []
    for j in [5, 9, 17]:
        if joint_valid[j]:
            d = float(np.linalg.norm(hand_xyz_wrist_rel[j]))
            if np.isfinite(d) and d > EPS:
                dists.append(d)
    return float(max(np.mean(dists) if dists else EPS, EPS))


# ------------------------------------------------------------------ Model
def build_model_from_artifact(artifact):
    """Build AttentionBiLSTM from a train_xyz.py checkpoint artifact."""
    torch = importlib.import_module("torch")
    nn = torch.nn

    mtype = artifact.get("type", "")
    if "attention_bilstm" not in mtype:
        raise ValueError(
            f"Unsupported model type '{mtype}'. "
            "This recognizer requires a train_xyz.py checkpoint (attention_bilstm_xyz)."
        )

    class TemporalAttentionPool(nn.Module):
        def __init__(self, embed_dim, num_heads=4):
            super().__init__()
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        def forward(self, x, key_padding_mask=None):
            B = x.size(0)
            q = self.query.expand(B, -1, -1)
            out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
            return out.squeeze(1)

    class AttentionBiLSTM(nn.Module):
        def __init__(self, in_dim, n_classes, hidden=128, lstm_layers=2,
                     attn_heads=4, proj_dim=128, dropout=0.4):
            super().__init__()
            self.proj_dim = proj_dim
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
            )
            self.lstm = nn.LSTM(
                proj_dim, hidden, lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0.0,
                batch_first=True, bidirectional=True,
            )
            lstm_out_dim = hidden * 2
            self.res_proj = (nn.Linear(lstm_out_dim, proj_dim)
                             if lstm_out_dim != proj_dim else nn.Identity())
            self.res_norm = nn.LayerNorm(proj_dim)
            self.attn_pool = TemporalAttentionPool(proj_dim, num_heads=attn_heads)
            self.head = nn.Sequential(
                nn.LayerNorm(proj_dim), nn.Dropout(dropout), nn.Linear(proj_dim, n_classes),
            )

        def forward(self, x, mask):
            h_proj = self.input_proj(x)
            lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64)
            packed = nn.utils.rnn.pack_padded_sequence(
                h_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            T_proj = h_proj.size(1)
            if lstm_out.size(1) < T_proj:
                pad = torch.zeros(lstm_out.size(0), T_proj - lstm_out.size(1),
                                  lstm_out.size(2), device=lstm_out.device)
                lstm_out = torch.cat([lstm_out, pad], dim=1)
            h = self.res_norm(self.res_proj(lstm_out) + h_proj)
            kpm = ~(mask[:, :T_proj].bool())
            return self.head(self.attn_pool(h, key_padding_mask=kpm))

    in_dim     = int(artifact["in_dim"])
    n_classes  = int(artifact["n_classes"])
    hidden     = int(artifact.get("hidden", 128))
    lstm_layers= int(artifact.get("lstm_layers", 2))
    attn_heads = int(artifact.get("attn_heads", 4))
    proj_dim   = int(artifact.get("proj_dim", 128))
    dropout    = float(artifact.get("dropout", 0.4))

    model = AttentionBiLSTM(in_dim, n_classes, hidden=hidden, lstm_layers=lstm_layers,
                            attn_heads=attn_heads, proj_dim=proj_dim, dropout=dropout)
    model.load_state_dict(artifact["state_dict"])
    return model


def load_checkpoint(path):
    """Load a train_xyz.py or train_with_angles.py checkpoint."""
    if not path.endswith(".pt"):
        raise ValueError("Only .pt checkpoints are supported.")
    torch = importlib.import_module("torch")
    art = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model_from_artifact(art)
    model.eval()
    mtype = art.get("type", "")
    has_angles = "angles" in mtype  # attention_bilstm_xyz_angles
    return {
        "model":        model,
        "mu":           np.array(art["mu"], dtype=np.float32),
        "sd":           np.array(art["sd"], dtype=np.float32),
        "n_continuous": int(art.get("n_continuous", 148)),
        "n_classes":    int(art["n_classes"]),
        "in_dim":       int(art["in_dim"]),
        "has_angles":   has_angles,
        # scale_mean is stored when --normalize_scale_channel was used at training
        "scale_mean":   float(art["scale_mean"]) if art.get("scale_mean") is not None else None,
    }


# ------------------------------------------------------------------ Feature buffer
class FeatureBufferXYZ:
    """Accumulates per-frame XYZ features, assembles 175-dim windows matching landmark_with_xyz.py."""

    def __init__(self, intr, t_target=64):
        self.intr = intr          # rs.intrinsics from the connected camera
        self.t_target = t_target
        # Each entry: (hand_xyz_norm[21,3], arm_xyz_norm[3,3], wrist_xyz[3],
        #              scale_m, valid_joint[21], valid_arm_joint[3], valid)
        self.buf = collections.deque(maxlen=256)

    def append_frame(self, bgr, depth, hol_result):
        """Extract per-frame XYZ features from a Holistic result and push to buffer."""
        H, W = bgr.shape[:2]

        # ── Arm joints (shoulder, elbow, wrist) — per-joint, independent validity ──
        arm_uvd        = np.full((3, 3), np.nan, dtype=np.float32)
        arm_joint_valid = np.zeros(3, dtype=np.uint8)
        if hol_result.pose_landmarks:
            lm = hol_result.pose_landmarks.landmark
            for ji, mp_idx in enumerate([R_SHOULDER, R_ELBOW, R_WRIST]):
                jx = int(np.clip(lm[mp_idx].x * W, 0, W - 1))
                jy = int(np.clip(lm[mp_idx].y * H, 0, H - 1))
                if lm[mp_idx].visibility > POSE_VIS_THRESH:
                    dz = get_depth_m(depth, jx, jy)
                    if dz is not None:
                        arm_uvd[ji] = [jx, jy, dz]
                        arm_joint_valid[ji] = 1

        # ── Hand landmarks ──
        chosen = hol_result.right_hand_landmarks if hasattr(hol_result, "right_hand_landmarks") else None

        hand_xyz_norm   = np.zeros((21, 3), dtype=np.float32)
        arm_xyz_norm    = np.zeros((3,  3), dtype=np.float32)
        wrist_xyz       = np.full(3, np.nan, dtype=np.float32)
        scale_m         = np.nan
        joint_valid     = np.zeros(21, dtype=np.uint8)
        frame_valid     = False

        if chosen is not None:
            hand_uvd = np.full((21, 3), np.nan, dtype=np.float32)
            for j, jlm in enumerate(chosen.landmark):
                u_j  = jlm.x * W
                v_j  = jlm.y * H
                u_px = int(np.clip(round(u_j), 0, W - 1))
                v_px = int(np.clip(round(v_j), 0, H - 1))
                dz   = get_depth_m(depth, u_px, v_px)
                hand_uvd[j, 0] = u_j
                hand_uvd[j, 1] = v_j
                if dz is not None:
                    hand_uvd[j, 2] = dz
                    joint_valid[j] = 1

            # Wrist must have depth and enough joints must be valid
            wrist_d = hand_uvd[0, 2]
            if (np.isfinite(wrist_d) and wrist_d > 0
                    and joint_valid.sum() >= HAND_MIN_VALID):
                wrist_u_dep = float(np.clip(hand_uvd[0, 0], 0, W - 1))
                wrist_v_dep = float(np.clip(hand_uvd[0, 1], 0, H - 1))
                wrist_xyz[:] = deproject(self.intr, wrist_u_dep, wrist_v_dep, float(wrist_d))

                # Deproject all hand joints → wrist-relative metric XYZ
                hand_xyz_m = np.zeros((21, 3), dtype=np.float32)
                for j in range(21):
                    if joint_valid[j]:
                        u_dep = float(np.clip(hand_uvd[j, 0], 0, W - 1))
                        v_dep = float(np.clip(hand_uvd[j, 1], 0, H - 1))
                        j_xyz = np.array(deproject(self.intr, u_dep, v_dep, hand_uvd[j, 2]),
                                         dtype=np.float32)
                        hand_xyz_m[j] = j_xyz - wrist_xyz

                # Metric bone-length scale
                if any(joint_valid[j] for j in [5, 9, 17]):
                    scale_m = metric_hand_scale(hand_xyz_m, joint_valid)
                elif joint_valid[9]:
                    scale_m = float(max(np.linalg.norm(hand_xyz_m[9]), EPS))
                elif joint_valid[5]:
                    scale_m = float(max(np.linalg.norm(hand_xyz_m[5]), EPS))

                if np.isfinite(scale_m) and scale_m > EPS:
                    hand_xyz_norm = hand_xyz_m / scale_m
                    hand_xyz_norm[joint_valid == 0] = 0.0

                    # Arm joints — deproject independently
                    arm_xyz_m = np.zeros((3, 3), dtype=np.float32)
                    for j in range(3):
                        if arm_joint_valid[j]:
                            u_dep = float(np.clip(arm_uvd[j, 0], 0, W - 1))
                            v_dep = float(np.clip(arm_uvd[j, 1], 0, H - 1))
                            j_xyz = np.array(deproject(self.intr, u_dep, v_dep, arm_uvd[j, 2]),
                                             dtype=np.float32)
                            arm_xyz_m[j] = j_xyz - wrist_xyz
                    arm_xyz_norm = arm_xyz_m / scale_m
                    arm_xyz_norm[arm_joint_valid == 0] = 0.0

                    frame_valid = True

        self.buf.append((hand_xyz_norm, arm_xyz_norm, wrist_xyz, scale_m,
                         joint_valid, arm_joint_valid, frame_valid))

    def append_invalid(self):
        self.buf.append((
            np.zeros((21, 3), dtype=np.float32),
            np.zeros((3,  3), dtype=np.float32),
            np.full(3, np.nan, dtype=np.float32),
            np.nan,
            np.zeros(21, dtype=np.uint8),
            np.zeros(3,  dtype=np.uint8),
            False,
        ))

    def assemble_window(self):
        """Build (T, 175) feature array matching landmark_with_xyz.py output exactly.

        Also returns hand_motion: temporal std of wrist-relative hand joint positions
        over valid frames (bone-normalized units). Low value = hand is stationary = idle.
        """
        if len(self.buf) == 0:
            return None

        hand_xyz_norm_seq  = np.stack([b[0] for b in self.buf], 0)  # (t, 21, 3)
        arm_xyz_norm_seq   = np.stack([b[1] for b in self.buf], 0)  # (t, 3, 3)
        wrist_xyz_seq      = np.stack([b[2] for b in self.buf], 0)  # (t, 3)
        scale_m_seq        = np.array([b[3] for b in self.buf], dtype=np.float32)   # (t,)
        valid_joint_seq    = np.stack([b[4] for b in self.buf], 0)  # (t, 21)
        valid_arm_joint_seq= np.stack([b[5] for b in self.buf], 0)  # (t, 3)
        valid_seq          = np.array([b[6] for b in self.buf], dtype=bool)         # (t,)

        # ── Activity metric: std of hand joint positions across valid frames ──
        # joints [1:21] are wrist-relative & bone-normalized → captures all hand
        # configuration changes (finger bending, spreading, orientation) without
        # depending on whether the wrist itself translates.
        hand_valid = hand_xyz_norm_seq[valid_seq, 1:, :]  # (n_valid, 20, 3)
        hand_motion = float(hand_valid.std(axis=0).mean()) if len(hand_valid) > 1 else 0.0

        T = self.t_target

        # ── Pose feature ──
        pose_feat = np.concatenate([hand_xyz_norm_seq, arm_xyz_norm_seq], axis=1)  # (t, 24, 3)
        t_len = pose_feat.shape[0]
        pose_2d = pose_feat.reshape(t_len, 72).astype(np.float32)

        # ── Resample everything to T_TARGET ──
        pose_T      = resample_2d(pose_2d, T)
        wrist_xyz_T = resample_2d(np.nan_to_num(wrist_xyz_seq, nan=0.0), T)
        scale_T     = resample_2d(np.nan_to_num(scale_m_seq[:, None], nan=0.0), T)
        valid_T     = resample_mask(valid_seq, T)

        valid_joint_T     = (resample_2d(valid_joint_seq.astype(np.float32), T)
                             > 0.5).astype(np.float32)       # (T, 21)
        valid_arm_joint_T = (resample_2d(valid_arm_joint_seq.astype(np.float32), T)
                             > 0.5).astype(np.float32)       # (T, 3)

        # ── Velocity (valid-aware) ──
        d_pose  = make_velocity(pose_T)
        d_wrist = make_velocity(wrist_xyz_T)

        bad_now  = ~valid_T
        bad_prev = np.concatenate([[bad_now[0]], bad_now[:-1]])
        bad = (bad_now | bad_prev)[:, None]
        d_pose[bad.repeat(d_pose.shape[1],  axis=1)] = 0.0
        d_wrist[bad.repeat(3,               axis=1)] = 0.0
        d_pose  = normalize_velocity_per_clip(d_pose)
        d_wrist = normalize_velocity_per_clip(d_wrist)

        # ── Assemble X (175-dim, matching landmark_with_xyz.py) ──
        feat_chunks = [
            pose_T,                                           # [0:72]   pose
            d_pose,                                           # [72:144]  vel_pose
            d_wrist,                                          # [144:147] vel_wrist
            scale_T,                                          # [147:148] scale
            valid_T.astype(np.float32)[:, None],              # [148:149] frame_valid
            valid_joint_T,                                    # [149:170] joint_valid_hand
            valid_arm_joint_T,                                # [170:173] joint_valid_arm
            valid_joint_T.mean(axis=1, keepdims=True),        # [173:174] hand_coverage
            valid_arm_joint_T.mean(axis=1, keepdims=True),    # [174:175] arm_coverage
        ]
        X = np.concatenate(feat_chunks, axis=1).astype(np.float32)

        # Clip only continuous channels; validity channels are already binary
        X[:, :148] = np.clip(X[:, :148], -CLIP_ABS, CLIP_ABS)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, valid_T, hand_motion


# ------------------------------------------------------------------ MediaPipe Holistic
class HolisticProcessor:
    """Single-pass MediaPipe Holistic — matches landmark_with_xyz.py."""

    def __init__(self, draw=False):
        self.draw = draw
        mp_holistic = mp.solutions.holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        if draw:
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_styles  = mp.solutions.drawing_styles

    def process(self, bgr):
        """Run Holistic on a BGR frame. Returns the result object."""
        return self.holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def draw_landmarks(self, bgr, result):
        if not self.draw:
            return
        mp_holistic = mp.solutions.holistic
        if result.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                bgr, result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                bgr, result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
            )

    def close(self):
        self.holistic.close()


# ------------------------------------------------------------------ Prob accumulator
class ProbAccumulator:
    """Smooths predictions via EMA on softmax distributions with hysteresis + hold time.

    Replaces streak-based LabelSmoother. Each inference call updates an EMA of
    the full probability vector. The public label only switches when:
      1. The candidate has been ahead for at least min_hold_frames.
      2. Its EMA probability exceeds the current label's by switch_margin.
      3. The overall EMA confidence (max prob) exceeds confidence_thresh.
    """

    def __init__(self, n_classes, alpha=0.2, switch_margin=0.15,
                 min_hold_frames=15, confidence_thresh=0.45):
        self.n             = n_classes
        self.alpha         = alpha
        self.switch_margin = switch_margin
        self.min_hold      = min_hold_frames
        self.conf_thresh   = confidence_thresh
        self.reset()

    def reset(self):
        self.ema          = np.ones(self.n, dtype=np.float32) / self.n
        self.public_label = None
        self.hold_count   = 0   # frames held on current public_label

    def update(self, probs):
        """Update EMA with new softmax vector. Returns (public_label, ema_max_prob)."""
        self.ema = (1 - self.alpha) * self.ema + self.alpha * probs
        candidate  = int(np.argmax(self.ema))
        max_prob   = float(self.ema[candidate])
        self.hold_count += 1

        if max_prob < self.conf_thresh:
            # Model is uncertain — stay silent
            return None, max_prob

        if self.public_label is None:
            self.public_label = candidate
            self.hold_count   = 0
        elif (candidate != self.public_label
              and self.hold_count >= self.min_hold
              and self.ema[candidate] > self.ema[self.public_label] + self.switch_margin):
            self.public_label = candidate
            self.hold_count   = 0

        return self.public_label, max_prob


# ------------------------------------------------------------------ Main
def main():
    args = parse_args()

    # Label map
    label_map = None
    if args.label_map and os.path.isfile(args.label_map):
        with open(args.label_map, "r") as f:
            label_map = {int(k): str(v) for k, v in json.load(f).items()}

    # Checkpoint
    art = load_checkpoint(args.checkpoint)
    model        = art["model"]
    mu           = art["mu"]
    sd           = art["sd"]
    n_continuous = art["n_continuous"]
    has_angles   = art["has_angles"]
    scale_mean   = art["scale_mean"]
    print(f"[INFO] Loaded model: {art['n_classes']}-class, in_dim={art['in_dim']}, "
          f"n_continuous={n_continuous}, angles={has_angles}, "
          f"scale_mean={scale_mean}")

    # RealSense setup
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile = pipeline.start(config)
    align   = rs.align(rs.stream.color)

    # Camera intrinsics — must match the device used for training
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    print(f"[Camera] fx={intr.fx:.2f} fy={intr.fy:.2f} "
          f"cx={intr.ppx:.2f} cy={intr.ppy:.2f} model={intr.model}")

    # Processors
    hol_proc    = HolisticProcessor(draw=args.show_landmarks)
    fb          = FeatureBufferXYZ(intr, t_target=args.t_target)
    accumulator = ProbAccumulator(
        n_classes        = art["n_classes"],
        alpha            = args.ema_alpha,
        switch_margin    = args.switch_margin,
        min_hold_frames  = args.min_hold_frames,
        confidence_thresh= args.confidence_thresh,
    )

    torch       = importlib.import_module("torch")
    frame_idx   = 0
    infer_idx   = 0   # counts MediaPipe-processed frames for inference stride
    idle_frames = 0   # consecutive frames below hand_motion_thresh

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                fb.append_invalid()
                continue

            bgr   = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())  # uint16, mm

            # Stride: skip frames to reduce CPU load
            if (frame_idx % max(1, args.stride_frames)) != 0:
                frame_idx += 1
                fb.append_invalid()
                continue
            frame_idx += 1

            # Run Holistic and extract XYZ features
            result = hol_proc.process(bgr)
            fb.append_frame(bgr, depth, result)
            if args.show_landmarks:
                hol_proc.draw_landmarks(bgr, result)

            # Assemble feature window
            out = fb.assemble_window()
            if out is None:
                cv2.imshow("Online Recognizer XYZ", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            X, valid_T, hand_motion = out

            vratio      = float(valid_T.mean())
            is_active   = hand_motion >= args.hand_motion_thresh
            public_label= accumulator.public_label
            ema_conf    = float(accumulator.ema.max())

            # ── Idle tracking: reset accumulator after sustained stillness ──
            if not is_active:
                idle_frames += 1
                if idle_frames >= args.idle_reset_frames:
                    accumulator.reset()
                    public_label = None
            else:
                idle_frames = 0

            # ── Inference (only every inference_stride MediaPipe frames) ──
            infer_idx += 1
            if (is_active
                    and vratio >= args.min_valid_ratio
                    and infer_idx % args.inference_stride == 0):
                Xi = X.copy()
                if has_angles:
                    Xi = _insert_angles(Xi)
                if scale_mean is not None and scale_mean > 1e-8:
                    Xi[:, 147] /= scale_mean
                Xi[:, :n_continuous] = (Xi[:, :n_continuous] - mu) / (sd + 1e-8)

                xt = torch.from_numpy(Xi[None]).float()
                mt = torch.from_numpy(valid_T[None].astype(np.float32))
                with torch.no_grad():
                    probs = torch.softmax(model(xt, mt), dim=1).cpu().numpy()[0]
                public_label, ema_conf = accumulator.update(probs)

            # ── HUD ──
            def label_name(idx):
                if idx is None: return "—"
                return label_map.get(idx, str(idx)) if label_map else str(idx)

            state = "IDLE" if not is_active else f"motion={hand_motion:.3f}"
            txt1  = f"valid={vratio:.2f}  {state}"
            txt2  = f"gesture={label_name(public_label)}  conf={ema_conf:.2f}"

            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 62), (0, 0, 0), -1)
            cv2.putText(bgr, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr, txt2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0) if public_label is not None else (128, 128, 128),
                        2, cv2.LINE_AA)

            cv2.imshow("Online Recognizer XYZ", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        hol_proc.close()


if __name__ == "__main__":
    main()
