#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_with_angles.py — Attention-BiLSTM with joint angle features appended.

Based on train_xyz.py. Adds 16 joint bend angles computed from the XYZ pose
channels already in X, inserted before the validity channels.

Feature layout (207 channels):
  [0:72]    pose        — hand (21j×3) + arm (3j×3), metric XYZ, wrist-relative,
                          bone-length normalized
  [72:144]  vel_pose    — frame-to-frame velocity of pose block
  [144:147] vel_wrist   — velocity of absolute wrist XYZ (meters/frame)
  [147:148] scale       — metric bone-length scale (meters)
  [148:163] hand_angles — 15 bend angles in radians [0,π], 0 where invalid
                          order: thumb(CMC,MCP,IP), index(MCP,PIP,DIP),
                                 middle(MCP,PIP,DIP), ring(MCP,PIP,DIP),
                                 pinky(MCP,PIP,DIP)
  [163:164] arm_angle   — elbow bend angle in radians [0,π], 0 where invalid
  [164:165] frame_valid — binary frame validity
  [165:186] joint_valid — per-joint depth validity, 21 hand joints
  [186:189] arm_valid   — per-joint depth validity, 3 arm joints
  [189:190] hand_cov    — mean hand joint validity
  [190:191] arm_cov     — mean arm joint validity
  [191:206] hand_angle_valid — binary validity for 15 hand angles
  [206:207] arm_angle_valid  — binary validity for arm elbow angle

N_CONTINUOUS = 164 (all channels before the binary validity blocks)
"""

import os, argparse, random, glob, json, logging, time, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Continuous channels: pose(72) + vel_pose(72) + vel_wrist(3) + scale(1) + angles(16) = 164
N_CONTINUOUS = 164

# Hand skeleton triplets (parent, joint, child) for angle computation.
# Joint indices follow MediaPipe hand landmarks (0=wrist).
HAND_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),       # thumb:  CMC, MCP, IP
    (0, 5, 6), (5, 6, 7), (6, 7, 8),        # index:  MCP, PIP, DIP
    (0, 9, 10), (9, 10, 11), (10, 11, 12),  # middle: MCP, PIP, DIP
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# ring:   MCP, PIP, DIP
    (0, 17, 18), (17, 18, 19), (18, 19, 20),# pinky:  MCP, PIP, DIP
]
# Arm triplet: shoulder(0), elbow(1), wrist(2) — wrist is reference [0,0,0]
ARM_TRIPLETS = [(0, 1, 2)]

EPS = 1e-6


# ================================================================
# Angle computation
# ================================================================
def compute_joint_angles(X):
    """Compute 16 bend angles from the XYZ pose channels in X.

    Uses pre-standardization X (raw values from npz). Angles are computed as
    arccos of the normalized dot product of the two bone vectors meeting at
    each joint. Returns 0 and invalid flag for any joint lacking valid depth.

    Args:
        X: (T, 175) raw feature array from train_xyz.py npz files.

    Returns:
        angles:       (T, 16) float32, radians in [0, pi], 0 where invalid.
        angle_valid:  (T, 16) float32, binary.
    """
    T = X.shape[0]

    # Extract pose and validity from existing X channels (pre-standardization)
    hand_xyz = X[:, 0:63].reshape(T, 21, 3)    # (T, 21, 3)
    arm_xyz  = X[:, 63:72].reshape(T, 3, 3)    # (T, 3, 3)
    hand_jv  = X[:, 149:170].astype(bool)       # (T, 21) joint validity
    arm_jv   = X[:, 170:173].astype(bool)       # (T, 3)  arm joint validity

    angles      = np.zeros((T, 16), dtype=np.float32)
    angle_valid = np.zeros((T, 16), dtype=np.float32)

    for ai, (p, j, c) in enumerate(HAND_TRIPLETS):
        valid = hand_jv[:, p] & hand_jv[:, j] & hand_jv[:, c]
        if valid.any():
            v1 = hand_xyz[valid, p] - hand_xyz[valid, j]   # parent - joint
            v2 = hand_xyz[valid, c] - hand_xyz[valid, j]   # child  - joint
            n1 = np.linalg.norm(v1, axis=1, keepdims=True).clip(EPS)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True).clip(EPS)
            cos_a = ((v1 / n1) * (v2 / n2)).sum(axis=1).clip(-1.0, 1.0)
            angles[valid, ai]      = np.arccos(cos_a)
            angle_valid[valid, ai] = 1.0

    for ai, (p, j, c) in enumerate(ARM_TRIPLETS):
        valid = arm_jv[:, p] & arm_jv[:, j] & arm_jv[:, c]
        if valid.any():
            v1 = arm_xyz[valid, p] - arm_xyz[valid, j]
            v2 = arm_xyz[valid, c] - arm_xyz[valid, j]
            n1 = np.linalg.norm(v1, axis=1, keepdims=True).clip(EPS)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True).clip(EPS)
            cos_a = ((v1 / n1) * (v2 / n2)).sum(axis=1).clip(-1.0, 1.0)
            angles[valid, 15 + ai]      = np.arccos(cos_a)
            angle_valid[valid, 15 + ai] = 1.0

    return angles, angle_valid


def insert_angles(X_raw):
    """Build 207-dim feature array from 175-dim train_xyz.py X.

    Inserts angle channels at [148:164] and angle validity at [191:207].

    Args:
        X_raw: (T, 175) array as saved by landmark_with_xyz.py pipeline.

    Returns:
        (T, 207) array with angles inserted.
    """
    angles, angle_valid = compute_joint_angles(X_raw)
    return np.concatenate([
        X_raw[:, :148],    # continuous: pose, vel_pose, vel_wrist, scale
        angles,             # (T, 16) new continuous: joint angles
        X_raw[:, 148:],    # original validity channels (27)
        angle_valid,        # (T, 16) new binary: angle validity
    ], axis=1).astype(np.float32)


# ================================================================
# CLI
# ================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Attention-BiLSTM trainer with joint angle features (LOSO / K-fold)"
    )
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Directories with processed *.npz clips from landmark_with_xyz.py")
    ap.add_argument("--save_path", type=str, default="runs/xyz_angles")
    ap.add_argument("--n_classes", type=int, default=7, choices=[7, 42])
    ap.add_argument("--model", type=str, default="bilstm",
                    choices=["bilstm", "unet1d", "resnet1d", "unet1d_lstm", "resnet1d_lstm", "stgcn"],
                    help="Model architecture: bilstm, unet1d, resnet1d, unet1d_lstm, resnet1d_lstm, stgcn")

    ap.add_argument("--eval_mode", type=str, default="loso", choices=["loso", "kfold"])
    ap.add_argument("--k", type=int, default=5)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--aug_speed", action="store_true", default=True)
    ap.add_argument("--aug_warp", action="store_true", default=True)
    ap.add_argument("--aug_noise_std", type=float, default=0.02)
    ap.add_argument("--aug_frame_drop_prob", type=float, default=0.05)
    ap.add_argument("--aug_block_mask_prob", type=float, default=0.05)
    ap.add_argument("--aug_scale_jitter", action="store_true", default=True)
    ap.add_argument("--no_aug_scale_jitter", action="store_true")
    ap.add_argument("--aug_scale_lo", type=float, default=0.7,
                    help="Lower bound for scale_jitter multiplier (default 0.7 to cover small hands)")
    ap.add_argument("--aug_scale_hi", type=float, default=1.2,
                    help="Upper bound for scale_jitter multiplier")
    ap.add_argument("--aug_rotate_prob", type=float, default=0.5,
                    help="Probability of applying in-plane pose rotation per sample")
    ap.add_argument("--aug_rotate_max_deg", type=float, default=15.0,
                    help="Max rotation angle in degrees for rotate_pose augmentation")
    ap.add_argument("--aug_temporal_crop_prob", type=float, default=0.5,
                    help="Probability of applying random temporal crop per sample")
    ap.add_argument("--mixup_alpha", type=float, default=0.2,
                    help="Beta distribution alpha for Mixup (0 = disabled)")
    ap.add_argument("--normalize_scale_channel", action="store_true", default=True,
                    help="Normalize channel 147 (bone-length scale) by training-fold mean, "
                         "reducing between-subject OOD shift. No conflict with bone-normalized XYZ.")
    ap.add_argument("--no_normalize_scale_channel", action="store_true")

    ap.add_argument("--clip_norm", action="store_true", default=True)
    ap.add_argument("--no_clip_norm", action="store_true")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--compress_joints", action="store_true", default=False,
                    help="Compress first 144 joint features via CNN before temporal model")
    ap.add_argument("--compress_joints_dim", type=int, default=32,
                    help="Output dim of joint compression CNN (default 32)")
    ap.add_argument("--condition_vec", action="store_true", default=False,
                    help="Append 6-dim one-hot condition vector (y42 // 7) to features")

    args = ap.parse_args()
    if args.no_aug_scale_jitter:
        args.aug_scale_jitter = False
    if args.no_clip_norm:
        args.clip_norm = False
    if args.no_normalize_scale_channel:
        args.normalize_scale_channel = False
    return args


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ================================================================
# Data loading
# ================================================================
def extract_participant_id(path):
    parent = os.path.basename(os.path.dirname(path))
    if parent.isdigit():
        return parent
    m = re.match(r"gesture_(\d+)_", os.path.basename(path))
    if m:
        return m.group(1)
    return parent


def _expand_roots(roots):
    """Expand any glob patterns in roots (needed on Windows where the shell
    does not expand wildcards before passing args to Python)."""
    expanded = []
    for r in roots:
        if any(c in r for c in ("*", "?", "[")):
            matches = sorted(glob.glob(r))
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(r)   # keep as-is so the missing-dir warning fires
        else:
            expanded.append(r)
    return expanded


def load_items(roots):
    roots = _expand_roots(roots)
    items = []
    for r in roots:
        print(f"[LOAD] {r}")
        for path in sorted(glob.glob(os.path.join(r, "*.npz"))):
            z = np.load(path)
            if "X" not in z:
                continue
            if "valid_T" in z:
                valid_arr = z["valid_T"]
            elif "valid" in z:
                valid_arr = z["valid"]
            else:
                continue
            X_raw = z["X"].astype(np.float32)   # (T, 175)
            if X_raw.shape[1] != 175:
                print(f"[WARN] {path}: expected 175-dim X, got {X_raw.shape[1]} — skipping")
                continue
            valid = np.asarray(valid_arr).squeeze()
            valid = valid.astype(bool) if valid.dtype == np.bool_ else (valid > 0.5)
            if "label" in z:
                y42 = int(z["label"])
            elif "gid" in z:
                y42 = int(z["gid"])
            else:
                continue
            vr = valid.mean() if valid.size else 1.0
            if vr < 0.25:
                continue
            # Compute and insert angle channels before standardization
            X = insert_angles(X_raw)            # (T, 207)
            pid = str(int(z["subject_id"])) if "subject_id" in z else extract_participant_id(path)
            items.append(dict(path=path, X=X, valid=valid, y42=y42, pid=pid, condition=y42 // 7))
    print(f"Loaded {len(items)} clips from {len(set(it['pid'] for it in items))} participants.")
    return items


# ================================================================
# Augmentation
# ================================================================
def _fix_binary_channels(X):
    """Round binary validity channels back to 0/1 after temporal interpolation.

    Channels [N_CONTINUOUS:] contain binary flags that get fractional values
    when frames are interpolated. Channels [189:191] (hand_cov, arm_cov) are
    continuous means — they are also rounded here since they are means of binary
    values and rounding is safe for augmented data.
    """
    X[:, N_CONTINUOUS:] = (X[:, N_CONTINUOUS:] >= 0.5).astype(np.float32)
    return X


def speed_perturb(X, valid, lo=0.85, hi=1.15):
    T, F = X.shape
    speed = np.random.uniform(lo, hi)
    new_T = max(4, int(round(T / speed)))
    idx = np.linspace(0, T - 1, new_T)
    lo_idx = np.floor(idx).astype(int)
    hi_idx = np.clip(lo_idx + 1, 0, T - 1)
    w = (idx - lo_idx)[:, None]
    X_new = ((1 - w) * X[lo_idx] + w * X[hi_idx]).astype(np.float32)
    v_new = np.interp(np.linspace(0, T - 1, new_T), np.arange(T), valid.astype(np.float32)) > 0.5
    idx2 = np.linspace(0, new_T - 1, T)
    lo2 = np.floor(idx2).astype(int)
    hi2 = np.clip(lo2 + 1, 0, new_T - 1)
    w2 = (idx2 - lo2)[:, None]
    X_out = ((1 - w2) * X_new[lo2] + w2 * X_new[hi2]).astype(np.float32)
    X_out = _fix_binary_channels(X_out)
    v_out = np.interp(np.linspace(0, new_T - 1, T), np.arange(new_T), v_new.astype(np.float32)) > 0.5
    return X_out, v_out


def time_warp(X, valid, sigma=0.2):
    T, F = X.shape
    warp = np.cumsum(np.random.randn(T) * sigma)
    warp = warp - warp.min()
    warp = warp / (warp.max() + 1e-8) * (T - 1)
    lo_idx = np.floor(warp).astype(int).clip(0, T - 2)
    hi_idx = lo_idx + 1
    w = (warp - lo_idx)[:, None]
    X_out = ((1 - w) * X[lo_idx] + w * X[hi_idx]).astype(np.float32)
    X_out = _fix_binary_channels(X_out)
    v_float = valid.astype(np.float32)
    v_out = ((1 - w.squeeze()) * v_float[lo_idx] + w.squeeze() * v_float[hi_idx]) > 0.5
    return X_out, v_out


def gaussian_noise(X, std=0.02):
    """Add noise to continuous channels only — binary validity channels are untouched."""
    noise = np.zeros_like(X)
    noise[:, :N_CONTINUOUS] = np.random.randn(X.shape[0], N_CONTINUOUS).astype(np.float32) * std
    return X + noise


def frame_dropout(X, valid, prob=0.05):
    T = X.shape[0]
    drop = np.random.rand(T) < prob
    X_out = X.copy(); v_out = valid.copy()
    X_out[drop] = 0.0; v_out[drop] = False
    return X_out, v_out


def feature_block_mask(X, prob=0.05):
    """Randomly zero one continuous feature block.

    Feature layout for 207-dim (continuous portion 0–163):
      pose(72) + vel_pose(72) + vel_wrist(3) + scale(1) + angles(16)
    """
    if np.random.rand() > prob:
        return X
    X_out = X.copy()
    blocks = [
        (0, 72),     # pose
        (72, 144),   # velocity_pose
        (144, 147),  # velocity_wrist
        (147, 148),  # scale
        (148, 164),  # joint angles
    ]
    start, end = blocks[np.random.randint(len(blocks))]
    X_out[:, start:end] = 0.0
    return X_out


def scale_jitter(X, lo=0.5, hi=1.2):
    """Scale pose+velocity XYZ channels. Angles are scale-invariant — not touched.

    Default lo=0.5 (was 0.8) to cover subjects with small hands / low bone-length scale.
    """
    X_out = X.copy()
    X_out[:, :144] *= np.random.uniform(lo, hi)
    return X_out


def rotate_pose(X, max_angle_deg=15.0):
    """Rotate hand/arm pose in the XY plane by a random angle.

    Simulates natural variation in hand orientation across subjects/sessions.
    Applied to pose [0:72], vel_pose [72:144], and vel_wrist [144:147].
    Bend angles [148:164] are rotation-invariant — not touched.
    Binary validity channels are not touched.

    Args:
        X: (T, 207) feature array (post-standardization or pre).
        max_angle_deg: maximum rotation magnitude in degrees.
    """
    theta = np.random.uniform(-max_angle_deg, max_angle_deg) * (np.pi / 180.0)
    c, s = np.cos(theta), np.sin(theta)
    X_out = X.copy()
    # Rotate every (x, y, z) triplet in pose and vel_pose blocks
    for block_start, block_end in [(0, 72), (72, 144)]:
        x_idx = np.arange(block_start, block_end, 3)
        y_idx = x_idx + 1
        x = X_out[:, x_idx].copy()
        y = X_out[:, y_idx].copy()
        X_out[:, x_idx] = c * x - s * y
        X_out[:, y_idx] = s * x + c * y
    # vel_wrist (144:147) is a single (vx, vy, vz)
    vx = X_out[:, 144].copy()
    vy = X_out[:, 145].copy()
    X_out[:, 144] = c * vx - s * vy
    X_out[:, 145] = s * vx + c * vy
    return X_out


def temporal_crop(X, valid, min_frac=0.7):
    """Take a random contiguous temporal crop and resample back to original T.

    Distinct from speed_perturb: the crop window is random (not center-anchored)
    and always resamples to the original sequence length.
    """
    T = X.shape[0]
    crop_len = max(4, int(T * np.random.uniform(min_frac, 1.0)))
    start = np.random.randint(0, max(1, T - crop_len + 1))
    X_crop = X[start:start + crop_len]
    v_crop = valid[start:start + crop_len].astype(np.float32)
    idx = np.linspace(0, crop_len - 1, T)
    lo_i = np.floor(idx).astype(int)
    hi_i = np.clip(lo_i + 1, 0, crop_len - 1)
    w = (idx - lo_i)[:, None]
    X_out = ((1 - w) * X_crop[lo_i] + w * X_crop[hi_i]).astype(np.float32)
    X_out = _fix_binary_channels(X_out)
    v_out = np.interp(np.linspace(0, crop_len - 1, T),
                      np.arange(crop_len), v_crop) > 0.5
    return X_out, v_out


# ================================================================
# Dataset
# ================================================================
class AugmentedClipSet(Dataset):
    def __init__(self, items, augment=False, aug_cfg=None, clip_norm=False):
        self.items = items
        self.augment = augment
        self.cfg = aug_cfg or {}
        self.clip_norm = clip_norm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        X = it["X"].copy()
        valid = it["valid"].copy().astype(bool)
        y = it["y42"]

        if self.augment:
            if self.cfg.get("speed", True) and np.random.rand() < 0.5:
                X, valid = speed_perturb(X, valid)
            if self.cfg.get("warp", True) and np.random.rand() < 0.5:
                X, valid = time_warp(X, valid)
            if self.cfg.get("temporal_crop_prob", 0.0) > 0 and np.random.rand() < self.cfg.get("temporal_crop_prob", 0.0):
                X, valid = temporal_crop(X, valid)
            if self.cfg.get("noise_std", 0.02) > 0:
                X = gaussian_noise(X, self.cfg.get("noise_std", 0.02))
            if self.cfg.get("frame_drop_prob", 0.05) > 0:
                X, valid = frame_dropout(X, valid, self.cfg.get("frame_drop_prob", 0.05))
            if self.cfg.get("block_mask_prob", 0.05) > 0:
                X = feature_block_mask(X, self.cfg.get("block_mask_prob", 0.05))
            if self.cfg.get("scale_jitter", True):
                X = scale_jitter(X, lo=self.cfg.get("scale_lo", 0.5),
                                    hi=self.cfg.get("scale_hi", 1.2))
            if self.cfg.get("rotate_prob", 0.0) > 0 and np.random.rand() < self.cfg.get("rotate_prob", 0.0):
                X = rotate_pose(X, self.cfg.get("rotate_max_deg", 15.0))

        if self.clip_norm:
            mu = X[:, :N_CONTINUOUS].mean(axis=0)
            sd = X[:, :N_CONTINUOUS].std(axis=0) + 1e-6
            X[:, :N_CONTINUOUS] = (X[:, :N_CONTINUOUS] - mu) / sd

        return X, valid.astype(np.float32), y


def collate(batch):
    Xs, Ms, Ys = zip(*batch)
    return (torch.from_numpy(np.stack(Xs, 0)).float(),
            torch.from_numpy(np.stack(Ms, 0)).float(),
            torch.tensor(Ys, dtype=torch.long))


# ================================================================
# Model: Attention-BiLSTM (unchanged from train_xyz.py)
# ================================================================
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
        self.res_proj = nn.Linear(lstm_out_dim, proj_dim) if lstm_out_dim != proj_dim else nn.Identity()
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
            pad = torch.zeros(lstm_out.size(0), T_proj - lstm_out.size(1), lstm_out.size(2),
                              device=lstm_out.device, dtype=lstm_out.dtype)
            lstm_out = torch.cat([lstm_out, pad], dim=1)
        h = self.res_norm(self.res_proj(lstm_out) + h_proj)
        pooled = self.attn_pool(h, key_padding_mask=~(mask[:, :T_proj].bool()))
        return self.head(pooled)


# ================================================================
# Model: 1D UNet for temporal classification
# ================================================================
class _ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    def __init__(self, in_dim, n_classes, base_channels=64, dropout=0.4):
        super().__init__()
        c = base_channels
        self.input_proj = nn.Linear(in_dim, c)
        # Encoder
        self.enc1 = _ConvBlock1D(c, c)
        self.enc2 = _ConvBlock1D(c, c * 2)
        self.enc3 = _ConvBlock1D(c * 2, c * 4)
        self.pool = nn.MaxPool1d(2)
        # Bottleneck
        self.bottleneck = _ConvBlock1D(c * 4, c * 8)
        # Decoder
        self.up3 = nn.ConvTranspose1d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock1D(c * 8, c * 4)   # concat with enc3
        self.up2 = nn.ConvTranspose1d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock1D(c * 4, c * 2)   # concat with enc2
        self.up1 = nn.ConvTranspose1d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock1D(c * 2, c)       # concat with enc1
        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(c), nn.Dropout(dropout), nn.Linear(c, n_classes),
        )

    def _pad_to_divisible(self, x, divisor=8):
        """Pad time dimension so it's divisible by divisor (needed for 3 pool layers)."""
        T = x.size(2)
        remainder = T % divisor
        if remainder == 0:
            return x, T
        pad_len = divisor - remainder
        x = nn.functional.pad(x, (0, pad_len))
        return x, T

    def forward(self, x, mask):
        # x: (B, T, F), mask: (B, T)
        x = self.input_proj(x)          # (B, T, C)
        x = x * mask.unsqueeze(-1)      # zero out padded frames
        x = x.transpose(1, 2)           # (B, C, T) for conv1d
        x, orig_T = self._pad_to_divisible(x)
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        # Trim back to original length
        d1 = d1[:, :, :orig_T]         # (B, C, T)
        d1 = d1.transpose(1, 2)        # (B, T, C)
        # Masked global average pool
        mask_f = mask[:, :orig_T].unsqueeze(-1)   # (B, T, 1)
        pooled = (d1 * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.head(pooled)


# ================================================================
# Model: 1D ResNet for temporal classification
# ================================================================
class _ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + x)


class _ResStage1D(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=2):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        ) if in_ch != out_ch else nn.Identity()
        self.blocks = nn.Sequential(*[_ResBlock1D(out_ch) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class ResNet1D(nn.Module):
    def __init__(self, in_dim, n_classes, base_channels=64, dropout=0.4):
        super().__init__()
        c = base_channels
        self.input_proj = nn.Linear(in_dim, c)
        self.stages = nn.Sequential(
            _ResStage1D(c, c, n_blocks=2),
            _ResStage1D(c, c * 2, n_blocks=2),
            _ResStage1D(c * 2, c * 4, n_blocks=2),
            _ResStage1D(c * 4, c * 8, n_blocks=2),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(c * 8), nn.Dropout(dropout), nn.Linear(c * 8, n_classes),
        )

    def forward(self, x, mask):
        # x: (B, T, F), mask: (B, T)
        x = self.input_proj(x)          # (B, T, C)
        x = x * mask.unsqueeze(-1)
        x = x.transpose(1, 2)           # (B, C, T)
        x = self.stages(x)              # (B, C*8, T')
        # Global average pool over time
        x = x.mean(dim=2)              # (B, C*8)
        return self.head(x)


# ================================================================
# Model: UNet1D encoder + BiLSTM + Attention
# ================================================================
class UNet1DLstm(nn.Module):
    def __init__(self, in_dim, n_classes, base_channels=64, hidden=128,
                 lstm_layers=2, attn_heads=4, dropout=0.4):
        super().__init__()
        c = base_channels
        self.input_proj = nn.Linear(in_dim, c)
        # Encoder
        self.enc1 = _ConvBlock1D(c, c)
        self.enc2 = _ConvBlock1D(c, c * 2)
        self.enc3 = _ConvBlock1D(c * 2, c * 4)
        self.pool = nn.MaxPool1d(2)
        # Bottleneck
        self.bottleneck = _ConvBlock1D(c * 4, c * 8)
        # Decoder
        self.up3 = nn.ConvTranspose1d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock1D(c * 8, c * 4)
        self.up2 = nn.ConvTranspose1d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock1D(c * 4, c * 2)
        self.up1 = nn.ConvTranspose1d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock1D(c * 2, c)
        # BiLSTM + Attention
        self.lstm = nn.LSTM(c, hidden, lstm_layers,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            batch_first=True, bidirectional=True)
        lstm_out_dim = hidden * 2
        self.attn_pool = TemporalAttentionPool(lstm_out_dim, num_heads=attn_heads)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim), nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, n_classes),
        )

    def _pad_to_divisible(self, x, divisor=8):
        T = x.size(2)
        remainder = T % divisor
        if remainder == 0:
            return x, T
        pad_len = divisor - remainder
        x = nn.functional.pad(x, (0, pad_len))
        return x, T

    def forward(self, x, mask):
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        x = x.transpose(1, 2)
        x, orig_T = self._pad_to_divisible(x)
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d1 = d1[:, :, :orig_T].transpose(1, 2)  # (B, T, C)
        # BiLSTM
        lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(
            d1, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        T_out = d1.size(1)
        if lstm_out.size(1) < T_out:
            pad = torch.zeros(lstm_out.size(0), T_out - lstm_out.size(1), lstm_out.size(2),
                              device=lstm_out.device, dtype=lstm_out.dtype)
            lstm_out = torch.cat([lstm_out, pad], dim=1)
        # Attention pool
        pooled = self.attn_pool(lstm_out, key_padding_mask=~(mask[:, :T_out].bool()))
        return self.head(pooled)


# ================================================================
# Model: ResNet1D encoder + BiLSTM + Attention
# ================================================================
class ResNet1DLstm(nn.Module):
    def __init__(self, in_dim, n_classes, base_channels=64, hidden=128,
                 lstm_layers=2, attn_heads=4, dropout=0.4):
        super().__init__()
        c = base_channels
        self.input_proj = nn.Linear(in_dim, c)
        self.stages = nn.Sequential(
            _ResStage1D(c, c, n_blocks=2),
            _ResStage1D(c, c * 2, n_blocks=2),
            _ResStage1D(c * 2, c * 4, n_blocks=2),
            _ResStage1D(c * 4, c * 8, n_blocks=2),
        )
        # BiLSTM + Attention
        self.lstm = nn.LSTM(c * 8, hidden, lstm_layers,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            batch_first=True, bidirectional=True)
        lstm_out_dim = hidden * 2
        self.attn_pool = TemporalAttentionPool(lstm_out_dim, num_heads=attn_heads)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim), nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, n_classes),
        )
        self.mask_pool = nn.MaxPool1d(16, stride=16)  # 4 stages, each stride 2 → 2^4=16

    def forward(self, x, mask):
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)
        x = x.transpose(1, 2)              # (B, C, T)
        x = self.stages(x)                  # (B, C*8, T')
        x = x.transpose(1, 2)              # (B, T', C*8)
        # Downsample mask to match T'
        T_prime = x.size(1)
        mask_ds = self.mask_pool(mask.unsqueeze(1).float()).squeeze(1)  # (B, T')
        mask_ds = mask_ds[:, :T_prime]
        # BiLSTM
        lengths = mask_ds.sum(dim=1).clamp(min=1).to(torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        if lstm_out.size(1) < T_prime:
            pad = torch.zeros(lstm_out.size(0), T_prime - lstm_out.size(1), lstm_out.size(2),
                              device=lstm_out.device, dtype=lstm_out.dtype)
            lstm_out = torch.cat([lstm_out, pad], dim=1)
        # Attention pool
        pooled = self.attn_pool(lstm_out, key_padding_mask=~(mask_ds[:, :T_prime].bool()))
        return self.head(pooled)


# ================================================================
# Model: Spatio-Temporal Graph Convolutional Network (ST-GCN)
# ================================================================
# Skeleton: 21 MediaPipe hand joints + 2 arm joints (shoulder=21, elbow=22).
# Arm wrist is merged with hand wrist (node 0) to avoid redundancy.
# Total: 23 nodes.
SKELETON_EDGES = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Arm: shoulder(21) → elbow(22) → wrist(0)
    (21, 22), (22, 0),
]
NUM_JOINTS = 23  # 21 hand + 2 arm (wrist merged)

# Distance from wrist (node 0) for spatial partitioning.
# Used to partition neighbors into centripetal (closer to wrist) vs centrifugal (farther).
_JOINT_DEPTH = {0: 0,
    1: 1, 2: 2, 3: 3, 4: 4,           # thumb
    5: 1, 6: 2, 7: 3, 8: 4,           # index
    9: 1, 10: 2, 11: 3, 12: 4,        # middle
    13: 1, 14: 2, 15: 3, 16: 4,       # ring
    17: 1, 18: 2, 19: 3, 20: 4,       # pinky
    21: 2, 22: 1,                       # arm: elbow closer to wrist than shoulder
}


def _build_adjacency_partitions():
    """Build 3-partition adjacency matrices for ST-GCN (Yan et al. 2018).

    Partitions:
      A_self:        identity (self-loops)
      A_centripetal: neighbor is closer to wrist (root)
      A_centrifugal: neighbor is farther from wrist

    Each is D^{-1/2} A_k D^{-1/2} normalized independently.
    """
    A_self = np.eye(NUM_JOINTS, dtype=np.float32)
    A_cen  = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    A_cfu  = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)

    for i, j in SKELETON_EDGES:
        # i→j edge: which direction is centripetal?
        if _JOINT_DEPTH[j] < _JOINT_DEPTH[i]:
            # j is closer to wrist → centripetal for i
            A_cen[i, j] = 1.0
            A_cfu[j, i] = 1.0
        else:
            A_cfu[i, j] = 1.0
            A_cen[j, i] = 1.0

    partitions = []
    for A in [A_self, A_cen, A_cfu]:
        D = A.sum(axis=1).clip(1e-6)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
        partitions.append(D_inv_sqrt @ A @ D_inv_sqrt)
    return np.stack(partitions, axis=0)  # (3, V, V)


class _STGCNBlock(nn.Module):
    """One ST-GCN block with 3-partition graph conv → temporal conv + residual."""
    def __init__(self, in_ch, out_ch, num_partitions=3, temporal_stride=1):
        super().__init__()
        self.num_partitions = num_partitions
        # One Conv2d per partition (1×1 to transform node features)
        self.gcn_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1) for _ in range(num_partitions)
        ])
        self.gcn_bn = nn.BatchNorm2d(out_ch)
        # Learnable edge importance per partition
        # Initialized to ones so initial behavior ≈ uniform weighting
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(NUM_JOINTS, NUM_JOINTS))
            for _ in range(num_partitions)
        ])
        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(9, 1), padding=(4, 0),
                      stride=(temporal_stride, 1)),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.GELU()
        # Residual
        if in_ch != out_ch or temporal_stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(temporal_stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A_partitions):
        # x: (B, C, T, V), A_partitions: (3, V, V)
        res = self.residual(x)
        # Multi-partition graph conv: sum contributions from each partition
        out = 0
        for k in range(self.num_partitions):
            A_k = A_partitions[k] * self.edge_importance[k]
            h = self.gcn_convs[k](x)                        # (B, out_ch, T, V)
            out = out + torch.einsum('bctv,vw->bctw', h, A_k)
        out = self.act(self.gcn_bn(out))
        # Temporal conv + residual
        out = self.act(self.tcn(out) + res)
        return out


class SpatioTemporalGCN(nn.Module):
    def __init__(self, in_dim, n_classes, dropout=0.4):
        super().__init__()
        A_parts = _build_adjacency_partitions()
        self.register_buffer('A_partitions', torch.from_numpy(A_parts))  # (3, V, V)

        # Input channels per joint: xyz_pos(3) + xyz_vel(3) + joint_valid(1) = 7
        n_in_ch = 7

        # ST-GCN blocks: 7 → 64 → 128 → 256
        self.temporal_strides = [1, 2, 2]
        self.blocks = nn.ModuleList([
            _STGCNBlock(n_in_ch, 64, temporal_stride=1),
            _STGCNBlock(64, 128, temporal_stride=2),
            _STGCNBlock(128, 256, temporal_stride=2),
        ])

        # Learnable per-joint importance for pooling (not uniform)
        self.joint_importance = nn.Parameter(torch.ones(NUM_JOINTS))

        # Non-joint features: vel_wrist(3) + scale(1) + angles(16) +
        #   frame_valid(1) + hand_cov(1) + arm_cov(1) = 23 kept channels
        # (drop per-joint validity and angle validity since graph uses them directly)
        self._rest_idx = (
            list(range(144, 165))  # vel_wrist(3), scale(1), angles(16), frame_valid(1)
            + [189, 190]           # hand_cov, arm_cov
        )
        n_rest = len(self._rest_idx)

        head_dim = 256 + n_rest
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Dropout(dropout), nn.Linear(head_dim, n_classes),
        )

    def _downsample_mask(self, mask):
        """Downsample temporal mask to match stride-2 reductions in ST-GCN blocks.

        A downsampled frame is valid if ANY of the original frames it covers were valid.
        """
        m = mask
        for stride in self.temporal_strides:
            if stride > 1:
                T = m.size(1)
                # Pad to even length if needed
                if T % 2 == 1:
                    m = nn.functional.pad(m, (0, 1), value=0.0)
                # Max-pool: valid if any constituent frame was valid
                m = m.reshape(m.size(0), -1, 2).max(dim=2).values
        return m

    def forward(self, x, mask):
        # x: (B, T, F), mask: (B, T)
        B, T, _ = x.shape

        # Extract joint features: pose(72) + vel(72) → (B, T, 23, 3) each
        # Hand joints: 21 joints from pose[0:63] and vel[72:135]
        # Arm joints: shoulder and elbow from pose[63:69] and vel[135:141]
        # (arm wrist pose[69:72] / vel[141:144] merged with hand wrist node 0)
        hand_pose = x[:, :, :63].reshape(B, T, 21, 3)                # (B, T, 21, 3)
        arm_se_pose = x[:, :, 63:69].reshape(B, T, 2, 3)             # shoulder, elbow
        hand_vel = x[:, :, 72:135].reshape(B, T, 21, 3)
        arm_se_vel = x[:, :, 135:141].reshape(B, T, 2, 3)

        # Merge arm wrist into hand wrist (node 0) by averaging
        arm_wrist_pose = x[:, :, 69:72].unsqueeze(2)                  # (B, T, 1, 3)
        arm_wrist_vel = x[:, :, 141:144].unsqueeze(2)
        hand_pose_merged = hand_pose.clone()
        hand_vel_merged = hand_vel.clone()
        hand_pose_merged[:, :, 0:1, :] = (hand_pose[:, :, 0:1, :] + arm_wrist_pose) / 2
        hand_vel_merged[:, :, 0:1, :] = (hand_vel[:, :, 0:1, :] + arm_wrist_vel) / 2

        pose_all = torch.cat([hand_pose_merged, arm_se_pose], dim=2)  # (B, T, 23, 3)
        vel_all = torch.cat([hand_vel_merged, arm_se_vel], dim=2)     # (B, T, 23, 3)

        # Per-joint validity as a node feature: hand(21) + arm(2 for shoulder,elbow)
        hand_jv = x[:, :, 165:186].unsqueeze(-1)                     # (B, T, 21, 1)
        arm_jv = x[:, :, 186:189]                                    # (B, T, 3)
        arm_se_jv = arm_jv[:, :, :2].unsqueeze(-1)                   # (B, T, 2, 1) shoulder,elbow
        # Wrist validity: valid if either hand or arm wrist is valid
        wrist_jv = torch.max(hand_jv[:, :, 0:1, :], arm_jv[:, :, 2:3].unsqueeze(-1))
        hand_jv_merged = hand_jv.clone()
        hand_jv_merged[:, :, 0:1, :] = wrist_jv
        jv_all = torch.cat([hand_jv_merged, arm_se_jv], dim=2)       # (B, T, 23, 1)

        # Gate joint features by per-joint validity
        pose_all = pose_all * jv_all
        vel_all = vel_all * jv_all

        # Build graph input: (B, T, 23, 7) → (B, 7, T, 23)
        graph_in = torch.cat([pose_all, vel_all, jv_all], dim=-1)     # (B, T, 23, 7)
        graph_in = graph_in.permute(0, 3, 1, 2)                       # (B, 7, T, 23)

        # Also mask by frame-level validity
        graph_in = graph_in * mask.unsqueeze(1).unsqueeze(-1)

        # ST-GCN blocks
        h = graph_in
        for block in self.blocks:
            h = block(h, self.A_partitions)                            # (B, 256, T', 23)

        # Mask-aware weighted pooling over time and joints
        ds_mask = self._downsample_mask(mask)                          # (B, T')
        T_prime = h.size(2)
        ds_mask = ds_mask[:, :T_prime]                                 # align lengths

        # Weighted joint pooling (learnable importance, softmax-normalized)
        joint_w = torch.softmax(self.joint_importance, dim=0)          # (V,)
        h = (h * joint_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=3)  # (B, 256, T')

        # Masked temporal pooling
        mask_t = ds_mask.unsqueeze(1)                                  # (B, 1, T')
        pooled = (h * mask_t).sum(dim=2) / mask_t.sum(dim=2).clamp(min=1)   # (B, 256)

        # Concat non-joint features (masked mean over original time)
        rest = x[:, :, self._rest_idx]                                 # (B, T, 23)
        mask_f = mask.unsqueeze(-1)
        rest_pooled = (rest * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        pooled = torch.cat([pooled, rest_pooled], dim=-1)

        return self.head(pooled)


# ================================================================
# Joint compression: CNN over joint dimension
# ================================================================
class JointCompressor(nn.Module):
    """Compress 144 pose+velocity features via 1D CNN over the joint axis.

    Reshapes (B, T, 144) → (B*T, 2, 72) treating pose and velocity as
    2 channels over 24 joints × 3 coords, then compresses to out_dim.
    """
    def __init__(self, out_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, joints):
        # joints: (B, T, 144)
        B, T, _ = joints.shape
        x = joints.reshape(B * T, 2, 72)   # pose(72) + vel(72) as 2 channels
        x = self.cnn(x).squeeze(-1)         # (B*T, 32)
        x = self.fc(x)                      # (B*T, out_dim)
        return x.reshape(B, T, -1)


class JointCompressWrapper(nn.Module):
    """Wraps any temporal model with joint compression on the first 144 features.

    Drops per-joint validity channels that are meaningless after compression.
    Keeps: vel_wrist(3), scale(1), angles(16), frame_valid(1), hand_cov(1), arm_cov(1).
    Drops: joint_valid(21), arm_valid(3), hand_angle_valid(15), arm_angle_valid(1) = 40 channels.
    """
    # Indices into the 63 remaining channels (i.e., x[:, :, 144:])
    # that we KEEP: vel_wrist[0:3], scale[3], angles[4:20], frame_valid[20],
    #               hand_cov[45], arm_cov[46]
    _KEEP_IDX = list(range(0, 21)) + [45, 46]  # 21 + 2 = 23 channels kept

    def __init__(self, compressor, temporal_model):
        super().__init__()
        self.compressor = compressor
        self.temporal_model = temporal_model

    def forward(self, x, mask):
        joints = x[:, :, :144]
        rest = x[:, :, 144:][:, :, self._KEEP_IDX]
        compressed = self.compressor(joints)
        x_new = torch.cat([compressed, rest], dim=-1)
        return self.temporal_model(x_new, mask)


# ================================================================
# Model factory
# ================================================================
def build_model(args, in_dim, n_classes):
    model_in_dim = in_dim
    if args.compress_joints:
        # compress_joints_dim + 23 kept channels (vel_wrist, scale, angles, frame_valid, hand_cov, arm_cov)
        model_in_dim = args.compress_joints_dim + len(JointCompressWrapper._KEEP_IDX)

    if args.model == "bilstm":
        temporal = AttentionBiLSTM(model_in_dim, n_classes, hidden=args.hidden,
                                   lstm_layers=args.lstm_layers, attn_heads=args.attn_heads,
                                   proj_dim=args.proj_dim, dropout=args.dropout)
    elif args.model == "unet1d":
        temporal = UNet1D(model_in_dim, n_classes, base_channels=args.proj_dim, dropout=args.dropout)
    elif args.model == "resnet1d":
        temporal = ResNet1D(model_in_dim, n_classes, base_channels=args.proj_dim, dropout=args.dropout)
    elif args.model == "unet1d_lstm":
        temporal = UNet1DLstm(model_in_dim, n_classes, base_channels=args.proj_dim,
                              hidden=args.hidden, lstm_layers=args.lstm_layers,
                              attn_heads=args.attn_heads, dropout=args.dropout)
    elif args.model == "resnet1d_lstm":
        temporal = ResNet1DLstm(model_in_dim, n_classes, base_channels=args.proj_dim,
                                hidden=args.hidden, lstm_layers=args.lstm_layers,
                                attn_heads=args.attn_heads, dropout=args.dropout)
    elif args.model == "stgcn":
        # ST-GCN handles its own feature extraction; bypass compress_joints
        return SpatioTemporalGCN(in_dim, n_classes, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.compress_joints:
        return JointCompressWrapper(JointCompressor(args.compress_joints_dim), temporal)
    return temporal


# ================================================================
# Standardization — continuous channels only
# ================================================================
# Channel index of the bone-length scale within the 207-dim feature vector.
# Layout: pose(72) + vel_pose(72) + vel_wrist(3) + scale(1) = index 147.
_SCALE_CH = 147


def standardize_fit(train_items, normalize_scale=True):
    Xtr = np.stack([it["X"] for it in train_items], 0)
    Xc = Xtr[:, :, :N_CONTINUOUS]
    mu = Xc.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    sd = Xc.std(axis=(0, 1), keepdims=True).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    # Fit per-fold mean of the raw (pre-standardization) scale channel so we
    # can divide it out in standardize_apply, removing between-subject OOD shift.
    scale_mean = float(Xtr[:, :, _SCALE_CH].mean()) if normalize_scale else None
    return mu, sd, scale_mean


def standardize_apply(items, mu, sd, scale_mean=None):
    out = []
    for it in items:
        X = it["X"].copy()
        # Optionally normalize scale channel BEFORE global standardization so
        # that between-subject absolute scale differences don't cause OOD shifts.
        # The XYZ pose channels are already bone-length normalized, so this is
        # safe — it only removes the redundant absolute-size information.
        if scale_mean is not None and scale_mean > 1e-8:
            X[:, _SCALE_CH] /= scale_mean
        X[:, :N_CONTINUOUS] = (X[:, :N_CONTINUOUS] - mu.squeeze(0)) / (sd.squeeze(0) + 1e-8)
        out.append(dict(X=X.astype(np.float32), valid=it["valid"],
                        y42=it["y42"], pid=it.get("pid", "?"), path=it.get("path", "")))
    return out


# ================================================================
# 7-class head derivation
# ================================================================
def derive_head_7_from_42(state_dict):
    head_w_name = head_b_name = None
    for k in state_dict.keys():
        if "head" in k and k.endswith("weight"):
            head_w_name = k
            head_b_name = k.replace("weight", "bias")
            break
    if head_w_name is None:
        raise RuntimeError("Could not find classifier head in state_dict")
    W42 = state_dict[head_w_name].clone()
    b42 = state_dict[head_b_name].clone()
    if W42.size(0) != 42:
        raise RuntimeError(f"Expected 42 output rows, got {W42.size(0)}")
    D = W42.size(1)
    W7 = torch.zeros(7, D, dtype=W42.dtype)
    b7 = torch.zeros(7, dtype=b42.dtype)
    for j in range(7):
        idx = torch.tensor([j, j+7, j+14, j+21, j+28, j+35], dtype=torch.long)
        W7[j] = W42.index_select(0, idx).mean(dim=0)
        b7[j] = b42.index_select(0, idx).mean()
    return head_w_name, head_b_name, W7, b7


# ================================================================
# Training & Evaluation
# ================================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, device, clip_grad, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0; n_samples = 0; n_skipped = 0
    for X, M, y in loader:
        X, M, y = X.to(device), M.to(device), y.to(device)
        if mixup_alpha > 0 and np.random.rand() < 0.5:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(X.size(0), device=device)
            X_mix = lam * X + (1 - lam) * X[idx]
            M_mix = lam * M + (1 - lam) * M[idx]
            logits = model(X_mix, M_mix)
            loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[idx])
        else:
            loss = criterion(model(X, M), y)
        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad()
            scheduler.step()
            continue
        optimizer.zero_grad(); loss.backward()
        if clip_grad > 0:
            clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step(); scheduler.step()
        total_loss += loss.item() * X.size(0); n_samples += X.size(0)
    if n_skipped:
        logging.getLogger(__name__).warning(f"Skipped {n_skipped} NaN/Inf batches this epoch")
    return total_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for X, M, y in loader:
        X, M = X.to(device), M.to(device)
        all_p.append(model(X, M).argmax(dim=1).cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_p)


def macro_f1(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)))


# ================================================================
# Plotting
# ================================================================
def plot_confusion_matrix(y_true, y_pred, n_classes, title, save_path):
    if not HAS_MPL: return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.3), max(8, n_classes * 0.3)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_training_curves(history, save_path):
    if not HAS_MPL or not history: return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"], label="train"); axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(epochs, history["val_f1"], label="val F1"); axes[1].set_title("Macro F1"); axes[1].legend()
    axes[2].plot(epochs, history["val_acc"], label="val Acc"); axes[2].set_title("Accuracy"); axes[2].legend()
    for ax in axes: ax.set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# ================================================================
# LOSO / K-fold splits
# ================================================================
def generate_loso_splits(items):
    pids = sorted(set(it["pid"] for it in items), key=lambda x: int(x))
    for i, test_pid in enumerate(pids):
        val_pid = pids[(i + 1) % len(pids)]
        yield (
            [it for it in items if it["pid"] != test_pid and it["pid"] != val_pid],
            [it for it in items if it["pid"] == val_pid],
            [it for it in items if it["pid"] == test_pid],
            f"LOSO-{test_pid}",
        )


def generate_kfold_splits(items, k, seed):
    ys = np.array([it["y42"] for it in items])
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(np.zeros(len(ys)), ys)):
        tr_idx = list(tr_idx); np.random.shuffle(tr_idx)
        n_val = max(1, int(0.1 * len(tr_idx)))
        yield ([items[i] for i in tr_idx[n_val:]],
               [items[i] for i in tr_idx[:n_val]],
               [items[i] for i in te_idx],
               f"fold-{fold_idx + 1}")


# ================================================================
# Main
# ================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "training.log", mode="w"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    items = load_items(args.roots)
    if not items:
        log.error("No training clips found."); return

    if args.condition_vec:
        for it in items:
            T = it["X"].shape[0]
            cond_onehot = np.zeros((T, 6), dtype=np.float32)
            cond_onehot[:, it["condition"]] = 1.0
            it["X"] = np.concatenate([it["X"], cond_onehot], axis=1)
        log.info("Appended 6-dim condition one-hot → feature dim %d", items[0]["X"].shape[1])

    n_classes = args.n_classes
    if n_classes == 7:
        for it in items:
            it["y42"] = it["y42"] % 7

    in_dim = items[0]["X"].shape[-1]
    log.info(f"N={len(items)}, in_dim={in_dim}, n_classes={n_classes}, N_CONTINUOUS={N_CONTINUOUS}")
    if in_dim != 207:
        log.warning(f"Expected 207-dim but got {in_dim}.")

    aug_cfg = None if args.no_aug else dict(
        speed=args.aug_speed, warp=args.aug_warp,
        noise_std=args.aug_noise_std, frame_drop_prob=args.aug_frame_drop_prob,
        block_mask_prob=args.aug_block_mask_prob, scale_jitter=args.aug_scale_jitter,
        rotate_prob=args.aug_rotate_prob,
        rotate_max_deg=args.aug_rotate_max_deg,
        temporal_crop_prob=args.aug_temporal_crop_prob,
        scale_lo=args.aug_scale_lo,
        scale_hi=args.aug_scale_hi,
    )
    mixup_alpha = 0.0 if args.no_aug else args.mixup_alpha

    splits = list(generate_loso_splits(items) if args.eval_mode == "loso"
                  else generate_kfold_splits(items, args.k, args.seed))
    log.info(f"Evaluation: {args.eval_mode} — {len(splits)} folds")

    all_y_test, all_p_test, all_preds_with_pid = [], [], []
    fold_results = []
    best_global_f1, best_global_state, best_global_mu, best_global_sd, best_global_scale_mean = -1.0, None, None, None, None
    best_global_val_f1 = -1.0
    best_global_fold = None

    for fold_i, (train_items, val_items, test_items, fold_name) in enumerate(splits):
        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold_i+1}/{len(splits)}: {fold_name}  "
                 f"(train={len(train_items)} val={len(val_items)} test={len(test_items)})")
        log.info(f"{'='*60}")

        mu, sd, scale_mean = standardize_fit(train_items, args.normalize_scale_channel)
        train_std = standardize_apply(train_items, mu, sd, scale_mean)
        val_std   = standardize_apply(val_items,   mu, sd, scale_mean)
        test_std  = standardize_apply(test_items,  mu, sd, scale_mean)

        ds_train = AugmentedClipSet(train_std, augment=(aug_cfg is not None),
                                    aug_cfg=aug_cfg, clip_norm=args.clip_norm)
        ds_val   = AugmentedClipSet(val_std,   augment=False, clip_norm=args.clip_norm)
        ds_test  = AugmentedClipSet(test_std,  augment=False, clip_norm=args.clip_norm)

        dl_train = DataLoader(ds_train, args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers)
        dl_val   = DataLoader(ds_val,   args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=args.num_workers)
        dl_test  = DataLoader(ds_test,  args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=args.num_workers)

        model = build_model(args, in_dim, n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=args.epochs * len(dl_train),
            pct_start=0.1, anneal_strategy="cos")
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best_state, best_val_f1, no_improve = None, -1.0, 0
        history = {"train_loss": [], "val_f1": [], "val_acc": []}

        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, dl_train, optimizer, scheduler,
                                  criterion, device, args.clip_grad, mixup_alpha)
            y_val, p_val = evaluate(model, dl_val, device)
            vf1 = macro_f1(y_val, p_val, n_classes)
            vacc = accuracy_score(y_val, p_val)
            history["train_loss"].append(tr_loss)
            history["val_f1"].append(vf1)
            history["val_acc"].append(vacc)
            log.info(f"[{fold_name} | Ep {ep:03d}] loss={tr_loss:.4f}  "
                     f"F1={vf1:.3f}  Acc={vacc:.3f}  ({time.time()-t0:.1f}s)")
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    log.info(f"[{fold_name}] Early stop at epoch {ep} (best F1={best_val_f1:.3f})")
                    break

        plot_training_curves(history, out_dir / f"curves_{fold_name}.png")
        model.load_state_dict(best_state)
        y_test, p_test = evaluate(model, dl_test, device)
        tf1  = macro_f1(y_test, p_test, n_classes)
        tacc = accuracy_score(y_test, p_test)
        log.info(f"[{fold_name} DONE] Test F1={tf1:.3f}  Acc={tacc:.3f}  "
                 f"(best val F1={best_val_f1:.3f})")

        fold_results.append(dict(fold=fold_name, val_f1=float(best_val_f1),
                                 test_f1=float(tf1), test_acc=float(tacc),
                                 n_train=len(train_items), n_val=len(val_items),
                                 n_test=len(test_items)))
        all_y_test.append(y_test); all_p_test.append(p_test)
        for idx, it in enumerate(test_std):
            all_preds_with_pid.append(dict(pid=it["pid"], y=y_test[idx], p=p_test[idx]))
        plot_confusion_matrix(y_test, p_test, n_classes,
                              f"Confusion: {fold_name}", out_dir / f"cm_{fold_name}.png")
        # Save per-fold model immediately
        fold_artifact = dict(type=f"{args.model}_xyz_angles", in_dim=int(in_dim),
                             n_continuous=N_CONTINUOUS, n_classes=n_classes,
                             hidden=args.hidden, lstm_layers=args.lstm_layers,
                             attn_heads=args.attn_heads, proj_dim=args.proj_dim, dropout=args.dropout,
                             mu=mu.squeeze().astype(np.float32),
                             sd=sd.squeeze().astype(np.float32),
                             scale_mean=float(scale_mean) if scale_mean is not None else None,
                             clip_norm=args.clip_norm,
                             state_dict=best_state,
                             fold=fold_name, val_f1=float(best_val_f1),
                             test_f1=float(tf1), test_acc=float(tacc))
        torch.save(fold_artifact, out_dir / f"model_{n_classes}_{fold_name}.pt")
        log.info(f"Saved model_{n_classes}_{fold_name}.pt  (test F1={tf1:.3f})")

        if tf1 > best_global_f1:
            best_global_f1 = tf1
            best_global_val_f1 = float(best_val_f1)
            best_global_fold = fold_name
            best_global_state, best_global_mu, best_global_sd, best_global_scale_mean = best_state, mu, sd, scale_mean

    # Aggregate
    all_y = np.concatenate(all_y_test); all_p = np.concatenate(all_p_test)
    agg_f1 = macro_f1(all_y, all_p, n_classes)
    agg_acc = accuracy_score(all_y, all_p)
    log.info(f"\n{'='*60}\nAGGREGATE: F1={agg_f1:.3f}  Acc={agg_acc:.3f}\n{'='*60}")
    plot_confusion_matrix(all_y, all_p, n_classes,
                          f"Aggregate ({args.eval_mode})", out_dir / "cm_aggregate.png")

    # Per-participant report
    lines = ["Participant  N_test  Accuracy  Macro-F1", "-" * 50]
    for pid in sorted(set(d["pid"] for d in all_preds_with_pid), key=lambda x: int(x)):
        sub = [d for d in all_preds_with_pid if d["pid"] == pid]
        ys = np.array([d["y"] for d in sub]); ps = np.array([d["p"] for d in sub])
        lines.append(f"{pid:>11s}  {len(sub):>6d}  {accuracy_score(ys,ps):>8.3f}  "
                     f"{macro_f1(ys,ps,n_classes):>8.3f}")
    report = "\n".join(lines)
    log.info(f"\nPer-participant report:\n{report}")
    (out_dir / "per_participant_report.txt").write_text(report + "\n")

    if n_classes == 42:
        clines = ["Condition  N_test  Accuracy  Macro-F1", "-" * 50]
        for cond in range(6):
            mask = (all_y // 7) == cond
            if not mask.any(): continue
            clines.append(f"{cond:>9d}  {int(mask.sum()):>6d}  "
                          f"{accuracy_score(all_y[mask],all_p[mask]):>8.3f}  "
                          f"{f1_score(all_y[mask],all_p[mask],average='macro',labels=list(range(42))):>8.3f}")
        cond_report = "\n".join(clines)
        log.info(f"\nPer-condition report:\n{cond_report}")
        (out_dir / "per_condition_report.txt").write_text(cond_report + "\n")

    with open(out_dir / "results.json", "w") as f:
        json.dump(dict(eval_mode=args.eval_mode, n_classes=n_classes,
                       aggregate_f1=float(agg_f1), aggregate_acc=float(agg_acc),
                       folds=fold_results), f, indent=2)

    if best_global_state is None:
        log.error("No model to save."); return

    model_key = f"model_{n_classes}.pt"
    artifact = dict(type=f"{args.model}_xyz_angles", in_dim=int(in_dim),
                    n_continuous=N_CONTINUOUS, n_classes=n_classes,
                    hidden=args.hidden, lstm_layers=args.lstm_layers,
                    attn_heads=args.attn_heads, proj_dim=args.proj_dim, dropout=args.dropout,
                    mu=best_global_mu.squeeze().astype(np.float32),
                    sd=best_global_sd.squeeze().astype(np.float32),
                    scale_mean=float(best_global_scale_mean) if best_global_scale_mean is not None else None,
                    clip_norm=args.clip_norm,
                    state_dict=best_global_state,
                    # provenance / eval metadata
                    roots=[str(r) for r in args.roots],
                    eval_mode=args.eval_mode,
                    best_fold=best_global_fold,
                    best_val_f1=best_global_val_f1,
                    best_test_f1=float(best_global_f1),
                    aggregate_f1=float(agg_f1),
                    aggregate_acc=float(agg_acc),
                    model_path=str(out_dir / model_key))
    torch.save(artifact, out_dir / model_key)
    log.info(f"Saved {model_key}")

    if n_classes == 42:
        _, _, W7, b7 = derive_head_7_from_42(best_global_state)
        m7 = build_model(args, in_dim, 7)
        sd7 = m7.state_dict()
        for k, v in sd7.items():
            if "head" in k: continue
            if k in best_global_state and best_global_state[k].shape == v.shape:
                sd7[k] = best_global_state[k].clone()
        sd7[[k for k in sd7 if "head" in k and k.endswith("weight")][0]] = W7
        sd7[[k for k in sd7 if "head" in k and k.endswith("bias")][0]] = b7
        m7.load_state_dict(sd7)
        torch.save(dict(**{k: artifact[k] for k in artifact if k not in ("n_classes", "state_dict", "model_path")},
                        n_classes=7, state_dict=m7.state_dict(),
                        model_path=str(out_dir / "model_7.pt"),
                        derived_from=str(out_dir / model_key)),
                   out_dir / "model_7.pt")
        log.info("Saved model_7.pt")

    log.info(f"\nAll outputs in: {out_dir}")
    log.info(f"Aggregate F1={agg_f1:.3f}  Acc={agg_acc:.3f}")


if __name__ == "__main__":
    main()
