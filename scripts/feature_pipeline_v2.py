#!/usr/bin/env python3
"""Shared feature construction for Home+ gesture pipeline v2.

The canonical representation remains structured. Sequence models may flatten
it, ST-GCN keeps the joint axis, and SVM summarizes the same information.
All functions in this module are NumPy-only so offline processing, training,
and live recognition can import exactly the same feature code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


T_TARGET = 64
EPS = 1e-6

# MediaPipe hand topology. Parent 0 for the five roots, then along each finger.
HAND_PARENTS = np.array([
    0, 0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
], dtype=np.int64)

FLEXION_TRIPLETS = np.array([
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
], dtype=np.int64)

# Cosines between adjacent palm rays. These complement within-finger flexion.
SPREAD_PAIRS = np.array([(1, 5), (5, 9), (9, 13), (13, 17)], dtype=np.int64)
ANGLE_COUNT = len(FLEXION_TRIPLETS) + len(SPREAD_PAIRS)  # 19


@dataclass(frozen=True)
class FeatureLayout:
    feature_set: str
    slices: Dict[str, Tuple[int, int]]
    dim: int


def _safe_unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, np.maximum(n, EPS), out=np.zeros_like(x), where=n > EPS)


def causal_ema(values: np.ndarray, valid: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Causal validity-aware EMA, suitable for both offline and live use."""
    out = np.zeros_like(values, dtype=np.float32)
    state = np.zeros(values.shape[1:], dtype=np.float32)
    initialized = False
    for t in range(len(values)):
        if bool(valid[t]):
            v = values[t].astype(np.float32)
            state = v if not initialized else alpha * v + (1.0 - alpha) * state
            initialized = True
            out[t] = state
        elif initialized:
            out[t] = state
    return out


def masked_resample(values: np.ndarray, valid: np.ndarray, target: int = T_TARGET) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly resample valid observations and return a nearest mask.

    Invalid output steps are explicitly zeroed. Interpolation provides a stable
    value through short holes, while the mask continues to tell learners that
    the observation itself was absent.
    """
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    n = len(values)
    out_shape = (target,) + values.shape[1:]
    if n == 0 or not valid.any():
        return np.zeros(out_shape, np.float32), np.zeros(target, bool)

    src_t = np.arange(n, dtype=np.float32)
    dst_t = np.linspace(0, max(n - 1, 0), target, dtype=np.float32)
    valid_idx = np.flatnonzero(valid)
    flat = values.reshape(n, -1)
    out = np.empty((target, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst_t, src_t[valid_idx], flat[valid_idx, j])

    nearest = np.clip(np.rint(dst_t).astype(np.int64), 0, n - 1)
    out_valid = valid[nearest]
    out[~out_valid] = 0.0
    return out.reshape(out_shape), out_valid


def velocity(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dx = np.diff(x, axis=0, prepend=x[0:1]).astype(np.float32)
    good = valid & np.concatenate(([False], valid[:-1]))
    dx[~good] = 0.0
    return dx


def compute_hand_bones(hand: np.ndarray, joint_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bones = np.zeros_like(hand, dtype=np.float32)
    bone_valid = np.zeros(joint_valid.shape, dtype=bool)
    for j in range(1, 21):
        p = int(HAND_PARENTS[j])
        good = joint_valid[:, j] & joint_valid[:, p]
        bones[good, j] = hand[good, j] - hand[good, p]
        bone_valid[good, j] = True
    return bones, bone_valid


def compute_angle_cosines(hand: np.ndarray, joint_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return 15 flexion and 4 finger-spread cosine features."""
    t = hand.shape[0]
    out = np.zeros((t, ANGLE_COUNT), dtype=np.float32)
    valid = np.zeros((t, ANGLE_COUNT), dtype=bool)
    for i, (p, j, c) in enumerate(FLEXION_TRIPLETS):
        good = joint_valid[:, p] & joint_valid[:, j] & joint_valid[:, c]
        v1 = hand[:, p] - hand[:, j]
        v2 = hand[:, c] - hand[:, j]
        cos = (_safe_unit(v1) * _safe_unit(v2)).sum(axis=1).clip(-1.0, 1.0)
        out[good, i] = cos[good]
        valid[good, i] = True
    offset = len(FLEXION_TRIPLETS)
    for i, (a, b) in enumerate(SPREAD_PAIRS):
        good = joint_valid[:, 0] & joint_valid[:, a] & joint_valid[:, b]
        va = hand[:, a] - hand[:, 0]
        vb = hand[:, b] - hand[:, 0]
        cos = (_safe_unit(va) * _safe_unit(vb)).sum(axis=1).clip(-1.0, 1.0)
        out[good, offset + i] = cos[good]
        valid[good, offset + i] = True
    return out, valid


def palm_orientation(hand: np.ndarray, joint_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Palm forward axis and normal, six channels in the camera frame."""
    good = joint_valid[:, 0] & joint_valid[:, 5] & joint_valid[:, 9] & joint_valid[:, 17]
    forward = _safe_unit(hand[:, 9] - hand[:, 0])
    across = _safe_unit(hand[:, 5] - hand[:, 17])
    normal = _safe_unit(np.cross(across, forward))
    out = np.concatenate((forward, normal), axis=1).astype(np.float32)
    out[~good] = 0.0
    return out, good


def robust_hand_scale(hand_wrist_relative_m: np.ndarray, joint_valid: np.ndarray) -> np.ndarray:
    """Per-frame median wrist-to-MCP scale with robust clip-level fallback."""
    n = len(hand_wrist_relative_m)
    raw = np.full(n, np.nan, dtype=np.float32)
    for t in range(n):
        ds = []
        for j in (5, 9, 17):
            if joint_valid[t, j]:
                d = float(np.linalg.norm(hand_wrist_relative_m[t, j]))
                if 0.02 <= d <= 0.20:
                    ds.append(d)
        if ds:
            raw[t] = float(np.median(ds))
    finite = np.isfinite(raw)
    fallback = float(np.median(raw[finite])) if finite.any() else 0.075
    raw[~finite] = fallback
    return causal_ema(raw[:, None], np.ones(n, bool), alpha=0.25).squeeze(1)


def robust_arm_scale(arm_wrist_relative_m: np.ndarray, arm_valid: np.ndarray) -> np.ndarray:
    """Shoulder-elbow plus elbow-wrist length, with clip-level fallback."""
    n = len(arm_wrist_relative_m)
    raw = np.full(n, np.nan, dtype=np.float32)
    for t in range(n):
        if arm_valid[t].all():
            upper = np.linalg.norm(arm_wrist_relative_m[t, 0] - arm_wrist_relative_m[t, 1])
            fore = np.linalg.norm(arm_wrist_relative_m[t, 1] - arm_wrist_relative_m[t, 2])
            total = float(upper + fore)
            if 0.20 <= total <= 1.20:
                raw[t] = total
    finite = np.isfinite(raw)
    fallback = float(np.median(raw[finite])) if finite.any() else 0.60
    raw[~finite] = fallback
    return causal_ema(raw[:, None], np.ones(n, bool), alpha=0.25).squeeze(1)


def build_canonical_features(
    hand_xyz_m: np.ndarray,
    arm_xyz_m: np.ndarray,
    wrist_xyz_m: np.ndarray,
    joint_valid: np.ndarray,
    arm_valid: np.ndarray,
    frame_valid: np.ndarray,
    target: int = T_TARGET,
    smooth_alpha: float = 0.45,
) -> Dict[str, np.ndarray]:
    """Build the canonical v2 tensors from cleaned metric coordinates."""
    hand_xyz_m = np.asarray(hand_xyz_m, np.float32)
    arm_xyz_m = np.asarray(arm_xyz_m, np.float32)
    wrist_xyz_m = np.asarray(wrist_xyz_m, np.float32)
    joint_valid = np.asarray(joint_valid, bool)
    arm_valid = np.asarray(arm_valid, bool)
    frame_valid = np.asarray(frame_valid, bool)

    # Smooth each joint only over frames where that joint is trusted.
    hand_s = np.zeros_like(hand_xyz_m)
    for j in range(21):
        hand_s[:, j] = causal_ema(hand_xyz_m[:, j], joint_valid[:, j], smooth_alpha)
    arm_s = np.zeros_like(arm_xyz_m)
    for j in range(3):
        arm_s[:, j] = causal_ema(arm_xyz_m[:, j], arm_valid[:, j], smooth_alpha)
    wrist_s = causal_ema(wrist_xyz_m, frame_valid, smooth_alpha)

    hand_scale_raw = robust_hand_scale(hand_s, joint_valid)
    arm_scale_raw = robust_arm_scale(arm_s, arm_valid)
    hand_local_raw = hand_s / np.maximum(hand_scale_raw[:, None, None], EPS)
    hand_local_raw[~joint_valid] = 0.0

    # Shoulder-centred arm, scaled by arm length rather than hand size.
    arm_shoulder = arm_s - arm_s[:, 0:1]
    arm_local_raw = arm_shoulder / np.maximum(arm_scale_raw[:, None, None], EPS)
    arm_local_raw[~arm_valid] = 0.0

    bones_raw, bone_valid_raw = compute_hand_bones(hand_local_raw, joint_valid)
    angles_raw, angle_valid_raw = compute_angle_cosines(hand_local_raw, joint_valid)
    palm_raw, palm_valid_raw = palm_orientation(hand_s, joint_valid)

    first = int(np.flatnonzero(frame_valid)[0]) if frame_valid.any() else 0
    wrist_disp_raw = wrist_s - wrist_s[first:first + 1]
    wrist_disp_raw[~frame_valid] = 0.0

    hand, frame_t = masked_resample(hand_local_raw, frame_valid, target)
    arm, _ = masked_resample(arm_local_raw, frame_valid, target)
    wrist_disp, _ = masked_resample(wrist_disp_raw, frame_valid, target)
    palm, palm_t = masked_resample(palm_raw, palm_valid_raw, target)
    angles = np.zeros((target, ANGLE_COUNT), np.float32)
    scales, _ = masked_resample(
        np.stack((hand_scale_raw, arm_scale_raw, wrist_s[:, 2]), axis=1), frame_valid, target
    )

    joint_t = np.zeros((target, 21), bool)
    bone_t = np.zeros((target, 21), bool)
    angle_t = np.zeros((target, ANGLE_COUNT), bool)
    arm_t = np.zeros((target, 3), bool)
    for j in range(21):
        _, joint_t[:, j] = masked_resample(hand_local_raw[:, j], joint_valid[:, j], target)
        _, bone_t[:, j] = masked_resample(bones_raw[:, j], bone_valid_raw[:, j], target)
    for j in range(ANGLE_COUNT):
        angle_j, angle_t[:, j] = masked_resample(
            angles_raw[:, j:j + 1], angle_valid_raw[:, j], target
        )
        angles[:, j] = angle_j[:, 0]
    for j in range(3):
        _, arm_t[:, j] = masked_resample(arm_local_raw[:, j], arm_valid[:, j], target)

    hand[~joint_t] = 0.0
    arm[~arm_t] = 0.0
    angles[~angle_t] = 0.0
    palm[~palm_t] = 0.0
    hand_vel = velocity(hand.reshape(target, -1), frame_t).reshape(target, 21, 3)
    bones, bone_t2 = compute_hand_bones(hand, joint_t)
    bone_t &= bone_t2
    arm_vel = velocity(arm.reshape(target, -1), frame_t).reshape(target, 3, 3)
    wrist_vel = velocity(wrist_disp, frame_t)
    wrist_acc = velocity(wrist_vel, frame_t)

    # Global stream: displacement, velocity, acceleration, palm basis, and
    # three scale/distance diagnostics. Scale channels are optional at load.
    global_features = np.concatenate(
        (wrist_disp, wrist_vel, wrist_acc, palm, scales), axis=1
    ).astype(np.float32)  # 18

    return {
        "hand_pos": hand.astype(np.float32),
        "hand_vel": hand_vel.astype(np.float32),
        "hand_bone": bones.astype(np.float32),
        "hand_angles": angles.astype(np.float32),
        "arm_pos": arm.astype(np.float32),
        "arm_vel": arm_vel.astype(np.float32),
        "global_features": global_features,
        "frame_valid": frame_t.astype(np.float32),
        "joint_valid": joint_t.astype(np.float32),
        "bone_valid": bone_t.astype(np.float32),
        "angle_valid": angle_t.astype(np.float32),
        "arm_valid": arm_t.astype(np.float32),
    }


def flatten_sequence(data: Dict[str, np.ndarray], feature_set: str = "full") -> Tuple[np.ndarray, FeatureLayout]:
    """Flatten canonical tensors for TCN/BiLSTM with named feature sets."""
    allowed = {"core", "bones", "angles", "full"}
    if feature_set not in allowed:
        raise ValueError(f"feature_set must be one of {sorted(allowed)}, got {feature_set!r}")
    parts = [
        ("hand_pos", data["hand_pos"].reshape(T_TARGET, -1)),
        ("hand_vel", data["hand_vel"].reshape(T_TARGET, -1)),
        ("arm_pos", data["arm_pos"].reshape(T_TARGET, -1)),
        ("arm_vel", data["arm_vel"].reshape(T_TARGET, -1)),
        # Scale diagnostics (last 3) are stored for explicit ablations only.
        ("global", data["global_features"][:, :15]),
    ]
    if feature_set in {"bones", "full"}:
        parts.append(("hand_bone", data["hand_bone"].reshape(T_TARGET, -1)))
    if feature_set in {"angles", "full"}:
        parts.extend([
            ("angles", data["hand_angles"]),
            ("angle_valid", data["angle_valid"]),
        ])
    parts.extend([
        ("joint_valid", data["joint_valid"]),
        ("arm_valid", data["arm_valid"]),
        ("frame_valid", data["frame_valid"][:, None]),
    ])
    slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    arrays = []
    for name, arr in parts:
        arr = np.asarray(arr, np.float32)
        arrays.append(arr)
        slices[name] = (cursor, cursor + arr.shape[1])
        cursor += arr.shape[1]
    return np.concatenate(arrays, axis=1), FeatureLayout(feature_set, slices, cursor)


def svm_summary(sequence: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fixed descriptor: moments, endpoints, and two-bin temporal means."""
    sequence = np.asarray(sequence, np.float32)
    valid = np.asarray(valid, bool)
    x = sequence[valid] if valid.any() else sequence
    mid = max(1, len(x) // 2)
    stats = [
        x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0),
        x[-1] - x[0], x[:mid].mean(axis=0), x[mid:].mean(axis=0),
    ]
    return np.concatenate(stats).astype(np.float32)


def canonical_keys() -> Iterable[str]:
    return (
        "hand_pos", "hand_vel", "hand_bone", "hand_angles", "arm_pos",
        "arm_vel", "global_features", "frame_valid", "joint_valid",
        "bone_valid", "angle_valid", "arm_valid",
    )
