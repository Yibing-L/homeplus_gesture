#!/usr/bin/env python3
"""EDA: Time-series comparison of gestures across conditions.

Analyzes how gesture trajectories differ between experimental conditions:
  1. Mean trajectory envelopes per gesture x condition (pose, velocity, angles)
  2. DTW distance matrix: same gesture across conditions
  3. Velocity magnitude profiles over normalized time
  4. Joint angle evolution over normalized time
  5. Variance-over-time: where in the gesture does condition matter most?
"""

import os, glob, argparse, sys
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    print("matplotlib required"); sys.exit(1)

try:
    from dtw import dtw as dtw_func
    HAS_DTW = True
except ImportError:
    try:
        from scipy.spatial.distance import cdist
        HAS_DTW = False
    except ImportError:
        HAS_DTW = False

# ── Feature layout ──
BLOCK_SLICES = {
    "pose_xyz":   (0, 72),
    "vel_pose":   (72, 144),
    "vel_wrist":  (144, 147),
    "scale":      (147, 148),
    "angles":     (148, 164),
}
N_CONTINUOUS = 164

ANGLE_NAMES = [
    "thumb_CMC", "thumb_MCP", "thumb_IP",
    "index_MCP", "index_PIP", "index_DIP",
    "mid_MCP",   "mid_PIP",   "mid_DIP",
    "ring_MCP",  "ring_PIP",  "ring_DIP",
    "pinky_MCP", "pinky_PIP", "pinky_DIP",
    "elbow",
]

# Angle computation
HAND_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]
ARM_TRIPLETS = [(0, 1, 2)]
EPS = 1e-6


def compute_joint_angles(X):
    T = X.shape[0]
    hand_xyz = X[:, 0:63].reshape(T, 21, 3)
    arm_xyz  = X[:, 63:72].reshape(T, 3, 3)
    hand_jv  = X[:, 149:170].astype(bool)
    arm_jv   = X[:, 170:173].astype(bool)
    angles      = np.zeros((T, 16), dtype=np.float32)
    angle_valid = np.zeros((T, 16), dtype=np.float32)
    for ai, (p, j, c) in enumerate(HAND_TRIPLETS):
        valid = hand_jv[:, p] & hand_jv[:, j] & hand_jv[:, c]
        if valid.any():
            v1 = hand_xyz[valid, p] - hand_xyz[valid, j]
            v2 = hand_xyz[valid, c] - hand_xyz[valid, j]
            n1 = np.linalg.norm(v1, axis=1, keepdims=True).clip(EPS)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True).clip(EPS)
            cos_a = ((v1 / n1) * (v2 / n2)).sum(axis=1).clip(-1.0, 1.0)
            angles[valid, ai] = np.arccos(cos_a)
            angle_valid[valid, ai] = 1.0
    for ai, (p, j, c) in enumerate(ARM_TRIPLETS):
        valid = arm_jv[:, p] & arm_jv[:, j] & arm_jv[:, c]
        if valid.any():
            v1 = arm_xyz[valid, p] - arm_xyz[valid, j]
            v2 = arm_xyz[valid, c] - arm_xyz[valid, j]
            n1 = np.linalg.norm(v1, axis=1, keepdims=True).clip(EPS)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True).clip(EPS)
            cos_a = ((v1 / n1) * (v2 / n2)).sum(axis=1).clip(-1.0, 1.0)
            angles[valid, 15 + ai] = np.arccos(cos_a)
            angle_valid[valid, 15 + ai] = 1.0
    return angles, angle_valid


def insert_angles(X_raw):
    angles, angle_valid = compute_joint_angles(X_raw)
    return np.concatenate([
        X_raw[:, :148], angles, X_raw[:, 148:], angle_valid,
    ], axis=1).astype(np.float32)


def load_items(roots):
    items = []
    for r in roots:
        expanded = sorted(glob.glob(r)) if any(c in r for c in ("*", "?", "[")) else [r]
        for d in expanded:
            for path in sorted(glob.glob(os.path.join(d, "*.npz"))):
                z = np.load(path)
                if "X" not in z:
                    continue
                X_raw = z["X"].astype(np.float32)
                if X_raw.shape[1] != 175:
                    continue
                if "label" in z:
                    y42 = int(z["label"])
                elif "gid" in z:
                    y42 = int(z["gid"])
                else:
                    continue
                X = insert_angles(X_raw)
                cond = y42 // 7
                gesture = y42 % 7
                pid = str(int(z["subject_id"])) if "subject_id" in z else "?"
                items.append(dict(X=X, cond=cond, gesture=gesture, y42=y42,
                                  pid=pid, T=X.shape[0], path=path))
    print(f"Loaded {len(items)} clips")
    return items


def resample_to_length(X, target_T):
    """Linearly resample (T, D) array to (target_T, D)."""
    T = X.shape[0]
    if T == target_T:
        return X.copy()
    idx = np.linspace(0, T - 1, target_T)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (idx - lo)[:, None]
    return ((1 - w) * X[lo] + w * X[hi]).astype(np.float32)


def compute_dtw_distance(seq1, seq2):
    """Euclidean DTW distance between two (T, D) sequences."""
    if HAS_DTW:
        alignment = dtw_func(seq1, seq2)
        return alignment.distance / max(len(seq1), len(seq2))
    else:
        # Simple DTW via DP
        n, m = len(seq1), len(seq2)
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return cost[n, m] / max(n, m)


# ================================================================
# Plot 1: Mean trajectory envelopes (gesture x condition)
# ================================================================
def plot_trajectory_envelopes(items, out_dir, norm_T=60):
    """For key feature channels, plot mean +/- std trajectory per condition for each gesture."""
    conds = sorted(set(it["cond"] for it in items))
    gestures = sorted(set(it["gesture"] for it in items))
    colors = cm.tab10(np.linspace(0, 1, max(len(conds), 1)))
    t_axis = np.linspace(0, 1, norm_T)

    # Channels to plot — pose is wrist-relative so wrist=[0,0,0].
    # Use fingertip joints and other meaningful signals instead.
    # Joint indices (21 hand joints x 3): index_tip=8*3=24, thumb_tip=4*3=12, middle_tip=12*3=36
    channel_groups = [
        ("index_tip_x", lambda X: X[:, 24:25]),
        ("index_tip_y", lambda X: X[:, 25:26]),
        ("index_tip_z", lambda X: X[:, 26:27]),
        ("thumb_tip_x", lambda X: X[:, 12:13]),
        ("thumb_tip_y", lambda X: X[:, 13:14]),
        ("pose_spread", lambda X: np.linalg.norm(X[:, 0:63].reshape(-1, 21, 3), axis=2).mean(axis=1, keepdims=True)),
        ("vel_magnitude", lambda X: np.linalg.norm(X[:, 72:144].reshape(-1, 24, 3), axis=2).mean(axis=1, keepdims=True)),
        ("vel_wrist_mag", lambda X: np.linalg.norm(X[:, 144:147], axis=1, keepdims=True)),
        ("index_MCP_angle", lambda X: X[:, 151:152]),
        ("thumb_CMC_angle", lambda X: X[:, 148:149]),
        ("elbow_angle", lambda X: X[:, 163:164]),
        ("scale", lambda X: X[:, 147:148]),
    ]

    for ch_name, ch_fn in channel_groups:
        n_gest = len(gestures)
        fig, axes = plt.subplots(1, n_gest, figsize=(4 * n_gest, 4), sharey=True)
        if n_gest == 1:
            axes = [axes]

        for gi, gest in enumerate(gestures):
            ax = axes[gi]
            for ci, cond in enumerate(conds):
                clips = [it for it in items if it["gesture"] == gest and it["cond"] == cond]
                if not clips:
                    continue
                # Resample each clip and extract channel
                resampled = []
                for it in clips:
                    Xr = resample_to_length(it["X"], norm_T)
                    val = ch_fn(Xr).squeeze(-1) if ch_fn(Xr).ndim > 1 else ch_fn(Xr)
                    resampled.append(val)
                stacked = np.stack(resampled, axis=0)  # (N_clips, norm_T)
                mu = stacked.mean(axis=0)
                sd = stacked.std(axis=0)
                ax.plot(t_axis, mu, color=colors[ci], label=f"cond {cond}", linewidth=1.5)
                ax.fill_between(t_axis, mu - sd, mu + sd, color=colors[ci], alpha=0.15)
            ax.set_title(f"gesture {gest}", fontsize=10)
            ax.set_xlabel("normalized time")
            if gi == 0:
                ax.set_ylabel(ch_name)

        axes[-1].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Trajectory envelopes: {ch_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_{ch_name}.png", dpi=150)
        plt.close(fig)
    print(f"Saved trajectory envelope plots ({len(channel_groups)} channels)")


# ================================================================
# Plot 2: Velocity magnitude profiles
# ================================================================
def plot_velocity_profiles(items, out_dir, norm_T=60):
    """Velocity magnitude over normalized time, per gesture x condition."""
    conds = sorted(set(it["cond"] for it in items))
    gestures = sorted(set(it["gesture"] for it in items))
    colors = cm.tab10(np.linspace(0, 1, max(len(conds), 1)))
    t_axis = np.linspace(0, 1, norm_T)

    n_gest = len(gestures)
    fig, axes = plt.subplots(2, (n_gest + 1) // 2, figsize=(5 * ((n_gest + 1) // 2), 8))
    axes = axes.flatten()

    for gi, gest in enumerate(gestures):
        ax = axes[gi]
        for ci, cond in enumerate(conds):
            clips = [it for it in items if it["gesture"] == gest and it["cond"] == cond]
            if not clips:
                continue
            vels = []
            for it in clips:
                Xr = resample_to_length(it["X"], norm_T)
                # vel_pose is 72 channels = 24 joints x 3
                v = np.linalg.norm(Xr[:, 72:144].reshape(norm_T, 24, 3), axis=2)
                vels.append(v.mean(axis=1))  # mean over joints → (norm_T,)
            stacked = np.stack(vels, axis=0)
            mu = stacked.mean(axis=0)
            sd = stacked.std(axis=0)
            ax.plot(t_axis, mu, color=colors[ci], label=f"cond {cond}", linewidth=1.5)
            ax.fill_between(t_axis, mu - sd, mu + sd, color=colors[ci], alpha=0.12)
        ax.set_title(f"gesture {gest}")
        ax.set_xlabel("normalized time")
        ax.set_ylabel("mean joint velocity")

    # Hide unused axes
    for i in range(n_gest, len(axes)):
        axes[i].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.suptitle("Velocity magnitude profiles by condition", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_profiles.png", dpi=150)
    plt.close(fig)
    print("Saved velocity_profiles.png")


# ================================================================
# Plot 3: Joint angle evolution
# ================================================================
def plot_angle_evolution(items, out_dir, norm_T=60):
    """All 16 angles over normalized time, one subplot per angle, lines per condition.
    Averaged across all gestures to show condition-level shift."""
    conds = sorted(set(it["cond"] for it in items))
    colors = cm.tab10(np.linspace(0, 1, max(len(conds), 1)))
    t_axis = np.linspace(0, 1, norm_T)

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()

    for ai in range(16):
        ax = axes[ai]
        for ci, cond in enumerate(conds):
            clips = [it for it in items if it["cond"] == cond]
            vals = []
            for it in clips:
                Xr = resample_to_length(it["X"], norm_T)
                vals.append(Xr[:, 148 + ai])
            stacked = np.stack(vals, axis=0)
            mu = stacked.mean(axis=0)
            sd = stacked.std(axis=0)
            ax.plot(t_axis, mu, color=colors[ci], label=f"c{cond}", linewidth=1.2)
            ax.fill_between(t_axis, mu - sd, mu + sd, color=colors[ci], alpha=0.1)
        ax.set_title(ANGLE_NAMES[ai], fontsize=9)
        ax.set_xlabel("t", fontsize=8)
        ax.set_ylabel("rad", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=6, ncol=3)
    fig.suptitle("Joint angle trajectories by condition (all gestures pooled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "angle_evolution_by_condition.png", dpi=150)
    plt.close(fig)

    # Also do per-gesture for a few key angles
    key_angles = [3, 0, 15]  # index_MCP, thumb_CMC, elbow
    gestures = sorted(set(it["gesture"] for it in items))
    for ai in key_angles:
        n_gest = len(gestures)
        fig, axes_g = plt.subplots(1, n_gest, figsize=(4 * n_gest, 4), sharey=True)
        if n_gest == 1:
            axes_g = [axes_g]
        for gi, gest in enumerate(gestures):
            ax = axes_g[gi]
            for ci, cond in enumerate(conds):
                clips = [it for it in items if it["gesture"] == gest and it["cond"] == cond]
                if not clips:
                    continue
                vals = [resample_to_length(it["X"], norm_T)[:, 148 + ai] for it in clips]
                stacked = np.stack(vals, axis=0)
                mu = stacked.mean(axis=0)
                sd = stacked.std(axis=0)
                ax.plot(t_axis, mu, color=colors[ci], label=f"c{cond}", linewidth=1.5)
                ax.fill_between(t_axis, mu - sd, mu + sd, color=colors[ci], alpha=0.12)
            ax.set_title(f"gesture {gest}", fontsize=10)
            ax.set_xlabel("normalized time")
        axes_g[0].set_ylabel(f"{ANGLE_NAMES[ai]} (rad)")
        axes_g[-1].legend(fontsize=7)
        fig.suptitle(f"{ANGLE_NAMES[ai]} trajectory per gesture x condition", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"angle_{ANGLE_NAMES[ai]}_per_gesture.png", dpi=150)
        plt.close(fig)

    print("Saved angle evolution plots")


# ================================================================
# Plot 4: Condition divergence over time
# ================================================================
def plot_divergence_over_time(items, out_dir, norm_T=60):
    """For each gesture, compute the between-condition variance at each time step
    relative to within-condition variance. High ratio = conditions differ at that moment."""
    conds = sorted(set(it["cond"] for it in items))
    gestures = sorted(set(it["gesture"] for it in items))
    t_axis = np.linspace(0, 1, norm_T)

    # Do this for: pose_xyz (0:72), vel_pose (72:144), angles (148:164)
    blocks = [("pose_xyz", 0, 72), ("vel_pose", 72, 144), ("angles", 148, 164)]

    for bname, bs, be in blocks:
        n_gest = len(gestures)
        fig, axes = plt.subplots(2, (n_gest + 1) // 2, figsize=(5 * ((n_gest + 1) // 2), 8))
        axes = axes.flatten()

        for gi, gest in enumerate(gestures):
            ax = axes[gi]
            # Collect resampled clips per condition
            cond_clips = {}
            for cond in conds:
                clips = [it for it in items if it["gesture"] == gest and it["cond"] == cond]
                if not clips:
                    continue
                resampled = np.stack([
                    resample_to_length(it["X"], norm_T)[:, bs:be] for it in clips
                ], axis=0)  # (N, norm_T, D)
                cond_clips[cond] = resampled

            if len(cond_clips) < 2:
                ax.set_visible(False)
                continue

            # Between-condition variance of means at each time step
            cond_means = np.stack([v.mean(axis=0) for v in cond_clips.values()], axis=0)  # (C, T, D)
            between_var = cond_means.var(axis=0).mean(axis=1)  # (T,) avg over features

            # Within-condition variance (averaged)
            within_var = np.mean([v.var(axis=0).mean(axis=1) for v in cond_clips.values()], axis=0)

            # F-ratio style
            ratio = between_var / (within_var + 1e-8)

            ax.plot(t_axis, ratio, color="black", linewidth=1.5)
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            ax.fill_between(t_axis, 0, ratio, alpha=0.3,
                            color="red" if ratio.max() > 2 else "steelblue")
            ax.set_title(f"gesture {gest}", fontsize=10)
            ax.set_xlabel("normalized time")
            ax.set_ylabel("between/within var ratio")

        for i in range(n_gest, len(axes)):
            axes[i].set_visible(False)
        fig.suptitle(f"Condition divergence over time: {bname}\n"
                     f"(ratio > 1 = conditions differ more than within-condition noise)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"divergence_over_time_{bname}.png", dpi=150)
        plt.close(fig)

    print("Saved divergence_over_time plots")


# ================================================================
# Plot 5: DTW distance matrix (condition x condition, per gesture)
# ================================================================
def plot_dtw_matrix(items, out_dir, norm_T=60, max_clips_per_cell=30):
    """Mean DTW distance between condition-averaged trajectories for each gesture."""
    conds = sorted(set(it["cond"] for it in items))
    gestures = sorted(set(it["gesture"] for it in items))
    n_conds = len(conds)

    # Use continuous features for DTW
    fig, axes = plt.subplots(2, (len(gestures) + 1) // 2,
                             figsize=(5 * ((len(gestures) + 1) // 2), 9))
    axes = axes.flatten()

    for gi, gest in enumerate(gestures):
        ax = axes[gi]
        # Compute mean trajectory per condition
        mean_trajs = {}
        for cond in conds:
            clips = [it for it in items if it["gesture"] == gest and it["cond"] == cond]
            if not clips:
                continue
            resampled = np.stack([
                resample_to_length(it["X"][:, :N_CONTINUOUS], norm_T) for it in clips
            ], axis=0)
            mean_trajs[cond] = resampled.mean(axis=0)  # (norm_T, D)

        dtw_mat = np.zeros((n_conds, n_conds))
        for i, ci in enumerate(conds):
            for j, cj in enumerate(conds):
                if i >= j or ci not in mean_trajs or cj not in mean_trajs:
                    continue
                d = compute_dtw_distance(mean_trajs[ci], mean_trajs[cj])
                dtw_mat[i, j] = d
                dtw_mat[j, i] = d

        im = ax.imshow(dtw_mat, cmap="YlOrRd")
        ax.set_xticks(range(n_conds))
        ax.set_xticklabels([f"c{c}" for c in conds], fontsize=8)
        ax.set_yticks(range(n_conds))
        ax.set_yticklabels([f"c{c}" for c in conds], fontsize=8)
        ax.set_title(f"gesture {gest}", fontsize=10)
        # Annotate cells
        for i in range(n_conds):
            for j in range(n_conds):
                if i != j:
                    ax.text(j, i, f"{dtw_mat[i,j]:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for i in range(len(gestures), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("DTW distance between mean condition trajectories\n(higher = more different)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "dtw_condition_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved dtw_condition_matrix.png")


def main():
    ap = argparse.ArgumentParser(description="EDA: time-series comparison across conditions")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="eda_conditions")
    ap.add_argument("--norm_T", type=int, default=60,
                    help="Resample all clips to this length for comparison")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.roots)
    if not items:
        print("No clips found."); return

    print(f"\nConditions: {sorted(set(it['cond'] for it in items))}")
    print(f"Gestures:   {sorted(set(it['gesture'] for it in items))}")
    print(f"Subjects:   {sorted(set(it['pid'] for it in items))}")
    print()

    plot_trajectory_envelopes(items, out_dir, args.norm_T)
    plot_velocity_profiles(items, out_dir, args.norm_T)
    plot_angle_evolution(items, out_dir, args.norm_T)
    plot_divergence_over_time(items, out_dir, args.norm_T)
    plot_dtw_matrix(items, out_dir, args.norm_T)

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
