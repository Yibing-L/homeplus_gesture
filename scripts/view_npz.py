#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_npz.py — quick viewer/sanity checker for gesture *.npz clips.

Features:
- Lists keys and shapes; finds the main 2D feature array (X) and a per-frame mask if present.
- Prints min/max/mean/std, counts NaN/Inf, per-feature zero-variance across time.
- Reports per-frame validity ratio and shows the first/last valid segments.
- Optional: plots time-series of a few features, a (T×F) feature heatmap, and the mask.
- Optional: batch mode via --glob "processed_*/gesture_*.npz"
- Optional: write a sanitized copy (NaN/Inf -> 0, clipping) with --write-fixed.

Usage:
  python inspect_npz.py path/to/gesture_11_11.npz
  python inspect_npz.py path/to/gesture_11_11.npz --plot
  python inspect_npz.py --glob "processed_0/*.npz" --summary-only
  python inspect_npz.py path/to/file.npz --write-fixed
"""

import argparse, os, sys, glob, re
import numpy as np

MASK_KEYS = ["valid_T","valid_mask","valid","mask","validity","validity_mask"]
SEQ_KEYS  = ["X","seq","features","data"]

def find_2d_key(z):
    for k in SEQ_KEYS:
        if k in z and z[k].ndim == 2:
            return k
    # fallback: first 2D array
    for k in z.files:
        if z[k].ndim == 2:
            return k
    return None

def find_mask(z, T):
    for k in MASK_KEYS:
        if k in z:
            a = np.array(z[k])
            if a.ndim == 1:
                if a.shape[0] == T:
                    return (a.astype(float) > 0).astype(np.float32), k
                else:
                    # soft resample to T
                    v = (a.astype(float) > 0).astype(np.float32)
                    t_src = np.linspace(0, len(v)-1, len(v))
                    t_dst = np.linspace(0, len(v)-1, T)
                    vv = np.interp(t_dst, t_src, v)
                    return (vv > 0.5).astype(np.float32), k + " (resampled)"
            if a.ndim == 0:
                val = float(a) > 0
                return (np.ones((T,), np.float32) if val else np.zeros((T,), np.float32)), k + " (scalar)"
    return np.ones((T,), np.float32), None

def summarize_array(name, A):
    A = np.asarray(A)
    finite = np.isfinite(A)
    n = A.size
    n_bad = n - finite.sum()
    A_safe = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {name}: shape={A.shape}, dtype={A.dtype}, finite={finite.sum()}/{n} "
          f"(bad={n_bad}, nan={np.isnan(A).sum()}, +inf={np.isposinf(A).sum()}, -inf={np.isneginf(A).sum()})")
    print(f"    min={A_safe.min():.6g}  max={A_safe.max():.6g}  mean={A_safe.mean():.6g}  std={A_safe.std():.6g}")

def zero_var_features_over_time(X):
    # X: (T,F)
    if X.ndim != 2: return 0, []
    zmask = X.std(axis=0) == 0
    idx = np.where(zmask)[0].tolist()
    return int(zmask.sum()), idx

def parse_gid_from_name(path):
    m = re.search(r"gesture_(\d+)_(\d+)\.npz$", os.path.basename(path))
    gid = int(m.group(1)) if m else None
    cmd = (gid % 7) if isinstance(gid, int) else None
    return gid, cmd

def inspect_file(path, plot=False, write_fixed=False, clip=50.0, summary_only=False):
    print(f"\n=== {path} ===")
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[ERROR] could not open: {e}")
        return

    print("Keys:", list(z.files))
    key = find_2d_key(z)
    if key is None:
        print("[ERROR] No 2D array key found."); return

    X = np.array(z[key])
    summarize_array(f"{key}", X)
    T, F = X.shape

    mask, mask_key = find_mask(z, T)
    vr = float(mask.mean())
    print(f"Mask: key={mask_key or 'None (assumed all valid)'}  valid_ratio={vr:.3f}")

    # Per-feature variance across time
    zv_count, zv_idx = zero_var_features_over_time(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
    print(f"Zero-variance features over time: {zv_count}{' (idx: '+str(zv_idx[:16]) + (' ...' if zv_count>16 else '') + ')' if zv_count else ''}")

    # Show a few metadata fields if present
    for meta in ["label","gid","include_arm","include_velocity","normalize_velocity","include_scale_channel","scale","T_target"]:
        if meta in z.files:
            arr = np.array(z[meta])
            if arr.ndim == 0:  print(f"meta.{meta} = {arr.item()}")
            else:              print(f"meta.{meta} shape={arr.shape}")

    gid, cmd = parse_gid_from_name(path)
    if gid is not None:
        print(f"Parsed from filename: gid={gid}  -> command (gid % 7) = {cmd}")

    # Show first/last few frames and mask runs (summary only)
    if not summary_only:
        print("\nFirst 2 frames (rows) and last 2 frames of X (truncated to first 8 features):")
        with np.printoptions(precision=4, suppress=True, linewidth=140):
            print("  X[0:2, 0:8] =\n", np.nan_to_num(X[0:2, :8]))
            print("  X[-2:, 0:8] =\n", np.nan_to_num(X[-2:, :8]))

    # Optionally write a sanitized copy
    if write_fixed:
        X_fix = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            X_fix = np.clip(X_fix, -clip, clip)
        data = {k: z[k] for k in z.files}
        data[key] = X_fix.astype(np.float32)
        out = path.replace(".npz", ".fixed.npz")
        np.savez_compressed(out, **data)
        print(f"[FIXED] wrote sanitized copy: {out}")

    # Optional plotting
    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[WARN] matplotlib not available ({e}); skipping plots."); return
        Xs = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # pick up to 6 feature indices spread across F
        idxs = np.linspace(0, F-1, num=min(6, F), dtype=int).tolist()
        t = np.arange(T)

        plt.figure(figsize=(10, 4))
        for i, fi in enumerate(idxs, 1):
            plt.subplot(len(idxs), 1, i)
            plt.plot(t, Xs[:, fi])
            if mask_key:
                bad = (mask < 0.5)
                if bad.any():
                    # visualize invalid frames as background
                    ylim = plt.ylim()
                    bad_runs = np.where(bad)[0]
                    if bad_runs.size:
                        plt.fill_between(t, ylim[0], ylim[1], where=bad, alpha=0.08, step="pre")
            plt.ylabel(f"f{fi}")
            if i == 1:
                plt.title(f"{os.path.basename(path)} — {key} time-series (subset)")
        plt.xlabel("frame")
        plt.tight_layout()

        # Heatmap of X (T×F)
        plt.figure(figsize=(8, 4))
        plt.imshow(Xs, aspect='auto', origin='lower')
        plt.colorbar(label="feature value")
        plt.title(f"{os.path.basename(path)} — {key} heatmap (T×F)")
        plt.xlabel("feature idx")
        plt.ylabel("frame")

        # Mask plot
        plt.figure(figsize=(8, 1.8))
        plt.step(t, mask, where='post')
        plt.ylim(-0.1, 1.1)
        plt.title(f"valid mask ({mask_key or 'assumed all-ones'})  valid_ratio={vr:.3f}")
        plt.xlabel("frame")
        plt.yticks([0,1])

        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="path to a single .npz file")
    ap.add_argument("--glob", help='glob pattern for batch mode, e.g. "processed_0/*.npz"')
    ap.add_argument("--plot", action="store_true", help="plot a few features and a heatmap")
    ap.add_argument("--summary-only", action="store_true", help="suppress sample rows")
    ap.add_argument("--write-fixed", action="store_true", help="write a sanitized .fixed.npz (NaN/Inf->0, clipped)")
    ap.add_argument("--clip", type=float, default=50.0, help="clip feature values to [-clip,+clip] when writing fixed")
    args = ap.parse_args()

    targets = []
    if args.path:
        targets.append(args.path)
    if args.glob:
        targets.extend(sorted(glob.glob(args.glob)))

    if not targets:
        print("Provide a file path or --glob pattern. See --help."); sys.exit(2)

    for p in targets:
        inspect_file(p, plot=args.plot, write_fixed=args.write_fixed, clip=args.clip, summary_only=args.summary_only)

if __name__ == "__main__":
    main()
