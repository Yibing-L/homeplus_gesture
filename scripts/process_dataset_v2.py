#!/usr/bin/env python3
"""One-command RGB-D -> canonical Home+ gesture dataset v2 processor.

Example (recommended when the original camera is connected):
  python scripts/process_dataset_v2.py --input D:/raw --output D:/processed_v2 --from-camera --resume

Or provide the recording camera intrinsics explicitly:
  python scripts/process_dataset_v2.py --input D:/raw --output D:/processed_v2 \
      --fx 604.97 --fy 604.87 --cx 322.1 --cy 244.7 --resume

The stable MediaPipe Holistic configuration from landmark_with_xyz.py is kept.
Depth sampling, plausibility checks, scale estimation, smoothing, and the
canonical feature schema are upgraded and shared with training/inference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from feature_pipeline_v2 import T_TARGET, build_canonical_features


MP = mp.solutions.holistic
PL = MP.PoseLandmark
ARM_LANDMARKS = (PL.RIGHT_SHOULDER.value, PL.RIGHT_ELBOW.value, PL.RIGHT_WRIST.value)
HAND_PARENTS = np.array([
    0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19
], dtype=np.int64)


def _finite_median(values: np.ndarray, valid: np.ndarray) -> Optional[float]:
    values = np.asarray(values, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    return float(np.median(values[valid])) if valid.any() else None


def clip_report(file_name: str, subject_id: int, label: int, status: str,
                data, error: Optional[str] = None) -> dict:
    """Compact, JSON-safe quality report for one processed or resumed clip."""
    if data is None:
        return dict(subject_id=subject_id, file=file_name, label=label, status=status,
                    error=error)
    frame = np.asarray(data["frame_valid"]) > 0.5
    raw_frame = np.asarray(data["valid_raw"]) > 0 if "valid_raw" in data else frame
    joint = np.asarray(data["joint_valid"]) > 0.5
    arm = np.asarray(data["arm_valid"]) > 0.5
    angle = np.asarray(data["angle_valid"]) > 0.5
    global_x = np.asarray(data["global_features"], np.float32)
    return dict(
        subject_id=subject_id, file=file_name, label=label, status=status, error=error,
        raw_frames=int(len(raw_frame)), raw_valid_frames=int(raw_frame.sum()),
        raw_valid_ratio=float(raw_frame.mean()) if len(raw_frame) else 0.0,
        resampled_valid_frames=int(frame.sum()), resampled_valid_ratio=float(frame.mean()),
        joint_valid_ratio=float(joint.mean()), arm_valid_ratio=float(arm.mean()),
        angle_valid_ratio=float(angle.mean()),
        hand_scale_median_m=_finite_median(global_x[:, 15], frame),
        arm_scale_median_m=_finite_median(global_x[:, 16], frame),
        wrist_depth_median_m=_finite_median(global_x[:, 17], frame),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Process every raw RGB-D gesture NPZ into canonical v2 features")
    ap.add_argument("--input", required=True, help="Raw root with numeric participant folders")
    ap.add_argument("--output", required=True, help="Destination root for canonical v2 NPZ files")
    intr = ap.add_mutually_exclusive_group(required=True)
    intr.add_argument("--from-camera", action="store_true", help="Use the connected study camera as the canonical calibration")
    intr.add_argument("--intrinsics-npz", help="Processed NPZ containing intrinsics=[fx,fy,cx,cy]")
    intr.add_argument("--fx", type=float, help="Recording camera fx (also requires --fy/--cx/--cy)")
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--target-frames", type=int, default=T_TARGET, choices=[T_TARGET],
                    help="Canonical length (fixed at 64 for all v2 models)")
    ap.add_argument("--patch-radius", type=int, default=3, help="Depth patch radius; 3 means 7x7")
    ap.add_argument("--hand-depth-tolerance", type=float, default=0.25, help="Max hand-joint depth difference from wrist, metres")
    ap.add_argument("--arm-depth-tolerance", type=float, default=0.70)
    ap.add_argument("--max-hand-radius", type=float, default=0.30, help="Max 3D distance from wrist to a hand joint")
    ap.add_argument("--max-hand-bone", type=float, default=0.13)
    ap.add_argument("--palm-root-bone-factor", type=float, default=1.75,
                    help="Multiplier for wrist-to-thumb/MCP edges, which span the palm")
    ap.add_argument("--min-depth", type=float, default=0.15)
    ap.add_argument("--max-depth", type=float, default=2.50)
    ap.add_argument("--min-hand-joints", type=int, default=12)
    ap.add_argument("--pose-visibility", type=float, default=0.40)
    ap.add_argument("--smooth-alpha", type=float, default=0.45)
    ap.add_argument("--mp-model-complexity", type=int, default=1, choices=(0, 1, 2))
    ap.add_argument("--mp-detection-confidence", type=float, default=0.70)
    ap.add_argument("--mp-tracking-confidence", type=float, default=0.70)
    ap.add_argument("--resume", action="store_true", help="Skip destinations that already exist")
    ap.add_argument("--workers", type=int, default=1,
                    help="Participants to process in parallel (start with 2 or 3)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N clips; useful for smoke tests")
    args = ap.parse_args()
    if args.fx is not None and any(v is None for v in (args.fy, args.cx, args.cy)):
        ap.error("--fx requires --fy, --cx, and --cy")
    if args.workers < 1:
        ap.error("--workers must be at least 1")
    if args.workers > 1 and args.limit:
        ap.error("--limit cannot be combined with --workers greater than 1")
    return args


def make_intrinsics(fx: float, fy: float, cx: float, cy: float,
                    model=rs.distortion.none, coeffs=None) -> rs.intrinsics:
    intr = rs.intrinsics()
    intr.width, intr.height = 640, 480
    intr.fx, intr.fy, intr.ppx, intr.ppy = float(fx), float(fy), float(cx), float(cy)
    intr.model = model
    intr.coeffs = list(coeffs) if coeffs is not None else [0.0] * 5
    return intr


def camera_intrinsics() -> Tuple[rs.intrinsics, float, Dict[str, object]]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    try:
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        device = profile.get_device()
        scale = device.first_depth_sensor().get_depth_scale()
        metadata = {
            "camera_name": device.get_info(rs.camera_info.name),
            "camera_serial": device.get_info(rs.camera_info.serial_number),
            "camera_firmware": device.get_info(rs.camera_info.firmware_version),
            "calibration_strategy": "canonical_study_camera",
        }
        return intr, float(scale), metadata
    finally:
        pipeline.stop()


def resolve_intrinsics(args: argparse.Namespace) -> Tuple[rs.intrinsics, float, Dict[str, object]]:
    scale = float(args.depth_scale)
    if args.from_camera:
        intr, camera_scale, metadata = camera_intrinsics()
        if abs(camera_scale - scale) > 1e-8:
            print(f"[INFO] using camera depth scale {camera_scale:g} instead of {scale:g}")
            scale = camera_scale
        return intr, scale, metadata
    if args.intrinsics_npz:
        z = np.load(args.intrinsics_npz, allow_pickle=True)
        fx, fy, cx, cy = np.asarray(z["intrinsics"], np.float64).tolist()
        model = rs.distortion(int(np.asarray(z["intrinsics_model"]))) if "intrinsics_model" in z else rs.distortion.none
        coeffs = np.asarray(z["intrinsics_coeffs"], np.float64).tolist() if "intrinsics_coeffs" in z else None
        if "depth_scale" in z:
            scale = float(z["depth_scale"])
        metadata = {
            "camera_name": str(np.asarray(z["camera_name"] ).item()) if "camera_name" in z else "unknown",
            "camera_serial": str(np.asarray(z["camera_serial"] ).item()) if "camera_serial" in z else "unknown",
            "camera_firmware": str(np.asarray(z["camera_firmware"] ).item()) if "camera_firmware" in z else "unknown",
            "calibration_strategy": str(np.asarray(z["calibration_strategy"]).item()) if "calibration_strategy" in z else "canonical_from_npz",
        }
        return make_intrinsics(fx, fy, cx, cy, model, coeffs), scale, metadata
    return make_intrinsics(args.fx, args.fy, args.cx, args.cy), scale, {
        "camera_name": "manual", "camera_serial": "unknown", "camera_firmware": "unknown",
        "calibration_strategy": "canonical_manual",
    }


def participant_sources(root: Path) -> Iterable[Tuple[int, Path]]:
    found = []
    for d in root.iterdir():
        if d.is_dir():
            m = re.fullmatch(r"(\d+)(?:_done)?", d.name)
            if m:
                found.append((int(m.group(1)), d))
    if found:
        yield from sorted(found)
    else:
        # Convenient for testing one folder; participant is inferred as 0.
        yield 0, root


def patch_depth(
    depth: np.ndarray,
    u: float,
    v: float,
    depth_scale: float,
    radius: int,
    min_depth: float,
    max_depth: float,
    reference: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> Optional[float]:
    """Median depth around a landmark, optionally constrained near a reference."""
    h, w = depth.shape[:2]
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < w and 0 <= y < h):
        return None
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    values = depth[y0:y1, x0:x1].astype(np.float32).reshape(-1) * depth_scale
    values = values[(values >= min_depth) & (values <= max_depth)]
    if reference is not None and tolerance is not None:
        values = values[np.abs(values - reference) <= tolerance]
    if values.size == 0:
        return None
    # Prefer samples near the landmark centre when they form a plausible cluster.
    center = float(depth[y, x]) * depth_scale
    if min_depth <= center <= max_depth:
        local = values[np.abs(values - center) <= 0.08]
        if local.size >= 3:
            values = local
    return float(np.median(values))


def deproject(intr: rs.intrinsics, u: float, v: float, z: float) -> np.ndarray:
    return np.asarray(rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z)), np.float32)


def invalidate_bad_hand_geometry(xyz: np.ndarray, valid: np.ndarray, max_radius: float,
                                 max_bone: float, palm_root_factor: float = 1.75) -> None:
    """In-place rejection of anatomically impossible hand observations."""
    if not valid[0]:
        valid[:] = False
        xyz[:] = 0.0
        return
    radius = np.linalg.norm(xyz, axis=1)
    valid[(~np.isfinite(radius)) | (radius > max_radius)] = False
    for j in range(1, 21):
        p = int(HAND_PARENTS[j])
        if not (valid[j] and valid[p]):
            valid[j] = False
            continue
        # Wrist-to-thumb/MCP links span the palm and are substantially longer
        # than phalanges; applying one threshold invalidates every descendant.
        limit = max_bone * palm_root_factor if p == 0 else max_bone
        if np.linalg.norm(xyz[j] - xyz[p]) > limit:
            valid[j] = False
    xyz[~valid] = 0.0


def extract_clip(
    rgb_frames: np.ndarray,
    depth_frames: np.ndarray,
    holistic,
    intr: rs.intrinsics,
    args: argparse.Namespace,
    depth_scale: float,
):
    n = min(len(rgb_frames), len(depth_frames))
    hand_xyz = np.zeros((n, 21, 3), np.float32)
    arm_xyz = np.zeros((n, 3, 3), np.float32)
    wrist_xyz = np.zeros((n, 3), np.float32)
    hand_uvd = np.full((n, 21, 3), np.nan, np.float32)
    arm_uvd = np.full((n, 3, 3), np.nan, np.float32)
    joint_valid = np.zeros((n, 21), bool)
    arm_valid = np.zeros((n, 3), bool)
    frame_valid = np.zeros(n, bool)

    for t, (bgr, depth) in enumerate(zip(rgb_frames[:n], depth_frames[:n])):
        result = holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        hand_lm = getattr(result, "right_hand_landmarks", None)
        if hand_lm is None:
            continue

        wrist_lm = hand_lm.landmark[0]
        wu, wv = wrist_lm.x * bgr.shape[1], wrist_lm.y * bgr.shape[0]
        wz = patch_depth(depth, wu, wv, depth_scale, args.patch_radius,
                         args.min_depth, args.max_depth)
        if wz is None:
            continue
        wrist = deproject(intr, wu, wv, wz)
        wrist_xyz[t] = wrist

        for j, lm in enumerate(hand_lm.landmark):
            u, v = lm.x * bgr.shape[1], lm.y * bgr.shape[0]
            z = patch_depth(depth, u, v, depth_scale, args.patch_radius,
                            args.min_depth, args.max_depth, wz, args.hand_depth_tolerance)
            hand_uvd[t, j, :2] = (u, v)
            if z is None:
                continue
            hand_uvd[t, j, 2] = z
            hand_xyz[t, j] = deproject(intr, u, v, z) - wrist
            joint_valid[t, j] = True
        hand_xyz[t, 0] = 0.0
        joint_valid[t, 0] = True
        invalidate_bad_hand_geometry(
            hand_xyz[t], joint_valid[t], args.max_hand_radius, args.max_hand_bone,
            args.palm_root_bone_factor,
        )

        pose = getattr(result, "pose_landmarks", None)
        if pose is not None:
            for j, mp_idx in enumerate(ARM_LANDMARKS):
                lm = pose.landmark[mp_idx]
                if lm.visibility < args.pose_visibility:
                    continue
                u, v = lm.x * bgr.shape[1], lm.y * bgr.shape[0]
                z = patch_depth(depth, u, v, depth_scale, args.patch_radius,
                                args.min_depth, args.max_depth, wz, args.arm_depth_tolerance)
                arm_uvd[t, j, :2] = (u, v)
                if z is None:
                    continue
                arm_uvd[t, j, 2] = z
                arm_xyz[t, j] = deproject(intr, u, v, z) - wrist
                arm_valid[t, j] = True

        frame_valid[t] = joint_valid[t].sum() >= args.min_hand_joints
        if not frame_valid[t]:
            joint_valid[t] = False
            hand_xyz[t] = 0.0

    features = build_canonical_features(
        hand_xyz, arm_xyz, wrist_xyz, joint_valid, arm_valid, frame_valid,
        target=args.target_frames, smooth_alpha=args.smooth_alpha,
    )
    raw = dict(
        hand_xyz_m=hand_xyz, arm_xyz_m=arm_xyz, wrist_xyz=wrist_xyz,
        hand_uvd=hand_uvd, arm_uvd=arm_uvd,
        valid_raw=frame_valid.astype(np.uint8),
        valid_joint_raw=joint_valid.astype(np.uint8),
        valid_arm_raw=arm_valid.astype(np.uint8),
    )
    return features, raw


def make_holistic(args):
    """Create an independent tracker for one worker."""
    return MP.Holistic(
        static_image_mode=False,
        model_complexity=args.mp_model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=args.mp_detection_confidence,
        min_tracking_confidence=args.mp_tracking_confidence,
    )


def reset_holistic_quietly(holistic) -> None:
    """Reset clip history while hiding MediaPipe's harmless native warning."""
    saved_stderr = os.dup(2)
    null_stderr = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_stderr, 2)
        holistic.reset()
    finally:
        os.dup2(saved_stderr, 2)
        os.close(null_stderr)
        os.close(saved_stderr)


def intrinsics_spec(intr: rs.intrinsics) -> dict:
    """Convert the native RealSense object to data that Windows workers can pickle."""
    return {
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.ppx), "cy": float(intr.ppy),
        "model": int(intr.model.value), "coeffs": list(intr.coeffs),
    }


def process_participant(subject_id: int, src_dir_value: str, output_root_value: str,
                        args: argparse.Namespace, intr_spec: dict, depth_scale: float,
                        camera_metadata: dict, clip_limit: int = 0) -> dict:
    """Process one participant in one process; no output file is shared by workers."""
    src_dir = Path(src_dir_value)
    output_root = Path(output_root_value)
    dst_dir = output_root / str(subject_id)
    dst_dir.mkdir(parents=True, exist_ok=True)
    intr = make_intrinsics(
        intr_spec["fx"], intr_spec["fy"], intr_spec["cx"], intr_spec["cy"],
        rs.distortion(intr_spec["model"]), intr_spec["coeffs"],
    )
    holistic = make_holistic(args)
    subject_rows = []
    stop_requested = False
    try:
        for src in sorted(src_dir.glob("*.npz")):
            if clip_limit and len(subject_rows) >= clip_limit:
                break
            dst = dst_dir / src.name
            temp_dst = dst.with_name(dst.name + ".tmp.npz")
            label = -1
            if args.resume and dst.exists():
                try:
                    with np.load(dst, allow_pickle=True) as existing:
                        label = int(np.asarray(existing["label"]).item())
                        row = clip_report(src.name, subject_id, label, "existing", existing)
                except Exception as exc:
                    row = clip_report(src.name, subject_id, -1, "error_existing", None, str(exc))
                    print(f"[ERROR] existing {dst}: {exc}", file=sys.stderr, flush=True)
                subject_rows.append(row)
                continue
            try:
                with np.load(src, allow_pickle=True) as z:
                    if not {"rgb", "depth", "label"}.issubset(z.files):
                        raise ValueError("missing rgb/depth/label")
                    rgb = np.asarray(z["rgb"])
                    depth = np.asarray(z["depth"])
                    label = int(np.asarray(z["label"]).item())
                try:
                    features, raw = extract_clip(rgb, depth, holistic, intr, args, depth_scale)
                finally:
                    reset_holistic_quietly(holistic)
                if temp_dst.exists():
                    temp_dst.unlink()
                np.savez_compressed(
                    temp_dst, **features, **raw, label=np.int64(label), subject_id=np.int64(subject_id),
                    intrinsics=np.array([intr.fx, intr.fy, intr.ppx, intr.ppy], np.float32),
                    intrinsics_model=np.int64(intr.model.value),
                    intrinsics_model_str=np.array(str(intr.model)),
                    intrinsics_coeffs=np.asarray(intr.coeffs, np.float32),
                    camera_name=np.array(camera_metadata["camera_name"]),
                    camera_serial=np.array(camera_metadata["camera_serial"]),
                    camera_firmware=np.array(camera_metadata["camera_firmware"]),
                    calibration_strategy=np.array(camera_metadata["calibration_strategy"]),
                    depth_scale=np.float64(depth_scale), target_frames=np.int64(args.target_frames),
                    schema_version=np.array("homeplus_v2"),
                )
                os.replace(temp_dst, dst)
                merged = {**features, **raw}
                row = clip_report(src.name, subject_id, label, "processed", merged)
                print(f"[OK] subject={subject_id} {src.name} "
                      f"valid={row['resampled_valid_ratio']:.2f}", flush=True)
            except KeyboardInterrupt:
                if temp_dst.exists():
                    temp_dst.unlink()
                stop_requested = True
                row = clip_report(src.name, subject_id, label, "interrupted", None,
                                  "Stopped by user")
                subject_rows.append(row)
                print(f"\n[STOP] subject={subject_id}; completed clips are safe", flush=True)
                break
            except Exception as exc:
                if temp_dst.exists():
                    temp_dst.unlink()
                row = clip_report(src.name, subject_id, label, "error", None, str(exc))
                print(f"[ERROR] {src}: {exc}", file=sys.stderr, flush=True)
            subject_rows.append(row)
    finally:
        holistic.close()

    good_rows = [r for r in subject_rows if r["status"] in {"processed", "existing"}]
    label_counts = {}
    for row in good_rows:
        key = str(row["label"])
        label_counts[key] = label_counts.get(key, 0) + 1
    raw_lengths = [r["raw_frames"] for r in good_rows]
    raw_ratios = [r["raw_valid_ratio"] for r in good_rows]
    res_ratios = [r["resampled_valid_ratio"] for r in good_rows]
    worst = sorted(good_rows, key=lambda r: r["resampled_valid_ratio"])[:5]
    participant_report = {
        "schema_version": "homeplus_processing_report_v1",
        "subject_id": subject_id,
        "source_directory": str(src_dir.resolve()),
        "output_directory": str(dst_dir.resolve()),
        "calibration": {
            **camera_metadata,
            "intrinsics": [intr.fx, intr.fy, intr.ppx, intr.ppy],
            "intrinsics_model": str(intr.model),
            "intrinsics_coeffs": list(intr.coeffs),
            "depth_scale": depth_scale,
        },
        "totals": {
            "seen": len(subject_rows),
            "processed": sum(r["status"] == "processed" for r in subject_rows),
            "existing": sum(r["status"] == "existing" for r in subject_rows),
            "errors": sum(r["status"].startswith("error") for r in subject_rows),
            "label_counts": label_counts,
            "total_raw_frames": int(sum(raw_lengths)),
            "mean_raw_frames": float(np.mean(raw_lengths)) if raw_lengths else None,
            "mean_raw_valid_ratio": float(np.mean(raw_ratios)) if raw_ratios else None,
            "mean_resampled_valid_ratio": float(np.mean(res_ratios)) if res_ratios else None,
        },
        "worst_clips_by_resampled_validity": [
            {k: r[k] for k in ("file", "label", "raw_frames", "raw_valid_ratio",
                               "resampled_valid_ratio")}
            for r in worst
        ],
        "clips": subject_rows,
    }
    (dst_dir / "processing_summary.json").write_text(
        json.dumps(participant_report, indent=2), encoding="utf-8"
    )
    return {
        "subject_id": subject_id,
        "rows": subject_rows,
        "report": participant_report,
        "interrupted": stop_requested,
    }


def stop_executor(executor) -> None:
    """Cancel queued work and stop active workers after Ctrl+C on Windows."""
    processes = list(getattr(executor, "_processes", {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    args = parse_args()
    input_root, output_root = Path(args.input), Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    intr, depth_scale, camera_metadata = resolve_intrinsics(args)
    print(f"[INTRINSICS] fx={intr.fx:.4f} fy={intr.fy:.4f} "
          f"cx={intr.ppx:.4f} cy={intr.ppy:.4f}")
    print(f"[DEPTH] scale={depth_scale:g}")
    config = vars(args).copy()
    config.update(
        schema_version="homeplus_v2",
        intrinsics=[intr.fx, intr.fy, intr.ppx, intr.ppy],
        intrinsics_model=int(intr.model.value), intrinsics_model_str=str(intr.model),
        intrinsics_coeffs=list(intr.coeffs), actual_depth_scale=depth_scale,
        tracker_reset_per_clip=True, parallelization="participant_processes",
        **camera_metadata,
    )
    (output_root / "processing_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    sources = list(participant_sources(input_root))
    spec = intrinsics_spec(intr)
    results = []
    stop_requested = False
    started = time.time()

    if args.workers == 1:
        remaining = args.limit
        for subject_id, src_dir in sources:
            result = process_participant(
                subject_id, str(src_dir), str(output_root), args, spec, depth_scale,
                camera_metadata, remaining,
            )
            results.append(result)
            if result["interrupted"]:
                stop_requested = True
                break
            if args.limit:
                remaining -= len(result["rows"])
                if remaining <= 0:
                    break
    else:
        active_workers = min(args.workers, len(sources))
        print(f"[PARALLEL] workers={active_workers} participants={len(sources)}", flush=True)
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=active_workers)
        futures = {
            executor.submit(
                process_participant, subject_id, str(src_dir), str(output_root), args,
                spec, depth_scale, camera_metadata, 0,
            ): subject_id
            for subject_id, src_dir in sources
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                if result["interrupted"]:
                    stop_requested = True
        except KeyboardInterrupt:
            stop_requested = True
            for future in futures:
                future.cancel()
            print("\n[STOP] stopping workers; completed clip files are safe", flush=True)
            stop_executor(executor)
            executor = None
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    results.sort(key=lambda item: item["subject_id"])
    dataset_rows = [row for result in results for row in result["rows"]]
    participant_summaries = [
        {k: v for k, v in result["report"].items() if k != "clips"}
        for result in results
    ]
    total = len(dataset_rows)
    ok = sum(row["status"] == "processed" for row in dataset_rows)
    skipped = sum(row["status"] == "existing" for row in dataset_rows)
    errors = sum(row["status"].startswith("error") for row in dataset_rows)
    elapsed = time.time() - started
    dataset_report = {
        "schema_version": "homeplus_dataset_processing_report_v1",
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "elapsed_seconds": elapsed,
        "totals": {
            "seen": total, "processed": ok, "existing": skipped, "errors": errors,
            "participants_reported": len(participant_summaries),
            "interrupted": stop_requested,
        },
        "processing_config": config,
        "participants": participant_summaries,
    }
    (output_root / "dataset_processing_summary.json").write_text(
        json.dumps(dataset_report, indent=2), encoding="utf-8"
    )
    csv_fields = [
        "subject_id", "file", "label", "status", "error", "raw_frames",
        "raw_valid_frames", "raw_valid_ratio", "resampled_valid_frames",
        "resampled_valid_ratio", "joint_valid_ratio", "arm_valid_ratio",
        "angle_valid_ratio", "hand_scale_median_m", "arm_scale_median_m",
        "wrist_depth_median_m",
    ]
    with (output_root / "clip_processing_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dataset_rows)
    print(f"[DONE] seen={total} processed={ok} skipped={skipped} errors={errors} "
          f"elapsed={elapsed:.1f}s reports={output_root}")


if __name__ == "__main__":
    main()