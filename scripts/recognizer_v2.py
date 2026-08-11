#!/usr/bin/env python3
"""Live RealSense recognizer for Home+ v2 SVM/TCN/BiLSTM/ST-GCN models."""

from __future__ import annotations

import argparse
import collections
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import torch

from feature_pipeline_v2 import build_canonical_features, flatten_sequence, svm_summary
from models_v2 import build_model
from process_dataset_v2 import ARM_LANDMARKS, deproject, invalidate_bad_hand_geometry, patch_depth
from recognizer_runtime_v2 import EventGate, JsonlLogger, recent_motion_score
from train_all_v2 import apply_flat_norm, apply_graph_norm


MP = mp.solutions.holistic


def parse_args():
    ap = argparse.ArgumentParser(description="Live recognition using any Home+ v2 final model")
    ap.add_argument("--checkpoint", required=True, help="model_final.pt or model_final.joblib")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--recognizer-config", help="Optional JSON defaults calibrated for this checkpoint")
    ap.add_argument("--window-frames", type=int, default=90, help="Raw-frame rolling window before resampling to 64")
    ap.add_argument("--stride", type=int, default=1, help="Run MediaPipe every N camera frames")
    ap.add_argument("--min-valid-ratio", type=float)
    ap.add_argument("--prob-threshold", type=float)
    ap.add_argument("--stable-count", type=int)
    ap.add_argument("--motion-on", type=float)
    ap.add_argument("--motion-off", type=float)
    ap.add_argument("--motion-start-count", type=int)
    ap.add_argument("--active-timeout-seconds", type=float)
    ap.add_argument("--cooldown-seconds", type=float)
    ap.add_argument("--rearm-quiet-count", type=int)
    ap.add_argument("--display-seconds", type=float)
    ap.add_argument("--log-jsonl", help="Prediction/event log (default: runs/live_logs/<timestamp>.jsonl)")
    ap.add_argument("--no-log", action="store_true", help="Disable JSONL logging")
    ap.add_argument("--record-bag", "--record-file", dest="record_file",
                    help="Optional RealSense .db3 recording for identical model replay")
    ap.add_argument("--camera-serial", help="Select a RealSense by serial number")
    ap.add_argument("--camera-bus-id", help="Select a RealSense whose physical-port string contains this value")
    ap.add_argument("--label-map", help="JSON mapping class index to display name")
    ap.add_argument("--show-landmarks", action="store_true")
    return ap.parse_args()


def load_runtime_settings(args, predictor):
    checkpoint = Path(args.checkpoint).resolve()
    config_path = Path(args.recognizer_config).resolve() if args.recognizer_config else checkpoint.with_name("recognizer_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    def choose(name, cli_value, fallback):
        return cli_value if cli_value is not None else config.get(name, fallback)

    settings = dict(
        min_valid_ratio=float(choose("min_valid_ratio", args.min_valid_ratio, predictor.min_valid_ratio)),
        probability_threshold=float(choose("probability_threshold", args.prob_threshold, 0.90)),
        stable_count=int(choose("stable_count", args.stable_count, 3)),
        motion_on=float(choose("motion_on", args.motion_on, 0.020)),
        motion_off=float(choose("motion_off", args.motion_off, 0.010)),
        motion_start_count=int(choose("motion_start_count", args.motion_start_count, 2)),
        active_timeout_seconds=float(choose("active_timeout_seconds", args.active_timeout_seconds, 1.0)),
        cooldown_seconds=float(choose("cooldown_seconds", args.cooldown_seconds, 1.25)),
        rearm_quiet_count=int(choose("rearm_quiet_count", args.rearm_quiet_count, 8)),
        display_seconds=float(choose("display_seconds", args.display_seconds, 1.5)),
    )
    return settings, config, config_path if config_path.exists() else None


def select_realsense_device(camera_serial=None, camera_bus_id=None):
    devices = list(rs.context().query_devices())
    rows = []
    for device in devices:
        def info(key, fallback=""):
            try:
                return device.get_info(key)
            except Exception:
                return fallback
        rows.append(dict(
            name=info(rs.camera_info.name),
            serial=info(rs.camera_info.serial_number),
            physical_port=info(rs.camera_info.physical_port),
            firmware=info(rs.camera_info.firmware_version),
        ))
    matches = rows
    if camera_serial:
        matches = [row for row in matches if row["serial"] == str(camera_serial)]
    if camera_bus_id:
        needle = str(camera_bus_id).lower()
        matches = [row for row in matches if needle in row["physical_port"].lower()]
    if not camera_serial and not camera_bus_id:
        d455 = [row for row in rows if "D455" in row["name"].upper()]
        matches = d455 if d455 else rows
    if len(matches) != 1:
        available = "; ".join(
            f"{row['name']} serial={row['serial']} port={row['physical_port']}" for row in rows
        ) or "none"
        raise RuntimeError(
            f"Camera selection matched {len(matches)} devices. Available RealSense devices: {available}. "
            "Use --camera-serial with an exact serial number."
        )
    return matches[0]


def normalized_record_path(value):
    if not value:
        return None
    path = Path(value).resolve()
    if path.suffix.lower() != ".db3":
        replacement = path.with_suffix(".db3")
        print(f"[RECORD] This RealSense backend requires .db3; using {replacement}")
        path = replacement
    return path


class LiveGeometry:
    def __init__(self, intr, depth_scale: float, maxlen: int):
        self.intr = intr; self.depth_scale = depth_scale
        self.hand = collections.deque(maxlen=maxlen)
        self.hand_xy = collections.deque(maxlen=maxlen)
        self.arm = collections.deque(maxlen=maxlen)
        self.wrist = collections.deque(maxlen=maxlen)
        self.joint_valid = collections.deque(maxlen=maxlen)
        self.arm_valid = collections.deque(maxlen=maxlen)
        self.frame_valid = collections.deque(maxlen=maxlen)
        self.cfg = SimpleNamespace(
            patch_radius=3, min_depth=0.15, max_depth=2.50,
            hand_depth_tolerance=0.25, arm_depth_tolerance=0.70,
            max_hand_radius=0.30, max_hand_bone=0.13,
            min_hand_joints=12, pose_visibility=0.40,
        )

    def append_invalid(self):
        self.hand.append(np.zeros((21, 3), np.float32)); self.arm.append(np.zeros((3, 3), np.float32))
        self.hand_xy.append(np.zeros((21, 2), np.float32))
        self.wrist.append(np.zeros(3, np.float32)); self.joint_valid.append(np.zeros(21, bool))
        self.arm_valid.append(np.zeros(3, bool)); self.frame_valid.append(False)

    def append(self, bgr, depth, result):
        hand_xyz = np.zeros((21, 3), np.float32); arm_xyz = np.zeros((3, 3), np.float32)
        hand_xy = np.zeros((21, 2), np.float32)
        wrist_xyz = np.zeros(3, np.float32); jv = np.zeros(21, bool); av = np.zeros(3, bool)
        hand_lm = getattr(result, "right_hand_landmarks", None)
        if hand_lm is None:
            self.append_invalid(); return
        lm0 = hand_lm.landmark[0]; wu, wv = lm0.x * bgr.shape[1], lm0.y * bgr.shape[0]
        for j, lm in enumerate(hand_lm.landmark):
            hand_xy[j] = (lm.x * bgr.shape[1], lm.y * bgr.shape[0])
        wz = patch_depth(depth, wu, wv, self.depth_scale, self.cfg.patch_radius,
                         self.cfg.min_depth, self.cfg.max_depth)
        if wz is None:
            self.append_invalid(); return
        wrist_xyz = deproject(self.intr, wu, wv, wz)
        for j, lm in enumerate(hand_lm.landmark):
            u, v = lm.x * bgr.shape[1], lm.y * bgr.shape[0]
            z = patch_depth(depth, u, v, self.depth_scale, self.cfg.patch_radius,
                            self.cfg.min_depth, self.cfg.max_depth, wz, self.cfg.hand_depth_tolerance)
            if z is not None:
                hand_xyz[j] = deproject(self.intr, u, v, z) - wrist_xyz; jv[j] = True
        hand_xyz[0] = 0.0; jv[0] = True
        invalidate_bad_hand_geometry(hand_xyz, jv, self.cfg.max_hand_radius, self.cfg.max_hand_bone)
        pose = getattr(result, "pose_landmarks", None)
        if pose is not None:
            for j, idx in enumerate(ARM_LANDMARKS):
                lm = pose.landmark[idx]
                if lm.visibility < self.cfg.pose_visibility: continue
                u, v = lm.x * bgr.shape[1], lm.y * bgr.shape[0]
                z = patch_depth(depth, u, v, self.depth_scale, self.cfg.patch_radius,
                                self.cfg.min_depth, self.cfg.max_depth, wz, self.cfg.arm_depth_tolerance)
                if z is not None:
                    arm_xyz[j] = deproject(self.intr, u, v, z) - wrist_xyz; av[j] = True
        valid = int(jv.sum()) >= self.cfg.min_hand_joints
        if not valid: hand_xyz[:] = 0.0; jv[:] = False
        self.hand.append(hand_xyz); self.hand_xy.append(hand_xy)
        self.arm.append(arm_xyz); self.wrist.append(wrist_xyz)
        self.joint_valid.append(jv); self.arm_valid.append(av); self.frame_valid.append(valid)

    def features(self):
        if not self.hand: return None
        return build_canonical_features(
            np.asarray(self.hand), np.asarray(self.arm), np.asarray(self.wrist),
            np.asarray(self.joint_valid), np.asarray(self.arm_valid), np.asarray(self.frame_valid),
        )


class Predictor:
    def __init__(self, path: str, device: str):
        self.path = str(Path(path).resolve()); self.device = torch.device(device)
        self.min_valid_ratio = 0.25
        if path.lower().endswith(".joblib"):
            art = joblib.load(path); self.kind = "svm"; self.model = art["model"]
            self.feature_set = art["feature_set"]; self.n_classes = art["n_classes"]
            self.min_valid_ratio = float(art.get("min_valid_ratio", 0.25))
        else:
            art = torch.load(path, map_location="cpu", weights_only=False)
            if art.get("schema_version") != "homeplus_v2": raise ValueError("Not a Home+ v2 checkpoint")
            self.kind = art["model_type"]; self.feature_set = art["feature_set"]
            self.n_classes = int(art["n_classes"]); self.norm = art["normalization"]
            self.min_valid_ratio = float(art.get("min_valid_ratio", 0.25))
            self.model = build_model(self.kind, self.n_classes, art["model_config"])
            self.model.load_state_dict(art["state_dict"]); self.model.to(self.device).eval()

    def predict_proba(self, data):
        mask_np = data["frame_valid"].astype(np.float32)
        if self.kind == "svm":
            seq, _ = flatten_sequence(data, self.feature_set)
            x = svm_summary(seq, mask_np.astype(bool))[None]
            decision = np.asarray(self.model.decision_function(x)).reshape(-1)
            if decision.size == 1: decision = np.array([-decision[0], decision[0]])
            exp = np.exp(decision - decision.max()); probs = exp / exp.sum()
            return probs.astype(np.float32)
        if self.kind == "stgcn":
            x_np, g_np = apply_graph_norm(data, self.feature_set, self.norm)
            x = torch.from_numpy(x_np[None]).to(self.device); g = torch.from_numpy(g_np[None]).to(self.device)
        else:
            x_np = apply_flat_norm(data, self.feature_set, self.norm)
            x = torch.from_numpy(x_np[None]).to(self.device); g = torch.zeros((1, 64, 1), device=self.device)
        mask = torch.from_numpy(mask_np[None]).to(self.device)
        with torch.inference_mode(): probs = torch.softmax(self.model(x, mask, g), 1)[0].cpu().numpy()
        return probs.astype(np.float32)

    def predict(self, data):
        probs = self.predict_proba(data)
        return int(np.argmax(probs)), float(np.max(probs))


def main():
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment has a CPU-only PyTorch build. "
            "Install a CUDA-enabled PyTorch wheel, then verify torch.cuda.is_available()."
        )
    if args.stride != 1:
        raise ValueError("Continuous event recognition currently requires --stride 1")
    predictor = Predictor(args.checkpoint, args.device)
    runtime, calibration, calibration_path = load_runtime_settings(args, predictor)
    min_valid_ratio = runtime.pop("min_valid_ratio")
    gate = EventGate(**runtime)
    labels = {}
    if args.label_map:
        labels = {int(k): str(v) for k, v in json.loads(Path(args.label_map).read_text()).items()}
    checkpoint_path = Path(predictor.path)
    model_identity = f"{predictor.kind}/{predictor.feature_set}"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = None if args.no_log else Path(
        args.log_jsonl or f"runs/live_logs/{stamp}_{predictor.kind}_{predictor.feature_set}.jsonl"
    )
    bag_path = normalized_record_path(args.record_file)
    if bag_path:
        bag_path.parent.mkdir(parents=True, exist_ok=True)
        if bag_path.exists():
            raise FileExistsError(f"Refusing to overwrite RealSense recording: {bag_path}")

    camera = select_realsense_device(args.camera_serial, args.camera_bus_id)
    pipeline = rs.pipeline(); config = rs.config()
    config.enable_device(camera["serial"])
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if bag_path:
        config.enable_record_to_file(str(bag_path))
    profile = pipeline.start(config); align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    geometry = LiveGeometry(intr, float(depth_scale), args.window_frames)
    holistic = MP.Holistic(
        static_image_mode=False, model_complexity=1, smooth_landmarks=True,
        enable_segmentation=False, refine_face_landmarks=False,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    )
    drawing = mp.solutions.drawing_utils
    logger = JsonlLogger(log_path, dict(
        checkpoint=str(checkpoint_path), model_type=predictor.kind,
        feature_set=predictor.feature_set, n_classes=predictor.n_classes,
        device=str(predictor.device), window_frames=args.window_frames,
        min_valid_ratio=min_valid_ratio, gate=gate.settings(),
        calibration_file=str(calibration_path) if calibration_path else None,
        calibration=calibration, record_file=str(bag_path) if bag_path else None,
        camera=camera,
        camera_intrinsics=[intr.fx, intr.fy, intr.ppx, intr.ppy],
        depth_scale=float(depth_scale),
    )) if log_path else None
    print(f"[CAMERA] {camera['name']} serial={camera['serial']} port={camera['physical_port']}")
    print(f"[MODEL] checkpoint={checkpoint_path}")
    print(f"[MODEL] identity={model_identity} classes={predictor.n_classes} device={predictor.device}")
    print(f"[SETTINGS] min_valid_ratio={min_valid_ratio:.3f} gate={json.dumps(gate.settings(), sort_keys=True)}")
    print(f"[LOG] {log_path if log_path else 'disabled'}")
    print("[CONTROLS] 0-6=start ground-truth gesture, SPACE=end annotation, Q/ESC=quit")
    frame_no = 0; annotation = None; event_count = 0; last_recognized = None
    started = time.perf_counter()
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            cf = frames.get_color_frame(); df = frames.get_depth_frame()
            if not cf or not df:
                continue
            bgr = np.asanyarray(cf.get_data()); depth = np.asanyarray(df.get_data())
            result = holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frame_no += 1
            geometry.append(bgr, depth, result)
            if args.show_landmarks:
                if result.right_hand_landmarks:
                    drawing.draw_landmarks(bgr, result.right_hand_landmarks, MP.HAND_CONNECTIONS)
                if result.pose_landmarks:
                    drawing.draw_landmarks(bgr, result.pose_landmarks, MP.POSE_CONNECTIONS)

            now = time.perf_counter(); elapsed = now - started
            window_ready = len(geometry.hand) >= args.window_frames
            data = geometry.features(); pred = None; prob = 0.0; probs = None
            ratio = float(data["frame_valid"].mean()) if data is not None else 0.0
            motion = recent_motion_score(
                geometry.hand, geometry.wrist, geometry.joint_valid, geometry.frame_valid,
                geometry.hand_xy,
            )
            if window_ready and data is not None and ratio >= min_valid_ratio:
                probs = predictor.predict_proba(data)
                pred = int(np.argmax(probs)); prob = float(probs[pred])
            status = gate.update(
                now=now, window_ready=window_ready, valid=ratio >= min_valid_ratio,
                motion_score=motion, prediction=pred, probability=prob,
            )
            if status.emitted_event is not None:
                event_count += 1
                last_recognized = status.emitted_event
                if logger:
                    logger.write(
                        "prediction_event", monotonic_seconds=elapsed,
                        event_index=event_count, class_index=status.emitted_event,
                        class_name=labels.get(status.emitted_event, str(status.emitted_event)),
                        confidence=prob, probabilities=probs,
                        valid_ratio=ratio, motion_score=motion,
                        ground_truth_class=(annotation["class_index"] if annotation else None),
                    )
            if logger:
                logger.write(
                    "frame", monotonic_seconds=elapsed, frame_index=frame_no,
                    window_ready=window_ready, valid_ratio=ratio, motion_score=motion,
                    state=status.state, prediction=pred, confidence=prob,
                    probabilities=probs, candidate=status.candidate, streak=status.streak,
                    displayed_event=status.displayed_event, emitted_event=status.emitted_event,
                    ground_truth_class=(annotation["class_index"] if annotation else None),
                )

            name = lambda x: "-" if x is None else labels.get(x, str(x))
            cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 116), (0, 0, 0), -1)
            cv2.putText(
                bgr, f"state={status.state} ready={int(window_ready)} valid={ratio:.2f} motion={motion:.3f}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
            )
            cv2.putText(
                bgr, f"raw_pred={name(pred)} p={prob:.2f} candidate={name(status.candidate)} stable={status.streak}/{gate.stable_count}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
            )
            cv2.putText(
                bgr, f"LAST RECOGNIZED: {name(last_recognized)}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 255, 0), 3,
            )
            cv2.putText(
                bgr, f"ground_truth={name(annotation['class_index']) if annotation else '-'}",
                (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2,
            )
            cv2.imshow("Home+ Recognizer v2", bgr)
            key = cv2.waitKey(1) & 0xFF
            if ord("0") <= key <= ord("6") and key - ord("0") < predictor.n_classes:
                if annotation and logger:
                    logger.write(
                        "ground_truth_end", monotonic_seconds=elapsed,
                        **annotation, reason="replaced",
                    )
                annotation = {"class_index": key - ord("0"), "start_seconds": elapsed}
                if logger:
                    logger.write("ground_truth_start", monotonic_seconds=elapsed, **annotation)
            elif key == ord(" "):
                if annotation and logger:
                    logger.write(
                        "ground_truth_end", monotonic_seconds=elapsed,
                        **annotation, reason="space",
                    )
                annotation = None
            elif key in (ord("q"), 27):
                break
    finally:
        ended = time.perf_counter() - started
        if annotation and logger:
            logger.write(
                "ground_truth_end", monotonic_seconds=ended,
                **annotation, reason="session_end",
            )
        if logger:
            logger.close(monotonic_seconds=ended, emitted_events=event_count)
        holistic.close(); pipeline.stop(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
