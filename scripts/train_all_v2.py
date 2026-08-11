#!/usr/bin/env python3
"""Unified LOSO training for SVM, TCN, Attention-BiLSTM, and ST-GCN.

Example:
  python scripts/train_all_v2.py --data D:/processed_v2 --output runs/v2 \
      --models svm tcn bilstm stgcn --feature-set full --device cuda

All learners use the same canonical NPZ files and subject splits. Torch models
are selected only on validation macro-F1. After LOSO evaluation, a final model
is retrained on all subjects for the median selected epoch count.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

from feature_pipeline_v2 import canonical_keys, flatten_sequence, svm_summary
from models_v2 import build_model, prepare_graph


@dataclass
class Item:
    path: str
    subject: int
    label: int
    raw_label: int
    condition: int
    valid_ratio: float
    data: Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train all Home+ v2 gesture learners with subject-wise evaluation")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", nargs="+", default=["svm", "tcn", "bilstm", "stgcn"],
                    choices=["svm", "tcn", "bilstm", "stgcn"])
    ap.add_argument("--feature-set", default="full", choices=["core", "bones", "angles", "full"])
    ap.add_argument("--n-classes", type=int, default=7, choices=[7, 42])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--tcn-width", type=int, default=128)
    ap.add_argument("--tcn-blocks", type=int, default=5)
    ap.add_argument("--bilstm-hidden", type=int, default=128)
    ap.add_argument("--bilstm-layers", type=int, default=2)
    ap.add_argument("--bilstm-proj", type=int, default=128)
    ap.add_argument("--attention-heads", type=int, default=4)
    ap.add_argument("--stgcn-width", type=int, default=64)
    ap.add_argument("--augmentation-noise", type=float, default=0.01)
    ap.add_argument("--frame-dropout", type=float, default=0.025)
    ap.add_argument("--gradient-clip", type=float, default=1.0)
    ap.add_argument("--svm-c-grid", nargs="+", type=float,
                    default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--svm-max-iter", type=int, default=5000)
    ap.add_argument("--svm-tol", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min-valid-ratio", type=float, default=0.25)
    ap.add_argument("--exclude-subjects", nargs="*", type=int, default=[], metavar="ID",
                    help="Participant IDs to exclude before splitting (for example: 9)")
    ap.add_argument("--no-augmentation", action="store_true")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="Allow faster nondeterministic CUDA kernels")
    ap.add_argument("--no-final", action="store_true", help="Skip final all-subject retraining")
    ap.add_argument("--limit", type=int, default=0, help="Load at most N clips for smoke testing")
    return ap.parse_args()


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def load_items(root: Path, n_classes: int, min_valid: float,
               exclude_subjects=(), limit: int = 0) -> Tuple[List[Item], Dict]:
    items: List[Item] = []
    excluded = set(exclude_subjects)
    stats = dict(
        scanned_homeplus=0, excluded_subject_clips=0, low_valid_clips=0,
        excluded_subject_counts={}, low_valid_by_subject={}, retained_by_subject={},
        retained_by_task={}, retained_by_raw_label={}, exclusions=[], limited=bool(limit),
    )

    def bump(name, key):
        mapping = stats[name]
        key = str(key)
        mapping[key] = mapping.get(key, 0) + 1

    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=True) as z:
            if "schema_version" not in z or str(z["schema_version"]) != "homeplus_v2":
                continue
            stats["scanned_homeplus"] += 1
            subject = int(z["subject_id"])
            raw_label = int(z["label"])
            if subject in excluded:
                stats["excluded_subject_clips"] += 1
                bump("excluded_subject_counts", subject)
                stats["exclusions"].append(dict(
                    path=str(path), file=path.name, subject=subject, raw_label=raw_label,
                    task=raw_label % 7, condition=raw_label // 7,
                    valid_ratio=None, reason="excluded_subject",
                ))
                continue
            data = {k: np.asarray(z[k], np.float32) for k in canonical_keys()}
            valid_ratio = float(data["frame_valid"].mean())
            if valid_ratio < min_valid:
                stats["low_valid_clips"] += 1
                bump("low_valid_by_subject", subject)
                stats["exclusions"].append(dict(
                    path=str(path), file=path.name, subject=subject, raw_label=raw_label,
                    task=raw_label % 7, condition=raw_label // 7,
                    valid_ratio=valid_ratio, reason="low_validity",
                ))
                continue
        label = raw_label % 7 if n_classes == 7 else raw_label
        condition = raw_label // 7
        items.append(Item(str(path), subject, label, raw_label, condition, valid_ratio, data))
        bump("retained_by_subject", subject)
        bump("retained_by_task", label)
        bump("retained_by_raw_label", raw_label)
        if limit and len(items) >= limit:
            break
    if not items:
        raise RuntimeError("No homeplus_v2 clips found. Run process_dataset_v2.py first.")
    stats.update(
        retained_clips=len(items), subjects=sorted(set(x.subject for x in items)),
        n_subjects=len(set(x.subject for x in items)), n_classes=n_classes,
    )
    print(f"[DATA] {len(items)} clips, subjects={stats['subjects']}, "
          f"excluded_subject_clips={stats['excluded_subject_clips']}, "
          f"low_valid_clips={stats['low_valid_clips']}")
    return items, stats
def loso(items: Sequence[Item]):
    subjects = sorted(set(x.subject for x in items))
    if len(subjects) < 3:
        raise ValueError("LOSO requires at least three subjects (train, validation, test)")
    for i, test_subject in enumerate(subjects):
        val_subject = subjects[(i + 1) % len(subjects)]
        train = [x for x in items if x.subject not in (test_subject, val_subject)]
        val = [x for x in items if x.subject == val_subject]
        test = [x for x in items if x.subject == test_subject]
        yield test_subject, val_subject, train, val, test


def flat_mask(data: Dict[str, np.ndarray], layout) -> np.ndarray:
    t, d = len(data["frame_valid"]), layout.dim
    mask = np.zeros((t, d), bool)
    for name, (s, e) in layout.slices.items():
        if name in {"joint_valid", "arm_valid", "frame_valid", "angle_valid"}:
            continue
        if name == "hand_pos":
            mask[:, s:e] = np.repeat(data["joint_valid"].astype(bool), 3, axis=1)
        elif name == "hand_vel":
            valid = data["joint_valid"].astype(bool).copy()
            valid[1:] &= valid[:-1]
            valid[0] = False
            mask[:, s:e] = np.repeat(valid, 3, axis=1)
        elif name == "hand_bone":
            mask[:, s:e] = np.repeat(data["bone_valid"].astype(bool), 3, axis=1)
        elif name == "arm_pos":
            mask[:, s:e] = np.repeat(data["arm_valid"].astype(bool), 3, axis=1)
        elif name == "arm_vel":
            valid = data["arm_valid"].astype(bool).copy()
            valid[1:] &= valid[:-1]
            valid[0] = False
            mask[:, s:e] = np.repeat(valid, 3, axis=1)
        elif name == "angles":
            mask[:, s:e] = data["angle_valid"].astype(bool)
        else:
            mask[:, s:e] = data["frame_valid"][:, None].astype(bool)
    return mask


def graph_masks(data: Dict[str, np.ndarray], feature_set: str) -> Tuple[np.ndarray, np.ndarray]:
    """Validity masks for continuous graph-node and global channels."""
    joint = data["joint_valid"].astype(bool)
    joint_vel = joint.copy(); joint_vel[1:] &= joint_vel[:-1]; joint_vel[0] = False
    node_parts = [np.repeat(joint[None], 3, axis=0),
                  np.repeat(joint_vel[None], 3, axis=0)]
    if feature_set in {"bones", "full"}:
        node_parts.append(np.repeat(data["bone_valid"].astype(bool)[None], 3, axis=0))

    arm = data["arm_valid"].astype(bool)
    arm_vel = arm.copy(); arm_vel[1:] &= arm_vel[:-1]; arm_vel[0] = False
    frame = data["frame_valid"].astype(bool)[:, None]
    global_parts = [np.repeat(arm, 3, axis=1), np.repeat(arm_vel, 3, axis=1),
                    np.repeat(frame, 15, axis=1)]
    if feature_set in {"angles", "full"}:
        global_parts.append(data["angle_valid"].astype(bool))
    return np.concatenate(node_parts, axis=0), np.concatenate(global_parts, axis=1)


def fit_flat_norm(items: Sequence[Item], feature_set: str) -> Dict[str, np.ndarray]:
    sample, layout = flatten_sequence(items[0].data, feature_set)
    sums = np.zeros(layout.dim, np.float64); sums2 = np.zeros(layout.dim, np.float64)
    counts = np.zeros(layout.dim, np.float64)
    for item in items:
        x, _ = flatten_sequence(item.data, feature_set)
        m = flat_mask(item.data, layout)
        sums += (x * m).sum(0); sums2 += ((x * x) * m).sum(0); counts += m.sum(0)
    mean = np.divide(sums, np.maximum(counts, 1))
    var = np.divide(sums2, np.maximum(counts, 1)) - mean * mean
    sd = np.sqrt(np.maximum(var, 1e-6))
    continuous = counts > 0
    return dict(mean=mean.astype(np.float32), sd=sd.astype(np.float32),
                continuous=continuous, layout=layout)


def apply_flat_norm(data: Dict[str, np.ndarray], feature_set: str, norm: Dict) -> np.ndarray:
    x, layout = flatten_sequence(data, feature_set)
    m = flat_mask(data, layout)
    cont = norm["continuous"]
    x[:, cont] = (x[:, cont] - norm["mean"][cont]) / norm["sd"][cont]
    x[:, cont] *= m[:, cont]
    return x.astype(np.float32)


def fit_graph_norm(items: Sequence[Item], feature_set: str) -> Dict[str, np.ndarray]:
    n0, g0 = prepare_graph(items[0].data, feature_set)
    node_sum = np.zeros(n0.shape[0] - 1, np.float64)
    node_sum2 = np.zeros_like(node_sum); node_count = np.zeros_like(node_sum)
    global_cont = 33 + (19 if feature_set in {"angles", "full"} else 0)
    gs = np.zeros(global_cont, np.float64); gs2 = np.zeros_like(gs); gc = np.zeros_like(gs)
    for item in items:
        node, glob = prepare_graph(item.data, feature_set)
        node_mask, global_mask = graph_masks(item.data, feature_set)
        vals = node[:-1]
        node_sum += (vals * node_mask).sum((1, 2)); node_sum2 += ((vals * vals) * node_mask).sum((1, 2))
        node_count += node_mask.sum((1, 2))
        gv = glob[:, :global_cont]
        gs += (gv * global_mask).sum(0); gs2 += ((gv * gv) * global_mask).sum(0); gc += global_mask.sum(0)
    nm = node_sum / np.maximum(node_count, 1); ns = np.sqrt(np.maximum(node_sum2 / np.maximum(node_count, 1) - nm * nm, 1e-6))
    gm = gs / np.maximum(gc, 1); gsd = np.sqrt(np.maximum(gs2 / np.maximum(gc, 1) - gm * gm, 1e-6))
    return dict(node_mean=nm.astype(np.float32), node_sd=ns.astype(np.float32),
                global_mean=gm.astype(np.float32), global_sd=gsd.astype(np.float32),
                global_cont=global_cont)


def apply_graph_norm(data: Dict[str, np.ndarray], feature_set: str, norm: Dict):
    node, glob = prepare_graph(data, feature_set)
    node_mask, global_mask = graph_masks(data, feature_set)
    node[:-1] = (node[:-1] - norm["node_mean"][:, None, None]) / norm["node_sd"][:, None, None]
    node[:-1] *= node_mask
    c = norm["global_cont"]
    glob[:, :c] = (glob[:, :c] - norm["global_mean"]) / norm["global_sd"]
    glob[:, :c] *= global_mask
    return node.astype(np.float32), glob.astype(np.float32)


class GestureDataset(Dataset):
    def __init__(self, items: Sequence[Item], model_type: str, feature_set: str,
                 norm: Dict, augment: bool = False, augmentation_noise: float = 0.01,
                 frame_dropout: float = 0.025):
        self.items = list(items); self.model_type = model_type
        self.feature_set = feature_set; self.norm = norm; self.augment = augment
        self.augmentation_noise = augmentation_noise; self.frame_dropout = frame_dropout

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        mask = item.data["frame_valid"].copy().astype(np.float32)
        if self.model_type == "stgcn":
            x, glob = apply_graph_norm(item.data, self.feature_set, self.norm)
        else:
            x = apply_flat_norm(item.data, self.feature_set, self.norm); glob = np.zeros((len(mask), 1), np.float32)
        if self.augment:
            # Safe post-normalization augmentation: small noise on continuous
            # values and whole-frame dropout. Masks/validity channels remain interpretable.
            if self.model_type == "stgcn":
                x[:-1] += np.random.normal(
                    0, self.augmentation_noise, x[:-1].shape).astype(np.float32)
            else:
                cont = self.norm["continuous"]
                x[:, cont] += np.random.normal(
                    0, self.augmentation_noise, (len(x), int(cont.sum()))).astype(np.float32)
            drop = np.random.random(len(mask)) < self.frame_dropout
            mask[drop] = 0.0
            if self.model_type == "stgcn": x[:, drop] = 0.0
            else: x[drop] = 0.0
            glob[drop] = 0.0
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(glob), item.label


def evaluate(model, loader, device):
    model.eval(); ys, ps, scores = [], [], []
    with torch.no_grad():
        for x, mask, glob, y in loader:
            logits = model(x.to(device), mask.to(device), glob.to(device))
            probability = torch.softmax(logits, dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist()); ps.extend(probability.argmax(1).tolist())
            scores.extend(probability.tolist())
    return np.asarray(ys), np.asarray(ps), np.asarray(scores, np.float32)


def model_config(model_type: str, x0, g0, args) -> Dict:
    config = dict(dropout=args.dropout)
    if model_type == "stgcn":
        config.update(node_channels=x0.shape[0], global_dim=g0.shape[1],
                      width=args.stgcn_width)
    elif model_type == "tcn":
        config.update(in_dim=x0.shape[1], width=args.tcn_width, blocks=args.tcn_blocks)
    elif model_type == "bilstm":
        config.update(in_dim=x0.shape[1], hidden=args.bilstm_hidden,
                      layers=args.bilstm_layers, proj=args.bilstm_proj,
                      heads=args.attention_heads)
    return config


def torch_fold(model_type: str, feature_set: str, train: Sequence[Item], val: Sequence[Item],
               test: Sequence[Item], args, device, fold_seed: int):
    norm = fit_graph_norm(train, feature_set) if model_type == "stgcn" else fit_flat_norm(train, feature_set)
    ds_train = GestureDataset(
        train, model_type, feature_set, norm, not args.no_augmentation,
        args.augmentation_noise, args.frame_dropout,
    )
    ds_val = GestureDataset(val, model_type, feature_set, norm)
    ds_test = GestureDataset(test, model_type, feature_set, norm)
    generator = torch.Generator().manual_seed(fold_seed)
    dl_train = DataLoader(ds_train, args.batch_size, True, num_workers=args.num_workers,
                          generator=generator)
    dl_val = DataLoader(ds_val, args.batch_size, False, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, args.batch_size, False, num_workers=args.num_workers)
    x0, _, g0, _ = ds_train[0]
    config = model_config(model_type, x0, g0, args)
    model = build_model(model_type, args.n_classes, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_state = None; best_f1 = -1.0; best_epoch = 1; stale = 0; history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, mask, glob, y in dl_train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), mask.to(device), glob.to(device))
            loss = criterion(logits, y.to(device)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step(); losses.append(float(loss.detach()))
        yv, pv, _ = evaluate(model, dl_val, device)
        vf1 = f1_score(yv, pv, average="macro", zero_division=0)
        history.append(dict(
            epoch=epoch, train_loss=float(np.mean(losses)), val_macro_f1=float(vf1),
            val_accuracy=float(accuracy_score(yv, pv)),
            elapsed_seconds=float(time.perf_counter() - started),
        ))
        print(f"  epoch={epoch:03d} loss={np.mean(losses):.4f} val_f1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_epoch, stale = vf1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    yt, pt, scores = evaluate(model, dl_test, device)
    return dict(
        y_true=yt, y_pred=pt, scores=scores, best_epoch=best_epoch,
        best_val_f1=float(best_f1), config=config, history=history,
        parameter_count=sum(p.numel() for p in model.parameters()),
        fold_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=(float(torch.cuda.max_memory_allocated(device) / 1024**2)
                            if device.type == "cuda" else None),
    )
def prepare_svm(items: Sequence[Item], feature_set: str):
    xs, ys = [], []
    for item in items:
        seq, _ = flatten_sequence(item.data, feature_set)
        xs.append(svm_summary(seq, item.data["frame_valid"].astype(bool))); ys.append(item.label)
    return np.stack(xs), np.asarray(ys)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def metric_bundle(y_true, y_pred, n_classes: int) -> Dict:
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sum = cm.sum(1, keepdims=True)
    cm_normalized = np.divide(cm, row_sum, where=row_sum != 0,
                              out=np.zeros_like(cm, dtype=np.float64))
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0,
    )
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(report["macro avg"]["recall"]),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=labels, average="weighted",
                                   zero_division=0)),
        confusion_matrix_counts=cm.tolist(),
        confusion_matrix_row_normalized=cm_normalized.tolist(),
        classification_report=report,
    )


def grouped_metrics(y_true, y_pred, groups, n_classes: int) -> Dict:
    result = {}
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); groups = np.asarray(groups)
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        result[str(group)] = dict(n=int(mask.sum()), **metric_bundle(
            y_true[mask], y_pred[mask], n_classes,
        ))
    return result


def distribution(values) -> Dict:
    values = [float(v) for v in values]
    return dict(
        n=len(values), mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        min=float(np.min(values)), max=float(np.max(values)), median=float(np.median(values)),
        q25=float(np.quantile(values, 0.25)), q75=float(np.quantile(values, 0.75)),
    )


def benchmark_torch_model(model, sample, device, warmup: int = 10, runs: int = 50) -> Dict:
    x, mask, glob, _ = sample
    x=x[None].to(device); mask=mask[None].to(device); glob=glob[None].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): model(x, mask, glob)
        if device.type == "cuda": torch.cuda.synchronize(device)
        times = []
        for _ in range(runs):
            started = time.perf_counter(); model(x, mask, glob)
            if device.type == "cuda": torch.cuda.synchronize(device)
            times.append((time.perf_counter() - started) * 1000.0)
    return dict(batch_size=1, warmup=warmup, runs=runs,
                mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times, 0.95)))


def environment_report(args) -> Dict:
    git_commit = git_status = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        git_status = subprocess.check_output(
            ["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
    except Exception:
        pass
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        python=sys.version, platform=platform.platform(), numpy=np.__version__,
        sklearn=sklearn.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        cudnn_version=(torch.backends.cudnn.version() if torch.cuda.is_available() else None),
        deterministic=not args.nondeterministic, git_commit=git_commit, git_status=git_status,
    )

def train_model(model_type: str, items: List[Item], args, root: Path) -> Dict:
    out = root / model_type; (out / "folds").mkdir(parents=True, exist_ok=True)
    fold_rows = []; prediction_rows = []; all_y = []; all_p = []; all_conditions = []
    selected_epochs = []; selected_cs = []; device = torch.device(args.device)
    model_started = time.perf_counter(); parameter_count = None; fold_model_config = None
    print(f"\n===== {model_type.upper()} / {args.feature_set} =====")

    for fold_index, (test_subject, val_subject, train, val, test) in enumerate(loso(items)):
        fold_seed = args.seed + fold_index
        set_seed(fold_seed, deterministic=not args.nondeterministic)
        print(f"[FOLD] test={test_subject} val={val_subject} train={len(train)} seed={fold_seed}")
        best_c = None; c_trace = []; peak_gpu = None
        if model_type == "svm":
            fold_started = time.perf_counter()
            xtr, ytr = prepare_svm(train, args.feature_set)
            xval, yval = prepare_svm(val, args.feature_set)
            xte, yte = prepare_svm(test, args.feature_set)
            clf = None; val_f1 = -1.0
            for c in args.svm_c_grid:
                candidate = make_pipeline(
                    StandardScaler(), LinearSVC(
                        C=c, class_weight="balanced", dual="auto", random_state=fold_seed,
                        max_iter=args.svm_max_iter, tol=args.svm_tol,
                    ),
                )
                candidate.fit(xtr, ytr)
                val_pred = candidate.predict(xval)
                score = f1_score(yval, val_pred, average="macro", zero_division=0)
                c_trace.append(dict(C=float(c), val_macro_f1=float(score),
                                    val_accuracy=float(accuracy_score(yval, val_pred))))
                if score > val_f1:
                    clf, best_c, val_f1 = candidate, float(c), float(score)
            selected_cs.append(best_c)
            pred = clf.predict(xte); scores = np.asarray(clf.decision_function(xte))
            if scores.ndim == 1: scores = scores[:, None]
            best_epoch = 0; history = c_trace
            fold_seconds = time.perf_counter() - fold_started
            parameter_count = int(clf[-1].coef_.size + clf[-1].intercept_.size)
            fold_model_config = dict(
                descriptor_dim=int(xtr.shape[1]), C_grid=args.svm_c_grid,
                max_iter=args.svm_max_iter, tol=args.svm_tol,
            )
            score_type = "decision_function"
        else:
            run = torch_fold(
                model_type, args.feature_set, train, val, test, args, device, fold_seed,
            )
            yte=run["y_true"]; pred=run["y_pred"]; scores=run["scores"]
            best_epoch=run["best_epoch"]; val_f1=run["best_val_f1"]
            history=run["history"]; fold_seconds=run["fold_seconds"]
            peak_gpu=run["peak_gpu_memory_mb"]; parameter_count=run["parameter_count"]
            fold_model_config=run["config"]; selected_epochs.append(best_epoch)
            score_type = "softmax_probability"

        metrics = metric_bundle(yte, pred, args.n_classes)
        conditions = np.asarray([item.condition for item in test])
        condition_metrics = grouped_metrics(yte, pred, conditions, args.n_classes)
        print(f"[TEST] subject={test_subject} f1={metrics['macro_f1']:.4f} "
              f"acc={metrics['accuracy']:.4f}")
        fold_detail = dict(
            fold_index=fold_index, fold_seed=fold_seed, test_subject=test_subject,
            validation_subject=val_subject,
            training_subjects=sorted(set(x.subject for x in train)),
            n_train=len(train), n_validation=len(val), n_test=len(test),
            best_epoch=best_epoch, best_validation_macro_f1=float(val_f1), best_c=best_c,
            model_config=fold_model_config, parameter_count=parameter_count,
            fold_seconds=float(fold_seconds), peak_gpu_memory_mb=peak_gpu,
            score_type=score_type, selection_history=history,
            metrics=metrics, condition_metrics=condition_metrics,
        )
        save_json(out / "folds" / f"test_subject_{test_subject}.json", fold_detail)
        fold_rows.append(dict(
            test_subject=test_subject, validation_subject=val_subject,
            n_train=len(train), n_validation=len(val), n_test=len(test),
            accuracy=metrics["accuracy"], balanced_accuracy=metrics["balanced_accuracy"],
            macro_f1=metrics["macro_f1"], weighted_f1=metrics["weighted_f1"],
            best_epoch=best_epoch, best_validation_macro_f1=float(val_f1), best_c=best_c,
            fold_seed=fold_seed, fold_seconds=float(fold_seconds),
            peak_gpu_memory_mb=peak_gpu,
        ))
        for item, truth, prediction, score in zip(test, yte, pred, scores):
            row = dict(
                model=model_type, feature_set=args.feature_set, fold_index=fold_index,
                test_subject=test_subject, validation_subject=val_subject,
                file=Path(item.path).name, path=item.path, raw_label=item.raw_label,
                condition=item.condition, true_task=int(truth), predicted_task=int(prediction),
                correct=int(truth == prediction), valid_ratio=item.valid_ratio,
                score_type=score_type, confidence=float(np.max(score)),
            )
            for class_id, value in enumerate(np.asarray(score).tolist()):
                row[f"score_{class_id}"] = float(value)
            prediction_rows.append(row)
        all_y.extend(np.asarray(yte).tolist()); all_p.extend(np.asarray(pred).tolist())
        all_conditions.extend(conditions.tolist())

    all_y=np.asarray(all_y); all_p=np.asarray(all_p); all_conditions=np.asarray(all_conditions)
    pooled = metric_bundle(all_y, all_p, args.n_classes)
    f1_dist = distribution([row["macro_f1"] for row in fold_rows])
    acc_dist = distribution([row["accuracy"] for row in fold_rows])
    result = dict(
        model=model_type, feature_set=args.feature_set, n_classes=args.n_classes,
        evaluation_protocol="outer LOSO; one different subject for validation/early stopping",
        score_type=("decision_function" if model_type == "svm" else "softmax_probability"),
        model_config=fold_model_config, parameter_count=parameter_count,
        pooled_metrics=pooled, per_subject_macro_f1=f1_dist,
        per_subject_accuracy=acc_dist, condition_metrics=grouped_metrics(
            all_y, all_p, all_conditions, args.n_classes,
        ), folds=fold_rows, total_loso_seconds=float(time.perf_counter()-model_started),
        final_model=None,
    )
    write_csv(out / "predictions_out_of_fold.csv", prediction_rows)
    write_csv(out / "per_subject_metrics.csv", fold_rows)
    per_class_rows=[]
    for class_id in range(args.n_classes):
        values=pooled["classification_report"][str(class_id)]
        per_class_rows.append(dict(task=class_id, **values))
    write_csv(out / "per_task_metrics.csv", per_class_rows)
    condition_rows=[]
    for condition, values in result["condition_metrics"].items():
        condition_rows.append(dict(
            condition=int(condition), n=values["n"], accuracy=values["accuracy"],
            balanced_accuracy=values["balanced_accuracy"], macro_f1=values["macro_f1"],
            weighted_f1=values["weighted_f1"],
        ))
    write_csv(out / "per_condition_metrics.csv", condition_rows)
    save_json(out / "results.json", result)

    if args.no_final:
        return result

    print(f"[FINAL] fitting {model_type} on all {len(items)} clips")
    set_seed(args.seed + 10000, deterministic=not args.nondeterministic)
    final_started = time.perf_counter()
    if model_type == "svm":
        x, y = prepare_svm(items, args.feature_set)
        final_c = statistics.multimode(selected_cs)[0]
        clf = make_pipeline(
            StandardScaler(), LinearSVC(C=final_c, class_weight="balanced", dual="auto",
                                        random_state=args.seed + 10000,
                                        max_iter=args.svm_max_iter, tol=args.svm_tol),
        )
        clf.fit(x, y)
        times=[]
        for _ in range(100):
            started=time.perf_counter(); clf.predict(x[:1]); times.append((time.perf_counter()-started)*1000)
        artifact_path=out / "model_final.joblib"
        joblib.dump(dict(
            model=clf, model_type="svm", feature_set=args.feature_set,
            n_classes=args.n_classes, selected_c=final_c, schema_version="homeplus_v2",
            task_mapping="raw_label % 7", excluded_subjects=args.exclude_subjects,
            min_valid_ratio=args.min_valid_ratio,
        ), artifact_path)
        result["final_model"] = dict(
            selected_c=final_c, trained_clips=len(items),
            training_seconds=float(time.perf_counter()-final_started),
            artifact=str(artifact_path), artifact_bytes=artifact_path.stat().st_size,
            latency=dict(batch_size=1, runs=100, mean_ms=float(np.mean(times)),
                         median_ms=float(np.median(times)), p95_ms=float(np.quantile(times,.95))),
        )
    else:
        epochs=max(1,int(round(statistics.median(selected_epochs))))
        norm=fit_graph_norm(items,args.feature_set) if model_type=="stgcn" else fit_flat_norm(items,args.feature_set)
        ds=GestureDataset(items,model_type,args.feature_set,norm,not args.no_augmentation,
                          args.augmentation_noise,args.frame_dropout)
        generator=torch.Generator().manual_seed(args.seed+10000)
        dl=DataLoader(ds,args.batch_size,True,num_workers=args.num_workers,generator=generator)
        x0,_,g0,_=ds[0]; config=model_config(model_type,x0,g0,args)
        model=build_model(model_type,args.n_classes,config).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        criterion=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        final_history=[]
        for epoch in range(1,epochs+1):
            model.train(); losses=[]
            for x,mask,glob,y in dl:
                opt.zero_grad(set_to_none=True); logits=model(x.to(device),mask.to(device),glob.to(device))
                loss=criterion(logits,y.to(device)); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip); opt.step(); losses.append(float(loss.detach()))
            final_history.append(dict(epoch=epoch,train_loss=float(np.mean(losses))))
            print(f"  final_epoch={epoch:03d}/{epochs}")
        latency=benchmark_torch_model(model,ds[0],device)
        artifact_path=out / "model_final.pt"
        portable_norm = dict(norm)
        layout = portable_norm.pop("layout", None)
        portable_layout = None if layout is None else dict(
            feature_set=layout.feature_set, slices=layout.slices, dim=layout.dim,
        )
        torch.save(dict(
            schema_version="homeplus_v2",model_type=model_type,feature_set=args.feature_set,
            n_classes=args.n_classes,model_config=config,normalization=portable_norm,
            feature_layout=portable_layout,
            state_dict=model.cpu().state_dict(),trained_epochs=epochs,
            task_mapping="raw_label % 7",excluded_subjects=args.exclude_subjects,
            min_valid_ratio=args.min_valid_ratio,training_hyperparameters=vars(args),
        ),artifact_path)
        result["final_model"]=dict(
            trained_epochs=epochs,trained_clips=len(items),training_history=final_history,
            training_seconds=float(time.perf_counter()-final_started),artifact=str(artifact_path),
            artifact_bytes=artifact_path.stat().st_size,latency=latency,
        )
    save_json(out / "results.json",result)
    return result
def main() -> None:
    args=parse_args(); set_seed(args.seed,deterministic=not args.nondeterministic)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment cannot use CUDA. "
            "Install the documented CUDA-enabled PyTorch wheel first."
        )
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    experiment=dict(
        schema_version="homeplus_experiment_v2",created_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args),task_definition="7 commands; condition is nuisance variation",
        task_mapping="task = raw_label % 7",condition_mapping="condition = raw_label // 7",
        evaluation_protocol=("outer leave-one-subject-out; next remaining subject is validation; "
                             "normalization and model selection use training/validation only"),
    )
    save_json(output/"experiment_config.json",experiment)
    save_json(output/"environment.json",environment_report(args))

    items,load_stats=load_items(
        Path(args.data),args.n_classes,args.min_valid_ratio,args.exclude_subjects,args.limit,
    )
    valid_values=[item.valid_ratio for item in items]
    retained_by_condition={}
    for item in items:
        key=str(item.condition);retained_by_condition[key]=retained_by_condition.get(key,0)+1
    load_stats["retained_by_condition"]=retained_by_condition
    load_stats["valid_ratio_distribution"]=distribution(valid_values)
    load_stats["quality_rule"]=dict(
        minimum_valid_ratio=args.min_valid_ratio,excluded_subjects=args.exclude_subjects,
    )
    save_json(output/"dataset_summary.json",load_stats)
    write_csv(output/"excluded_clips.csv",load_stats["exclusions"])
    write_csv(output/"retained_clips.csv",[
        dict(path=x.path,file=Path(x.path).name,subject=x.subject,raw_label=x.raw_label,
             condition=x.condition,task=x.label,valid_ratio=x.valid_ratio) for x in items
    ])
    fold_manifest=[]
    for fold_index,(test_subject,val_subject,train,val,test) in enumerate(loso(items)):
        fold_manifest.append(dict(
            fold_index=fold_index,test_subject=test_subject,validation_subject=val_subject,
            training_subjects=sorted(set(x.subject for x in train)),n_train=len(train),
            n_validation=len(val),n_test=len(test),fold_seed=args.seed+fold_index,
        ))
    save_json(output/"fold_manifest.json",fold_manifest)
    write_csv(output/"fold_manifest.csv",[
        {**row,"training_subjects":" ".join(map(str,row["training_subjects"]))}
        for row in fold_manifest
    ])

    started=time.perf_counter();results=[]
    for model_type in args.models:
        results.append(train_model(model_type,items,args,output))
    comparison_rows=[]
    for result in results:
        comparison_rows.append(dict(
            model=result["model"],feature_set=result["feature_set"],
            n_classes=result["n_classes"],parameter_count=result["parameter_count"],
            pooled_accuracy=result["pooled_metrics"]["accuracy"],
            pooled_balanced_accuracy=result["pooled_metrics"]["balanced_accuracy"],
            pooled_macro_f1=result["pooled_metrics"]["macro_f1"],
            pooled_weighted_f1=result["pooled_metrics"]["weighted_f1"],
            subject_macro_f1_mean=result["per_subject_macro_f1"]["mean"],
            subject_macro_f1_std=result["per_subject_macro_f1"]["std"],
            subject_macro_f1_min=result["per_subject_macro_f1"]["min"],
            subject_macro_f1_max=result["per_subject_macro_f1"]["max"],
            loso_seconds=result["total_loso_seconds"],
            final_latency_median_ms=(result["final_model"]["latency"]["median_ms"]
                                     if result["final_model"] else None),
            final_artifact_bytes=(result["final_model"]["artifact_bytes"]
                                  if result["final_model"] else None),
        ))
    elapsed=time.perf_counter()-started
    comparison=dict(
        schema_version="homeplus_model_comparison_v2",elapsed_seconds=float(elapsed),
        dataset=dict(retained_clips=len(items),subjects=load_stats["subjects"],
                     excluded_subjects=args.exclude_subjects,
                     minimum_valid_ratio=args.min_valid_ratio),
        models=comparison_rows,
    )
    save_json(output/"comparison_summary.json",comparison)
    write_csv(output/"comparison_summary.csv",comparison_rows)
    save_json(output/"run_status.json",dict(
        status="complete",completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=float(elapsed),models=args.models,output=str(output.resolve()),
    ))
    print(f"[DONE] elapsed={elapsed:.1f}s outputs={output}")

if __name__ == "__main__":
    main()
