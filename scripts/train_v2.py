#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_v2.py — Attention-BiLSTM training with LOSO CV and data augmentations.

Changes from train_v1.py (targeting LOSO generalization gap):
  1. Participant-disjoint validation: hold out a second participant as val
     instead of random 10% of remaining clips (fixes leaky val set)
  2. Per-clip z-normalization: strips absolute scale/position per clip
  3. Scale jitter augmentation: random spatial scale on pose+velocity blocks
  4. Reduced default model capacity: hidden 128, lstm_layers 2, dropout 0.4,
     weight_decay 5e-4 (~800K params vs 2.4M)
  5. Increased patience: 15 → 20 (noisier participant-disjoint val signal)
"""

import os, argparse, random, glob, json, logging, time, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             confusion_matrix)

# Optional: matplotlib for plots (graceful fallback)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ================================================================
# CLI
# ================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Attention-BiLSTM trainer (LOSO / K-fold)")

    # Data
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Directories with processed *.npz clips (one per participant)")
    ap.add_argument("--save_path", type=str, default="runs/v2",
                    help="Output folder for checkpoints, plots, reports")
    ap.add_argument("--n_classes", type=int, default=42, choices=[7, 42],
                    help="Train on 42 fine-grained or 7 command classes")

    # Evaluation
    ap.add_argument("--eval_mode", type=str, default="loso", choices=["loso", "kfold"],
                    help="LOSO (leave-one-subject-out) or stratified K-fold")
    ap.add_argument("--k", type=int, default=5, help="K for K-fold (ignored in LOSO)")

    # Model — reduced defaults for less overfitting
    ap.add_argument("--hidden", type=int, default=128, help="LSTM hidden size per direction")
    ap.add_argument("--lstm_layers", type=int, default=2, help="Number of BiLSTM layers")
    ap.add_argument("--attn_heads", type=int, default=4, help="Attention heads for temporal pooling")
    ap.add_argument("--proj_dim", type=int, default=128, help="Input projection dimension")
    ap.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")

    # Training — increased regularization
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20,
                    help="Early stop after N epochs without val macro-F1 improvement")
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # Augmentation
    ap.add_argument("--no_aug", action="store_true", help="Disable all augmentations")
    ap.add_argument("--aug_speed", action="store_true", default=True)
    ap.add_argument("--aug_warp", action="store_true", default=True)
    ap.add_argument("--aug_noise_std", type=float, default=0.02)
    ap.add_argument("--aug_frame_drop_prob", type=float, default=0.05)
    ap.add_argument("--aug_block_mask_prob", type=float, default=0.05)
    ap.add_argument("--aug_scale_jitter", action="store_true", default=True,
                    help="Apply random scale jitter to pose+velocity blocks")
    ap.add_argument("--no_aug_scale_jitter", action="store_true",
                    help="Disable scale jitter augmentation")

    # Per-clip normalization
    ap.add_argument("--clip_norm", action="store_true", default=True,
                    help="Per-clip z-normalization after global standardization")
    ap.add_argument("--no_clip_norm", action="store_true",
                    help="Disable per-clip z-normalization")

    # Misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()

    # Handle --no_* flags
    if args.no_aug_scale_jitter:
        args.aug_scale_jitter = False
    if args.no_clip_norm:
        args.clip_norm = False

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
    """Extract participant ID from path like dataset/5/gesture_5_12.npz."""
    # Try directory name first
    parent = os.path.basename(os.path.dirname(path))
    if parent.isdigit():
        return parent
    # Fallback: parse filename  gesture_{pid}_{idx}.npz
    m = re.match(r"gesture_(\d+)_", os.path.basename(path))
    if m:
        return m.group(1)
    return parent  # best effort


def load_items(roots):
    items = []
    for r in roots:
        print(f"[LOAD] {r}")
        for path in sorted(glob.glob(os.path.join(r, "*.npz"))):
            z = np.load(path)
            if "X" not in z:
                continue
            if "valid" in z:
                valid_arr = z["valid"]
            elif "valid_T" in z:
                valid_arr = z["valid_T"]
            else:
                continue
            X = z["X"].astype(np.float32)           # (T, F)
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
            pid = extract_participant_id(path)
            items.append(dict(path=path, X=X, valid=valid, y42=y42, pid=pid))
    print(f"Loaded {len(items)} clips from {len(set(it['pid'] for it in items))} participants.")
    return items


# ================================================================
# Augmentation
# ================================================================
def speed_perturb(X, valid, lo=0.85, hi=1.15):
    """Resample to simulate different execution speeds."""
    T, F = X.shape
    speed = np.random.uniform(lo, hi)
    new_T = max(4, int(round(T / speed)))
    idx = np.linspace(0, T - 1, new_T)
    lo_idx = np.floor(idx).astype(int)
    hi_idx = np.clip(lo_idx + 1, 0, T - 1)
    w = (idx - lo_idx)[:, None]
    X_new = ((1 - w) * X[lo_idx] + w * X[hi_idx]).astype(np.float32)
    # Resample valid mask
    v_float = valid.astype(np.float32)
    v_new = np.interp(np.linspace(0, T - 1, new_T), np.arange(T), v_float)
    v_new = v_new > 0.5
    # Resample back to original T
    idx2 = np.linspace(0, new_T - 1, T)
    lo2 = np.floor(idx2).astype(int)
    hi2 = np.clip(lo2 + 1, 0, new_T - 1)
    w2 = (idx2 - lo2)[:, None]
    X_out = ((1 - w2) * X_new[lo2] + w2 * X_new[hi2]).astype(np.float32)
    v2 = np.interp(np.linspace(0, new_T - 1, T), np.arange(new_T), v_new.astype(np.float32))
    v_out = v2 > 0.5
    return X_out, v_out


def time_warp(X, valid, sigma=0.2):
    """Non-linear temporal distortion via smooth warp field."""
    T, F = X.shape
    # Generate smooth random warp
    warp = np.cumsum(np.random.randn(T) * sigma)
    warp = warp - warp.min()
    warp = warp / (warp.max() + 1e-8) * (T - 1)
    lo_idx = np.floor(warp).astype(int).clip(0, T - 2)
    hi_idx = lo_idx + 1
    w = (warp - lo_idx)[:, None]
    X_out = ((1 - w) * X[lo_idx] + w * X[hi_idx]).astype(np.float32)
    v_float = valid.astype(np.float32)
    v_warp = (1 - w.squeeze()) * v_float[lo_idx] + w.squeeze() * v_float[hi_idx]
    v_out = v_warp > 0.5
    return X_out, v_out


def gaussian_noise(X, std=0.02):
    return X + np.random.randn(*X.shape).astype(np.float32) * std


def frame_dropout(X, valid, prob=0.05):
    """Randomly zero out entire frames and mark them invalid."""
    T = X.shape[0]
    drop = np.random.rand(T) < prob
    X_out = X.copy()
    v_out = valid.copy()
    X_out[drop] = 0.0
    v_out[drop] = False
    return X_out, v_out


def feature_block_mask(X, prob=0.05):
    """Randomly zero entire feature blocks (pose/velocity/wrist/scale).

    Feature layout for 148-dim: pose(72) + vel_pose(72) + vel_wrist(3) + scale(1)
    """
    if np.random.rand() > prob:
        return X
    X_out = X.copy()
    # Define blocks
    blocks = [
        (0, 72),       # pose
        (72, 144),      # velocity_pose
        (144, 147),     # velocity_wrist
        (147, 148),     # scale
    ]
    # Pick one block to mask
    start, end = blocks[np.random.randint(len(blocks))]
    X_out[:, start:end] = 0.0
    return X_out


def scale_jitter(X, lo=0.8, hi=1.2):
    """Multiply pose and velocity feature blocks by a random scale factor.

    Breaks the link between body proportions and class identity.
    Feature layout for 148-dim: pose(72) + vel_pose(72) + vel_wrist(3) + scale(1)
    """
    X_out = X.copy()
    factor = np.random.uniform(lo, hi)
    # Scale pose block (0:72) and velocity_pose block (72:144)
    X_out[:, :144] *= factor
    return X_out


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
            noise_std = self.cfg.get("noise_std", 0.02)
            if noise_std > 0:
                X = gaussian_noise(X, noise_std)
            fdrop = self.cfg.get("frame_drop_prob", 0.05)
            if fdrop > 0:
                X, valid = frame_dropout(X, valid, fdrop)
            bmask = self.cfg.get("block_mask_prob", 0.05)
            if bmask > 0:
                X = feature_block_mask(X, bmask)
            if self.cfg.get("scale_jitter", True):
                X = scale_jitter(X)

        # Per-clip z-normalization: strip absolute scale/position
        if self.clip_norm:
            mu = X.mean(axis=0)
            sd = X.std(axis=0) + 1e-6
            X = (X - mu) / sd

        return X, valid.astype(np.float32), y


def collate(batch):
    Xs, Ms, Ys = zip(*batch)
    X = torch.from_numpy(np.stack(Xs, 0)).float()
    M = torch.from_numpy(np.stack(Ms, 0)).float()
    y = torch.tensor(Ys, dtype=torch.long)
    return X, M, y


# ================================================================
# Model: Attention-BiLSTM
# ================================================================
class TemporalAttentionPool(nn.Module):
    """Learned-query multi-head attention pooling over time steps."""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, T, D)
        key_padding_mask: (B, T) True = ignore
        Returns: (B, D)
        """
        B = x.size(0)
        q = self.query.expand(B, -1, -1)              # (B, 1, D)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)                          # (B, D)


class AttentionBiLSTM(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128, lstm_layers=2,
                 attn_heads=4, proj_dim=128, dropout=0.4):
        super().__init__()
        self.proj_dim = proj_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            proj_dim, hidden, lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True, bidirectional=True,
        )
        lstm_out_dim = hidden * 2

        # Residual alignment: project LSTM output back to proj_dim for residual add
        self.res_proj = nn.Linear(lstm_out_dim, proj_dim) if lstm_out_dim != proj_dim else nn.Identity()
        self.res_norm = nn.LayerNorm(proj_dim)

        # Temporal attention pooling
        self.attn_pool = TemporalAttentionPool(proj_dim, num_heads=attn_heads)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, n_classes),
        )

    def forward(self, x, mask):
        """
        x: (B, T, F)
        mask: (B, T) float, 1=valid 0=invalid
        """
        # Input projection
        h_proj = self.input_proj(x)                     # (B, T, proj_dim)

        # Pack for LSTM
        lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(
            h_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Pad lstm_out to match h_proj length if needed
        T_proj = h_proj.size(1)
        T_lstm = lstm_out.size(1)
        if T_lstm < T_proj:
            pad = torch.zeros(lstm_out.size(0), T_proj - T_lstm, lstm_out.size(2),
                              device=lstm_out.device, dtype=lstm_out.dtype)
            lstm_out = torch.cat([lstm_out, pad], dim=1)

        # Residual connection
        h = self.res_norm(self.res_proj(lstm_out) + h_proj)

        # Attention pooling (mask: True = ignore for MHA)
        key_padding_mask = ~(mask[:, :T_proj].bool())
        pooled = self.attn_pool(h, key_padding_mask=key_padding_mask)

        return self.head(pooled)


# ================================================================
# Standardization
# ================================================================
def standardize_fit(train_items):
    Xtr = np.stack([it["X"] for it in train_items], 0)
    mu = Xtr.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    sd = Xtr.std(axis=(0, 1), keepdims=True).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def standardize_apply(items, mu, sd):
    out = []
    for it in items:
        X = (it["X"] - mu.squeeze(0)) / (sd.squeeze(0) + 1e-8)
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
        idx = torch.tensor([j, j + 7, j + 14, j + 21, j + 28, j + 35], dtype=torch.long)
        W7[j] = W42.index_select(0, idx).mean(dim=0)
        b7[j] = b42.index_select(0, idx).mean()
    return head_w_name, head_b_name, W7, b7


# ================================================================
# Training & Evaluation
# ================================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, device, clip_grad):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for X, M, y in loader:
        X, M, y = X.to(device), M.to(device), y.to(device)
        logits = model(X, M)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * X.size(0)
        n_samples += X.size(0)
    return total_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p, all_pid = [], [], []
    for X, M, y in loader:
        X, M = X.to(device), M.to(device)
        logits = model(X, M)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_y.append(y.numpy())
        all_p.append(preds)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    return y, p


def macro_f1(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)))


# ================================================================
# Plotting helpers
# ================================================================
def plot_confusion_matrix(y_true, y_pred, n_classes, title, save_path):
    if not HAS_MPL:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.3), max(8, n_classes * 0.3)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_training_curves(history, save_path):
    if not HAS_MPL or not history:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_f1"], label="val F1")
    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["val_acc"], label="val Acc")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ================================================================
# LOSO / K-fold split generation
# ================================================================
def generate_loso_splits(items):
    """Yield (train_items, val_items, test_items, fold_name) per participant.

    Participant-disjoint validation: for each test participant, the next
    participant (cyclic) is held out as validation.  Train uses the remaining
    participants.  This prevents early stopping from selecting an overfit
    checkpoint via a leaky (participant-shared) validation set.
    """
    pids = sorted(set(it["pid"] for it in items))
    for i, test_pid in enumerate(pids):
        val_pid = pids[(i + 1) % len(pids)]
        test_items = [it for it in items if it["pid"] == test_pid]
        val_items = [it for it in items if it["pid"] == val_pid]
        train_items = [it for it in items if it["pid"] != test_pid and it["pid"] != val_pid]
        yield train_items, val_items, test_items, f"LOSO-{test_pid}"


def generate_kfold_splits(items, k, seed):
    ys = np.array([it["y42"] for it in items])
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(np.zeros(len(ys)), ys)):
        tr_idx = list(tr_idx)
        np.random.shuffle(tr_idx)
        n_val = max(1, int(0.1 * len(tr_idx)))
        va_idx = tr_idx[:n_val]
        tr_idx = tr_idx[n_val:]
        train_items = [items[i] for i in tr_idx]
        val_items = [items[i] for i in va_idx]
        test_items = [items[i] for i in te_idx]
        yield train_items, val_items, test_items, f"fold-{fold_idx + 1}"


# ================================================================
# Main
# ================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Output directory
    out_dir = Path(args.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_path = out_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    # Save config
    config = vars(args)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Config: {json.dumps(config, indent=2)}")

    # Load data
    items = load_items(args.roots)
    if not items:
        log.error("No training clips found.")
        return

    # Map labels if training on 7 classes
    n_classes = args.n_classes
    if n_classes == 7:
        for it in items:
            it["y42"] = it["y42"] % 7  # reuse y42 field for 7-class label
        log.info("Mapped labels to 7 command classes.")

    in_dim = items[0]["X"].shape[-1]
    log.info(f"N={len(items)}, in_dim={in_dim}, n_classes={n_classes}")

    # Augmentation config
    aug_cfg = None
    if not args.no_aug:
        aug_cfg = dict(
            speed=args.aug_speed,
            warp=args.aug_warp,
            noise_std=args.aug_noise_std,
            frame_drop_prob=args.aug_frame_drop_prob,
            block_mask_prob=args.aug_block_mask_prob,
            scale_jitter=args.aug_scale_jitter,
        )

    # Generate splits
    if args.eval_mode == "loso":
        splits = list(generate_loso_splits(items))
    else:
        splits = list(generate_kfold_splits(items, args.k, args.seed))

    log.info(f"Evaluation: {args.eval_mode} — {len(splits)} folds")

    # Aggregate results
    all_y_test, all_p_test = [], []
    all_test_items_with_preds = []
    fold_results = []
    best_global_f1 = -1.0
    best_global_state = None
    best_global_mu = None
    best_global_sd = None

    for fold_i, (train_items, val_items, test_items, fold_name) in enumerate(splits):
        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold_i + 1}/{len(splits)}: {fold_name}  "
                 f"(train={len(train_items)} val={len(val_items)} test={len(test_items)})")
        log.info(f"{'='*60}")

        # Standardize
        mu, sd = standardize_fit(train_items)
        train_std = standardize_apply(train_items, mu, sd)
        val_std = standardize_apply(val_items, mu, sd)
        test_std = standardize_apply(test_items, mu, sd)

        # Datasets
        ds_train = AugmentedClipSet(train_std, augment=(aug_cfg is not None),
                                    aug_cfg=aug_cfg, clip_norm=args.clip_norm)
        ds_val = AugmentedClipSet(val_std, augment=False, clip_norm=args.clip_norm)
        ds_test = AugmentedClipSet(test_std, augment=False, clip_norm=args.clip_norm)

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=collate, num_workers=args.num_workers)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=args.num_workers)

        # Model
        model = AttentionBiLSTM(
            in_dim, n_classes,
            hidden=args.hidden, lstm_layers=args.lstm_layers,
            attn_heads=args.attn_heads, proj_dim=args.proj_dim, dropout=args.dropout,
        ).to(device)

        # Optimizer, scheduler, loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        total_steps = args.epochs * len(dl_train)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy="cos",
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        # Training loop
        best_state = None
        best_val_f1 = -1.0
        no_improve = 0
        history = {"train_loss": [], "val_f1": [], "val_acc": []}

        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, dl_train, optimizer, scheduler,
                                  criterion, device, args.clip_grad)
            y_val, p_val = evaluate(model, dl_val, device)
            val_f1_score = macro_f1(y_val, p_val, n_classes)
            val_acc = accuracy_score(y_val, p_val)
            dt = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["val_f1"].append(val_f1_score)
            history["val_acc"].append(val_acc)

            log.info(f"[{fold_name} | Ep {ep:03d}] loss={tr_loss:.4f}  "
                     f"F1={val_f1_score:.3f}  Acc={val_acc:.3f}  ({dt:.1f}s)")

            if val_f1_score > best_val_f1:
                best_val_f1 = val_f1_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    log.info(f"[{fold_name}] Early stop at epoch {ep} "
                             f"(best val F1={best_val_f1:.3f})")
                    break

        # Plot training curves for this fold
        plot_training_curves(history, out_dir / f"curves_{fold_name}.png")

        # Evaluate on test set with best weights
        model.load_state_dict(best_state)
        y_test, p_test = evaluate(model, dl_test, device)
        test_f1 = macro_f1(y_test, p_test, n_classes)
        test_acc = accuracy_score(y_test, p_test)

        log.info(f"[{fold_name} DONE] Test F1={test_f1:.3f}  Acc={test_acc:.3f}  "
                 f"(best val F1={best_val_f1:.3f})")

        fold_results.append(dict(
            fold=fold_name,
            val_f1=float(best_val_f1),
            test_f1=float(test_f1),
            test_acc=float(test_acc),
            n_train=len(train_items),
            n_val=len(val_items),
            n_test=len(test_items),
        ))

        # Accumulate test predictions
        all_y_test.append(y_test)
        all_p_test.append(p_test)

        # Track per-item predictions for per-participant report
        for idx, it in enumerate(test_std):
            all_test_items_with_preds.append(dict(
                pid=it["pid"], y=y_test[idx], p=p_test[idx],
            ))

        # Per-fold confusion matrix
        plot_confusion_matrix(y_test, p_test, n_classes,
                              f"Confusion: {fold_name}", out_dir / f"cm_{fold_name}.png")

        # Keep best model across folds (by test F1 — for checkpoint saving)
        if test_f1 > best_global_f1:
            best_global_f1 = test_f1
            best_global_state = best_state
            best_global_mu = mu
            best_global_sd = sd

    # ================================================================
    # Aggregate evaluation
    # ================================================================
    all_y = np.concatenate(all_y_test)
    all_p = np.concatenate(all_p_test)
    agg_f1 = macro_f1(all_y, all_p, n_classes)
    agg_acc = accuracy_score(all_y, all_p)

    log.info(f"\n{'='*60}")
    log.info(f"AGGREGATE: F1={agg_f1:.3f}  Acc={agg_acc:.3f}")
    log.info(f"{'='*60}")

    # Aggregate confusion matrix
    plot_confusion_matrix(all_y, all_p, n_classes,
                          f"Aggregate Confusion ({args.eval_mode})", out_dir / "cm_aggregate.png")

    # Per-participant report
    pid_report_lines = ["Participant  N_test  Accuracy  Macro-F1"]
    pid_report_lines.append("-" * 50)
    pids_in_test = sorted(set(d["pid"] for d in all_test_items_with_preds))
    for pid in pids_in_test:
        subset = [d for d in all_test_items_with_preds if d["pid"] == pid]
        ys = np.array([d["y"] for d in subset])
        ps = np.array([d["p"] for d in subset])
        acc = accuracy_score(ys, ps)
        f1 = macro_f1(ys, ps, n_classes)
        pid_report_lines.append(f"{pid:>11s}  {len(subset):>6d}  {acc:>8.3f}  {f1:>8.3f}")
    pid_report = "\n".join(pid_report_lines)
    log.info(f"\nPer-participant report:\n{pid_report}")
    with open(out_dir / "per_participant_report.txt", "w") as f:
        f.write(pid_report + "\n")

    # Per-condition report (only meaningful for 42-class)
    if n_classes == 42:
        cond_report_lines = ["Condition  N_test  Accuracy  Macro-F1"]
        cond_report_lines.append("-" * 50)
        for cond in range(6):
            mask = (all_y // 7) == cond
            if mask.sum() == 0:
                continue
            ys_c = all_y[mask]
            ps_c = all_p[mask]
            acc_c = accuracy_score(ys_c, ps_c)
            f1_c = f1_score(ys_c, ps_c, average="macro", labels=list(range(42)))
            cond_report_lines.append(f"{cond:>9d}  {int(mask.sum()):>6d}  {acc_c:>8.3f}  {f1_c:>8.3f}")
        cond_report = "\n".join(cond_report_lines)
        log.info(f"\nPer-condition report:\n{cond_report}")
        with open(out_dir / "per_condition_report.txt", "w") as f:
            f.write(cond_report + "\n")

    # ================================================================
    # Save results JSON
    # ================================================================
    results = dict(
        eval_mode=args.eval_mode,
        n_classes=n_classes,
        aggregate_f1=float(agg_f1),
        aggregate_acc=float(agg_acc),
        folds=fold_results,
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ================================================================
    # Save checkpoints
    # ================================================================
    if best_global_state is None:
        log.error("No model to save.")
        return

    model_key = f"model_{n_classes}.pt"
    artifact = {
        "type": "attention_bilstm",
        "in_dim": int(in_dim),
        "n_classes": n_classes,
        "hidden": args.hidden,
        "lstm_layers": args.lstm_layers,
        "attn_heads": args.attn_heads,
        "proj_dim": args.proj_dim,
        "dropout": args.dropout,
        "mu": best_global_mu.squeeze().astype(np.float32),
        "sd": best_global_sd.squeeze().astype(np.float32),
        "state_dict": best_global_state,
    }
    torch.save(artifact, out_dir / model_key)
    log.info(f"Saved {model_key} to {out_dir / model_key}")

    # If trained on 42 classes, also derive and save a 7-class model
    if n_classes == 42:
        head_w_name, head_b_name, W7, b7 = derive_head_7_from_42(best_global_state)
        model_7 = AttentionBiLSTM(
            in_dim, 7,
            hidden=args.hidden, lstm_layers=args.lstm_layers,
            attn_heads=args.attn_heads, proj_dim=args.proj_dim, dropout=args.dropout,
        )
        sd7 = model_7.state_dict()
        for k, v in sd7.items():
            if k.endswith("weight") and "head" in k:
                continue
            if k.endswith("bias") and "head" in k:
                continue
            if k in best_global_state and best_global_state[k].shape == v.shape:
                sd7[k] = best_global_state[k].clone()
        # Set 7-class head
        k_w7 = [k for k in sd7.keys() if "head" in k and k.endswith("weight")][0]
        k_b7 = [k for k in sd7.keys() if "head" in k and k.endswith("bias")][0]
        sd7[k_w7] = W7
        sd7[k_b7] = b7
        model_7.load_state_dict(sd7)
        art7 = {
            "type": "attention_bilstm",
            "in_dim": int(in_dim),
            "n_classes": 7,
            "hidden": args.hidden,
            "lstm_layers": args.lstm_layers,
            "attn_heads": args.attn_heads,
            "proj_dim": args.proj_dim,
            "dropout": args.dropout,
            "mu": best_global_mu.squeeze().astype(np.float32),
            "sd": best_global_sd.squeeze().astype(np.float32),
            "state_dict": model_7.state_dict(),
        }
        torch.save(art7, out_dir / "model_7.pt")
        log.info(f"Saved model_7.pt (derived from 42-class head)")

    log.info(f"\nAll outputs in: {out_dir}")
    log.info(f"Aggregate F1={agg_f1:.3f}  Acc={agg_acc:.3f}")


if __name__ == "__main__":
    main()
