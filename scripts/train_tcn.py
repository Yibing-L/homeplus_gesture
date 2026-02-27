#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_tcn.py — TCN training for metric XYZ gesture features.

Same data pipeline as train_xyz.py (175-dim XYZ format from landmark_with_xyz.py),
but uses a Temporal Convolutional Network instead of Attention-BiLSTM.

TCNs are often faster to train and better at capturing local temporal patterns,
which can outperform RNNs on smaller datasets.

Architecture: linear input projection → stack of residual dilated TCN blocks
              (dilation doubles each block) → masked mean pooling → classification head.

Feature layout (175 channels):
  [0:72]    pose        — hand (21j×3) + arm (3j×3)
  [72:144]  vel_pose    — frame-to-frame velocity
  [144:147] vel_wrist   — absolute wrist velocity (m/frame)
  [147:148] scale       — metric bone-length scale (m)
  [148:149] frame_valid — binary frame validity
  [149:170] joint_valid — per-joint depth validity (21 hand joints)
  [170:173] arm_valid   — per-joint depth validity (3 arm joints)
  [173:174] hand_cov    — mean hand joint validity
  [174:175] arm_cov     — mean arm joint validity
"""

import os, argparse, random, glob, json, logging, time, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Continuous channels — validity channels [N_CONTINUOUS:] must not be augmented.
# XYZ layout: pose(72) + vel_pose(72) + vel_wrist(3) + scale(1) = 148
N_CONTINUOUS = 148


# ================================================================
# CLI
# ================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="TCN trainer for XYZ features (LOSO / K-fold)")

    # Data
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Directories with processed *.npz clips from landmark_with_xyz.py")
    ap.add_argument("--save_path", type=str, default="runs/tcn_xyz",
                    help="Output folder for checkpoints, plots, reports")
    ap.add_argument("--n_classes", type=int, default=7, choices=[7, 42])

    # Evaluation
    ap.add_argument("--eval_mode", type=str, default="loso", choices=["loso", "kfold"])
    ap.add_argument("--k", type=int, default=5)

    # Model
    ap.add_argument("--width", type=int, default=128,
                    help="Number of channels in TCN blocks")
    ap.add_argument("--n_blocks", type=int, default=5,
                    help="Number of TCN residual blocks (dilation = 2^i for block i, "
                         "giving receptive field ~2*kernel*(2^n_blocks - 1))")
    ap.add_argument("--kernel_size", type=int, default=3,
                    help="Convolution kernel size in each TCN block")
    ap.add_argument("--proj_dim", type=int, default=128,
                    help="Input projection dimension (set to width for no bottleneck)")
    ap.add_argument("--dropout", type=float, default=0.3)

    # Training
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # Augmentation
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--aug_speed", action="store_true", default=True)
    ap.add_argument("--aug_warp", action="store_true", default=True)
    ap.add_argument("--aug_noise_std", type=float, default=0.02)
    ap.add_argument("--aug_frame_drop_prob", type=float, default=0.05)
    ap.add_argument("--aug_block_mask_prob", type=float, default=0.05)
    ap.add_argument("--aug_scale_jitter", action="store_true", default=True)
    ap.add_argument("--no_aug_scale_jitter", action="store_true")

    # Per-clip normalization
    ap.add_argument("--clip_norm", action="store_true", default=True)
    ap.add_argument("--no_clip_norm", action="store_true")

    # Misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()
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
    parent = os.path.basename(os.path.dirname(path))
    if parent.isdigit():
        return parent
    m = re.match(r"gesture_(\d+)_", os.path.basename(path))
    if m:
        return m.group(1)
    return parent


def _expand_roots(roots):
    """Expand glob patterns in roots — needed on Windows where the shell
    does not expand wildcards before passing args to Python."""
    expanded = []
    for r in roots:
        if any(c in r for c in ("*", "?", "[")):
            matches = sorted(glob.glob(r))
            expanded.extend(matches) if matches else expanded.append(r)
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
            valid_arr = z.get("valid_T", z.get("valid", None))
            if valid_arr is None:
                continue
            X = z["X"].astype(np.float32)
            if X.shape[1] != 175:
                print(f"[WARN] {path}: expected 175-dim X, got {X.shape[1]} — skipping")
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
            pid = str(int(z["subject_id"])) if "subject_id" in z else extract_participant_id(path)
            items.append(dict(path=path, X=X, valid=valid, y42=y42, pid=pid))
    print(f"Loaded {len(items)} clips from {len(set(it['pid'] for it in items))} participants.")
    return items


# ================================================================
# Augmentation
# ================================================================
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
    v_warp = (1 - w.squeeze()) * valid.astype(np.float32)[lo_idx] + w.squeeze() * valid.astype(np.float32)[hi_idx]
    return X_out, v_warp > 0.5


def gaussian_noise(X, std=0.02):
    noise = np.zeros_like(X)
    noise[:, :N_CONTINUOUS] = np.random.randn(X.shape[0], N_CONTINUOUS).astype(np.float32) * std
    return X + noise


def frame_dropout(X, valid, prob=0.05):
    drop = np.random.rand(X.shape[0]) < prob
    X_out = X.copy(); v_out = valid.copy()
    X_out[drop] = 0.0; v_out[drop] = False
    return X_out, v_out


def feature_block_mask(X, prob=0.05):
    if np.random.rand() > prob:
        return X
    X_out = X.copy()
    blocks = [(0, 72), (72, 144), (144, 147), (147, 148)]
    start, end = blocks[np.random.randint(len(blocks))]
    X_out[:, start:end] = 0.0
    return X_out


def scale_jitter(X, lo=0.8, hi=1.2):
    X_out = X.copy()
    X_out[:, :144] *= np.random.uniform(lo, hi)
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
            if self.cfg.get("noise_std", 0.02) > 0:
                X = gaussian_noise(X, self.cfg["noise_std"])
            if self.cfg.get("frame_drop_prob", 0.05) > 0:
                X, valid = frame_dropout(X, valid, self.cfg["frame_drop_prob"])
            if self.cfg.get("block_mask_prob", 0.05) > 0:
                X = feature_block_mask(X, self.cfg["block_mask_prob"])
            if self.cfg.get("scale_jitter", True):
                X = scale_jitter(X)

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
# Model: TCN
# ================================================================
class TCNBlock(nn.Module):
    """Residual block with two causal dilated convolutions."""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation   # causal: trim output to original length
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)   # instance norm (groups=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        self._pad  = pad

    def forward(self, x):
        T = x.shape[2]
        h = self.act(self.norm1(self.conv1(x)[..., :T]))
        h = self.drop(h)
        h = self.norm2(self.conv2(h)[..., :T])
        h = self.drop(h)
        return self.act(h + x)   # residual connection


class GestureTCN(nn.Module):
    def __init__(self, in_dim, n_classes, proj_dim=128, width=128,
                 kernel_size=3, n_blocks=5, dropout=0.3):
        super().__init__()
        # Input projection: per-timestep linear
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        # Channel adapter if proj_dim != width
        self.chan_adapt = nn.Conv1d(proj_dim, width, 1) if proj_dim != width else nn.Identity()

        # TCN stack — dilation doubles each block
        self.blocks = nn.ModuleList([
            TCNBlock(width, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x, mask=None):
        # x: (B, T, F), mask: (B, T) float 1=valid
        h = self.input_proj(x)        # (B, T, proj_dim)
        h = h.transpose(1, 2)         # (B, proj_dim, T)
        h = self.chan_adapt(h)         # (B, width, T)
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)         # (B, T, width)

        # Masked mean pooling — invalid frames don't contribute
        if mask is not None:
            m = mask[:, :h.size(1)].unsqueeze(-1)   # (B, T, 1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = h.mean(1)

        return self.head(pooled)


# ================================================================
# Standardization — continuous channels only
# ================================================================
def standardize_fit(train_items):
    Xtr = np.stack([it["X"] for it in train_items], 0)
    Xc = Xtr[:, :, :N_CONTINUOUS]
    mu = Xc.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    sd = Xc.std(axis=(0, 1), keepdims=True).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def standardize_apply(items, mu, sd):
    out = []
    for it in items:
        X = it["X"].copy()
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
        idx = torch.tensor([j, j + 7, j + 14, j + 21, j + 28, j + 35], dtype=torch.long)
        W7[j] = W42.index_select(0, idx).mean(dim=0)
        b7[j] = b42.index_select(0, idx).mean()
    return head_w_name, head_b_name, W7, b7


# ================================================================
# Training & Evaluation
# ================================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, device, clip_grad):
    model.train()
    total_loss = n_samples = 0
    for X, M, y in loader:
        X, M, y = X.to(device), M.to(device), y.to(device)
        loss = criterion(model(X, M), y)
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
    all_y, all_p = [], []
    for X, M, y in loader:
        all_p.append(model(X.to(device), M.to(device)).argmax(1).cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_p)


def macro_f1(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)))


# ================================================================
# Plotting
# ================================================================
def plot_confusion_matrix(y_true, y_pred, n_classes, title, save_path):
    if not HAS_MPL:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.3), max(8, n_classes * 0.3)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_training_curves(history, save_path):
    if not HAS_MPL or not history:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"]); axes[0].set_title("Loss")
    axes[1].plot(epochs, history["val_f1"]);    axes[1].set_title("Val Macro F1")
    axes[2].plot(epochs, history["val_acc"]);   axes[2].set_title("Val Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# ================================================================
# LOSO / K-fold splits
# ================================================================
def generate_loso_splits(items):
    pids = sorted(set(it["pid"] for it in items), key=lambda x: int(x))
    for i, test_pid in enumerate(pids):
        val_pid = pids[(i + 1) % len(pids)]
        yield ([it for it in items if it["pid"] != test_pid and it["pid"] != val_pid],
               [it for it in items if it["pid"] == val_pid],
               [it for it in items if it["pid"] == test_pid],
               f"LOSO-{test_pid}")


def generate_kfold_splits(items, k, seed):
    ys = np.array([it["y42"] for it in items])
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(np.zeros(len(ys)), ys)):
        tr_idx = list(tr_idx); np.random.shuffle(tr_idx)
        n_val = max(1, int(0.1 * len(tr_idx)))
        yield ([items[i] for i in tr_idx[n_val:]], [items[i] for i in tr_idx[:n_val]],
               [items[i] for i in te_idx], f"fold-{fold_idx + 1}")


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

    n_classes = args.n_classes
    if n_classes == 7:
        for it in items:
            it["y42"] = it["y42"] % 7
        log.info("Mapped labels to 7 command classes.")

    in_dim = items[0]["X"].shape[-1]
    log.info(f"N={len(items)}, in_dim={in_dim}, n_classes={n_classes}, N_CONTINUOUS={N_CONTINUOUS}")
    if in_dim != 175:
        log.warning(f"Expected 175-dim XYZ features, got {in_dim}.")

    aug_cfg = None if args.no_aug else dict(
        speed=args.aug_speed, warp=args.aug_warp,
        noise_std=args.aug_noise_std, frame_drop_prob=args.aug_frame_drop_prob,
        block_mask_prob=args.aug_block_mask_prob, scale_jitter=args.aug_scale_jitter,
    )

    splits = list(generate_loso_splits(items) if args.eval_mode == "loso"
                  else generate_kfold_splits(items, args.k, args.seed))
    log.info(f"Evaluation: {args.eval_mode} — {len(splits)} folds")

    all_y_test, all_p_test, all_preds_with_pid = [], [], []
    fold_results = []
    best_global_f1 = best_global_val_f1 = -1.0
    best_global_fold = None
    best_global_state = best_global_mu = best_global_sd = None

    for fold_i, (train_items, val_items, test_items, fold_name) in enumerate(splits):
        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold_i+1}/{len(splits)}: {fold_name}  "
                 f"(train={len(train_items)} val={len(val_items)} test={len(test_items)})")
        log.info(f"{'='*60}")

        mu, sd = standardize_fit(train_items)
        train_std = standardize_apply(train_items, mu, sd)
        val_std   = standardize_apply(val_items,   mu, sd)
        test_std  = standardize_apply(test_items,  mu, sd)

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

        model = GestureTCN(
            in_dim, n_classes,
            proj_dim=args.proj_dim, width=args.width,
            kernel_size=args.kernel_size, n_blocks=args.n_blocks,
            dropout=args.dropout,
        ).to(device)

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
                                  criterion, device, args.clip_grad)
            y_val, p_val = evaluate(model, dl_val, device)
            vf1  = macro_f1(y_val, p_val, n_classes)
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
                    log.info(f"[{fold_name}] Early stop ep {ep} (best val F1={best_val_f1:.3f})")
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

        if tf1 > best_global_f1:
            best_global_f1 = tf1
            best_global_val_f1 = float(best_val_f1)
            best_global_fold = fold_name
            best_global_state, best_global_mu, best_global_sd = best_state, mu, sd

    # ================================================================
    # Aggregate evaluation
    # ================================================================
    all_y = np.concatenate(all_y_test); all_p = np.concatenate(all_p_test)
    agg_f1  = macro_f1(all_y, all_p, n_classes)
    agg_acc = accuracy_score(all_y, all_p)
    log.info(f"\n{'='*60}\nAGGREGATE: F1={agg_f1:.3f}  Acc={agg_acc:.3f}\n{'='*60}")
    plot_confusion_matrix(all_y, all_p, n_classes,
                          f"Aggregate ({args.eval_mode})", out_dir / "cm_aggregate.png")

    lines = ["Participant  N_test  Accuracy  Macro-F1", "-" * 50]
    for pid in sorted(set(d["pid"] for d in all_preds_with_pid), key=lambda x: int(x)):
        sub = [d for d in all_preds_with_pid if d["pid"] == pid]
        ys = np.array([d["y"] for d in sub]); ps = np.array([d["p"] for d in sub])
        lines.append(f"{pid:>11s}  {len(sub):>6d}  {accuracy_score(ys,ps):>8.3f}  "
                     f"{macro_f1(ys,ps,n_classes):>8.3f}")
    report = "\n".join(lines)
    log.info(f"\nPer-participant report:\n{report}")
    (out_dir / "per_participant_report.txt").write_text(report + "\n")

    with open(out_dir / "results.json", "w") as f:
        json.dump(dict(eval_mode=args.eval_mode, n_classes=n_classes,
                       aggregate_f1=float(agg_f1), aggregate_acc=float(agg_acc),
                       folds=fold_results), f, indent=2)

    if best_global_state is None:
        log.error("No model to save."); return

    # ================================================================
    # Save checkpoint
    # ================================================================
    model_key = f"model_{n_classes}.pt"
    artifact = dict(
        type="tcn_xyz",
        in_dim=int(in_dim),
        n_continuous=N_CONTINUOUS,
        n_classes=n_classes,
        width=args.width,
        proj_dim=args.proj_dim,
        kernel_size=args.kernel_size,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        mu=best_global_mu.squeeze().astype(np.float32),
        sd=best_global_sd.squeeze().astype(np.float32),
        state_dict=best_global_state,
        # provenance / eval metadata
        roots=[str(r) for r in args.roots],
        eval_mode=args.eval_mode,
        best_fold=best_global_fold,
        best_val_f1=best_global_val_f1,
        best_test_f1=float(best_global_f1),
        aggregate_f1=float(agg_f1),
        aggregate_acc=float(agg_acc),
        model_path=str(out_dir / model_key),
    )
    torch.save(artifact, out_dir / model_key)
    log.info(f"Saved {model_key}")

    if n_classes == 42:
        _, _, W7, b7 = derive_head_7_from_42(best_global_state)
        m7 = GestureTCN(in_dim, 7, proj_dim=args.proj_dim, width=args.width,
                        kernel_size=args.kernel_size, n_blocks=args.n_blocks,
                        dropout=args.dropout)
        sd7 = m7.state_dict()
        for k, v in sd7.items():
            if "head" in k: continue
            if k in best_global_state and best_global_state[k].shape == v.shape:
                sd7[k] = best_global_state[k].clone()
        sd7[[k for k in sd7 if "head" in k and k.endswith("weight")][0]] = W7
        sd7[[k for k in sd7 if "head" in k and k.endswith("bias")][0]] = b7
        m7.load_state_dict(sd7)
        torch.save(dict(**{k: artifact[k] for k in artifact
                           if k not in ("n_classes", "state_dict", "model_path")},
                        n_classes=7, state_dict=m7.state_dict(),
                        model_path=str(out_dir / "model_7.pt"),
                        derived_from=str(out_dir / model_key)),
                   out_dir / "model_7.pt")
        log.info("Saved model_7.pt (derived from 42-class head)")

    log.info(f"\nAll outputs in: {out_dir}")
    log.info(f"Aggregate F1={agg_f1:.3f}  Acc={agg_acc:.3f}")


if __name__ == "__main__":
    main()
