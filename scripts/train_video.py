#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_video.py — Video-frame gesture classifier for Home+.

Trains a pretrained CNN backbone + temporal aggregator on raw RGB clips.
Loads raw .npz files from data_collection.py (keys: rgb, depth, label),
extracts per-frame visual features, and classifies gestures.

Pipeline:
  1. Load raw .npz clips containing RGB frames (640x480 BGR)
  2. Sample T frames uniformly per clip
  3. Extract per-frame features via pretrained CNN (e.g. ResNet-18)
  4. Aggregate temporally via BiLSTM + attention (or mean pooling)
  5. Classify gesture from aggregated representation

Raw .npz format (from data_collection.py):
  rgb:   (N, 480, 640, 3) uint8 BGR
  depth: (N, 480, 640)    uint16 (mm)
  label: gesture label (int)

Supports LOSO and K-fold evaluation (same as train_with_angles.py).

Usage:
  python scripts/train_video.py --roots /path/to/raw/*
  python scripts/train_video.py --roots /path/to/raw/* --backbone resnet34 --temporal mean
  python scripts/train_video.py --roots /path/to/raw/* --no_freeze_backbone --backbone_lr_scale 0.01

Requires: pip install torchvision  (in addition to requirements.txt)
"""

import os, argparse, random, glob, json, logging, time, re
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torchvision.transforms as T
import torchvision.models as models

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
    ap = argparse.ArgumentParser(
        description="Video-frame gesture classifier (CNN backbone + temporal aggregator)"
    )
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Directories with raw .npz clips (rgb + depth + label)")
    ap.add_argument("--save_path", type=str, default="runs/video")
    ap.add_argument("--n_classes", type=int, default=7, choices=[7, 42])

    ap.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50",
                             "efficientnet_b0", "mobilenet_v3_small"],
                    help="CNN backbone for per-frame feature extraction")
    ap.add_argument("--freeze_backbone", action="store_true", default=True,
                    help="Freeze backbone weights (train only temporal head)")
    ap.add_argument("--no_freeze_backbone", action="store_true",
                    help="Fine-tune backbone end-to-end (slower, needs GPU)")
    ap.add_argument("--temporal", type=str, default="bilstm",
                    choices=["bilstm", "mean"],
                    help="Temporal aggregation: bilstm (BiLSTM+attention) or mean pooling")

    ap.add_argument("--t_target", type=int, default=16,
                    help="Number of frames to sample per clip (default 16)")
    ap.add_argument("--frame_size", type=int, default=224,
                    help="Input frame size for backbone (default 224)")

    ap.add_argument("--eval_mode", type=str, default="loso", choices=["loso", "kfold"])
    ap.add_argument("--k", type=int, default=5)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.4)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    ap.add_argument("--backbone_lr_scale", type=float, default=0.01,
                    help="LR multiplier for backbone when fine-tuning (default 0.01)")
    ap.add_argument("--backbone_chunk_size", type=int, default=64,
                    help="Max frames through backbone at once (controls GPU memory)")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=2)

    args = ap.parse_args()
    if args.no_freeze_backbone:
        args.freeze_backbone = False
    return args


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ================================================================
# Data loading
# ================================================================
def _expand_roots(roots):
    """Expand any glob patterns (needed on Windows where the shell doesn't expand)."""
    expanded = []
    for r in roots:
        if any(c in r for c in ("*", "?", "[")):
            matches = sorted(glob.glob(r))
            expanded.extend(matches if matches else [r])
        else:
            expanded.append(r)
    return expanded


def extract_participant_id(path):
    """Extract participant ID from parent directory name (e.g. '0', '2_done')."""
    parent = os.path.basename(os.path.dirname(path))
    m = re.match(r"(\d+)", parent)
    return m.group(1) if m else parent


def load_items(roots):
    """Scan directories for raw .npz files containing RGB frames.

    Only reads metadata (shape, label) — full frames loaded lazily in Dataset.
    """
    roots = _expand_roots(roots)
    items = []
    for r in roots:
        print(f"[LOAD] {r}")
        for path in sorted(glob.glob(os.path.join(r, "*.npz"))):
            try:
                z = np.load(path, allow_pickle=True)
                if "rgb" not in z or "label" not in z:
                    continue
                n_frames = z["rgb"].shape[0]
                if n_frames < 4:
                    continue
                y42 = int(z["label"])
                pid = extract_participant_id(path)
                items.append(dict(path=path, n_frames=n_frames, y42=y42, pid=pid))
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")
    pids = set(it["pid"] for it in items)
    print(f"Loaded {len(items)} clips from {len(pids)} participants.")
    return items


# ================================================================
# Transforms
# ================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transforms(frame_size, train=True):
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return T.Compose([
            T.Resize(frame_size + 32),
            T.RandomCrop(frame_size),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomRotation(10),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([
        T.Resize(frame_size + 32),
        T.CenterCrop(frame_size),
        T.ToTensor(),
        normalize,
    ])


# ================================================================
# Dataset
# ================================================================
class VideoClipDataset(Dataset):
    """Lazily loads raw .npz clips and returns uniformly sampled RGB frames."""

    def __init__(self, items, transform, t_target=16):
        self.items = items
        self.transform = transform
        self.t_target = t_target

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        z = np.load(item["path"])
        rgb = z["rgb"]  # (N, H, W, 3) BGR uint8
        N = rgb.shape[0]

        # Uniform temporal sampling
        indices = np.linspace(0, N - 1, self.t_target, dtype=int)

        frames = []
        for i in indices:
            img = Image.fromarray(cv2.cvtColor(rgb[i], cv2.COLOR_BGR2RGB))
            frames.append(self.transform(img))

        frames_t = torch.stack(frames)  # (T, 3, H, W)
        mask = torch.ones(self.t_target, dtype=torch.float32)
        return frames_t, mask, item["y42"]


def collate(batch):
    frames, masks, labels = zip(*batch)
    return (torch.stack(frames), torch.stack(masks),
            torch.tensor(labels, dtype=torch.long))


# ================================================================
# Model
# ================================================================
class TemporalAttentionPool(nn.Module):
    """Learnable-query multi-head attention pooling over time."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


def _make_backbone(name):
    """Create pretrained backbone, return (features_module, feature_dim)."""
    if name == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = base.fc.in_features
        features = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    elif name == "resnet34":
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        feat_dim = base.fc.in_features
        features = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    elif name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = base.fc.in_features
        features = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    elif name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = base.classifier[1].in_features
        base.classifier = nn.Identity()
        features = base
    elif name == "mobilenet_v3_small":
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feat_dim = base.classifier[0].in_features
        base.classifier = nn.Identity()
        features = base
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return features, feat_dim


class VideoModel(nn.Module):
    """CNN backbone (per-frame) + temporal aggregator + classification head."""

    def __init__(self, backbone_name, n_classes, freeze_backbone=True,
                 temporal="bilstm", hidden=128, lstm_layers=2,
                 attn_heads=4, dropout=0.4, chunk_size=64):
        super().__init__()
        self.backbone, self.feat_dim = _make_backbone(backbone_name)
        self._freeze = freeze_backbone
        self.chunk_size = chunk_size
        self.temporal_mode = temporal

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        if temporal == "bilstm":
            self.lstm = nn.LSTM(
                self.feat_dim, hidden, lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0.0,
                batch_first=True, bidirectional=True,
            )
            pool_dim = hidden * 2
            self.attn_pool = TemporalAttentionPool(pool_dim, num_heads=attn_heads)
        else:  # mean
            pool_dim = self.feat_dim

        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim), nn.Dropout(dropout),
            nn.Linear(pool_dim, n_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        if self._freeze:
            self.backbone.eval()
        return self

    def _extract_features(self, flat_frames):
        """Extract backbone features in chunks to control GPU memory."""
        chunks = []
        for i in range(0, flat_frames.size(0), self.chunk_size):
            chunks.append(self.backbone(flat_frames[i:i + self.chunk_size]))
        return torch.cat(chunks, dim=0)

    def forward(self, frames, mask):
        # frames: (B, T, C, H, W), mask: (B, T)
        B, T_len = frames.shape[:2]
        flat = frames.reshape(B * T_len, *frames.shape[2:])

        if self._freeze:
            with torch.no_grad():
                feats = self._extract_features(flat).detach()
        else:
            feats = self._extract_features(flat)

        feats = feats.reshape(B, T_len, -1)  # (B, T, feat_dim)

        if self.temporal_mode == "bilstm":
            lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64)
            packed = nn.utils.rnn.pack_padded_sequence(
                feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True)
            if lstm_out.size(1) < T_len:
                pad = torch.zeros(B, T_len - lstm_out.size(1), lstm_out.size(2),
                                  device=lstm_out.device, dtype=lstm_out.dtype)
                lstm_out = torch.cat([lstm_out, pad], dim=1)
            pooled = self.attn_pool(
                lstm_out, key_padding_mask=~mask[:, :lstm_out.size(1)].bool())
        else:
            mask_f = mask.unsqueeze(-1)
            pooled = (feats * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return self.head(pooled)


# ================================================================
# Training & Evaluation
# ================================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, device, clip_grad):
    model.train()
    total_loss, n_samples = 0.0, 0
    for frames, mask, y in loader:
        frames, mask, y = frames.to(device), mask.to(device), y.to(device)
        loss = criterion(model(frames, mask), y)
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            scheduler.step()
            continue
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0:
            clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * frames.size(0)
        n_samples += frames.size(0)
    return total_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for frames, mask, y in loader:
        frames, mask = frames.to(device), mask.to(device)
        all_p.append(model(frames, mask).argmax(dim=1).cpu().numpy())
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
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.3),) * 2)
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
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
    axes[0].legend()
    axes[1].plot(epochs, history["val_f1"], label="val F1")
    axes[1].set_title("Macro F1")
    axes[1].legend()
    axes[2].plot(epochs, history["val_acc"], label="val Acc")
    axes[2].set_title("Accuracy")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ================================================================
# LOSO / K-fold splits
# ================================================================
def generate_loso_splits(items):
    pids = sorted(set(it["pid"] for it in items), key=lambda x: int(x))
    for i, test_pid in enumerate(pids):
        val_pid = pids[(i + 1) % len(pids)]
        yield (
            [it for it in items if it["pid"] not in (test_pid, val_pid)],
            [it for it in items if it["pid"] == val_pid],
            [it for it in items if it["pid"] == test_pid],
            f"LOSO-{test_pid}",
        )


def generate_kfold_splits(items, k, seed):
    ys = np.array([it["y42"] for it in items])
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(np.zeros(len(ys)), ys)):
        tr_idx = list(tr_idx)
        np.random.shuffle(tr_idx)
        n_val = max(1, int(0.1 * len(tr_idx)))
        yield (
            [items[i] for i in tr_idx[n_val:]],
            [items[i] for i in tr_idx[:n_val]],
            [items[i] for i in te_idx],
            f"fold-{fold_idx + 1}",
        )


# ================================================================
# Main
# ================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = "cuda" in args.device

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
        log.error("No clips found.")
        return

    n_classes = args.n_classes
    if n_classes == 7:
        for it in items:
            it["y42"] = it["y42"] % 7

    log.info(f"N={len(items)}, n_classes={n_classes}, backbone={args.backbone}, "
             f"temporal={args.temporal}, t_target={args.t_target}, "
             f"freeze={args.freeze_backbone}")

    train_tf = make_transforms(args.frame_size, train=True)
    eval_tf = make_transforms(args.frame_size, train=False)

    splits = list(generate_loso_splits(items) if args.eval_mode == "loso"
                  else generate_kfold_splits(items, args.k, args.seed))
    log.info(f"Evaluation: {args.eval_mode} — {len(splits)} folds")

    all_y_test, all_p_test = [], []
    fold_results = []
    best_global_f1, best_global_state = -1.0, None
    best_global_val_f1, best_global_fold = -1.0, None

    for fold_i, (train_items, val_items, test_items, fold_name) in enumerate(splits):
        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold_i+1}/{len(splits)}: {fold_name}  "
                 f"(train={len(train_items)} val={len(val_items)} test={len(test_items)})")
        log.info(f"{'='*60}")

        ds_train = VideoClipDataset(train_items, train_tf, args.t_target)
        ds_val = VideoClipDataset(val_items, eval_tf, args.t_target)
        ds_test = VideoClipDataset(test_items, eval_tf, args.t_target)

        dl_train = DataLoader(ds_train, args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=use_cuda)
        dl_val = DataLoader(ds_val, args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=use_cuda)
        dl_test = DataLoader(ds_test, args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=args.num_workers,
                             pin_memory=use_cuda)

        model = VideoModel(
            args.backbone, n_classes, freeze_backbone=args.freeze_backbone,
            temporal=args.temporal, hidden=args.hidden,
            lstm_layers=args.lstm_layers, attn_heads=args.attn_heads,
            dropout=args.dropout, chunk_size=args.backbone_chunk_size,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable params: {n_params:,} "
                 f"(backbone {'frozen' if args.freeze_backbone else 'fine-tuned'})")

        # Separate LR for backbone when fine-tuning
        if not args.freeze_backbone:
            backbone_params = list(model.backbone.parameters())
            other_params = [p for n, p in model.named_parameters()
                           if not n.startswith("backbone") and p.requires_grad]
            param_groups = [
                {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
                {"params": other_params},
            ]
            max_lrs = [args.lr * args.backbone_lr_scale, args.lr]
        else:
            param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
            max_lrs = args.lr

        optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs,
            total_steps=args.epochs * len(dl_train),
            pct_start=0.1, anneal_strategy="cos")
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best_state, best_val_f1, no_improve = None, -1.0, 0
        history = {"train_loss": [], "val_f1": [], "val_acc": []}

        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, dl_train, optimizer, scheduler,
                                  criterion, device, args.clip_grad)
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
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    log.info(f"[{fold_name}] Early stop at epoch {ep} "
                             f"(best F1={best_val_f1:.3f})")
                    break

        plot_training_curves(history, out_dir / f"curves_{fold_name}.png")

        model.load_state_dict(best_state)
        y_test, p_test = evaluate(model, dl_test, device)
        tf1 = macro_f1(y_test, p_test, n_classes)
        tacc = accuracy_score(y_test, p_test)
        log.info(f"[{fold_name} DONE] Test F1={tf1:.3f}  Acc={tacc:.3f}  "
                 f"(best val F1={best_val_f1:.3f})")

        fold_results.append(dict(fold=fold_name, val_f1=float(best_val_f1),
                                 test_f1=float(tf1), test_acc=float(tacc),
                                 n_train=len(train_items), n_val=len(val_items),
                                 n_test=len(test_items)))
        all_y_test.append(y_test)
        all_p_test.append(p_test)
        plot_confusion_matrix(y_test, p_test, n_classes,
                              f"Confusion: {fold_name}", out_dir / f"cm_{fold_name}.png")

        # Save per-fold checkpoint
        torch.save(dict(
            backbone=args.backbone, temporal=args.temporal,
            n_classes=n_classes, hidden=args.hidden,
            lstm_layers=args.lstm_layers, attn_heads=args.attn_heads,
            dropout=args.dropout, freeze_backbone=args.freeze_backbone,
            t_target=args.t_target, frame_size=args.frame_size,
            state_dict=best_state, fold=fold_name,
            val_f1=float(best_val_f1), test_f1=float(tf1), test_acc=float(tacc),
        ), out_dir / f"model_{n_classes}_{fold_name}.pt")
        log.info(f"Saved model_{n_classes}_{fold_name}.pt")

        if tf1 > best_global_f1:
            best_global_f1 = tf1
            best_global_val_f1 = float(best_val_f1)
            best_global_fold = fold_name
            best_global_state = best_state

    # ── Aggregate ──
    all_y = np.concatenate(all_y_test)
    all_p = np.concatenate(all_p_test)
    agg_f1 = macro_f1(all_y, all_p, n_classes)
    agg_acc = accuracy_score(all_y, all_p)
    log.info(f"\n{'='*60}\nAGGREGATE: F1={agg_f1:.3f}  Acc={agg_acc:.3f}\n{'='*60}")
    plot_confusion_matrix(all_y, all_p, n_classes,
                          f"Aggregate ({args.eval_mode})", out_dir / "cm_aggregate.png")

    with open(out_dir / "results.json", "w") as f:
        json.dump(dict(eval_mode=args.eval_mode, n_classes=n_classes,
                       aggregate_f1=float(agg_f1), aggregate_acc=float(agg_acc),
                       folds=fold_results), f, indent=2)

    if best_global_state is not None:
        torch.save(dict(
            backbone=args.backbone, temporal=args.temporal,
            n_classes=n_classes, hidden=args.hidden,
            lstm_layers=args.lstm_layers, attn_heads=args.attn_heads,
            dropout=args.dropout, freeze_backbone=args.freeze_backbone,
            t_target=args.t_target, frame_size=args.frame_size,
            state_dict=best_global_state,
            best_fold=best_global_fold,
            best_val_f1=best_global_val_f1,
            best_test_f1=float(best_global_f1),
            aggregate_f1=float(agg_f1), aggregate_acc=float(agg_acc),
        ), out_dir / f"model_{n_classes}.pt")
        log.info(f"Saved model_{n_classes}.pt")

    log.info(f"\nAll outputs in: {out_dir}")
    log.info(f"Aggregate F1={agg_f1:.3f}  Acc={agg_acc:.3f}")


if __name__ == "__main__":
    main()
