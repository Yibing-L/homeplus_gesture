#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random, math, glob, json, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import re

# ----------------------------
# Args / Repro
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Directories with processed *.npz clips")
    ap.add_argument("--model", type=str, default="tcn", choices=["tcn", "gru", "lstm", "transformer"])
    ap.add_argument("--save_path", type=str, default="bestcheckpoint",
                    help="Will create a folder containing model_42.pt, model_7.pt and eval files")
    ap.add_argument("--eval_mode", type=str, default="loso", choices=["loso", "kfold"],
                    help="LOSO (leave-one-subject-out) or stratified K-fold")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=5, help="K-fold CV (ignored in LOSO)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--patience", type=int, default=8, help="Early stop after N epochs without val F1 improvement")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ----------------------------
# Data loading
# ----------------------------
def extract_participant_id(path):
    """Extract participant ID from path like dataset/5/gesture_5_12.npz."""
    parent = os.path.basename(os.path.dirname(path))
    if parent.isdigit():
        return parent
    m = re.match(r"gesture_(\d+)_", os.path.basename(path))
    if m:
        return m.group(1)
    return parent

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
            X = z["X"].astype(np.float32)     # (T,F)
            valid = np.asarray(valid_arr).squeeze()
            valid = valid.astype(bool) if valid.dtype == np.bool_ else (valid > 0.5)  # (T,)
            # label: 42-way fine-grained index (condition-major, command-minor)
            if "label" in z:
                y42 = int(z["label"])
            elif "gid" in z:
                y42 = int(z["gid"])
            else:
                continue

            vr = valid.mean() if valid.size else 1.0
            if vr < 0.25:
                print(f"[DROP] {os.path.basename(path)} valid_ratio={vr:.2f}")
                continue
            pid = extract_participant_id(path)
            items.append(dict(path=path, X=X, valid=valid, y42=y42, pid=pid))
    print(f"Loaded {len(items)} clips from {len(set(it['pid'] for it in items))} participants.")
    return items

class ClipSet(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return it["X"], it["valid"].astype(np.float32), it["y42"]

def collate(batch):
    Xs, Ms, Ys42 = zip(*batch)
    X = torch.from_numpy(np.stack(Xs, 0)).float()                 # [B,T,F]
    M = torch.from_numpy(np.stack(Ms, 0)).float()                 # [B,T]
    y42 = torch.tensor(Ys42, dtype=torch.long)                    # [B]
    y7 = (y42 % 7)                                                # collapse to commands
    return X, M, y42, y7

# ----------------------------
# Models (mirror online inference)
# ----------------------------
class TCN(nn.Module):
    def __init__(self, in_dim, n_classes, width=128, dropout=0.05, dilations=(1,2,4,8)):
        super().__init__()
        layers=[]; c_in=in_dim
        for d in dilations:
            layers += [
                nn.Conv1d(c_in, width, 3, padding=d, dilation=d),
                nn.GELU(),
                nn.Conv1d(width, width, 3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            c_in = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)
    def forward(self, x, mask=None):
        x = x.transpose(1,2)  # [B,F,T]
        h = self.net(x).mean(dim=2)
        return self.head(h)

class BiRNN(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=128, layers=2, dropout=0.1, cell="gru"):
        super().__init__()
        rnn_cls = nn.GRU if cell=="gru" else nn.LSTM
        self.rnn = rnn_cls(in_dim, hidden, layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Dropout(dropout), nn.Linear(hidden*2, n_classes))
    def forward(self, x, mask):
        lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out,_ = self.rnn(packed)
        out,_ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        m = mask[:, :out.size(1)].unsqueeze(-1)
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)

class PosEnc(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerEnc(nn.Module):
    def __init__(self, in_dim, n_classes, d_model=128, nhead=4, ff=256, layers=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos  = PosEnc(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls = nn.Parameter(torch.zeros(1,1,d_model)); nn.init.normal_(self.cls, std=0.02)
        self.head = nn.Linear(d_model, n_classes)
    def forward(self, x, mask):
        B,T,F = x.shape; h = self.pos(self.proj(x))
        cls = self.cls.expand(B,1,-1); h = torch.cat([cls, h], 1)
        kpm = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=h.device), (~mask.bool())],1)
        h = self.enc(h, src_key_padding_mask=kpm)
        return self.head(h[:,0,:])

# ----------------------------
# Training / Eval
# ----------------------------
def standardize_fit(train_X):
    mu = train_X.mean(axis=(0,1), keepdims=True)  # [1,1,F]
    sd = train_X.std(axis=(0,1), keepdims=True)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)

def standardize_apply(X, mu, sd):
    return (X - mu) / (sd + 1e-8)

def build_model(name, in_dim, n_classes):
    if name == "tcn":
        return TCN(in_dim, n_classes)
    elif name in ("gru","lstm"):
        return BiRNN(in_dim, n_classes, cell=name)
    else:
        return TransformerEnc(in_dim, n_classes)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys42, ps42 = [], []
    for X, M, y42, y7 in loader:
        X = X.to(device); M = M.to(device)
        logits = model(X, M) if model.__class__.__name__ != "TCN" else model(X)
        p42 = torch.argmax(logits, dim=1).cpu().numpy()
        ys42.append(y42.numpy()); ps42.append(p42)
    y42 = np.concatenate(ys42); p42 = np.concatenate(ps42)
    y7  = y42 % 7
    p7  = p42 % 7
    return (y42, p42, y7, p7)

def macro_f1(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)))

def train_epoch(model, loader, opt, ce, device):
    model.train()
    total = 0.0
    for X, M, y42, y7 in loader:
        X = X.to(device); M = M.to(device); y42 = y42.to(device)
        logits = model(X, M) if model.__class__.__name__ != "TCN" else model(X)
        loss = ce(logits, y42)
        opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.item()) * X.size(0)
    return total / max(1, len(loader.dataset))

# ----------------------------
# Utilities to derive a 7-class head from 42-class weights
# ----------------------------
def derive_head_7_from_42(model_42_state_dict):
    """
    Given a trained model state_dict with final head weight of shape [42, D],
    produce averaged weights/bias for 7 classes by averaging every 6 rows
    corresponding to the same command across conditions:
      indices for command j: [j, j+7, j+14, j+21, j+28, j+35]
    """
    sd = model_42_state_dict
    # find the classifier layer name
    head_w_name = None; head_b_name = None
    for k in sd.keys():
        if k.endswith("head.weight"):
            head_w_name = k
            head_b_name = k.replace("weight","bias")
            break
    if head_w_name is None:
        raise RuntimeError("Could not find classifier head in state_dict")
    W42 = sd[head_w_name].clone()   # [42, D]
    b42 = sd[head_b_name].clone()   # [42]
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

# ----------------------------
# LOSO / K-fold split generation
# ----------------------------
def generate_loso_splits(items):
    """Yield (train_items, val_items, test_items, fold_name) per participant.

    Participant-disjoint validation: for each test participant, the next
    participant (cyclic) is held out as validation.
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

# ----------------------------
# Main (LOSO/K-fold, best fold selection, saving both models & evals)
# ----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    items = load_items(args.roots)
    if len(items) == 0:
        print("No training clips found."); return

    # Stats
    print(f"Global: N={len(items)}, min_valid~{min(it['valid'].mean() for it in items):.2f}")

    # Generate splits
    if args.eval_mode == "loso":
        splits = list(generate_loso_splits(items))
    else:
        splits = list(generate_kfold_splits(items, args.k, args.seed))
    print(f"Evaluation: {args.eval_mode} — {len(splits)} folds")

    best_macro42 = -1.0
    best_artifact_42 = None
    best_reports = None

    for fold, (train_items, val_items, test_items, fold_name) in enumerate(splits, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{len(splits)}: {fold_name}  "
              f"(train={len(train_items)} val={len(val_items)} test={len(test_items)})")
        print(f"{'='*60}")

        # Fit standardization on train
        Xtr = np.stack([it["X"] for it in train_items], 0)  # [N,T,F]
        mu, sd = standardize_fit(Xtr)
        def apply_std(ds):
            out = []
            for it in ds:
                X = standardize_apply(it["X"], mu, sd)
                out.append(dict(X=X, valid=it["valid"], y42=it["y42"]))
            return out

        tr = ClipSet(apply_std(train_items))
        va = ClipSet(apply_std(val_items))
        te = ClipSet(apply_std(test_items))

        in_dim = tr[0][0].shape[-1]
        n_classes_42 = 42

        model = build_model(args.model, in_dim, n_classes_42).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        ce = nn.CrossEntropyLoss()

        dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate)
        dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        dl_te = DataLoader(te, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        best_state = None
        best_f1_va_42 = None
        noimp = 0

        for ep in range(1, args.epochs+1):
            tr_loss = train_epoch(model, dl_tr, opt, ce, device)
            y42_val, p42_val, y7_val, p7_val = evaluate(model, dl_va, device)
            f1_va_42 = macro_f1(y42_val, p42_val, 42)
            f1_va_7  = macro_f1(y7_val,  p7_val,  7)

            print(f"[{fold_name} | Epoch {ep:03d}] loss={tr_loss:.4f}  F1_42={f1_va_42:.3f}  F1_7={f1_va_7:.3f}")

            if (best_f1_va_42 is None) or (f1_va_42 > best_f1_va_42):
                best_f1_va_42 = f1_va_42
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1
                if noimp >= args.patience:
                    print(f"[{fold_name}] Early stop at epoch {ep} (best F1_42={best_f1_va_42:.3f})")
                    break

        # Test with best weights for this fold
        model.load_state_dict(best_state)
        y42_val, p42_val, y7_val, p7_val = evaluate(model, dl_va, device)
        y42_te,  p42_te,  y7_te,  p7_te  = evaluate(model, dl_te, device)

        f1_va_42 = macro_f1(y42_val, p42_val, 42)
        f1_va_7  = macro_f1(y7_val,  p7_val,  7)
        f1_te_42 = macro_f1(y42_te,  p42_te,  42)
        f1_te_7  = macro_f1(y7_te,   p7_te,   7)

        print(f"[{fold_name} DONE]  VAL: F1_42={f1_va_42:.3f} F1_7={f1_va_7:.3f}   "
              f"TEST: F1_42={f1_te_42:.3f} F1_7={f1_te_7:.3f}")

        # Keep best fold by 42-way macro-F1 (primary metric)
        if f1_va_42 > best_macro42:
            best_macro42 = float(f1_va_42)
            # build artifact for 42
            artifact_42 = {
                "type": args.model,
                "in_dim": int(in_dim),
                "n_classes": 42,
                "mu": mu.squeeze().tolist(),   # store as lists to be JSON-friendly if needed
                "sd": sd.squeeze().tolist(),
                "state_dict": best_state,
            }
            # also compute textual reports for later saving
            rep_val_42 = classification_report(y42_val, p42_val, labels=list(range(42)), zero_division=0, output_dict=True)
            rep_val_7  = classification_report(y7_val,  p7_val,  labels=list(range(7)),  zero_division=0, output_dict=True)
            rep_te_42  = classification_report(y42_te,  p42_te,  labels=list(range(42)), zero_division=0, output_dict=True)
            rep_te_7   = classification_report(y7_te,   p7_te,   labels=list(range(7)),  zero_division=0, output_dict=True)
            best_artifact_42 = artifact_42
            best_reports = dict(
                val=dict(F1_42=f1_va_42, F1_7=f1_va_7, report_42=rep_val_42, report_7=rep_val_7),
                test=dict(F1_42=f1_te_42, F1_7=f1_te_7, report_42=rep_te_42, report_7=rep_te_7),
                fold=fold_name
            )

    # -------------- Save outputs --------------
    if best_artifact_42 is None:
        print("No best artifact to save."); return

    # Folder path from save_path (strip extension)
    out_dir = Path(args.save_path)
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model_42.pt (full artifact)
    art42 = best_artifact_42.copy()
    # convert mu/sd back to numpy for saving; keep state_dict tensors
    art42["mu"] = np.array(art42["mu"], dtype=np.float32)
    art42["sd"] = np.array(art42["sd"], dtype=np.float32)
    torch.save(art42, out_dir / "model_42.pt")

    # Create a 7-class model by averaging 42-class head weights (no retraining)
    model_name = art42["type"]
    in_dim = int(art42["in_dim"])
    # Build 42-class model just to inspect shapes if needed (not strictly necessary)
    # Derive W7/b7
    head_w_name, head_b_name, W7, B7 = derive_head_7_from_42(art42["state_dict"])
    # Build a fresh 7-class model with the same backbone hyperparams
    # We assume default hyperparams used above; if you added custom widths etc, put them into the artifact and read here.
    if model_name == "tcn":
        model7 = TCN(in_dim, 7)
    elif model_name in ("gru","lstm"):
        model7 = BiRNN(in_dim, 7, cell=model_name)
    else:
        model7 = TransformerEnc(in_dim, 7)
    # transfer matching weights except head.*
    sd7 = model7.state_dict()
    for k, v in sd7.items():
        if k.endswith("head.weight") or k.endswith("head.bias"):
            continue
        # copy from 42-model if exists with same shape
        if k in art42["state_dict"] and art42["state_dict"][k].shape == v.shape:
            sd7[k] = art42["state_dict"][k].clone()
    # set head weights
    # find sd7 head names matching head_w_name pattern
    k_w7 = [k for k in sd7.keys() if k.endswith("head.weight")][0]
    k_b7 = [k for k in sd7.keys() if k.endswith("head.bias")][0]
    sd7[k_w7] = W7
    sd7[k_b7] = B7
    model7.load_state_dict(sd7)
    # Build artifact for 7
    art7 = {
        "type": model_name,
        "in_dim": in_dim,
        "n_classes": 7,
        "mu": art42["mu"],  # same standardization
        "sd": art42["sd"],
        "state_dict": model7.state_dict()
    }
    torch.save(art7, out_dir / "model_7.pt")

    # Save evaluations
    with open(out_dir / "eval_val.json", "w", encoding="utf-8") as f:
        json.dump(best_reports["val"], f, indent=2)
    with open(out_dir / "eval_test.json", "w", encoding="utf-8") as f:
        json.dump(best_reports["test"], f, indent=2)

    # Save a small readme
    with open(out_dir / "readme.txt", "w", encoding="utf-8") as f:
        f.write(
            "This folder was produced from --save_path on the command line.\n"
            "We trained a 42-class model and derived a 7-class model by averaging\n"
            "the 42-class classifier head across the 6 conditions per command.\n\n"
            "Files:\n"
            " - model_42.pt : PyTorch artifact (type, in_dim, n_classes=42, mu, sd, state_dict)\n"
            " - model_7.pt  : PyTorch artifact (n_classes=7) derived from model_42 without retraining\n"
            " - eval_val.json, eval_test.json : metrics for both 42 and 7\n"
        )

    print(f"[BEST] Saved best fold (by VAL macro-F1 42) to folder: {out_dir}")
    print("  - model_42.pt")
    print("  - model_7.pt")
    print("  - eval_val.json")
    print("  - eval_test.json")
    print(f"Best fold: {best_reports['fold']}  VAL F1_42={best_reports['val']['F1_42']:.3f}  F1_7={best_reports['val']['F1_7']:.3f}")

if __name__ == "__main__":
    main()
