#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
online_recognizer_v1.py — Live RGB–D gesture recognition for Home+.

This is an updated version of online_recognizer.py that adds support for the
AttentionBiLSTM model architecture from train_v1.py, while retaining full
backward compatibility with TCN, BiRNN, and Transformer checkpoints.

Usage
------
python online_recognizer_v1.py \
  --checkpoint runs/v1_loso/model_42.pt \
  --desired_hand Auto \
  --min_valid_ratio 0.25 \
  --label_map labels.json

Press 'q' to quit.
"""

# --- Force MediaPipe to run without TensorFlow (prevents protobuf/TensorFlow import errors) ---
import os
os.environ["MEDIAPIPE_DISABLE_TF"] = "1"

import sys, json, argparse, math, collections
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import importlib

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to neural checkpoint saved by train.py or train_v1.py (PyTorch .pt only)")
    ap.add_argument("--desired_hand", type=str, default="Auto", choices=["Left","Right","Auto"],
                    help="Which hand to prioritize; 'Auto' picks nearest to right wrist")
    ap.add_argument("--min_valid_ratio", type=float, default=0.25,
                    help="Require at least this fraction of valid frames in the 64-frame window")
    ap.add_argument("--t_target", type=int, default=64, help="Sequence length the model expects")
    ap.add_argument("--stride_frames", type=int, default=2,
                    help="Process every Nth captured frame to reduce CPU load (>=1)")
    ap.add_argument("--smooth_k", type=int, default=3,
                    help="Require K consecutive identical predictions to change public label")
    ap.add_argument("--prob_tau", type=float, default=0.60,
                    help="Min probability to accept immediate label change")
    ap.add_argument("--label_map", type=str, default=None,
                    help="Optional JSON mapping from class index (str/int) to name")
    ap.add_argument("--show_landmarks", action="store_true", help="Draw pose/hand landmarks")
    return ap.parse_args()

# ---------------- Utils (mirrors landmark.py) ----------------
EPS = 1e-6

def resample_2d(arr, T):
    """Linear resample 2D (t, d) → (T, d)."""
    t = arr.shape[0]
    arr = arr.astype(np.float32)
    if t == T:
        return arr
    if t < 2:
        return np.repeat(arr, T, axis=0)
    idx = np.linspace(0, t - 1, T)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, t - 1)
    w = (idx - lo)[:, None]
    return ((1 - w) * arr[lo] + w * arr[hi]).astype(np.float32)

def resample_mask(mask_bool, T):
    m = mask_bool.astype(np.float32)[:, None]
    return (resample_2d(m, T).squeeze() > 0.5)

def make_velocity(x):
    return np.diff(x, axis=0, prepend=x[0:1])

def normalize_velocity_per_clip(dx):
    mag = np.sqrt((dx * dx).sum(axis=1) + EPS)
    m = float(mag.mean())
    return dx if m < EPS else (dx / m)

def get_depth_m(depth_img, x, y):
    """Depth in meters with small neighborhood median fallback."""
    h, w = depth_img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    d = depth_img[y, x]
    if d == 0:
        y0, y1 = max(0, y - 2), min(h, y + 3)
        x0, x1 = max(0, x - 2), min(w, x + 3)
        neigh = depth_img[y0:y1, x0:x1]
        nz = neigh[neigh > 0]
        if nz.size == 0:
            return None
        d = int(np.median(nz))
    return d * 0.001

def hand_scale_from_landmarks(hand21_px):
    wrist = hand21_px[0, :2]
    idxM = hand21_px[5, :2]
    d = np.linalg.norm(idxM - wrist)
    xmin, ymin = hand21_px[:, 0].min(), hand21_px[:, 1].min()
    xmax, ymax = hand21_px[:, 0].max(), hand21_px[:, 1].max()
    box = max(xmax - xmin, ymax - ymin)
    return float(max(d, box, EPS))

# ---------------- Models (mirror train.py + train_v1.py) ----------------
def build_model_from_artifact(artifact):
    mtype = artifact.get('type', None)
    if mtype not in ('tcn', 'gru', 'lstm', 'transformer', 'attention_bilstm'):
        raise ValueError(f"Unsupported model type: {mtype}")

    torch = importlib.import_module('torch')
    nn = torch.nn

    # --- Original model classes (from train.py) ---

    class TCN(nn.Module):
        def __init__(self, in_dim, n_classes, width=128, dropout=0.05, dilations=(1,2,4,8)):
            super().__init__()
            layers = []; c_in = in_dim
            for d in dilations:
                layers += [nn.Conv1d(c_in, width, 3, padding=d, dilation=d),
                           nn.GELU(),
                           nn.Conv1d(width, width, 3, padding=1),
                           nn.GELU(),
                           nn.Dropout(dropout)]
                c_in = width
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(width, n_classes)
        def forward(self, x, mask=None):
            x = x.transpose(1,2)  # [B,F,T]
            h = self.net(x).mean(dim=2)  # global average pool
            return self.head(h)

    class BiRNN(nn.Module):
        def __init__(self, in_dim, n_classes, hidden=128, layers=2, dropout=0.1, cell="gru"):
            super().__init__()
            rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
            self.rnn = rnn_cls(in_dim, hidden, layers, dropout=dropout, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Dropout(dropout), nn.Linear(hidden*2, n_classes))
        def forward(self, x, mask):
            lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            m = mask[:, :out.size(1)].unsqueeze(-1)
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
            return self.head(pooled)

    class PosEnc(nn.Module):
        def __init__(self, d, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0)/d))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))
        def forward(self, x): return x + self.pe[:, :x.size(1)]

    class TransformerEnc(nn.Module):
        def __init__(self, in_dim, n_classes, d_model=128, nhead=4, ff=256, layers=4, dropout=0.1):
            super().__init__()
            self.proj = nn.Linear(in_dim, d_model); self.pos = PosEnc(d_model)
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

    # --- AttentionBiLSTM (from train_v1.py) ---

    class TemporalAttentionPool(nn.Module):
        """Learned-query multi-head attention pooling over time steps."""
        def __init__(self, embed_dim, num_heads=4):
            super().__init__()
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        def forward(self, x, key_padding_mask=None):
            B = x.size(0)
            q = self.query.expand(B, -1, -1)
            out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
            return out.squeeze(1)

    class AttentionBiLSTM(nn.Module):
        def __init__(self, in_dim, n_classes, hidden=192, lstm_layers=3,
                     attn_heads=4, proj_dim=128, dropout=0.3):
            super().__init__()
            self.proj_dim = proj_dim
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
            )
            self.lstm = nn.LSTM(
                proj_dim, hidden, lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0.0,
                batch_first=True, bidirectional=True,
            )
            lstm_out_dim = hidden * 2
            self.res_proj = nn.Linear(lstm_out_dim, proj_dim) if lstm_out_dim != proj_dim else nn.Identity()
            self.res_norm = nn.LayerNorm(proj_dim)
            self.attn_pool = TemporalAttentionPool(proj_dim, num_heads=attn_heads)
            self.head = nn.Sequential(
                nn.LayerNorm(proj_dim),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, n_classes),
            )
        def forward(self, x, mask):
            h_proj = self.input_proj(x)
            lengths = mask.sum(dim=1).clamp(min=1).to(torch.int64)
            packed = nn.utils.rnn.pack_padded_sequence(
                h_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            T_proj = h_proj.size(1)
            T_lstm = lstm_out.size(1)
            if T_lstm < T_proj:
                pad = torch.zeros(lstm_out.size(0), T_proj - T_lstm, lstm_out.size(2),
                                  device=lstm_out.device, dtype=lstm_out.dtype)
                lstm_out = torch.cat([lstm_out, pad], dim=1)
            h = self.res_norm(self.res_proj(lstm_out) + h_proj)
            key_padding_mask = ~(mask[:, :T_proj].bool())
            pooled = self.attn_pool(h, key_padding_mask=key_padding_mask)
            return self.head(pooled)

    # --- Model instantiation ---

    in_dim = int(artifact['in_dim']); n_classes = int(artifact['n_classes'])
    if mtype == 'tcn':
        width = int(artifact.get('width', 128))
        dropout = float(artifact.get('dropout', 0.05))
        dilations = tuple(artifact.get('dilations', (1,2,4,8)))
        model = TCN(in_dim, n_classes, width=width, dropout=dropout, dilations=dilations)
    elif mtype in ('gru','lstm'):
        hidden = int(artifact.get('hidden', 128))
        layers = int(artifact.get('layers', 2))
        dropout = float(artifact.get('dropout', 0.1))
        model = BiRNN(in_dim, n_classes, hidden=hidden, layers=layers, dropout=dropout, cell=mtype)
    elif mtype == 'transformer':
        d_model = int(artifact.get('d_model', 128))
        nhead = int(artifact.get('nhead', 4))
        ff = int(artifact.get('ff', 256))
        layers = int(artifact.get('layers', 4))
        dropout = float(artifact.get('dropout', 0.1))
        model = TransformerEnc(in_dim, n_classes, d_model=d_model, nhead=nhead, ff=ff, layers=layers, dropout=dropout)
    elif mtype == 'attention_bilstm':
        hidden = int(artifact.get('hidden', 192))
        lstm_layers = int(artifact.get('lstm_layers', 3))
        attn_heads = int(artifact.get('attn_heads', 4))
        proj_dim = int(artifact.get('proj_dim', 128))
        dropout = float(artifact.get('dropout', 0.3))
        model = AttentionBiLSTM(in_dim, n_classes, hidden=hidden, lstm_layers=lstm_layers,
                                attn_heads=attn_heads, proj_dim=proj_dim, dropout=dropout)

    # load weights
    state = artifact['state_dict']
    torch = importlib.import_module('torch')
    model.load_state_dict(state)
    return model

# ---------------- Loader for checkpoint (.pt only) ----------------
def load_checkpoint(path):
    """Returns a dict with fields:
      - 'model': torch.nn.Module
      - 'mu', 'sd': np arrays
      - 'n_classes': int
      - 'in_dim': int
    """
    ext = os.path.splitext(path)[1].lower()
    if ext != ".pt":
        raise ValueError("Only .pt checkpoints are supported; GBDT/.joblib disabled.")
    torch = importlib.import_module('torch')
    # PyTorch 2.6+: allow full artifact (trusted source)
    art = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model_from_artifact(art)
    model.eval()
    return {
        'model': model,
        'mu': np.array(art['mu'], dtype=np.float32),
        'sd': np.array(art['sd'], dtype=np.float32),
        'n_classes': int(art['n_classes']),
        'in_dim': int(art['in_dim'])
    }

# ---------------- Online feature buffer ----------------
class FeatureBuffer:
    """Stores per-frame raw pieces, then assembles a (T,F) window matching landmark.py."""
    def __init__(self, include_arm=True, include_velocity=True, normalize_velocity=True,
                 include_scale_ch=True, clip_abs=50.0, t_target=64):
        self.include_arm = include_arm
        self.include_velocity = include_velocity
        self.normalize_velocity = normalize_velocity
        self.include_scale_ch = include_scale_ch
        self.clip_abs = clip_abs
        self.t_target = t_target
        self.buf = collections.deque(maxlen=256)  # (hand_norm[21,3], arm_norm[3,3], wrist_uvz[3], scale, valid)
    def append(self, hand_norm, arm_norm, wrist_uvz, scale, valid):
        self.buf.append((hand_norm.astype(np.float32),
                         arm_norm.astype(np.float32),
                         np.array(wrist_uvz, np.float32),
                         np.float32(scale) if np.isfinite(scale) else np.float32(np.nan),
                         bool(valid)))
    def append_invalid(self):
        hand_norm = np.zeros((21,3), np.float32)
        arm_norm  = np.zeros((3,3),  np.float32)
        wrist_uvz = np.array([np.nan, np.nan, np.nan], np.float32)
        scale     = np.float32(np.nan)
        self.append(hand_norm, arm_norm, wrist_uvz, scale, False)
    def _stack(self):
        if len(self.buf) == 0:
            return None
        hand_seq = np.stack([b[0] for b in self.buf], 0)  # (t,21,3)
        arm_seq  = np.stack([b[1] for b in self.buf], 0)  # (t,3,3)
        wrist_uvz_seq = np.stack([b[2] for b in self.buf], 0)  # (t,3)
        scale_seq     = np.array([b[3] for b in self.buf], np.float32)  # (t,)
        valid_seq     = np.array([b[4] for b in self.buf], bool)        # (t,)
        return hand_seq, arm_seq, wrist_uvz_seq, scale_seq, valid_seq
    def assemble_window(self):
        st = self._stack()
        if st is None:
            return None
        hand_seq, arm_seq, wrist_uvz_seq, scale_seq, valid_seq = st
        pose_feat = np.concatenate([hand_seq, arm_seq], axis=1)  # (t,J,3)
        t_len, J, C = pose_feat.shape
        pose_2d = pose_feat.reshape(t_len, J*C).astype(np.float32)

        scale_col = scale_seq[:, None]
        wrist_uvz = wrist_uvz_seq

        # Resample
        pose_T      = resample_2d(pose_2d,   self.t_target)
        wrist_uvz_T = resample_2d(wrist_uvz, self.t_target)
        valid_T     = resample_mask(valid_seq, self.t_target)
        scale_T     = resample_2d(scale_col,  self.t_target)

        # Sanitize
        wrist_uvz_T = np.nan_to_num(wrist_uvz_T, nan=0.0, posinf=0.0, neginf=0.0)
        scale_T     = np.nan_to_num(scale_T,     nan=0.0, posinf=0.0, neginf=0.0)

        # Velocities
        feat_chunks = [pose_T]
        d_pose  = make_velocity(pose_T)
        d_wrist = make_velocity(wrist_uvz_T)

        bad_now  = (~valid_T).astype(np.bool_)
        bad_prev = np.concatenate([[bad_now[0]], bad_now[:-1]])
        bad = (bad_now | bad_prev)[:, None]
        d_pose[bad.repeat(d_pose.shape[1], axis=1)] = 0.0
        d_wrist[bad.repeat(3, axis=1)] = 0.0

        d_pose  = normalize_velocity_per_clip(d_pose)
        d_wrist = normalize_velocity_per_clip(d_wrist)
        feat_chunks.extend([d_pose, d_wrist, scale_T])

        X = np.concatenate(feat_chunks, axis=1).astype(np.float32)
        X = np.clip(X, -50.0, 50.0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, valid_T

# ---------------- Hand + Pose processing ----------------
class MPProcessors:
    def __init__(self, desired_hand='Auto', pose_vis_thresh=0.40, draw=False):
        self.desired_hand = desired_hand
        self.pose_vis_thresh = pose_vis_thresh
        self.draw = draw

        self.mp_hands = mp.solutions.hands
        self.mp_pose  = mp.solutions.pose

        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.pose_det = self.mp_pose.Pose(static_image_mode=False,
                                          model_complexity=1,
                                          enable_segmentation=False,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

        self.R_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        self.R_ELBOW    = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
        self.R_WRIST    = self.mp_pose.PoseLandmark.RIGHT_WRIST.value

    def close(self):
        self.hands.close(); self.pose_det.close()

    def _px(self, lm, idx, W, H):
        return int(lm[idx].x * W), int(lm[idx].y * H), lm[idx].visibility

    def process_frame(self, bgr, depth, overlay=None):
        """Returns hand_norm(21,3), arm_norm(3,3), wrist_uvz(3), scale(float), valid(bool)."""
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = bgr.shape[:2]

        # Pose
        pose_results = self.pose_det.process(image_rgb)
        right_wrist_px = None
        arm_xyz = None
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            sx, sy, svis = self._px(lm, self.R_SHOULDER, W, H)
            ex, ey, evis = self._px(lm, self.R_ELBOW,    W, H)
            wx, wy, wvis = self._px(lm, self.R_WRIST,    W, H)
            right_wrist_px = (wx, wy)
            if (svis > self.pose_vis_thresh) and (evis > self.pose_vis_thresh) and (wvis > self.pose_vis_thresh):
                sz = get_depth_m(depth, sx, sy)
                ez = get_depth_m(depth, ex, ey)
                wz = get_depth_m(depth, wx, wy)
                if (sz is not None) and (ez is not None) and (wz is not None):
                    arm_xyz = np.array([[sx, sy, sz],
                                        [ex, ey, ez],
                                        [wx, wy, wz]], dtype=np.float32)

        # Hands
        hand_norm = np.zeros((21,3), dtype=np.float32)
        arm_norm  = np.zeros((3,3),  dtype=np.float32)
        wrist_uvz = [np.nan, np.nan, np.nan]
        scale     = np.nan
        valid     = False

        hand_results = self.hands.process(image_rgb)
        chosen = None
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            cands = list(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness))
            if self.desired_hand in ('Left','Right'):
                filtered = [c for c in cands if c[1].classification[0].label == self.desired_hand]
                if filtered:
                    cands = filtered
            if right_wrist_px is not None:
                def wrist_dist(hlm):
                    x = int(hlm.landmark[0].x * W); y = int(hlm.landmark[0].y * H)
                    dx = x - right_wrist_px[0]; dy = y - right_wrist_px[1]
                    return dx*dx + dy*dy
                cands.sort(key=lambda c: wrist_dist(c[0]))
            chosen = cands[0][0] if cands else None

        if chosen is not None:
            hand_px = np.array([[lm.x * W, lm.y * H, lm.z] for lm in chosen.landmark], dtype=np.float32)
            wrist_px = hand_px[0, :2]
            scale = hand_scale_from_landmarks(hand_px)

            hand_xy = (hand_px[:, :2] - wrist_px[None, :]) / max(scale, EPS)
            wz = get_depth_m(depth, int(round(wrist_px[0])), int(round(wrist_px[1])))
            wz = 0.0 if (wz is None or not np.isfinite(wz)) else float(wz)

            hand_norm[:,:2] = hand_xy
            hand_norm[:,2]  = hand_px[:,2]

            if arm_xyz is not None:
                arm_xy = (arm_xyz[:, :2] - wrist_px[None, :]) / max(scale, EPS)
                arm_norm[:, :2] = arm_xy
                arm_norm[:,  2] = arm_xyz[:, 2]

            wrist_uvz = [wrist_px[0], wrist_px[1], wz]
            valid = True

        return hand_norm, arm_norm, wrist_uvz, scale, valid

# ---------------- Smoother ----------------
class LabelSmoother:
    def __init__(self, k_consecutive=3, prob_tau=0.60):
        self.k = k_consecutive
        self.tau = prob_tau
        self.last_pred = None
        self.streak = 0
        self.public_label = None
    def update(self, pred_idx, pred_prob=None):
        if pred_idx == self.last_pred:
            self.streak += 1
        else:
            self.last_pred = pred_idx
            self.streak = 1
        changed = False
        if pred_prob is not None and pred_prob >= self.tau:
            self.public_label = pred_idx; changed = True
        elif self.streak >= self.k:
            self.public_label = pred_idx; changed = True
        return self.public_label, changed, self.streak

# ---------------- Main loop ----------------
def main():
    args = parse_args()

    # Load label map if provided
    label_map = None
    if args.label_map and os.path.isfile(args.label_map):
        with open(args.label_map, 'r', encoding='utf-8') as f:
            lm = json.load(f)
        label_map = {int(k): str(v) for k, v in lm.items()}

    # Load checkpoint (.pt only)
    art = load_checkpoint(args.checkpoint)
    model = art['model']
    mu, sd = art['mu'], art['sd']

    # RealSense setup (aligned to color)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # MediaPipe processors
    mp_proc = MPProcessors(desired_hand=args.desired_hand, draw=args.show_landmarks)

    # Feature buffer & smoother
    fb = FeatureBuffer(t_target=args.t_target)
    smoother = LabelSmoother(k_consecutive=args.smooth_k, prob_tau=args.prob_tau)

    frame_idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                fb.append_invalid()
                continue

            bgr = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            # Skip frames to save compute
            if (frame_idx % max(1, args.stride_frames)) != 0:
                frame_idx += 1
                fb.append_invalid()
                continue
            frame_idx += 1

            # Per-frame features
            hand_norm, arm_norm, wrist_uvz, scale, valid = mp_proc.process_frame(bgr, depth)
            fb.append(hand_norm, arm_norm, wrist_uvz, scale, valid)

            # Assemble window
            out = fb.assemble_window()
            if out is None:
                cv2.imshow('Online Recognizer', bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            X, valid_T = out

            # Gate by valid ratio
            vratio = float(valid_T.mean()) if valid_T.size else 1.0
            pred_idx = None
            pred_prob = None

            if vratio >= args.min_valid_ratio:
                torch = importlib.import_module('torch')
                xz = (X - mu) / (sd + 1e-8)
                xt = torch.from_numpy(xz[None]).float()
                mt = torch.from_numpy(valid_T[None].astype(np.float32))
                with torch.no_grad():
                    # All models accept (x, mask); TCN ignores mask internally
                    logits = model(xt, mt)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    pred_prob = float(np.max(probs))

            # Update smoother
            if pred_idx is not None:
                public_label, changed, streak = smoother.update(pred_idx, pred_prob)
            else:
                public_label, changed, streak = (smoother.public_label, False, smoother.streak)

            # ---- UI ----
            def label_name(idx):
                if idx is None:
                    return "—"
                return label_map.get(idx, str(idx)) if label_map else str(idx)

            txt1 = f"valid={vratio:.2f}  pred={label_name(pred_idx)}"
            txt2 = f"stable={label_name(smoother.public_label)}"
            if pred_prob is not None:
                txt1 += f"  p={pred_prob:.2f}"
            cv2.rectangle(bgr, (0,0), (bgr.shape[1], 62), (0,0,0), -1)
            cv2.putText(bgr, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(bgr, txt2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow('Online Recognizer', bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        mp_proc.close()

if __name__ == "__main__":
    main()
