# Home+ Gesture Pipeline (RGB-D + MediaPipe)

Scripts to record RGB-D gesture clips, featurize them into fixed-length sequences, train a classifier, and run live recognition.

The dataset is not stored in git. Put raw/processed data under `data/` (ignored by `.gitignore`) or distribute via GitHub Releases.

## Layout

- `scripts/`:
  - `data_collection.py`: record raw `.npz` clips (RealSense)
  - `landmark.py`: offline preprocessing, raw `.npz` -> processed `.npz` (writes `X`, `valid_T`)
  - `train.py`: train a model from processed clips
  - `online_recognizer.py`: live recognition (RealSense)
  - `view_npz.py`: inspect `.npz` files
  - `cleaning.py`, `validation.py`: dataset utilities
- `docs/`: format + feature documentation
- `data/` (ignored): `raw/`, `processed/`

## Setup

Python 3.10 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

RealSense-only scripts require `pyrealsense2`:

```bash
pip install -r requirements-realsense.txt
```

## Data format (short)

### Raw clip `.npz` (from `data_collection.py`)
- `rgb`: `(N, H, W, 3)` frames (OpenCV BGR)
- `depth`: `(N, H, W)` depth frames (RealSense z16)
- `label`: gesture label

### Processed clip `.npz` (from `landmark.py`)
- `X`: `(64, 148)` float32 feature sequence (see below)
- `valid_T`: `(64,)` validity mask aligned with `X`
- `label`: gesture label

Full schema: `docs/DATA_FORMAT.md`.

## What is inside `X` (64, 148)

`landmark.py` forces every clip to `T_TARGET=64` timesteps, then concatenates feature blocks:

1) **pose_T** (columns `0:72`, shape `(64, 72)`)
- Flattened `[x,y,z]` per landmark in this order:
  - 21 right-hand landmarks (MediaPipe indices 0..20), then
  - 3 right-arm joints: shoulder, elbow, wrist
- For the **hand**:
  - `x,y` are pixel coordinates centered at the hand wrist and divided by a per-frame `scale`
  - `z` is MediaPipe hand landmark `z` (not metric depth)
- For the **arm** (when pose landmarks are visible and depth is valid):
  - `x,y` are pixel coordinates centered at the hand wrist and divided by the same `scale`
  - `z` is RealSense depth in meters at that joint
- When tracking fails, landmarks are stored as zeros for that raw frame and later resampled.

2) **d_pose** (columns `72:144`, shape `(64, 72)`, if `INCLUDE_VELOCITY=True`)
- First-order difference of `pose_T` over time (same column order).
- Velocities are set to 0 whenever the current or previous timestep is invalid per `valid_T`.
- If `NORMALIZE_VELOCITY=True`, the entire clip velocity block is divided by its mean per-timestep magnitude.

3) **d_wrist** (columns `144:147`, shape `(64, 3)`, if `INCLUDE_VELOCITY=True`)
- First-order difference of `wrist_uvz_T = [u, v, z]` where:
  - `u,v` are absolute wrist pixel coordinates from the hand landmarks
  - `z` is RealSense depth (meters) at the wrist pixel
- Also masked to 0 on invalid timesteps and optionally normalized per clip.

4) **scale_T** (column `147`, shape `(64, 1)`, if `INCLUDE_SCALE_CH=True`)
- Resampled per-frame `scale` used for hand normalization.

Finally, `X` is clipped to `[-50, 50]` and NaNs/Infs are replaced with 0.

## Why `valid_T` exists

`X` is always length 64 even when tracking fails. Missing portions are represented by zeros/interpolation after resampling.
`valid_T` marks which of the 64 timesteps are trustworthy (hand landmarks existed in the raw clip). Use it to:
- mask loss/pooling during training
- filter out low-quality clips (e.g., mean(valid_T) below a threshold)

## Typical workflow

1) Put raw clips under `data/raw/`
2) Run preprocessing:
```bash
python scripts/landmark.py
```
3) Train:
```bash
python scripts/train.py --roots data/processed --save_path runs/exp1
```
4) Live recognition (RealSense):
```bash
python scripts/online_recognizer.py --checkpoint runs/exp1/model_best.pt
```

Notes:
- Offline preprocessing uses MediaPipe Holistic, while live recognition uses Hands + Pose in this codebase. Training will run, but live accuracy can drop due to feature distribution mismatch.

## Dataset hosting

Upload `dataset.zip` as a GitHub Release asset (do not commit it). See `scripts/download_data.ps1` and `scripts/download_data.sh` for templates.
