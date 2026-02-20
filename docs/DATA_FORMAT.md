# Data format

This project uses two `.npz` formats: raw recorded clips and processed feature clips.

## Raw clip `.npz` (from `scripts/data_collection.py`)

Typical keys:
- `rgb`: `(N, H, W, 3)` frames, dtype `uint8` (OpenCV BGR ordering)
- `depth`: `(N, H, W)` frames, dtype typically `uint16` (RealSense z16)
- `label`: gesture label (int-like)

Raw files may contain additional metadata depending on your collection script version.

## Processed clip `.npz` (from `scripts/landmark.py`)

Required keys:
- `X`: `(64, 148)` float32 feature matrix
- `valid_T`: `(64,)` float32 mask (0/1) aligned with `X`
- `label`: gesture label

Debug/provenance keys (saved by the current script):
- `landmarks`: `(t_raw, 21, 3)` float32, right-hand landmarks
  - `x,y`: centered at wrist and divided by `scale`
  - `z`: MediaPipe hand landmark z (not metric depth)
- `arm_landmarks`: `(t_raw, 3, 3)` float32, right shoulder/elbow/wrist
  - `x,y`: centered at hand wrist and divided by `scale`
  - `z`: RealSense depth in meters at that joint (0 if unavailable)
- `wrist_uvz`: `(t_raw, 3)` float32, `[u, v, z]` wrist pixel coordinates + wrist depth (meters)
- `scale`: `(t_raw,)` float32, per-frame hand scale used for normalization
- `valid_raw`: `(t_raw,)` uint8, per-frame validity (1 when right hand landmarks exist)

Metadata keys:
- `T_target` (int)
- `include_arm` (bool-like)
- `include_velocity` (bool-like)
- `normalize_velocity` (bool-like)
- `include_scale_channel` (bool-like)

Feature layout details: `docs/FEATURES.md`.
