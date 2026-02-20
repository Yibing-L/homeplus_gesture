# Feature vector `X`

`X` is produced by `scripts/landmark.py`.

## Shape

- `X`: `(T_TARGET, F)` = `(64, 148)` for the default config:
  - `T_TARGET = 64`
  - `INCLUDE_ARM = True`
  - `INCLUDE_VELOCITY = True`
  - `INCLUDE_SCALE_CH = True`

## Column blocks

### 1) pose_T (0:72)

Built from per-frame landmarks, then resampled to 64 steps.

Landmark sources:
- Right hand: 21 landmarks from MediaPipe Holistic right hand.
- Right arm: 3 joints from MediaPipe pose landmarks (right shoulder, right elbow, right wrist), with depth from the RealSense frame when available.

Normalization:
- For hand and arm `x,y`:
  - Convert to pixels, subtract the hand wrist pixel `(u0,v0)`, divide by `scale`.
- Hand `z`:
  - MediaPipe hand landmark z (unitless, camera-relative).
- Arm `z`:
  - RealSense depth at that joint in meters.

Flattening order (per timestep):
`[lm0_x, lm0_y, lm0_z, lm1_x, lm1_y, lm1_z, ..., lm23_x, lm23_y, lm23_z]`
where lm0..lm20 are hand landmarks, lm21..lm23 are [shoulder, elbow, wrist].

### 2) d_pose (72:144)

`d_pose[t] = pose_T[t] - pose_T[t-1]` with `d_pose[0]=0`.

Masked to 0 if timestep `t` or `t-1` is invalid according to `valid_T`.

If `NORMALIZE_VELOCITY=True`, divided by the mean per-timestep magnitude across the clip.

### 3) d_wrist (144:147)

Velocity of the resampled wrist stream `wrist_uvz_T = [u, v, z]`:
- `u,v`: absolute wrist pixel coordinates
- `z`: RealSense wrist depth in meters

Same masking and optional normalization as `d_pose`.

### 4) scale_T (147)

Resampled per-frame `scale` used for normalization, included as a single channel.

## Post-processing

`X` is clipped to `[-50, 50]`, then NaNs/Infs are replaced by 0.
