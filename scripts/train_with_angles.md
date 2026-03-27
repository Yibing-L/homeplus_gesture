# train_with_angles.py

Gesture classification trainer supporting multiple temporal/spatial architectures. Loads preprocessed `.npz` clips (207-dim features with joint angles), trains with LOSO or K-fold evaluation, and saves per-fold and best-overall model checkpoints.

## Quick Start

```bash
# Default: BiLSTM with LOSO evaluation
python scripts/train_with_angles.py --roots dataset/*_done

# UNet1D (faster on CPU)
python scripts/train_with_angles.py --roots dataset/*_done --model unet1d --proj_dim 32

# ST-GCN (best with GPU)
python scripts/train_with_angles.py --roots dataset/*_done --model stgcn
```

## Feature Layout (207 channels)

| Channels    | Name              | Dim | Description                                      |
|-------------|-------------------|-----|--------------------------------------------------|
| `[0:72]`    | pose              | 72  | 24 joints x 3 (XYZ), wrist-relative, bone-normalized |
| `[72:144]`  | vel_pose          | 72  | Frame-to-frame velocity of pose                  |
| `[144:147]` | vel_wrist         | 3   | Absolute wrist velocity (meters/frame)           |
| `[147:148]` | scale             | 1   | Bone-length scale (meters)                       |
| `[148:163]` | hand_angles       | 15  | Hand joint bend angles (radians, 0 if invalid)   |
| `[163:164]` | arm_angle         | 1   | Elbow bend angle (radians, 0 if invalid)         |
| `[164:165]` | frame_valid       | 1   | Binary frame validity                            |
| `[165:186]` | joint_valid       | 21  | Per-joint depth validity (hand)                  |
| `[186:189]` | arm_valid         | 3   | Per-joint depth validity (arm)                   |
| `[189:190]` | hand_cov          | 1   | Mean hand joint validity                         |
| `[190:191]` | arm_cov           | 1   | Mean arm joint validity                          |
| `[191:206]` | hand_angle_valid  | 15  | Validity flags for hand angles                   |
| `[206:207]` | arm_angle_valid   | 1   | Validity flag for elbow angle                    |

First 164 channels are continuous (`N_CONTINUOUS=164`), remaining 43 are binary.

## Models (`--model`)

| Model           | Description                                                        | Key Flags Used                              |
|-----------------|--------------------------------------------------------------------|---------------------------------------------|
| `bilstm`        | Attention-BiLSTM (default). Linear projection -> BiLSTM -> multi-head attention pool -> classify. | `--hidden`, `--lstm_layers`, `--attn_heads`, `--proj_dim` |
| `unet1d`        | 1D UNet. 3-level Conv1d encoder/decoder with skip connections, masked global avg pool. Fast on CPU. | `--proj_dim` (base channels)                |
| `resnet1d`      | 1D ResNet. 4 residual stages with stride-2 downsampling, global avg pool.                          | `--proj_dim` (base channels)                |
| `unet1d_lstm`   | UNet encoder/decoder -> BiLSTM -> attention pool. Hybrid CNN+RNN.  | `--proj_dim`, `--hidden`, `--lstm_layers`, `--attn_heads` |
| `resnet1d_lstm` | ResNet encoder -> BiLSTM -> attention pool. Hybrid CNN+RNN.        | `--proj_dim`, `--hidden`, `--lstm_layers`, `--attn_heads` |
| `stgcn`         | Spatio-Temporal GCN. 3-partition graph conv over 23-node hand+arm skeleton, temporal stride-2 downsampling, mask-aware pooling. Ignores `--proj_dim`. Best with GPU. | `--dropout` only |

## Flags

### Required

| Flag       | Description                                |
|------------|--------------------------------------------|
| `--roots`  | One or more directories containing `.npz` clips |

### Model & Evaluation

| Flag                   | Default   | Description                                  |
|------------------------|-----------|----------------------------------------------|
| `--model`              | `bilstm`  | Model architecture (see table above)         |
| `--n_classes`          | `7`       | 7 (gesture only) or 42 (gesture x condition) |
| `--eval_mode`          | `loso`    | `loso` (leave-one-subject-out) or `kfold`    |
| `--k`                  | `5`       | Number of folds for kfold mode               |
| `--save_path`          | `runs/xyz_angles` | Output directory for checkpoints & logs |

### Architecture

| Flag             | Default | Used By                       |
|------------------|---------|-------------------------------|
| `--hidden`       | `256`   | bilstm, unet1d_lstm, resnet1d_lstm |
| `--lstm_layers`  | `2`     | bilstm, unet1d_lstm, resnet1d_lstm |
| `--attn_heads`   | `4`     | bilstm, unet1d_lstm, resnet1d_lstm |
| `--proj_dim`     | `256`   | All except stgcn (base channel count for CNNs, projection dim for BiLSTM) |
| `--dropout`      | `0.4`   | All models                    |

### Training

| Flag                | Default | Description                         |
|---------------------|---------|-------------------------------------|
| `--epochs`          | `120`   | Max training epochs per fold        |
| `--batch_size`      | `64`    | Batch size                          |
| `--lr`              | `1e-3`  | Learning rate (OneCycleLR max)      |
| `--weight_decay`    | `5e-4`  | AdamW weight decay                  |
| `--label_smoothing` | `0.1`   | Cross-entropy label smoothing       |
| `--patience`        | `30`    | Early stopping patience (epochs)    |
| `--clip_grad`       | `1.0`   | Gradient clipping max norm          |
| `--seed`            | `1337`  | Random seed                         |
| `--device`          | auto    | `cuda` if available, else `cpu`     |
| `--num_workers`     | `0`     | DataLoader workers                  |

### Feature Modifiers

| Flag                     | Default | Description                                      |
|--------------------------|---------|--------------------------------------------------|
| `--condition_vec`        | off     | Append 6-dim one-hot condition vector (from `y42 // 7`) to features. Changes in_dim from 207 to 213. |
| `--compress_joints`      | off     | Compress first 144 joint features via CNN before temporal model. Drops per-joint validity flags. |
| `--compress_joints_dim`  | `32`    | Output dimension of joint compression CNN        |
| `--clip_norm`            | on      | Per-sample instance normalization of continuous channels |
| `--normalize_scale_channel` | on   | Normalize bone-length scale by training-fold mean |

### Augmentation

All augmentations are on by default. Use `--no_aug` to disable all.

| Flag                      | Default | Description                              |
|---------------------------|---------|------------------------------------------|
| `--no_aug`                | off     | Disable all augmentation                 |
| `--aug_speed`             | on      | Speed perturbation (0.85x-1.15x)         |
| `--aug_warp`              | on      | Random time warping                      |
| `--aug_noise_std`         | `0.02`  | Gaussian noise on continuous channels    |
| `--aug_frame_drop_prob`   | `0.05`  | Random frame dropout probability         |
| `--aug_block_mask_prob`   | `0.05`  | Probability of zeroing a feature block   |
| `--aug_scale_jitter`      | on      | Random scaling of pose+velocity channels |
| `--aug_scale_lo/hi`       | `0.7/1.2` | Scale jitter range                    |
| `--aug_rotate_prob`       | `0.5`   | Probability of in-plane rotation         |
| `--aug_rotate_max_deg`    | `15.0`  | Max rotation angle (degrees)             |
| `--aug_temporal_crop_prob`| `0.5`   | Probability of random temporal crop      |
| `--mixup_alpha`           | `0.2`   | Mixup beta distribution alpha (0=off)    |

## Outputs

All saved to `--save_path` (default `runs/xyz_angles/`):

| File                          | Description                                |
|-------------------------------|--------------------------------------------|
| `config.json`                 | Full argument dump                         |
| `training.log`                | Training log                               |
| `model_7.pt` / `model_42.pt` | Best overall model checkpoint              |
| `model_{N}_{fold}.pt`        | Per-fold model checkpoints                 |
| `results.json`                | Aggregate and per-fold metrics             |
| `cm_aggregate.png`            | Aggregate confusion matrix                 |
| `cm_{fold}.png`               | Per-fold confusion matrices                |
| `curves_{fold}.png`           | Training curves (loss, F1, accuracy)       |
| `per_participant_report.txt`  | Per-subject accuracy and F1                |
| `per_condition_report.txt`    | Per-condition breakdown (42-class only)    |

## Model Checkpoint Contents

Each `.pt` file is a dict containing:

- `type`: model identifier (e.g. `bilstm_xyz_angles`, `stgcn_xyz_angles`)
- `state_dict`: model weights
- `in_dim`, `n_classes`, `n_continuous`: feature dimensions
- `mu`, `sd`: standardization parameters (mean, std of training fold)
- `scale_mean`: bone-length scale normalization factor
- `clip_norm`: whether instance normalization was used
- Architecture hyperparameters (`hidden`, `lstm_layers`, `attn_heads`, `proj_dim`, `dropout`)
- Eval metadata (`best_fold`, `best_val_f1`, `best_test_f1`, `aggregate_f1`, `aggregate_acc`)

## Example Commands

```bash
# Fast iteration on CPU
python scripts/train_with_angles.py --roots dataset/*_done \
    --model unet1d --proj_dim 32 --epochs 30 --patience 10

# BiLSTM with joint compression
python scripts/train_with_angles.py --roots dataset/*_done \
    --model bilstm --compress_joints --proj_dim 32 --hidden 64

# ST-GCN (needs GPU for reasonable speed)
python scripts/train_with_angles.py --roots dataset/*_done \
    --model stgcn --epochs 30 --patience 10

# UNet+LSTM hybrid with condition encoding
python scripts/train_with_angles.py --roots dataset/*_done \
    --model unet1d_lstm --proj_dim 32 --hidden 64 --condition_vec

# 42-class training (gesture x condition)
python scripts/train_with_angles.py --roots dataset/*_done \
    --n_classes 42 --model bilstm
```
