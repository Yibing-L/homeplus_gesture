# Home+ gesture pipeline v2

The v2 pipeline uses one canonical processed NPZ schema for every learner.
Re-run MediaPipe on **all old and new raw RGB-D clips** so every subject uses the
same depth rejection, smoothing, geometry, and validity rules. Do not mix v1 and v2 NPZ files.
TCN and BiLSTM flatten the sequence, ST-GCN preserves the hand-joint graph,
and SVM summarizes the same canonical signals.

## CUDA setup

The tested study machine uses an RTX 4080 Laptop GPU and PyTorch 2.11 with CUDA
12.8. Install the official wheel inside `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Then verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Do not start `--device cuda` until this prints `True`.

## Process all raw RGB-D clips

With the original recording camera connected:

```powershell
python scripts/process_dataset_v2.py `
  --input "D:\path\to\raw" `
  --output "D:\path\to\processed_v2" `
  --from-camera `
  --resume
```

Without the camera, reuse intrinsics saved in a previous XYZ NPZ:

```powershell
python scripts/process_dataset_v2.py `
  --input "D:\path\to\raw" `
  --output "D:\path\to\processed_v2" `
  --intrinsics-npz "D:\path\to\sample_xyz.npz" `
  --resume
```

All numeric subject directories, including `12_done`, are found automatically.

## Feature ablation

```powershell
foreach ($featureSet in @("core", "bones", "angles", "full")) {
  python scripts/train_all_v2.py `
    --data "D:\path\to\processed_v2" `
    --output "runs\ablation_$featureSet" `
    --models tcn `
    --feature-set $featureSet `
    --n-classes 7 `
    --exclude-subjects 9 `
    --min-valid-ratio 0.25 `
    --device cuda `
    --no-final
}
```

| Set | Flat shape | Adds |
| --- | --- | --- |
| `core` | `64 x 184` | joints, motion, arm/global motion, masks |
| `bones` | `64 x 247` | core plus bone vectors |
| `angles` | `64 x 222` | core plus 19 angle cosines and masks |
| `full` | `64 x 285` | core plus bones and angles |

## Compare learners

```powershell
python scripts/train_all_v2.py `
  --data "D:\path\to\processed_v2" `
  --output "runs\model_comparison" `
  --models svm tcn bilstm stgcn `
  --feature-set full `
  --n-classes 7 `
  --exclude-subjects 9 `
  --min-valid-ratio 0.25 `
  --device cuda
```

ST-GCN receives `C x 64 x 21` hand nodes plus a global temporal branch. SVM
receives a fixed summary descriptor. Shapes differ, but the canonical data and
subject splits are shared.

## Paper-ready evaluation protocol

Use subject-independent leave-one-subject-out (LOSO) evaluation. In every outer
fold, one subject is the untouched test subject and one different subject is the
validation subject. Hyperparameters, SVM `C`, epoch selection, and early stopping
use only the training/validation subjects. Do not change settings after inspecting
outer-test results.

A defensible workflow is:

1. Run a short pipeline smoke test and the feature ablation using validation
   results only.
2. Select and record one feature set and one configuration per learner.
3. Run the complete four-learner LOSO comparison once with the settings locked.
4. If stochastic robustness is needed, repeat the locked neural configurations
   with three declared seeds; do not choose the best seed.

The default model widths are deliberately in a comparable parameter range:
TCN width 128 with 5 blocks, Attention-BiLSTM hidden size 128 with 2 layers and
projection size 128, and ST-GCN width 64. Every value is exposed as a command-line
argument and copied into the run metadata.

## Saved research artifacts

Every run directory contains:

- `experiment_config.json`: command, label mapping, split policy, feature set,
  exclusions, thresholds, and every hyperparameter.
- `environment.json`: Python/package versions, OS, GPU, CUDA, deterministic mode,
  Git revision, and worktree status.
- `dataset_summary.json`, `retained_clips.csv`, and `excluded_clips.csv`: the exact
  analyzed sample and every QC exclusion with its reason.
- `fold_manifest.json` and `fold_manifest.csv`: train, validation, and test subjects
  for every outer fold.
- `comparison_summary.json` and `comparison_summary.csv`: pooled and subject-level
  results for all requested learners.
- `<model>/predictions_out_of_fold.csv`: one row per held-out clip, including true
  task, prediction, raw label, condition, validity, and model scores.
- `<model>/per_subject_metrics.csv`, `per_task_metrics.csv`, and
  `per_condition_metrics.csv`: accuracy, balanced accuracy, macro-F1, precision,
  recall, F1, support, and dispersion needed for tables.
- `<model>/folds/test_subject_*.json`: selected epoch or SVM `C`, validation trace,
  complete learning history, timing, memory, and confusion matrices for that fold.
- `<model>/results.json`: pooled confusion matrices, classification report,
  mean/SD/median/min/max subject performance, training time, artifact size, and
  inference latency.
- `<model>/model_final.pt` or `model_final.joblib`: the deployable recognizer model
  plus feature schema, task mapping, QC policy, and hyperparameters.

The out-of-fold predictions are the source of truth for paper tables, confidence
intervals, paired model tests, and confusion-matrix figures. Final deployable-model
performance must not be reported as LOSO performance.
## Live recognition

```powershell
python scripts/recognizer_v2.py `
  --checkpoint "runs\model_comparison\stgcn\model_final.pt" `
  --device cuda `
  --show-landmarks
```

For SVM, pass `runs\model_comparison\svm\model_final.joblib`. Press `q` or
Escape to exit.
