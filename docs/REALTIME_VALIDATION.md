# Home+ real-time validation

## Status

The provisional deployment checkpoint is the bones TCN:

`runs/paper_20260808_locked_v2/tcn_bones/tcn/model_final.pt`

Its adjacent `recognizer_config.json` is loaded automatically. The initial
probability threshold is 0.90. On 11,648 positive out-of-fold gesture clips,
this retained 70.69% of predictions with 93.77% accepted-prediction precision.
This is not a final continuous-use calibration because the training dataset has
no dedicated rest/non-gesture class.

## Pilot command

Connect the study D455, then run from the repository root in PowerShell:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

.\.venv\Scripts\python.exe scripts\recognizer_v2.py `
  --checkpoint "runs\paper_20260808_locked_v2\tcn_bones\tcn\model_final.pt" `
  --device cuda `
  --camera-serial "241122300549" `
  --log-jsonl "runs\realtime_sessions\$stamp.bones.jsonl" `
  --record-file "runs\realtime_sessions\$stamp.db3" `
  --show-landmarks
```

The program refuses to overwrite an existing recording. JSONL output is line-buffered
so completed records survive an interruption.

## Controls and protocol

- Wait until `window_ready` becomes true (90 camera frames, approximately three
  seconds at 30 Hz).
- Press `0` through `6` immediately before starting the corresponding gesture.
- Press Space immediately after completing it.
- Pause with the hand quiet between gestures so the recognizer can leave
  cooldown and rearm.
- Press `Q` or Escape to finish.
- Include long annotated-free periods containing rest and ordinary non-command
  hand movements. These periods measure false activations.

The overlay must say `model=tcn/bones`. Startup output must show the absolute
bones checkpoint path, `classes=7`, `device=cuda`, and threshold 0.90.

For a pilot, perform every class at least five times and include at least five
minutes of non-gesture activity. Do not use a pilot as the final paper result;
use it to tune motion and confidence thresholds without touching the eventual
held-out real-time evaluation sessions.

## Metrics

The primary metric is gesture-event macro-F1 over all seven classes. A predicted
event is matched one-to-one to a same-class ground-truth event within the
predefined temporal tolerance. Unmatched/duplicate/wrong triggers are false
positives; unmatched ground-truth gestures are false negatives.

Mandatory secondary metrics are:

- false activations per minute during non-gesture intervals;
- median and 95th-percentile detection latency;
- per-class event precision, recall, and F1;
- missed and wrong gesture counts.

Generate a report with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_recognizer_log_v2.py `
  --log "runs\realtime_sessions\SESSION.bones.jsonl" `
  --output "runs\realtime_sessions\SESSION.bones.metrics.json"
```

Macro-F1 is meaningful only when the session includes all seven gesture classes.

## Final bones-versus-full comparison

Record each continuous RGB-D session once and use that same `.db3` recording for both
checkpoints. Use separate calibration sessions to choose thresholds, then lock
the thresholds and evaluate on untouched sessions. Select full instead of bones
if it gives meaningfully fewer false activations or misses, even though bones
has the highest isolated-clip macro-F1.

