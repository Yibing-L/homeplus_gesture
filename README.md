# Home+ Gesture Pipeline

This repository contains scripts to (1) collect RGB-D gesture clips, (2) featurize clips into fixed-length sequences, (3) train a gesture classifier, and (4) run live recognition with an Intel RealSense camera.

## Repository layout

Recommended layout:

- `scripts/`
  - `data_collection.py` : record raw clips from RealSense and save `.npz`
  - `landmark.py` : offline preprocessing, raw `.npz` -> processed `.npz` with `X` and `valid_T`
  - `train.py` : train model on processed clips
  - `online_recognizer.py` : run live recognition (RealSense)
  - `view_npz.py` : inspect a single `.npz`
  - `cleaning.py`, `validation.py` : dataset utilities
- `data/`
  - `raw/` : raw recorded `.npz` clips
  - `processed/` : processed `.npz` clips produced by `landmark.py`

## Dataset

Download `dataset.zip` from the GitHub Release:

https://github.com/Yibing-L/homeplus_gesture/releases/tag/data-v3

## Setup

Python 3.10 is recommended.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Install the Intel RealSense bindings when processing RGB-D recordings or using
the live recognizer:

```bash
pip install -r requirements-realsense.txt
```

## Recommended v2 pipeline

See [`docs/PIPELINE_V2.md`](docs/PIPELINE_V2.md) for the unified RGB-D processor,
SVM/TCN/Attention-BiLSTM/ST-GCN trainer, and compatible live recognizer.

## Live v2 deployment

Download the release's TCN-bones checkpoint and its adjacent
`recognizer_config.json` into the same directory. On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\recognizer_v2.py `
  --checkpoint "D:\path\to\homeplus_tcn_bones_v2.pt" `
  --device cuda `
  --no-log `
  --show-landmarks
```

Use `--camera-serial SERIAL` when more than one RealSense camera is connected.
The continuously changing `raw_pred` is diagnostic only. An application command
may be driven only by a newly emitted event; `LAST RECOGNIZED` remains visible
until the next accepted gesture for readability.
