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

https://github.com/Yibing-L/homeplus_gesture/releases/tag/data-v1

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