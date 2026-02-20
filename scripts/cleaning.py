#!/usr/bin/env python3
# fix_output_labels_selected.py
#
# Only fixes labels for a chosen participant + chosen filename-label(s).
# Always trusts the filename label (gesture_<LABEL>_*.npz).
#
# You edit the CONFIG section below (no CLI args).
#
# Creates backups of modified files under:
#   ./output/<participant_id>/_backup_before_label_fix_YYYYMMDD_HHMMSS/
# Writes a report to:
#   ./output/label_fix_selected_report.txt

import os
import re
import shutil
from datetime import datetime
from io import BytesIO
import zipfile
import numpy as np

# ===================== CONFIG (edit these) =====================
TARGET_PARTICIPANT_ID = "5"      # participant folder under ./output/, e.g. "0"
TARGET_FILENAME_LABELS = {13}    # only fix files whose *filename* label is in this set
# Example: {13} means only gesture_13_*.npz files in output/0/

# If True, only report what would change; if False, actually modify files
DRY_RUN = False
# ===============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
REPORT_PATH = os.path.join(OUTPUT_ROOT, "label_fix_selected_report.txt")

FNAME_LABEL_RE = re.compile(r"^gesture_(\d+)", re.IGNORECASE)

def expected_label_from_filename(fname: str):
    base = os.path.splitext(os.path.basename(fname))[0]
    m = FNAME_LABEL_RE.match(base)
    if m:
        return int(m.group(1)), "OK"
    return None, "NO_MATCH_gesture_<label>"

def read_label_fast(npz_path: str):
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            if "label.npy" not in zf.namelist():
                return None, "NO_LABEL_KEY"
            with zf.open("label.npy") as f:
                arr = np.load(BytesIO(f.read()), allow_pickle=True)
            return int(arr), "OK"
    except zipfile.BadZipFile:
        return None, "BAD_ZIP"
    except Exception:
        return None, "READ_ERROR"

def write_label(npz_path: str, new_label: int, backup_dir: str):
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(npz_path, os.path.join(backup_dir, os.path.basename(npz_path)))

    # Close the original file handle before replacing (important on Windows)
    with np.load(npz_path, allow_pickle=True) as d:
        payload = {k: d[k] for k in d.files}
    payload["label"] = np.int32(new_label)

    # IMPORTANT: tmp path must end with .npz, otherwise numpy will append .npz
    tmp_path = npz_path + ".tmp.npz"
    np.savez_compressed(tmp_path, **payload)

    # Atomic replace
    os.replace(tmp_path, npz_path)

def main():
    target_dir = os.path.join(OUTPUT_ROOT, TARGET_PARTICIPANT_ID)
    if not os.path.isdir(target_dir):
        print(f"[ERROR] Participant output folder not found: {target_dir}")
        return

    files = sorted([f for f in os.listdir(target_dir) if f.endswith(".npz")])
    if not files:
        print(f"[WARN] No .npz files in {target_dir}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(target_dir, f"_backup_before_label_fix_{ts}")

    total_files = 0
    checked = 0
    changed = 0

    skipped_not_target_label = 0
    name_parse_fail = []
    label_read_fail = []
    fixes = []  # (fname, expected, actual_before)

    for f in files:
        total_files += 1

        expected, status = expected_label_from_filename(f)
        if expected is None:
            name_parse_fail.append((f, status))
            continue

        if expected not in TARGET_FILENAME_LABELS:
            skipped_not_target_label += 1
            continue

        npz_path = os.path.join(target_dir, f)
        actual, lstat = read_label_fast(npz_path)
        if actual is None:
            label_read_fail.append((f, lstat))
            continue

        checked += 1
        if actual == expected:
            continue

        print(f"[FIX] {f}: label {actual} -> {expected}")
        fixes.append((f, expected, actual))
        changed += 1

        if not DRY_RUN:
            write_label(npz_path, expected, backup_dir)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as out:
        out.write("Label fix report (selected participant + selected filename-labels)\n")
        out.write(f"OUTPUT_ROOT: {OUTPUT_ROOT}\n")
        out.write(f"TARGET_PARTICIPANT_ID: {TARGET_PARTICIPANT_ID}\n")
        out.write(f"TARGET_FILENAME_LABELS: {sorted(TARGET_FILENAME_LABELS)}\n")
        out.write(f"DRY_RUN: {DRY_RUN}\n\n")

        out.write("Summary\n")
        out.write(f"  total .npz found: {total_files}\n")
        out.write(f"  skipped (filename label not in TARGET_FILENAME_LABELS): {skipped_not_target_label}\n")
        out.write(f"  checked (name+label readable and targeted): {checked}\n")
        out.write(f"  changed: {changed}\n")
        out.write(f"  name parse failures: {len(name_parse_fail)}\n")
        out.write(f"  label read failures: {len(label_read_fail)}\n")
        if (not DRY_RUN) and changed > 0:
            out.write(f"  backup_dir: {backup_dir}\n")
        out.write("\n")

        out.write("Fixes applied (filename, expected_from_name, actual_before)\n")
        if fixes:
            for f, exp, act in fixes:
                out.write(f"  {f}\t{exp}\t{act}\n")
        else:
            out.write("  (none)\n")

        out.write("\nName parse failures (reason, filename)\n")
        if name_parse_fail:
            for f, reason in name_parse_fail:
                out.write(f"  {reason}\t{f}\n")
        else:
            out.write("  (none)\n")

        out.write("\nLabel read failures (reason, filename)\n")
        if label_read_fail:
            for f, reason in label_read_fail:
                out.write(f"  {reason}\t{f}\n")
        else:
            out.write("  (none)\n")

    print("\n[DONE] Selected label correction finished.")
    print(f"  participant: {TARGET_PARTICIPANT_ID}")
    print(f"  target filename labels: {sorted(TARGET_FILENAME_LABELS)}")
    print(f"  total found: {total_files}")
    print(f"  checked: {checked}")
    print(f"  changed: {changed}")
    print(f"  report: {REPORT_PATH}")
    if DRY_RUN:
        print("  DRY_RUN=True, no files were modified.")
    elif changed > 0:
        print(f"  backup: {backup_dir}")

if __name__ == "__main__":
    main()
