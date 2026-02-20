#!/usr/bin/env python3
# check_output_labels.py
#
# Scans ./output/<participant_id>/*.npz
# Checks whether filename label (e.g., gesture_41_10.npz -> expected label 41)
# matches the stored `label` inside the .npz.
# Writes a report to: ./output/label_mismatch_report.txt

import os
import re
import zipfile
from io import BytesIO
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
REPORT_PATH = os.path.join(OUTPUT_ROOT, "label_mismatch_report.txt")

FNAME_LABEL_RE = re.compile(r"^gesture_(\d+)", re.IGNORECASE)
INT_DIR_RE = re.compile(r"^\d+$")

def list_participant_dirs(root: str):
    if not os.path.isdir(root):
        return []
    items = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and INT_DIR_RE.fullmatch(name):
            items.append((int(name), name, full))
    items.sort(key=lambda x: x[0])
    return items

def read_label_fast(npz_path: str):
    # Read only label.npy from the .npz zip, avoiding loading big arrays
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            label_name = "label.npy"
            if label_name not in zf.namelist():
                return None, "NO_LABEL_KEY"
            with zf.open(label_name) as f:
                arr = np.load(BytesIO(f.read()), allow_pickle=True)
            try:
                return int(arr), "OK"
            except Exception:
                return None, "LABEL_NOT_INT"
    except zipfile.BadZipFile:
        return None, "BAD_ZIP"
    except Exception:
        return None, "READ_ERROR"

def expected_label_from_filename(fname: str):
    base = os.path.splitext(os.path.basename(fname))[0]
    m = FNAME_LABEL_RE.match(base)
    if m:
        return int(m.group(1)), "OK"
    # Fallback: first integer anywhere in filename
    ints = re.findall(r"\d+", base)
    if ints:
        return int(ints[0]), "FALLBACK_FIRST_INT"
    return None, "NO_INT_IN_NAME"

def main():
    if not os.path.isdir(OUTPUT_ROOT):
        print(f"[ERROR] output folder not found: {OUTPUT_ROOT}")
        return

    participants = list_participant_dirs(OUTPUT_ROOT)
    if not participants:
        print(f"[WARN] No participant folders found under: {OUTPUT_ROOT}")
        return

    mismatches = []
    name_parse_fail = []
    label_read_fail = []
    total_files = 0
    total_checked = 0

    for _, pid_str, pdir in participants:
        npz_files = sorted([f for f in os.listdir(pdir) if f.endswith(".npz")])
        for f in npz_files:
            total_files += 1
            expected, name_status = expected_label_from_filename(f)
            if expected is None:
                name_parse_fail.append((pid_str, f, name_status))
                continue

            npz_path = os.path.join(pdir, f)
            actual, label_status = read_label_fast(npz_path)
            if actual is None:
                label_read_fail.append((pid_str, f, label_status))
                continue

            total_checked += 1
            if actual != expected:
                mismatches.append((pid_str, f, expected, actual))

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as out:
        out.write("Label check report\n")
        out.write(f"OUTPUT_ROOT: {OUTPUT_ROOT}\n\n")

        out.write("Summary\n")
        out.write(f"  total .npz files found: {total_files}\n")
        out.write(f"  total checked (name+label readable): {total_checked}\n")
        out.write(f"  mismatches: {len(mismatches)}\n")
        out.write(f"  name parse failures: {len(name_parse_fail)}\n")
        out.write(f"  label read failures: {len(label_read_fail)}\n\n")

        out.write("Mismatches (participant_id, expected_label_from_name, actual_label_in_npz, filename)\n")
        if mismatches:
            for pid, fname, exp, act in mismatches:
                out.write(f"  {pid}\t{exp}\t{act}\t{fname}\n")
        else:
            out.write("  (none)\n")

        out.write("\nName parse failures (participant_id, reason, filename)\n")
        if name_parse_fail:
            for pid, fname, reason in name_parse_fail:
                out.write(f"  {pid}\t{reason}\t{fname}\n")
        else:
            out.write("  (none)\n")

        out.write("\nLabel read failures (participant_id, reason, filename)\n")
        if label_read_fail:
            for pid, fname, reason in label_read_fail:
                out.write(f"  {pid}\t{reason}\t{fname}\n")
        else:
            out.write("  (none)\n")

    print(f"[DONE] Checked output labels.")
    print(f"  total .npz found: {total_files}")
    print(f"  total checked: {total_checked}")
    print(f"  mismatches: {len(mismatches)}")
    print(f"  report saved to: {REPORT_PATH}")
    if mismatches:
        print("\nFirst 20 mismatches (participant_id expected actual filename):")
        for row in mismatches[:20]:
            print(" ", row[0], row[2], row[3], row[1])

if __name__ == "__main__":
    main()
