#!/usr/bin/env python3
"""Evaluate annotated Home+ recognizer JSONL logs at the gesture-event level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Event-level metrics for recognizer_v2 JSONL")
    ap.add_argument("--log", required=True)
    ap.add_argument("--classes", type=int, default=7)
    ap.add_argument("--pre-grace", type=float, default=0.25)
    ap.add_argument("--post-grace", type=float, default=1.0)
    ap.add_argument("--output", help="Optional JSON report path")
    return ap.parse_args()


def load_records(path: Path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def interval_union_seconds(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return float(sum(end - start for start, end in merged))


def main():
    args = parse_args(); path = Path(args.log)
    rows = load_records(path)
    starts = []
    ground_truth = []
    predictions = []
    duration = 0.0
    for row in rows:
        duration = max(duration, float(row.get("monotonic_seconds", 0.0)))
        kind = row.get("record_type")
        if kind == "ground_truth_start":
            starts.append(dict(class_index=int(row["class_index"]), start=float(row["monotonic_seconds"])))
        elif kind == "ground_truth_end":
            cls = int(row["class_index"]); declared = float(row.get("start_seconds", -1.0))
            match = next((i for i, item in enumerate(starts) if item["class_index"] == cls and abs(item["start"] - declared) < 1e-3), None)
            if match is None:
                raise ValueError(f"Ground-truth end without matching start: {row}")
            item = starts.pop(match); item["end"] = float(row["monotonic_seconds"])
            ground_truth.append(item)
        elif kind == "prediction_event":
            predictions.append(dict(
                class_index=int(row["class_index"]), time=float(row["monotonic_seconds"]),
                confidence=float(row.get("confidence", 0.0)),
            ))
    if starts:
        raise ValueError(f"{len(starts)} ground-truth annotation(s) were not closed")

    used = set(); matched = []
    for gt_index, gt in enumerate(sorted(ground_truth, key=lambda x: x["start"])):
        eligible = [
            (i, pred) for i, pred in enumerate(predictions)
            if i not in used and pred["class_index"] == gt["class_index"]
            and gt["start"] - args.pre_grace <= pred["time"] <= gt["end"] + args.post_grace
        ]
        if eligible:
            pred_index, pred = min(eligible, key=lambda pair: pair[1]["time"])
            used.add(pred_index)
            matched.append((gt, pred))

    per_class = {}
    f1s = []
    for cls in range(args.classes):
        tp = sum(gt["class_index"] == cls for gt, _ in matched)
        actual = sum(gt["class_index"] == cls for gt in ground_truth)
        predicted = sum(pred["class_index"] == cls for pred in predictions)
        fp = predicted - tp; fn = actual - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(cls)] = dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)
        f1s.append(f1)

    intervals = [(gt["start"], gt["end"]) for gt in ground_truth]
    negative_seconds = max(0.0, duration - interval_union_seconds(intervals))
    false_in_negative = sum(
        not any(start <= pred["time"] <= end for start, end in intervals)
        for pred in predictions
    )
    latencies = [pred["time"] - gt["start"] for gt, pred in matched]
    report = dict(
        schema_version="homeplus_realtime_metrics_v1", log=str(path.resolve()),
        ground_truth_events=len(ground_truth), predicted_events=len(predictions),
        matched_events=len(matched), event_macro_f1=float(np.mean(f1s)),
        false_activations_in_negative_intervals=false_in_negative,
        negative_minutes=negative_seconds / 60.0,
        false_activations_per_minute=(false_in_negative / (negative_seconds / 60.0) if negative_seconds else None),
        detection_latency_seconds=dict(
            mean=float(np.mean(latencies)) if latencies else None,
            median=float(np.median(latencies)) if latencies else None,
            p95=float(np.quantile(latencies, 0.95)) if latencies else None,
        ),
        matching=dict(pre_grace_seconds=args.pre_grace, post_grace_seconds=args.post_grace),
        per_class=per_class,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
