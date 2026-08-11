#!/usr/bin/env python3
"""State and logging helpers for the Home+ continuous recognizer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


class JsonlLogger:
    """Line-buffered event/frame logger that remains readable after interruption."""

    def __init__(self, path: str | Path, metadata: Dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.write("session_start", **metadata)

    def write(self, record_type: str, **fields) -> None:
        row = {"record_type": record_type, "utc": utc_now(), **fields}
        self.handle.write(json.dumps(row, default=_json_default, separators=(",", ":")) + "\n")

    def close(self, **fields) -> None:
        if self.handle.closed:
            return
        self.write("session_end", **fields)
        self.handle.close()


def recent_motion_score(hand, wrist, joint_valid, frame_valid, hand_xy=None) -> float:
    """Robust recent motion used only for gesture activity gating.

    Live recognition prefers the 75th-percentile displacement between two
    three-frame medians of screen-space hand landmarks, normalized by apparent
    hand size. The legacy 3-D calculation remains as a compatibility fallback.
    This score is never a classifier feature.
    """
    # Prefer robust screen-space motion when live 2-D landmarks are available.
    # Depth estimates can jump while a hand is stationary, which makes a
    # frame-to-frame 3-D gate look active even though the image is still.
    if hand_xy is not None and len(hand_xy) >= 6 and len(frame_valid) >= 6:
        if not all(bool(value) for value in list(frame_valid)[-6:]):
            return 0.0
        xy = np.asarray(list(hand_xy)[-6:], np.float32)
        valid = np.asarray(list(joint_valid)[-6:], bool)
        common = np.all(valid, axis=0)
        if int(common.sum()) < 4:
            return 0.0
        before = np.median(xy[:3], axis=0)
        after = np.median(xy[3:], axis=0)
        scale_points = [j for j in (5, 9, 17) if common[j]]
        if scale_points:
            scale = float(np.median(np.linalg.norm(after[scale_points] - after[0], axis=1)))
        else:
            scale = 50.0
        scale = max(scale, 12.0)
        displacement = np.linalg.norm(after[common] - before[common], axis=1) / scale
        return float(np.quantile(displacement, 0.75)) if displacement.size else 0.0

    if len(hand) < 2 or len(wrist) < 2 or len(frame_valid) < 2:
        return 0.0
    if not (bool(frame_valid[-1]) and bool(frame_valid[-2])):
        return 0.0
    current = np.asarray(hand[-1], np.float32)
    previous = np.asarray(hand[-2], np.float32)
    common = np.asarray(joint_valid[-1], bool) & np.asarray(joint_valid[-2], bool)
    if int(common.sum()) < 4:
        return 0.0
    scale_points = [j for j in (5, 9, 17) if common[j]]
    if scale_points:
        scale = float(np.median(np.linalg.norm(current[scale_points], axis=1)))
    else:
        scale = 0.075
    scale = max(scale, 0.02)
    joint_motion = np.linalg.norm(current[common] - previous[common], axis=1) / scale
    hand_score = float(np.quantile(joint_motion, 0.75)) if joint_motion.size else 0.0
    wrist_score = float(np.linalg.norm(np.asarray(wrist[-1]) - np.asarray(wrist[-2])) / scale)
    return max(hand_score, wrist_score)


@dataclass
class GateResult:
    state: str
    candidate: Optional[int]
    streak: int
    displayed_event: Optional[int]
    emitted_event: Optional[int]
    accepted: bool
    moving: bool
    quiet: bool


class EventGate:
    """IDLE -> ACTIVE -> COOLDOWN recognizer state machine."""

    def __init__(
        self,
        probability_threshold: float = 0.90,
        stable_count: int = 3,
        motion_on: float = 0.020,
        motion_off: float = 0.010,
        motion_start_count: int = 2,
        active_timeout_seconds: float = 1.0,
        cooldown_seconds: float = 1.25,
        rearm_quiet_count: int = 8,
        display_seconds: float = 1.5,
    ):
        if not 0.0 < probability_threshold < 1.0:
            raise ValueError("probability_threshold must be between zero and one")
        if stable_count < 1 or motion_start_count < 1 or rearm_quiet_count < 1:
            raise ValueError("count settings must be positive")
        if motion_off >= motion_on:
            raise ValueError("motion_off must be smaller than motion_on")
        self.probability_threshold = float(probability_threshold)
        self.stable_count = int(stable_count)
        self.motion_on = float(motion_on)
        self.motion_off = float(motion_off)
        self.motion_start_count = int(motion_start_count)
        self.active_timeout_seconds = float(active_timeout_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.rearm_quiet_count = int(rearm_quiet_count)
        self.display_seconds = float(display_seconds)
        self.state = "IDLE"
        self.candidate: Optional[int] = None
        self.streak = 0
        self.motion_streak = 0
        self.quiet_streak = 0
        self.last_motion_time = 0.0
        self.cooldown_until = 0.0
        self.displayed_event: Optional[int] = None
        self.display_until = 0.0

    def settings(self) -> Dict[str, Any]:
        keys = (
            "probability_threshold", "stable_count", "motion_on", "motion_off",
            "motion_start_count", "active_timeout_seconds", "cooldown_seconds",
            "rearm_quiet_count", "display_seconds",
        )
        return {key: getattr(self, key) for key in keys}

    def _reset_candidate(self) -> None:
        self.candidate = None
        self.streak = 0

    def update(
        self,
        now: float,
        window_ready: bool,
        valid: bool,
        motion_score: float,
        prediction: Optional[int],
        probability: float,
    ) -> GateResult:
        emitted = None
        moving = bool(motion_score >= self.motion_on)
        quiet = bool(motion_score <= self.motion_off)
        if self.displayed_event is not None and now >= self.display_until:
            self.displayed_event = None

        if self.state == "IDLE":
            self._reset_candidate()
            self.motion_streak = self.motion_streak + 1 if window_ready and valid and moving else 0
            if self.motion_streak >= self.motion_start_count:
                self.state = "ACTIVE"
                self.last_motion_time = now
                self.motion_streak = 0

        elif self.state == "ACTIVE":
            if moving:
                self.last_motion_time = now
            accepted = bool(
                window_ready and valid and prediction is not None
                and probability >= self.probability_threshold
            )
            if not accepted:
                # Strictly consecutive accepted predictions are required.
                self._reset_candidate()
            elif prediction == self.candidate:
                self.streak += 1
            else:
                self.candidate = int(prediction)
                self.streak = 1
            if self.streak >= self.stable_count:
                emitted = int(self.candidate)
                self.displayed_event = emitted
                self.display_until = now + self.display_seconds
                self.state = "COOLDOWN"
                self.cooldown_until = now + self.cooldown_seconds
                self.quiet_streak = 0
                self._reset_candidate()
            elif now - self.last_motion_time >= self.active_timeout_seconds:
                self.state = "IDLE"
                self._reset_candidate()

        else:  # COOLDOWN
            self._reset_candidate()
            self.quiet_streak = self.quiet_streak + 1 if quiet else 0
            if now >= self.cooldown_until and self.quiet_streak >= self.rearm_quiet_count:
                self.state = "IDLE"
                self.quiet_streak = 0
                self.motion_streak = 0

        accepted = bool(
            self.state == "ACTIVE" and window_ready and valid and prediction is not None
            and probability >= self.probability_threshold
        )
        return GateResult(
            state=self.state,
            candidate=self.candidate,
            streak=self.streak,
            displayed_event=self.displayed_event,
            emitted_event=emitted,
            accepted=accepted,
            moving=moving,
            quiet=quiet,
        )

