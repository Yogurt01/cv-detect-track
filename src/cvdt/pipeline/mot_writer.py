"""Write tracker output in MOTChallenge format.

One line per track and frame::

    frame,id,bb_left,bb_top,width,height,conf,-1,-1,-1

Both the frame index and the pixel coordinates are 1-based (MOT16 paper,
Tab. 5; README "Licenses & data" B3). Internal boxes are 0-based xyxy
(:mod:`cvdt.types`) and are converted with :func:`cvdt.data.mot.xyxy_to_mot`,
which adds 1 to left/top.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cvdt.data.mot import xyxy_to_mot
from cvdt.types import Tracks


def format_rows(frame: int, tracks: Tracks) -> list[str]:
    if frame < 1:
        raise ValueError("MOT frame indices are 1-based")
    if len(tracks) == 0:
        return []
    xywh = xyxy_to_mot(tracks.boxes)
    return [
        f"{frame},{int(i)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1"
        for i, (x, y, w, h), s in zip(tracks.ids, xywh, tracks.scores, strict=True)
    ]


def write_mot(path: Path, per_frame: Iterable[tuple[int, Tracks]]) -> int:
    """Write ``(frame, tracks)`` pairs to ``path``; returns the number of lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for frame, tracks in per_frame:
            for line in format_rows(frame, tracks):
                f.write(line + "\n")
                n += 1
    return n


def read_mot(path: Path) -> np.ndarray:
    """Read a MOT result file as a float array (K, 10); empty files give (0, 10)."""
    text = Path(path).read_text().strip()
    if not text:
        return np.zeros((0, 10))
    return np.array([[float(v) for v in line.split(",")] for line in text.splitlines()])
