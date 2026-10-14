"""Data containers shared by detectors, trackers, datasets and evaluation.

Box convention everywhere inside ``cvdt``: ``xyxy`` in float pixels, 0-based,
continuous coordinates (the top-left corner of the top-left pixel is (0, 0)).
Dataset-specific conventions (e.g. MOT's 1-based ``xywh``) are converted at
the dataset boundary, never inside detectors or trackers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _empty_boxes() -> np.ndarray:
    return np.zeros((0, 4), dtype=np.float32)


@dataclass(frozen=True)
class Detections:
    """Detections for one frame.

    Attributes:
        boxes: (N, 4) float32, xyxy pixels.
        scores: (N,) float32 confidence in [0, 1].
        classes: (N,) int64 class ids in the detector's own label space.
    """

    boxes: np.ndarray = field(default_factory=_empty_boxes)
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    classes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass(frozen=True)
class Tracks:
    """Confirmed tracks output by a tracker for one frame.

    Attributes:
        boxes: (M, 4) float32, xyxy pixels.
        ids: (M,) int64 track ids, unique within a frame and stable across frames.
        scores: (M,) float32 score of the detection associated in this frame.
        classes: (M,) int64 class ids.
    """

    boxes: np.ndarray = field(default_factory=_empty_boxes)
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    classes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.boxes)
