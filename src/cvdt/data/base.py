"""Dataset interface: a video sequence with optional per-frame ground truth.

Dataset-specific details (file layout, class tables, coordinate conventions)
stay inside each implementation; consumers see :class:`SequenceDataset` only.
This keeps a later small-object / drone dataset a new implementation rather
than a refactor.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import cv2
import numpy as np

CameraType = Literal["static", "moving"]


@dataclass(frozen=True)
class SequenceInfo:
    """Static metadata of one sequence."""

    name: str  # e.g. "MOT17-02"
    dataset: str  # e.g. "MOT17"
    num_frames: int
    width: int
    height: int
    fps: float
    camera: CameraType
    image_dir: Path
    extra: Mapping[str, str] = field(default_factory=dict)  # e.g. viewpoint, conditions


@dataclass(frozen=True)
class FrameAnnotations:
    """Ground truth of one frame, in the dataset's own label space.

    Attributes:
        boxes: (K, 4) float32, xyxy 0-based pixels; may extend beyond the image
            for datasets with full-extent (amodal) boxes.
        ids: (K,) int64 object ids.
        classes: (K,) int64 dataset class ids.
        attrs: dataset-specific per-box columns, e.g. ``{"conf_flag": ..., "visibility": ...}``.
    """

    boxes: np.ndarray
    ids: np.ndarray
    classes: np.ndarray
    attrs: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass(frozen=True)
class Frame:
    """One frame of a sequence. The image is loaded lazily via :meth:`load`."""

    index: int  # 1-based frame number, as in MOT files
    image_path: Path
    gt: FrameAnnotations | None

    def load(self) -> np.ndarray:
        """Read the image as (H, W, 3) uint8 BGR."""
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"cannot read image {self.image_path}")
        return img


@runtime_checkable
class SequenceDataset(Protocol):
    """A single video sequence: ordered frames plus metadata and ground truth."""

    @property
    def info(self) -> SequenceInfo: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames in temporal order."""
        ...

    def frame(self, index: int) -> Frame:
        """Random access by 1-based frame number."""
        ...
