"""Detector interface.

This package is the only place allowed to import a detection framework
(currently Ultralytics, AGPL-3.0). Tracking, evaluation and the pipeline depend
on :class:`Detector` only, so the backend can be swapped for an exported
ONNX/TensorRT engine or a differently licensed model without touching them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from cvdt.types import Detections


@runtime_checkable
class Detector(Protocol):
    """Single-image object detector.

    Implementations own their pre- and post-processing (resize, normalisation,
    NMS, score threshold) and must return boxes in the coordinates of the
    input frame, not of the network input.
    """

    name: str
    """Short identifier used in configs and result files, e.g. ``"yolo26n-coco-640"``."""

    def detect(self, frame: np.ndarray) -> Detections:
        """Detect objects in one frame.

        Args:
            frame: (H, W, 3) uint8 image in BGR order (OpenCV convention).

        Returns:
            Detections in xyxy pixel coordinates of ``frame``.
        """
        ...

    def warmup(self, shape: tuple[int, int, int], n: int = 1) -> None:
        """Run ``n`` dummy inferences on an input of ``shape`` (H, W, C) so that
        lazy initialisation and kernel autotuning do not pollute latency numbers."""
        ...
