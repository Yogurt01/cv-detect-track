"""Tracker interface.

Both library adapters and the project's own ByteTrack-style tracker implement
:class:`Tracker`. A tracker sees detections only (plus, optionally, the frame
for trackers that need appearance or camera-motion compensation), never the
detector itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from cvdt.types import Detections, Tracks


@runtime_checkable
class Tracker(Protocol):
    """Online multi-object tracker, fed one frame at a time in temporal order."""

    name: str
    """Short identifier used in configs and result files, e.g. ``"bytetrack-own"``."""

    def update(self, detections: Detections, frame: np.ndarray | None = None) -> Tracks:
        """Associate this frame's detections with existing tracks.

        Args:
            detections: Detections of the current frame, xyxy pixels.
            frame: Optional (H, W, 3) BGR image, for trackers that use appearance
                features or global motion compensation. Motion-only trackers
                ignore it.

        Returns:
            Tracks that are output for this frame (confirmed tracks matched in
            this frame, according to the tracker's own output policy).
        """
        ...

    def reset(self) -> None:
        """Drop all state and restart id assignment; call between sequences."""
        ...
