"""Per-stage wall-clock timing with optional CUDA synchronisation.

Stages used by the pipeline: ``read``, ``preprocess``, ``inference``,
``postprocess``, ``detect_total``, ``track``, ``render``, ``write`` and
``frame_total``. Detector-internal stages (preprocess / inference /
postprocess) come from Ultralytics' own profiler, which synchronises CUDA;
``detect_total`` is the wall clock of the whole detector call, including
framework overhead those three do not cover. ``frame_total`` is the end-to-end
wall clock per frame. This is a diagnostic tool, not the formal speed protocol.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class StageTimer:
    """Collects per-frame durations (ms) per stage."""

    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync
        self.samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        if self.cuda_sync:
            _cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda_sync:
                _cuda_sync()
            self.samples[name].append((time.perf_counter() - t0) * 1e3)

    def add(self, name: str, ms: float) -> None:
        self.samples[name].append(float(ms))

    def summary(self, skip_first: int = 0) -> dict[str, dict[str, float]]:
        """Median / p95 / mean (ms) per stage, ignoring the first ``skip_first`` samples."""
        out = {}
        for name, vals in self.samples.items():
            v = np.asarray(vals[skip_first:])
            if len(v):
                out[name] = {
                    "n": len(v),
                    "median_ms": round(float(np.median(v)), 3),
                    "p95_ms": round(float(np.percentile(v, 95)), 3),
                    "mean_ms": round(float(v.mean()), 3),
                }
        return out
