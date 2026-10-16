"""Speed benchmark following the D20 protocol.

Per frame, with CUDA synchronisation at the boundaries::

    t0 -> read + JPEG decode -> t1 -> detect (+ loose NMS) -> filter -> track -> t2

* perception latency = t2 - t1 (preprocess + inference + postprocess + tracking,
  from a decoded frame in memory; the D20 real-time criterion uses its p95);
* end-to-end latency = t2 - t0 (adds read / decode);
* throughput = timed frames / wall time of the timed loop (sequential loop, no
  pipelining).

A pipelined system (decode, inference and tracking as parallel stages joined by
bounded queues that drop frames when full) can reach a throughput above
1 / latency while per-frame latency stays the same or grows; this harness
measures the sequential loop only, so its throughput is roughly 1 / mean
end-to-end latency.

Rendering and file writing are not part of the loop. Warm-up: the first
``warmup`` frames run through detector and tracker, then the tracker is reset
and the whole sequence is timed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from cvdt.data.base import SequenceDataset
from cvdt.detect.yolo import YOLODetector
from cvdt.track.base import Tracker
from cvdt.types import Detections

REAL_TIME_P95_MS = 1000.0 / 30.0  # D20: 30 FPS, the highest MOT17 source frame rate


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _filter(det: Detections, min_conf: float) -> Detections:
    keep = det.scores >= min_conf
    return Detections(boxes=det.boxes[keep], scores=det.scores[keep], classes=det.classes[keep])


def stats(ms: list[float]) -> dict[str, float]:
    v = np.asarray(ms, dtype=np.float64)
    return {
        "n": int(len(v)),
        "p50_ms": round(float(np.percentile(v, 50)), 3),
        "p95_ms": round(float(np.percentile(v, 95)), 3),
        "p99_ms": round(float(np.percentile(v, 99)), 3),
        "mean_ms": round(float(v.mean()), 3),
        "max_ms": round(float(v.max()), 3),
    }


@dataclass
class SeqBench:
    perception_ms: list[float]
    end_to_end_ms: list[float]
    decode_ms: list[float]
    stage_ms: dict[str, list[float]]
    wall_s: float
    peak_mem_allocated_mb: float
    peak_mem_reserved_mb: float


def bench_sequence(
    ds: SequenceDataset,
    detector: YOLODetector,
    tracker: Tracker,
    tracker_input_conf: float,
    loose_iou: float | None,
    warmup: int = 50,
) -> SeqBench:
    import torch

    def step(img):
        if loose_iou is not None:
            det, loose = detector.detect_with_loose(img, loose_iou)
            loose = _filter(loose, tracker_input_conf)
        else:
            det, loose = detector.detect(img), None
        det = _filter(det, tracker_input_conf)
        t_track = time.perf_counter()
        tracker.update(det, img if tracker.needs_frame else None, loose=loose)
        return t_track

    tracker.reset()
    for idx in range(1, min(warmup, len(ds)) + 1):
        step(ds.frame(idx).load())
    tracker.reset()
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    perception, e2e, decode = [], [], []
    stage_ms: dict[str, list[float]] = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
        "track": [],
    }
    t_start = time.perf_counter()
    for idx in range(1, len(ds) + 1):
        fr = ds.frame(idx)
        _sync()
        t0 = time.perf_counter()
        img = fr.load()
        t1 = time.perf_counter()
        t_track = step(img)
        _sync()
        t2 = time.perf_counter()
        decode.append((t1 - t0) * 1e3)
        perception.append((t2 - t1) * 1e3)
        e2e.append((t2 - t0) * 1e3)
        for k in ("preprocess", "inference", "postprocess"):
            stage_ms[k].append(float(detector.last_speed.get(k, float("nan"))))
        stage_ms["track"].append((t2 - t_track) * 1e3)
    wall = time.perf_counter() - t_start
    mb = 1024**2
    alloc = torch.cuda.max_memory_allocated() / mb if torch.cuda.is_available() else 0.0
    reserved = torch.cuda.max_memory_reserved() / mb if torch.cuda.is_available() else 0.0
    return SeqBench(perception, e2e, decode, stage_ms, wall, round(alloc, 1), round(reserved, 1))


def summarize(b: SeqBench) -> dict:
    p = stats(b.perception_ms)
    return {
        "perception": p,
        "end_to_end": stats(b.end_to_end_ms),
        "read_decode": stats(b.decode_ms),
        "stages_ultralytics_profiler_and_track": {k: stats(v) for k, v in b.stage_ms.items()},
        "throughput_fps": round(len(b.perception_ms) / b.wall_s, 2),
        "peak_gpu_mem_allocated_mb": b.peak_mem_allocated_mb,
        "peak_gpu_mem_reserved_mb": b.peak_mem_reserved_mb,
        "real_time_threshold_p95_ms": round(REAL_TIME_P95_MS, 2),
        "real_time": bool(p["p95_ms"] <= REAL_TIME_P95_MS),
    }
