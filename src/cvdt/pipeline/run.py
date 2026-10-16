"""Run loop: frames -> detections (live or cached) -> tracker -> MOT txt.

The loop depends only on the interfaces in :mod:`cvdt.data.base`,
:mod:`cvdt.detect.base` and :mod:`cvdt.track.base` plus the detection cache
reader; it never imports a detection framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cvdt.data.base import SequenceDataset
from cvdt.detect.base import Detector
from cvdt.detect.cache import CachedSequence
from cvdt.pipeline.mot_writer import format_rows
from cvdt.track.base import Tracker
from cvdt.types import Detections, Tracks
from cvdt.utils.timing import StageTimer

# Tracker input threshold: the default ``conf`` of Ultralytics' tracking path (D18).
TRACKER_INPUT_CONF = 0.1


def _filter(det: Detections, min_conf: float) -> Detections:
    keep = det.scores >= min_conf
    return Detections(boxes=det.boxes[keep], scores=det.scores[keep], classes=det.classes[keep])


@dataclass
class RunResult:
    sequence: str
    frames: int
    mot_lines: int
    tracks_per_frame: list[Tracks] = field(repr=False)
    timer: StageTimer = field(repr=False)


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id))
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def draw_tracks(img: np.ndarray, tracks: Tracks) -> np.ndarray:
    for (x0, y0, x1, y1), tid in zip(tracks.boxes, tracks.ids, strict=True):
        c = _color(int(tid))
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, 2)
        cv2.putText(
            img,
            str(int(tid)),
            (int(x0), max(int(y0) - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            c,
            2,
            cv2.LINE_AA,
        )
    return img


def run_sequence(
    dataset: SequenceDataset,
    tracker: Tracker,
    out_txt: Path,
    *,
    detector: Detector | None = None,
    cache: CachedSequence | None = None,
    tracker_input_conf: float = TRACKER_INPUT_CONF,
    loose_iou: float | None = None,
    video_path: Path | None = None,
    max_frames: int | None = None,
) -> RunResult:
    """Track one sequence and write ``out_txt`` (MOT format).

    Exactly one of ``detector`` (live) or ``cache`` must be given. In live mode
    the detector is expected to run at ``tracker_input_conf`` already; the filter
    is applied in both modes so the two paths are interchangeable.
    ``loose_iou`` enables the live loose-NMS pass for TrackTrack (D16); it
    re-runs only NMS on the same forward pass (``detect_with_loose``).
    """
    if (detector is None) == (cache is None):
        raise ValueError("pass exactly one of detector= or cache=")
    info = dataset.info
    n = min(len(dataset), max_frames or len(dataset))
    timer = StageTimer()
    tracker.reset()
    need_image = detector is not None or tracker.needs_frame or video_path is not None
    writer = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
        )

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    all_tracks: list[Tracks] = []
    lines = 0
    with open(out_txt, "w") as fh:
        for idx in range(1, n + 1):
            t_frame = time.perf_counter()
            fr = dataset.frame(idx)
            img = None
            if need_image:
                with timer.stage("read"):
                    img = fr.load()
            loose = None
            if detector is not None:
                with timer.stage("detect_total"):  # wall clock incl. framework overhead
                    if loose_iou is not None:  # one forward pass, NMS re-run at loose_iou (D16)
                        det, loose = detector.detect_with_loose(img, loose_iou)
                    else:
                        det = detector.detect(img)
                for k in ("preprocess", "inference", "postprocess"):
                    if k in getattr(detector, "last_speed", {}):
                        timer.add(k, detector.last_speed[k])
            else:
                det = cache.tight(idx)
                loose = cache.loose(idx)
            det = _filter(det, tracker_input_conf)
            if loose is not None:
                loose = _filter(loose, tracker_input_conf)
            with timer.stage("track"):
                tracks = tracker.update(det, img if tracker.needs_frame else None, loose=loose)
            all_tracks.append(tracks)
            with timer.stage("write"):
                rows = format_rows(idx, tracks)
                for r in rows:
                    fh.write(r + "\n")
                lines += len(rows)
            if writer is not None:
                with timer.stage("render"):
                    writer.write(draw_tracks(img.copy(), tracks))
            timer.add("frame_total", (time.perf_counter() - t_frame) * 1e3)
    if writer is not None:
        writer.release()
    return RunResult(info.name, n, lines, all_tracks, timer)
