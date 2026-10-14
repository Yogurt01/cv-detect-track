"""MOT ground truth -> Ultralytics YOLO detection labels plus ignore sidecars.

Output layout, one directory per sequence so that split configs can list
sequences directly::

    <out>/MOT17-02/images/000001.jpg   hard link to the canonical image
    <out>/MOT17-02/labels/000001.txt   "0 cx cy w h" per POSITIVE box, normalised
    <out>/MOT17-02/ignore.csv          frame,x0,y0,x1,y1,mot_class,visibility,reason

Ultralytics finds labels by replacing ``/images/`` with ``/labels/`` in the image
path, so images are hard links (not symlinks, which could be resolved to the
canonical path). Every frame gets a label file; an empty file marks a frame
with no positives as background.

Boxes are clipped to the image before normalising (MOT boxes are full extent
and may extend past the border). A clipped box narrower or shorter than
``MIN_CLIPPED_SIDE`` pixels is not written.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from cvdt.data import filtering
from cvdt.data.filtering import Category
from cvdt.data.mot import load_gt, mot_to_xyxy, read_seqinfo, xyxy_to_mot

CLASS_NAMES = {0: "pedestrian"}
MIN_CLIPPED_SIDE = 2.0


def clip_xyxy(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)
    return xyxy


def xyxy_to_yolo(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """0-based xyxy pixels -> normalised (cx, cy, w, h). Clips to the image first."""
    b = clip_xyxy(xyxy, width, height)
    return np.stack(
        [
            (b[:, 0] + b[:, 2]) / 2 / width,
            (b[:, 1] + b[:, 3]) / 2 / height,
            (b[:, 2] - b[:, 0]) / width,
            (b[:, 3] - b[:, 1]) / height,
        ],
        axis=1,
    )


def yolo_to_xyxy(cxcywh: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised (cx, cy, w, h) -> 0-based xyxy pixels."""
    b = np.asarray(cxcywh, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] * width, b[:, 1] * height, b[:, 2] * width, b[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def mot_to_yolo(mot_xywh: np.ndarray, width: int, height: int) -> np.ndarray:
    return xyxy_to_yolo(mot_to_xyxy(mot_xywh), width, height)


def yolo_to_mot(cxcywh: np.ndarray, width: int, height: int) -> np.ndarray:
    return xyxy_to_mot(yolo_to_xyxy(cxcywh, width, height))


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_sequence(seq_dir: Path, out_dir: Path, min_visibility: float) -> dict:
    """Convert one canonical MOT17 sequence. Returns per-sequence counts."""
    si = read_seqinfo(seq_dir / "seqinfo.ini")
    W, H, n_frames = int(si["imWidth"]), int(si["imHeight"]), int(si["seqLength"])
    ext = si.get("imExt", ".jpg")
    img_src = seq_dir / si.get("imDir", "img1")
    df = load_gt(seq_dir / "gt" / "gt.txt")

    cat = filtering.categorize(
        df["cls"].to_numpy(), df["conf"].to_numpy(), df["vis"].to_numpy(), min_visibility
    )
    reason = filtering.ignore_reason(df["cls"].to_numpy(), cat)
    xyxy = mot_to_xyxy(df[["x", "y", "w", "h"]].to_numpy())
    clipped = clip_xyxy(xyxy, W, H)
    side_ok = ((clipped[:, 2] - clipped[:, 0]) >= MIN_CLIPPED_SIDE) & (
        (clipped[:, 3] - clipped[:, 1]) >= MIN_CLIPPED_SIDE
    )
    write_pos = (cat == Category.POSITIVE) & side_ok
    yolo = xyxy_to_yolo(xyxy, W, H)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    frames = df["frame"].to_numpy()
    pos_idx = np.flatnonzero(write_pos)
    by_frame: dict[int, list[int]] = {}
    for i in pos_idx:
        by_frame.setdefault(int(frames[i]), []).append(int(i))
    for f in range(1, n_frames + 1):
        stem = f"{f:06d}"
        _link(img_src / f"{stem}{ext}", out_dir / "images" / f"{stem}{ext}")
        lines = ["0 {:.6f} {:.6f} {:.6f} {:.6f}".format(*yolo[i]) for i in by_frame.get(f, [])]
        (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    ign = np.flatnonzero((cat == Category.IGNORE) | (cat == Category.LOW_VISIBILITY))
    sidecar = pd.DataFrame(
        {
            "frame": frames[ign],
            "x0": xyxy[ign, 0].round(2),
            "y0": xyxy[ign, 1].round(2),
            "x1": xyxy[ign, 2].round(2),
            "y1": xyxy[ign, 3].round(2),
            "mot_class": df["cls"].to_numpy()[ign],
            "visibility": df["vis"].to_numpy()[ign],
            "reason": reason[ign],
        }
    ).sort_values(["frame", "reason"], kind="stable")
    sidecar.to_csv(out_dir / "ignore.csv", index=False)

    counts = {
        "frames": n_frames,
        "positive": int((cat == Category.POSITIVE).sum()),
        "positive_written": int(write_pos.sum()),
        "positive_degenerate_after_clip": int(((cat == Category.POSITIVE) & ~side_ok).sum()),
        "positive_extends_beyond_image": int(
            ((cat == Category.POSITIVE) & np.any(clipped != xyxy, axis=1)).sum()
        ),
        "low_visibility": int((cat == Category.LOW_VISIBILITY).sum()),
        "ignore_distractor_class": int((reason == filtering.REASON_DISTRACTOR).sum()),
        "ignore_zero_flag": int((reason == filtering.REASON_ZERO_FLAG).sum()),
        "dropped": int((cat == Category.DROPPED).sum()),
        "frames_without_positive": int(n_frames - len(by_frame)),
    }
    return counts


def convert_all(canon_root: Path, out_root: Path, min_visibility: float) -> dict:
    """Convert every sequence under ``canon_root``; writes ``<out_root>/conversion.json``."""
    seqs = sorted(p for p in canon_root.iterdir() if p.is_dir() and p.name.startswith("MOT17-"))
    if not seqs:
        raise FileNotFoundError(f"no canonical sequences in {canon_root}; run dedupe first")
    summary = {
        "min_visibility": min_visibility,
        "min_clipped_side_px": MIN_CLIPPED_SIDE,
        "class_names": CLASS_NAMES,
        "sequences": {s.name: convert_sequence(s, out_root / s.name, min_visibility) for s in seqs},
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
