"""MOT16/MOT17 file formats and the MOT17 :class:`SequenceDataset`.

Ground-truth ``gt.txt`` columns (Milan et al., MOT16, arXiv:1603.00831, Tab. 5):

    frame, id, bb_left, bb_top, width, height, conf_flag, class, visibility

* ``frame`` is 1-based.
* Box coordinates are 1-based: the top-left image corner is (1, 1).
* ``conf_flag`` in GT is a flag: 1 = consider in evaluation, 0 = ignore.
* ``class`` follows :class:`MOTClass` (Tab. 6).
* ``visibility`` in [0, 1] is computed from occlusion by other annotated
  objects and from cropping at the image border.
* Boxes are full extent: occluded people are annotated with their estimated
  full body box, and boxes may extend beyond the image (Sec. 2.2).

Inside ``cvdt`` boxes are 0-based xyxy (see :mod:`cvdt.types`), so this module
subtracts 1 when reading. Anything written back in MOT format (tracker output)
must add 1 again; :func:`xyxy_to_mot` does that.
"""

from __future__ import annotations

import configparser
from collections.abc import Iterator
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from cvdt.data.base import CameraType, Frame, FrameAnnotations, SequenceInfo

GT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]


class MOTClass(IntEnum):
    """MOT16/MOT17 ground-truth classes (MOT16 paper, Tab. 6)."""

    PEDESTRIAN = 1
    PERSON_ON_VEHICLE = 2
    CAR = 3
    BICYCLE = 4
    MOTORBIKE = 5
    NON_MOTORIZED_VEHICLE = 6
    STATIC_PERSON = 7
    DISTRACTOR = 8
    OCCLUDER = 9
    OCCLUDER_ON_GROUND = 10
    OCCLUDER_FULL = 11
    REFLECTION = 12


def read_seqinfo(path: Path) -> dict[str, str]:
    """Parse ``seqinfo.ini`` into a flat dict (keys as in the file)."""
    cp = configparser.ConfigParser()
    cp.optionxform = str  # keep key case (frameRate, imWidth, ...)
    cp.read(path)
    return dict(cp["Sequence"])


def load_gt(path: Path) -> pd.DataFrame:
    """Load a MOT ``gt.txt`` with typed columns :data:`GT_COLUMNS` (raw MOT convention)."""
    df = pd.read_csv(path, header=None, names=GT_COLUMNS)
    return df.astype(
        {
            "frame": int,
            "id": int,
            "x": float,
            "y": float,
            "w": float,
            "h": float,
            "conf": int,
            "cls": int,
            "vis": float,
        }
    )


def mot_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """MOT 1-based (left, top, w, h) -> 0-based xyxy."""
    xywh = np.asarray(xywh, dtype=np.float64).reshape(-1, 4)
    x0 = xywh[:, 0] - 1.0
    y0 = xywh[:, 1] - 1.0
    return np.stack([x0, y0, x0 + xywh[:, 2], y0 + xywh[:, 3]], axis=1)


def xyxy_to_mot(xyxy: np.ndarray) -> np.ndarray:
    """0-based xyxy -> MOT 1-based (left, top, w, h). Inverse of :func:`mot_to_xyxy`."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return np.stack(
        [xyxy[:, 0] + 1.0, xyxy[:, 1] + 1.0, xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]],
        axis=1,
    )


class MOT17Sequence:
    """One canonical (deduplicated) MOT17 train sequence.

    Expected layout (produced by ``scripts/prepare_data.py dedupe``)::

        <root>/MOT17-02/seqinfo.ini
        <root>/MOT17-02/img1/000001.jpg ...
        <root>/MOT17-02/gt/gt.txt

    Ground truth is returned unfiltered (all classes, flags and visibilities);
    filtering for training or detection evaluation is :mod:`cvdt.data.filtering`'s job,
    and tracking evaluation applies TrackEval's own rules to the raw file.
    """

    def __init__(self, seq_dir: Path, camera: CameraType, extra: dict[str, str] | None = None):
        self.seq_dir = Path(seq_dir)
        si = read_seqinfo(self.seq_dir / "seqinfo.ini")
        self._info = SequenceInfo(
            name=self.seq_dir.name,
            dataset="MOT17",
            num_frames=int(si["seqLength"]),
            width=int(si["imWidth"]),
            height=int(si["imHeight"]),
            fps=float(si["frameRate"]),
            camera=camera,
            image_dir=self.seq_dir / si.get("imDir", "img1"),
            extra=dict(extra or {}),
        )
        self._ext = si.get("imExt", ".jpg")
        gt_path = self.seq_dir / "gt" / "gt.txt"
        self.gt_df = load_gt(gt_path) if gt_path.exists() else None
        self._by_frame = (
            {int(f): g for f, g in self.gt_df.groupby("frame")} if self.gt_df is not None else {}
        )

    @property
    def info(self) -> SequenceInfo:
        return self._info

    def __len__(self) -> int:
        return self._info.num_frames

    def __iter__(self) -> Iterator[Frame]:
        for i in range(1, self._info.num_frames + 1):
            yield self.frame(i)

    def image_path(self, index: int) -> Path:
        return self._info.image_dir / f"{index:06d}{self._ext}"

    def frame(self, index: int) -> Frame:
        if not 1 <= index <= self._info.num_frames:
            raise IndexError(f"frame {index} out of range 1..{self._info.num_frames}")
        gt = None
        if self.gt_df is not None:
            g = self._by_frame.get(index)
            if g is None:
                g = self.gt_df.iloc[0:0]
            gt = FrameAnnotations(
                boxes=mot_to_xyxy(g[["x", "y", "w", "h"]].to_numpy()).astype(np.float32),
                ids=g["id"].to_numpy(np.int64),
                classes=g["cls"].to_numpy(np.int64),
                attrs={
                    "conf_flag": g["conf"].to_numpy(np.int64),
                    "visibility": g["vis"].to_numpy(np.float32),
                },
            )
        return Frame(index=index, image_path=self.image_path(index), gt=gt)
