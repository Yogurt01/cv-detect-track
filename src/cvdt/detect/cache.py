"""Detection cache (DECISIONS.md D15, D16).

One directory per cache key, one ``.npz`` file per sequence::

    outputs/detections/<key_dir>/meta.json
    outputs/detections/<key_dir>/<sequence>.npz

``<key_dir>`` is ``<weights stem>_<imgsz>_<precision>_conf<floor>_<hash>``, where
``hash`` is the first 12 hex digits of the SHA256 of the canonical JSON of the
full key (weights SHA256, imgsz, precision, conf floor, NMS IoU, loose-NMS IoU,
classes, max_det). The sequence is the file name.

Each ``.npz`` holds, for the tight (``t_``) and loose (``l_``) NMS passes:
``frame`` (int32, 1-based), ``boxes`` (float32 N x 4, 0-based xyxy pixels, the
convention of :mod:`cvdt.types`), ``scores`` (float32), ``classes`` (int64), and
``count`` (int32 per frame, before any consumer threshold). Consumers apply
their own confidence thresholds on read.

This module does not import any detection framework.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cvdt.types import Detections

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPO_ROOT / "outputs" / "detections"


@dataclass(frozen=True)
class CacheKey:
    weights_sha256: str
    weights_stem: str
    imgsz: int
    precision: str  # "fp32" | "fp16"
    conf_floor: float
    nms_iou: float
    loose_nms_iou: float
    classes: tuple[int, ...]
    max_det: int

    def digest(self) -> str:
        d = asdict(self)
        d.pop("weights_stem")  # informational only; the SHA256 identifies the weights
        blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def dirname(self) -> str:
        return (
            f"{self.weights_stem}_{self.imgsz}_{self.precision}_conf{self.conf_floor:g}_"
            f"{self.digest()}"
        )


def key_dir(key: CacheKey, root: Path = CACHE_ROOT) -> Path:
    return root / key.dirname()


def _pack(prefix: str, per_frame: list[Detections], n_frames: int) -> dict[str, np.ndarray]:
    counts = np.array([len(d) for d in per_frame], dtype=np.int32)
    frames = np.repeat(np.arange(1, n_frames + 1, dtype=np.int32), counts)

    def cat(attr, shape, dtype):
        parts = [getattr(d, attr) for d in per_frame if len(d)]
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(shape, dtype)

    return {
        f"{prefix}frame": frames,
        f"{prefix}boxes": cat("boxes", (0, 4), np.float32),
        f"{prefix}scores": cat("scores", (0,), np.float32),
        f"{prefix}classes": cat("classes", (0,), np.int64),
        f"{prefix}count": counts,
    }


def write_sequence(
    path: Path, tight: list[Detections], loose: list[Detections] | None, n_frames: int
) -> None:
    arrays = _pack("t_", tight, n_frames)
    if loose is not None:
        arrays |= _pack("l_", loose, n_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)


class CachedSequence:
    """Read access to one cached sequence, with a consumer confidence threshold."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with np.load(self.path) as z:
            self._z = {k: z[k] for k in z.files}
        self.n_frames = len(self._z["t_count"])
        self.has_loose = "l_frame" in self._z

    def _get(self, prefix: str, frame: int, min_conf: float) -> Detections:
        fr = self._z[f"{prefix}frame"]
        lo, hi = np.searchsorted(fr, frame, "left"), np.searchsorted(fr, frame, "right")
        s = self._z[f"{prefix}scores"][lo:hi]
        keep = s >= min_conf
        return Detections(
            boxes=self._z[f"{prefix}boxes"][lo:hi][keep],
            scores=s[keep],
            classes=self._z[f"{prefix}classes"][lo:hi][keep],
        )

    def tight(self, frame: int, min_conf: float = 0.0) -> Detections:
        return self._get("t_", frame, min_conf)

    def loose(self, frame: int, min_conf: float = 0.0) -> Detections | None:
        return self._get("l_", frame, min_conf) if self.has_loose else None

    def max_count(self, prefix: str = "t_") -> int:
        return int(self._z[f"{prefix}count"].max(initial=0))


def write_meta(directory: Path, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_meta(directory: Path) -> dict:
    return json.loads((directory / "meta.json").read_text())
