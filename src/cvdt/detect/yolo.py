"""Ultralytics YOLO26 behind the :class:`~cvdt.detect.base.Detector` interface.

Inference mode (verified in ultralytics 8.4.177, DECISIONS.md D16)
-------------------------------------------------------------------
YOLO26 checkpoints contain two heads. With the predictor's default ``nms=None``
("external NMS") Ultralytics runs the one-to-many head followed by standard
class-aware NMS at ``iou`` (default 0.7); the NMS-free one-to-one head is used
only with ``nms=False``. ``model.track()`` uses the default as well, so this
wrapper keeps ``nms=None``.

Consequences for low-confidence detections: candidates are selected by their
best class score over all 80 classes and labelled with that class before the
person filter, and NMS is greedy by score. Kept boxes above a threshold t do not
depend on candidates below t, so filtering a low-floor result (e.g. 0.01) at t
gives the same boxes as running at t directly. ``max_det`` (300) truncates
after NMS and the class filter.

Coordinates: boxes are returned in the input frame's pixel coordinates, 0-based
continuous xyxy (:mod:`cvdt.types`), exactly as Ultralytics reports them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from cvdt.types import Detections

COCO_PERSON = 0


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _boxes_to_detections(b) -> Detections:
    return Detections(
        boxes=b.xyxy.cpu().numpy().astype(np.float32),
        scores=b.conf.cpu().numpy().astype(np.float32),
        classes=b.cls.cpu().numpy().astype(np.int64),
    )


class YOLODetector:
    """YOLO26 detector, person class only by default.

    Args:
        weights: Path to a local ``.pt`` file. Nothing is downloaded: a missing
            file raises instead of triggering Ultralytics' auto-download.
        imgsz: Inference size (letterboxed long side).
        conf: Confidence floor passed to Ultralytics.
        iou: NMS IoU threshold.
        half: FP16 inference (``False`` = FP32).
        device: CUDA index or ``"cpu"``.
        classes: COCO class ids to keep.
        max_det: Maximum detections per image after NMS.
        nms: Ultralytics ``nms`` argument. ``None`` (default) = one-to-many head
            + NMS (D16); ``False`` = NMS-free one-to-one head (speed-study variant).

    Loose-NMS pass (TrackTrack TAI, D16): :meth:`detect_with_loose` runs ONE
    forward pass and re-runs only NMS at ``loose_iou`` on the same raw
    predictions, like ``model.track()`` does. It wraps the predictor's
    ``postprocess`` to keep a clone of the raw predictions (NMS converts boxes in
    place). ``forward_passes`` counts model forward calls for verification.
    """

    def __init__(
        self,
        weights: str | Path,
        imgsz: int = 640,
        conf: float = 0.01,
        iou: float = 0.7,
        half: bool = False,
        device: int | str = 0,
        classes: tuple[int, ...] = (COCO_PERSON,),
        max_det: int = 300,
        nms: bool | None = None,
    ):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"weights not found: {weights}")
        self.weights = weights
        self.weights_sha256 = file_sha256(weights)
        self.model = YOLO(str(weights))
        self.imgsz, self.conf, self.iou, self.half = imgsz, conf, iou, half
        self.device, self.classes, self.max_det, self.nms = device, list(classes), max_det, nms
        head = "nmsfree" if nms is False else "nms"
        self.name = f"{weights.stem}-{imgsz}-{'fp16' if half else 'fp32'}-{head}"
        self.last_speed: dict[str, float] = {}
        self.forward_passes = 0
        self._wrapped = None  # predictor whose postprocess is wrapped
        self._orig_postprocess = None
        self._raw = None

    def _predict_results(self, frame: np.ndarray):
        r = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            quantize=16 if self.half else 32,  # replaces the deprecated `half` flag
            device=self.device,
            classes=self.classes,
            max_det=self.max_det,
            nms=self.nms,
            verbose=False,
        )[0]
        # Ultralytics' own per-stage timings (ms, CUDA-synchronised by its profiler).
        self.last_speed = dict(r.speed)
        return r

    def _ensure_hooks(self, frame: np.ndarray) -> None:
        """Wrap postprocess and count forward passes on the current predictor."""
        pred = self.model.predictor
        if pred is None:
            self._predict_results(frame)  # builds the predictor (once per detector)
            pred = self.model.predictor
        if self._wrapped is pred:
            return
        orig = pred.postprocess

        def wrapped(preds, img, orig_imgs, *args, **kwargs):
            raw = preds[0] if isinstance(preds, (list, tuple)) else preds
            self._raw = (raw.detach().clone(), img, orig_imgs)
            return orig(preds, img, orig_imgs, *args, **kwargs)

        def count(_module, _inputs):
            self.forward_passes += 1

        pred.postprocess = wrapped
        pred.model.register_forward_pre_hook(count)
        self._wrapped, self._orig_postprocess = pred, orig

    def detect(self, frame: np.ndarray) -> Detections:
        """Detect in one BGR frame with the configured confidence floor and NMS IoU."""
        self._ensure_hooks(frame)
        return _boxes_to_detections(self._predict_results(frame).boxes)

    def detect_with_loose(
        self, frame: np.ndarray, loose_iou: float
    ) -> tuple[Detections, Detections]:
        """Tight detections plus a loose-NMS pass from the same forward pass (D16)."""
        self._ensure_hooks(frame)
        self._raw = None
        tight = _boxes_to_detections(self._predict_results(frame).boxes)
        if self._raw is None or self.model.predictor is not self._wrapped:
            raise RuntimeError("raw predictions were not captured; the predictor was rebuilt")
        raw, img, orig_imgs = self._raw
        loose = self._orig_postprocess(raw, img, orig_imgs, iou=loose_iou)[0]
        return tight, _boxes_to_detections(loose.boxes)

    def warmup(self, shape: tuple[int, int, int], n: int = 1) -> None:
        dummy = np.zeros(shape, dtype=np.uint8)
        for _ in range(n):
            self.detect(dummy)
