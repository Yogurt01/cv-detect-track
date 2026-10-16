"""Ultralytics trackers behind the :class:`~cvdt.track.base.Tracker` interface.

This is the only module in ``cvdt.track`` allowed to import Ultralytics
(DECISIONS.md D15). The project's own tracker must not import it.

Standalone call pattern (pinned to ultralytics 8.4.177)
-------------------------------------------------------
``model.track()`` (``ultralytics/trackers/track.py``) does, per frame::

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("<type>.yaml")))
    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)          # once per video
    det = result.boxes.cpu().numpy()                            # Boxes: xyxy, conf, cls
    tracks = tracker.update(det, result.orig_img, feats=None[, dets_del=...])
    # tracks: float32 (M, 8) = [x1, y1, x2, y2, track_id, score, cls, det_idx]

This adapter does the same from :class:`~cvdt.types.Detections`, wrapping them
in a real ``ultralytics.engine.results.Boxes`` so every tracker reads the same
attributes (``xyxy``, ``xywh``, ``conf``, ``cls``) as under ``model.track()``.

TrackTrack additionally receives ``dets_del``: detections that the tight NMS
dropped but a loose NMS (IoU ``_LOOSE_NMS_IOU`` = 0.95) kept, i.e. loose boxes
whose max IoU with every tight box is < ``_LOOSE_NMS_DEDUP_IOU`` (0.97).
``model.track()`` computes it from internal predictor state
(``track_tracker.compute_dets_del``); here it is computed from the cached
loose-NMS pass with the same rule and the same ``box_iou`` (D16).

Differences from ``model.track()``, by design:
  * ``track_buffer`` is scaled by ``fps / 30`` unless ``scale_buffer_by_fps=False``
    (D17). The 8.4.177 trackers have no frame-rate input.
  * Track ids restart at 1 per sequence (``reset()``), as they do per video in
    ``model.track()``.
  * Track boxes are the tracker's own state and may extend beyond the image.
    ``model.track()`` clips them to the image in ``Results.update()``; set
    ``clip_boxes=True`` to reproduce that (D18). With clipping, output equals
    ``model.track()`` exactly (verified on MOT17-09 with ByteTrack).

Version-specific risks: ``TRACKER_MAP``, the tracker constructors' ``args``-only
signature, the ``update(results, img, feats=..., dets_del=...)`` keywords, the
8-column output and the two private TrackTrack constants are internal APIs.
``check_version()`` asserts the version these were verified against.
"""

from __future__ import annotations

import numpy as np

from cvdt.types import Detections, Tracks

VERIFIED_ULTRALYTICS = "8.4.177"
TRACKER_TYPES = ("bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack")


def check_version() -> None:
    import ultralytics

    if ultralytics.__version__ != VERIFIED_ULTRALYTICS:
        raise RuntimeError(
            f"tracker adapter verified against ultralytics {VERIFIED_ULTRALYTICS}, "
            f"found {ultralytics.__version__}; re-verify the call pattern before use"
        )


def load_tracker_config(tracker_type: str, overrides: dict | None = None):
    """Library default config for ``tracker_type`` (from ultralytics/cfg/trackers)."""
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    if tracker_type not in TRACKER_TYPES:
        raise ValueError(f"unknown tracker {tracker_type!r}; choose from {TRACKER_TYPES}")
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{tracker_type}.yaml")))
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    return cfg


def scaled_track_buffer(track_buffer: int, fps: float) -> int:
    """Original ByteTrack: ``int(frame_rate / 30.0 * track_buffer)`` (D17)."""
    return int(fps / 30.0 * track_buffer)


def _boxes(det: Detections, orig_shape: tuple[int, int]):
    from ultralytics.engine.results import Boxes

    data = np.zeros((len(det), 6), dtype=np.float32)
    if len(det):
        data[:, :4] = det.boxes
        data[:, 4] = det.scores
        data[:, 5] = det.classes
    return Boxes(data, orig_shape)


def _dets_del(tight: Detections, loose: Detections | None):
    """Same rule as ``ultralytics.trackers.track_tracker.compute_dets_del``."""
    import torch
    from ultralytics.trackers import track_tracker
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou

    if loose is None or len(loose) == 0 or len(tight) == 0:
        return None
    max_iou = (
        box_iou(torch.from_numpy(loose.boxes), torch.from_numpy(tight.boxes)).max(dim=1).values
    )
    mask = (max_iou < track_tracker._LOOSE_NMS_DEDUP_IOU).numpy()
    if not mask.any():
        return None
    xywh = ops.xyxy2xywh(torch.from_numpy(loose.boxes[mask])).numpy()
    return xywh, loose.scores[mask], loose.classes[mask].astype(np.float32)


class UltralyticsTracker:
    """One of the six Ultralytics trackers, fed from :class:`Detections`.

    Args:
        tracker_type: one of :data:`TRACKER_TYPES`.
        fps: sequence frame rate (``seqinfo.ini``).
        frame_shape: (H, W) of the sequence, used for ``Boxes.orig_shape``.
        scale_buffer_by_fps: scale ``track_buffer`` by ``fps / 30`` (D17).
        clip_boxes: clip output boxes to the image like ``model.track()`` (D18).
        overrides: config values replacing library defaults (Tier 2 only).
    """

    def __init__(
        self,
        tracker_type: str,
        fps: float,
        frame_shape: tuple[int, int],
        scale_buffer_by_fps: bool = True,
        clip_boxes: bool = False,
        overrides: dict | None = None,
    ):
        check_version()
        from ultralytics.trackers.track import TRACKER_MAP

        cfg = load_tracker_config(tracker_type, overrides)
        if getattr(cfg, "with_reid", False):
            raise ValueError("ReID is off in Tier 1 (D15); enable it only through a new decision")
        cfg.device = "cpu"  # only used by a ReID encoder, which is off
        self.default_track_buffer = int(cfg.track_buffer)
        if scale_buffer_by_fps:
            cfg.track_buffer = scaled_track_buffer(cfg.track_buffer, fps)
        self.cfg = cfg
        self.tracker_type = tracker_type
        self.fps = fps
        self.frame_shape = tuple(frame_shape)
        self.clip_boxes = clip_boxes
        self.name = f"{tracker_type}-ultralytics"
        self.needs_frame = getattr(cfg, "gmc_method", "none") not in (None, "none")
        self._impl = TRACKER_MAP[tracker_type](args=cfg)

    def update(
        self,
        detections: Detections,
        frame: np.ndarray | None = None,
        *,
        loose: Detections | None = None,
    ) -> Tracks:
        kwargs: dict = {"feats": None}
        if self.tracker_type == "tracktrack":
            kwargs["dets_del"] = _dets_del(detections, loose)
        out = self._impl.update(_boxes(detections, self.frame_shape).numpy(), frame, **kwargs)
        out = np.asarray(out, dtype=np.float32).reshape(-1, 8) if len(out) else np.zeros((0, 8))
        boxes = out[:, :4].astype(np.float32)
        if self.clip_boxes:
            h, w = self.frame_shape
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return Tracks(
            boxes=boxes,
            ids=out[:, 4].astype(np.int64),
            scores=out[:, 5].astype(np.float32),
            classes=out[:, 6].astype(np.int64),
        )

    def reset(self) -> None:
        self._impl.reset()


def reference_track(
    weights: str,
    image_paths: list[str],
    tracker_type: str,
    imgsz: int = 640,
    conf: float = 0.1,
    device: int | str = 0,
) -> list[Tracks]:
    """Run Ultralytics' own ``model.track()`` frame by frame (verification only).

    Used by the cache-vs-live integration test to check that this adapter drives
    the trackers exactly like the library does. Person class only, FP32.
    """
    check_version()
    from ultralytics import YOLO

    model = YOLO(weights)
    out = []
    for p in image_paths:
        r = model.track(
            p,
            persist=True,
            tracker=f"{tracker_type}.yaml",
            conf=conf,
            imgsz=imgsz,
            classes=[0],
            quantize=32,
            device=device,
            verbose=False,
        )[0]
        b = r.boxes
        if b.id is None:
            out.append(Tracks())
            continue
        out.append(
            Tracks(
                boxes=b.xyxy.cpu().numpy().astype(np.float32),
                ids=b.id.cpu().numpy().astype(np.int64),
                scores=b.conf.cpu().numpy().astype(np.float32),
                classes=b.cls.cpu().numpy().astype(np.int64),
            )
        )
    return out
