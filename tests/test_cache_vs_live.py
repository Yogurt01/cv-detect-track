"""Cache-vs-live consistency (DECISIONS.md D15): must pass before any official experiment.

Integration test on MOT17-09 with yolo26s (640, FP32) and ByteTrack:

1. Live detector at conf 0.1 vs cache (floor 0.01) filtered at 0.1: identical
   detections.
2. Tracking from the cache vs tracking live: identical track ids in every frame,
   boxes within 0.5 px.
3. The adapter vs Ultralytics' own ``model.track()``: identical ids, and boxes
   within 0.5 px once clipped like ``model.track()`` (D18).

Skips when the data, the weights, the cache or a CUDA GPU is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvdt.data.config import REPO_ROOT, dataset_config

SEQ = "MOT17-09"
WEIGHTS = REPO_ROOT / "weights" / "yolo26s.pt"
CANON = REPO_ROOT / dataset_config()["canonical_root"]
CACHES = sorted((REPO_ROOT / "outputs" / "detections").glob("yolo26s_640_fp32_conf0.01_*"))
TOL_PX = 0.5

pytestmark = pytest.mark.integration


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not (CANON / SEQ).exists() or not WEIGHTS.exists() or not _cuda():
        pytest.skip("MOT17 data, yolo26s weights or CUDA missing")
    if not CACHES or not (CACHES[0] / f"{SEQ}.npz").exists():
        pytest.skip("yolo26s detection cache missing; run scripts/cache_detections.py")
    from cvdt.data.config import open_sequence
    from cvdt.detect.cache import CachedSequence
    from cvdt.detect.yolo import YOLODetector
    from cvdt.pipeline.run import TRACKER_INPUT_CONF, run_sequence
    from cvdt.track.ultralytics_trackers import UltralyticsTracker, reference_track

    ds = open_sequence(SEQ)
    cache = CachedSequence(CACHES[0] / f"{SEQ}.npz")
    det = YOLODetector(WEIGHTS, imgsz=640, conf=TRACKER_INPUT_CONF)
    tmp = tmp_path_factory.mktemp("cvl")

    def tracker(clip=False):
        return UltralyticsTracker(
            "bytetrack",
            fps=ds.info.fps,
            frame_shape=(ds.info.height, ds.info.width),
            clip_boxes=clip,
        )

    det_pairs = [(det.detect(fr.load()), cache.tight(fr.index, TRACKER_INPUT_CONF)) for fr in ds]
    live = run_sequence(ds, tracker(), tmp / "live.txt", detector=det)
    cached = run_sequence(ds, tracker(), tmp / "cache.txt", cache=cache)
    live_clipped = run_sequence(ds, tracker(clip=True), tmp / "live_clip.txt", detector=det)
    paths = [str(ds.frame(i).image_path) for i in range(1, len(ds) + 1)]
    ref = reference_track(str(WEIGHTS), paths, "bytetrack", imgsz=640, conf=TRACKER_INPUT_CONF)
    return det_pairs, live, cached, live_clipped, ref


def _compare(a, b):
    """(frames with differing id sets, max box difference over frames with equal ids)."""
    bad, maxd = 0, 0.0
    for x, y in zip(a, b, strict=True):
        ox, oy = np.argsort(x.ids), np.argsort(y.ids)
        if not np.array_equal(x.ids[ox], y.ids[oy]):
            bad += 1
            continue
        if len(x):
            maxd = max(maxd, float(np.abs(x.boxes[ox] - y.boxes[oy]).max()))
    return bad, maxd


def test_cached_detections_equal_live_detections(runs):
    det_pairs = runs[0]
    for live, cached in det_pairs:
        assert len(live) == len(cached)
        ol, oc = np.argsort(-live.scores, kind="stable"), np.argsort(-cached.scores, kind="stable")
        np.testing.assert_allclose(live.boxes[ol], cached.boxes[oc], atol=TOL_PX)
        np.testing.assert_allclose(live.scores[ol], cached.scores[oc], atol=1e-6)


def test_tracking_from_cache_equals_live(runs):
    _, live, cached, _, _ = runs
    bad, maxd = _compare(cached.tracks_per_frame, live.tracks_per_frame)
    assert bad == 0 and maxd <= TOL_PX, (bad, maxd)
    assert cached.mot_lines == live.mot_lines


def test_adapter_reproduces_model_track(runs):
    _, live, _, live_clipped, ref = runs
    bad, maxd = _compare(live_clipped.tracks_per_frame, ref)
    assert bad == 0 and maxd <= TOL_PX, (bad, maxd)
    # Unclipped output differs only by clipping: same ids in every frame.
    assert _compare(live.tracks_per_frame, ref)[0] == 0


def test_tracktrack_live_uses_one_forward_pass_and_matches_cache_and_model_track(runs, tmp_path):
    """TrackTrack needs the loose-NMS pass (D16). Live: exactly one forward pass per
    frame; tracks identical to the cache path over the whole sequence and to
    model.track() over the first 200 frames."""
    from cvdt.data.config import open_sequence
    from cvdt.detect.cache import CachedSequence
    from cvdt.detect.yolo import YOLODetector
    from cvdt.pipeline.run import TRACKER_INPUT_CONF, run_sequence
    from cvdt.track.ultralytics_trackers import UltralyticsTracker, reference_track

    ds = open_sequence(SEQ)

    def tracker(clip):
        return UltralyticsTracker(
            "tracktrack",
            fps=ds.info.fps,
            frame_shape=(ds.info.height, ds.info.width),
            clip_boxes=clip,
        )

    det = YOLODetector(WEIGHTS, imgsz=640, conf=TRACKER_INPUT_CONF)
    det.warmup((ds.info.height, ds.info.width, 3), n=2)
    before = det.forward_passes
    live = run_sequence(ds, tracker(False), tmp_path / "l.txt", detector=det, loose_iou=0.95)
    assert det.forward_passes - before == len(ds)  # one forward pass per frame

    cache = CachedSequence(CACHES[0] / f"{SEQ}.npz")
    cached = run_sequence(ds, tracker(False), tmp_path / "c.txt", cache=cache)
    bad, maxd = _compare(cached.tracks_per_frame, live.tracks_per_frame)
    assert bad == 0 and maxd <= TOL_PX, (bad, maxd)

    n = 200
    live_clip = run_sequence(
        ds, tracker(True), tmp_path / "lc.txt", detector=det, loose_iou=0.95, max_frames=n
    )
    paths = [str(ds.frame(i).image_path) for i in range(1, n + 1)]
    ref = reference_track(str(WEIGHTS), paths, "tracktrack", imgsz=640, conf=TRACKER_INPUT_CONF)
    bad, maxd = _compare(live_clip.tracks_per_frame, ref)
    assert bad == 0 and maxd <= TOL_PX, (bad, maxd)


def test_detect_with_loose_equals_two_separate_passes(runs):
    """The single-pass loose NMS gives the same boxes as a second forward pass at loose IoU."""
    from cvdt.data.config import open_sequence
    from cvdt.detect.yolo import YOLODetector

    ds = open_sequence(SEQ)
    single = YOLODetector(WEIGHTS, imgsz=640, conf=0.01)
    separate = YOLODetector(WEIGHTS, imgsz=640, conf=0.01, iou=0.95)
    for idx in (1, 100, 300, 525):
        img = ds.frame(idx).load()
        tight, loose = single.detect_with_loose(img, 0.95)
        ref_loose = separate.detect(img)
        assert len(loose) == len(ref_loose) and len(loose) >= len(tight)
        np.testing.assert_allclose(loose.boxes, ref_loose.boxes, atol=1e-4)
