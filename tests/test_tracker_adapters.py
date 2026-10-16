"""All six Ultralytics trackers driven standalone through the Tracker interface."""

from __future__ import annotations

import numpy as np
import pytest

from cvdt.track.base import Tracker
from cvdt.track.ultralytics_trackers import (
    TRACKER_TYPES,
    UltralyticsTracker,
    load_tracker_config,
    scaled_track_buffer,
)
from cvdt.types import Detections

H, W = 720, 1280


def _two_movers(f: int) -> Detections:
    boxes = np.array(
        [[100 + 5 * f, 200, 160 + 5 * f, 350], [900 - 4 * f, 300, 970 - 4 * f, 480]], np.float32
    )
    return Detections(boxes=boxes, scores=np.array([0.9, 0.85], np.float32), classes=np.zeros(2))


@pytest.mark.parametrize("tracker_type", TRACKER_TYPES)
def test_two_linear_movers_keep_two_stable_ids(tracker_type):
    tr = UltralyticsTracker(tracker_type, fps=30, frame_shape=(H, W))
    assert isinstance(tr, Tracker)
    frame = np.full((H, W, 3), 128, np.uint8)
    ids = []
    for f in range(40):
        d = _two_movers(f)
        out = tr.update(d, frame if tr.needs_frame else None, loose=d)
        ids.append(tuple(sorted(out.ids.tolist())))
    confirmed = [x for x in ids if len(x) == 2]
    assert len(confirmed) >= 35, ids
    assert set(confirmed) == {confirmed[0]}, ids  # same two ids throughout
    # Track boxes follow the detections.
    np.testing.assert_allclose(
        np.sort(out.boxes[:, 0]), np.sort(_two_movers(39).boxes[:, 0]), atol=5
    )


@pytest.mark.parametrize("tracker_type", TRACKER_TYPES)
def test_reset_restarts_ids(tracker_type):
    tr = UltralyticsTracker(tracker_type, fps=30, frame_shape=(H, W))
    frame = np.full((H, W, 3), 128, np.uint8)
    for f in range(5):
        first = tr.update(_two_movers(f), frame if tr.needs_frame else None)
    tr.reset()
    for f in range(5):
        second = tr.update(_two_movers(f), frame if tr.needs_frame else None)
    assert sorted(first.ids.tolist()) == sorted(second.ids.tolist())


def test_empty_detections():
    for t in TRACKER_TYPES:
        tr = UltralyticsTracker(t, fps=30, frame_shape=(H, W))
        frame = np.full((H, W, 3), 128, np.uint8)
        out = tr.update(Detections(), frame if tr.needs_frame else None)
        assert len(out) == 0 and out.boxes.shape == (0, 4)


def test_fps_scaling_of_track_buffer():
    default = load_tracker_config("bytetrack").track_buffer
    assert default == 30
    assert scaled_track_buffer(30, 14) == 14
    assert scaled_track_buffer(30, 25) == 25
    assert scaled_track_buffer(30, 30) == 30
    tr = UltralyticsTracker("bytetrack", fps=14, frame_shape=(480, 640))
    assert tr.cfg.track_buffer == 14
    tr = UltralyticsTracker("bytetrack", fps=14, frame_shape=(480, 640), scale_buffer_by_fps=False)
    assert tr.cfg.track_buffer == 30


def test_verified_bytetrack_defaults():
    cfg = load_tracker_config("bytetrack")
    assert (cfg.track_high_thresh, cfg.track_low_thresh, cfg.new_track_thresh) == (0.25, 0.1, 0.25)
    assert (cfg.track_buffer, cfg.match_thresh, cfg.fuse_score) == (30, 0.8, True)


def test_reid_is_rejected_in_tier1():
    with pytest.raises(ValueError):
        UltralyticsTracker("botsort", fps=30, frame_shape=(H, W), overrides={"with_reid": True})
