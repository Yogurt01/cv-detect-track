"""Detection evaluator (D13, D19): hand-computable cases and a pycocotools cross-check."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from cvdt.data.config import REPO_ROOT, dataset_config
from cvdt.eval.detection import (
    FP,
    IGNORED,
    TP,
    ImageGT,
    evaluate_images,
    ioa_matrix,
    iou_matrix,
    load_eval_gt,
    match_image,
)

EMPTY = np.zeros((0, 4))


def _gt(pos=(), dis=(), low=()):
    def arr(x):
        return np.array(x, float).reshape(-1, 4)

    return ImageGT(positives=arr(pos), distractors=arr(dis), low_vis=arr(low))


def _dets(*rows):
    rows = np.array(rows, float).reshape(-1, 5)
    return rows[:, :4], rows[:, 4]


def _labels(dets, gt, thr, rule):
    return match_image(*dets, gt, thr, ignore_rule=rule)[1].tolist()


def test_iou_and_ioa_matrix():
    assert iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == 1.0
    assert iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(50 / 150)
    assert iou_matrix([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0] == 0.0
    # IoA is relative to the first (detection) box: a box inside a larger one has IoA 1.
    assert ioa_matrix([[0, 0, 10, 4]], [[0, 0, 10, 10]])[0, 0] == 1.0
    assert ioa_matrix([[0, 0, 10, 10]], [[0, 0, 10, 4]])[0, 0] == pytest.approx(0.4)


# --- D13 cases (still valid; distractor ignore regions behave the same under D19) ---


@pytest.mark.parametrize("rule", ["d19", "d13"])
def test_case1_detection_on_distractor_region(rule):
    # Higher-scoring detection sits on a distractor box (IoU 1): removed, not a FP.
    gt = {1: _gt(pos=[[0, 0, 100, 200]], dis=[[300, 0, 400, 200]])}
    dets = {1: _dets([300, 0, 400, 200, 0.9], [0, 0, 100, 200, 0.8])}
    assert _labels(dets[1], gt[1], 0.5, rule) == [IGNORED, TP]
    r = evaluate_images(dets, gt, ignore_rule=rule)
    assert r["AP50"] == pytest.approx(1.0)
    assert (r["TP"], r["FP"], r["FN"], r["ignored"]) == (1, 0, 0, 1)
    assert r["ignored_distractor"] == 1
    # Without the ignore region the same detection is a FP ranked first: AP50 = 0.5.
    r_no_ign = evaluate_images(dets, {1: _gt(pos=[[0, 0, 100, 200]])}, ignore_rule=rule)
    assert r_no_ign["AP50"] == pytest.approx(0.5, abs=1e-6)


def test_case2_detection_overlapping_positive_and_distractor():
    pos, dis = [0, 0, 100, 200], [10, 0, 110, 200]  # IoU(pos, dis) = 0.818
    gt = _gt(pos=[pos], dis=[dis])
    det = _dets([5, 0, 105, 200, 0.9])  # IoU 0.905 with pos, 0.905 with dis
    # Matches the positive first, so it is a TP under both rules.
    assert _labels(det, gt, 0.5, "d13") == [TP]
    assert _labels(det, gt, 0.5, "d19") == [TP]
    # At 0.95 the positive match fails. D13: falls through to the distractor box.
    # D19: IoU 0.905 >= 0.5 with a positive -> duplicate guard -> FP.
    assert _labels(det, gt, 0.95, "d13") == [IGNORED]
    assert _labels(det, gt, 0.95, "d19") == [FP]
    # A second detection on the same place (positive already taken).
    two = _dets([5, 0, 105, 200, 0.9], [8, 0, 108, 200, 0.8])
    assert _labels(two, gt, 0.5, "d13") == [TP, IGNORED]
    assert _labels(two, gt, 0.5, "d19") == [TP, FP]


@pytest.mark.parametrize("rule", ["d19", "d13"])
def test_case3_image_without_gt(rule):
    gt = {1: _gt(pos=[[0, 0, 100, 200]]), 2: _gt()}
    dets = {1: _dets([0, 0, 100, 200, 0.5]), 2: _dets([0, 0, 100, 200, 0.9])}
    r = evaluate_images(dets, gt, ignore_rule=rule)
    # Pooled ranking: FP (0.9) then TP (0.5) -> precision 0.5 at recall 1.
    assert r["AP50"] == pytest.approx(0.5, abs=1e-6)
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 0)
    # No GT anywhere: AP undefined.
    assert np.isnan(evaluate_images({1: _dets([0, 0, 1, 1, 0.9])}, {1: _gt()})["AP50"])


@pytest.mark.parametrize("rule", ["d19", "d13"])
def test_case4_two_detections_on_one_gt(rule):
    gt = {1: _gt(pos=[[0, 0, 100, 100]])}
    # IoU 0.88 (higher score) and 0.80: the first is TP, the duplicate is FP.
    dets = {1: _dets([0, 0, 100, 88, 0.9], [0, 0, 100, 80, 0.8])}
    assert _labels(dets[1], gt[1], 0.5, rule) == [TP, FP]
    r = evaluate_images(dets, gt, ignore_rule=rule)
    assert r["AP50"] == pytest.approx(1.0)
    # TP at thresholds 0.50..0.85 (8 of 10), nothing at 0.90 and 0.95.
    assert r["AP50_95"] == pytest.approx(0.8)
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 0)


# --- D19 cases ---


def test_d19_a_visible_part_inside_low_visibility_box_is_removed():
    # Full-body low-visibility box 100 x 200; detection of the visible upper part 100 x 80.
    gt = _gt(low=[[0, 0, 100, 200]])
    det = _dets([0, 0, 100, 80, 0.9])  # IoU 0.4, IoA 1.0
    assert iou_matrix(det[0], gt.low_vis)[0, 0] == pytest.approx(0.4)
    assert _labels(det, gt, 0.5, "d19") == [IGNORED]
    assert _labels(det, gt, 0.5, "d13") == [FP]  # the D13 problem D19 fixes
    r = evaluate_images({1: det}, {1: gt})
    assert r["ignored_low_vis"] == 1 and r["FP"] == 0


def test_d19_b_duplicate_of_visible_person_inside_low_visibility_box_is_fp():
    # Occluded person (low-vis, full-body) behind a visible person who lies inside that box.
    low, pos = [0, 0, 120, 300], [10, 50, 110, 290]
    gt = _gt(pos=[pos], low=[low])
    dets = _dets([10, 50, 110, 290, 0.9], [12, 55, 110, 290, 0.8])  # TP, then a duplicate
    # Duplicate: IoU 0.96 with the positive, IoA 1.0 with the low-vis box.
    assert _labels(dets, gt, 0.5, "d19") == [TP, FP]  # duplicate guard
    assert _labels(dets, gt, 0.5, "d13") == [TP, IGNORED]  # IoU 0.64 with low-vis box
    r = evaluate_images({1: dets}, {1: gt})
    assert (r["fp_duplicate_guard"], r["fp_guard_overrode_ignore"]) == (1, 1)


def test_d19_c_ioa_does_not_excuse_distractor_regions():
    gt = _gt(dis=[[0, 0, 100, 200]])
    det = _dets([0, 0, 100, 80, 0.9])  # IoA 1.0 but IoU 0.4 with the distractor box
    assert _labels(det, gt, 0.5, "d19") == [FP]
    assert _labels(det, gt, 0.5, "d13") == [FP]


def test_d19_d_guard_applies_at_strict_threshold_to_any_positive():
    # Positive inside a larger low-visibility box; detection with IoU 0.7 with the positive.
    pos, low = [0, 0, 100, 200], [0, 0, 150, 300]
    gt = _gt(pos=[pos], low=[low])
    det = _dets([0, 0, 100, 140, 0.9])  # IoU 0.7 with pos, IoA 1.0 with low-vis box
    assert _labels(det, gt, 0.5, "d19") == [TP]
    # At 0.95 it is unmatched; the guard (IoU 0.7 >= 0.5 with a positive) makes it a FP
    # even though it lies inside the low-visibility box.
    assert _labels(det, gt, 0.95, "d19") == [FP]
    _, _, reasons = match_image(*det, gt, 0.95, return_reasons=True)
    from cvdt.eval.detection import R_FP_GUARD_EXCUSABLE

    assert reasons.tolist() == [R_FP_GUARD_EXCUSABLE]


def test_unknown_ignore_rule_rejected():
    with pytest.raises(ValueError):
        match_image(*_dets([0, 0, 1, 1, 0.5]), _gt(), 0.5, ignore_rule="coco")


def _pycocotools(dets: dict, gts: dict) -> tuple[float, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, res = [], [], []
    for i in sorted(gts):
        images.append({"id": i, "width": 4000, "height": 4000})
        for b in gts[i].positives:
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": [b[0], b[1], w, h],
                    "area": w * h,
                }
            )
        boxes, scores = dets.get(i, (EMPTY, np.zeros(0)))
        for b, s in zip(boxes, scores, strict=True):
            res.append(
                {
                    "image_id": i,
                    "category_id": 1,
                    "score": float(s),
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1}]}
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(res), "bbox")
        ev.params.maxDets = [1, 10, 100000]  # no per-image cap, like our evaluator
        ev.evaluate()
        ev.accumulate()
    # summarize() hardcodes maxDets=100 for AP in pycocotools 2.0.11, so read the
    # accumulated precision directly: [iou thr, recall thr, category, area 'all', maxDets].
    prec = ev.eval["precision"][:, :, 0, 0, -1]
    return float(np.mean(prec[0][prec[0] > -1])), float(np.mean(prec[prec > -1]))


def test_cross_check_pycocotools_synthetic():
    pytest.importorskip("pycocotools")
    rng = np.random.default_rng(0)
    gts, dets = {}, {}
    for i in range(1, 61):
        n = int(rng.integers(0, 12))
        xy = rng.uniform(0, 1500, (n, 2))
        wh = rng.uniform(20, 300, (n, 2))
        pos = np.hstack([xy, xy + wh])
        gts[i] = _gt(pos=pos)
        keep = pos[rng.random(n) > 0.2]
        jit = keep + rng.normal(0, 8, keep.shape)
        fps = np.hstack([xy2 := rng.uniform(0, 1500, (int(rng.integers(0, 5)), 2)), xy2 + 80])
        boxes = np.vstack([jit, fps])
        dets[i] = (boxes, rng.uniform(0.01, 1.0, len(boxes)))
    ours = evaluate_images(dets, gts)
    ap50, ap = _pycocotools(dets, gts)
    assert ours["AP50"] == pytest.approx(ap50, abs=1e-12)
    assert ours["AP50_95"] == pytest.approx(ap, abs=1e-12)


def _cache_dir(stem: str):
    root = REPO_ROOT / "outputs" / "detections"
    hits = sorted(root.glob(f"{stem}_640_fp32_conf0.01_*")) if root.exists() else []
    return hits[0] if hits else None


@pytest.mark.parametrize("seq", ["MOT17-09", "MOT17-05"])
def test_cross_check_pycocotools_real_detections_without_ignore(seq):
    pytest.importorskip("pycocotools")
    cache_dir = _cache_dir("yolo26s")
    canon = REPO_ROOT / dataset_config()["canonical_root"] / seq
    if cache_dir is None or not (cache_dir / f"{seq}.npz").exists() or not canon.exists():
        pytest.skip("cached yolo26s detections or MOT17 data missing")
    from cvdt.detect.cache import CachedSequence

    full = load_eval_gt(canon, 0.25, protocol="secondary")
    gts = {f: _gt(pos=g.positives) for f, g in full.items()}  # no ignore regions
    cache = CachedSequence(cache_dir / f"{seq}.npz")
    dets = {
        f: (d.boxes.astype(float), d.scores.astype(float)) for f in gts for d in [cache.tight(f)]
    }
    ours = evaluate_images(dets, gts)
    ap50, ap = _pycocotools(dets, gts)
    assert ours["AP50"] == pytest.approx(ap50, abs=1e-12)
    assert ours["AP50_95"] == pytest.approx(ap, abs=1e-12)
