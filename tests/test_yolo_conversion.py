"""MOT box -> YOLO normalised -> MOT round trip, and label-rule sanity."""

from __future__ import annotations

import numpy as np
import pytest

from cvdt.data import filtering
from cvdt.data.filtering import Category
from cvdt.data.mot import MOTClass, mot_to_xyxy, xyxy_to_mot
from cvdt.data.yolo import mot_to_yolo, yolo_to_mot


@pytest.mark.parametrize("size", [(1920, 1080), (640, 480)])
def test_round_trip_within_one_pixel(size):
    w, h = size
    rng = np.random.default_rng(0)
    # MOT boxes are integer, 1-based, fully inside the image.
    bw = rng.integers(5, w // 4, 500)
    bh = rng.integers(10, h // 2, 500)
    left = rng.integers(1, w - bw + 1)
    top = rng.integers(1, h - bh + 1)
    mot = np.stack([left, top, bw, bh], axis=1).astype(float)

    yolo = mot_to_yolo(mot, w, h)
    # Labels are written with 6 decimals; round-trip through that precision.
    yolo = np.round(yolo, 6)
    assert np.all((yolo >= 0) & (yolo <= 1))
    back = yolo_to_mot(yolo, w, h)
    assert np.abs(back - mot).max() <= 1.0


def test_mot_coordinates_are_one_based():
    xyxy = mot_to_xyxy(np.array([[1.0, 1.0, 10.0, 20.0]]))
    np.testing.assert_allclose(xyxy, [[0.0, 0.0, 10.0, 20.0]])
    np.testing.assert_allclose(xyxy_to_mot(xyxy), [[1.0, 1.0, 10.0, 20.0]])


def test_boxes_beyond_border_are_clipped():
    w, h = 100, 100
    yolo = mot_to_yolo(np.array([[-19.0, 51.0, 40.0, 80.0]]), w, h)  # 0-based x0=-20, y1=130
    cx, cy, bw, bh = yolo[0]
    assert bw == pytest.approx(0.20) and bh == pytest.approx(0.50)
    assert cx == pytest.approx(0.10) and cy == pytest.approx(0.75)


def test_categorize_rule():
    ped, static, car = MOTClass.PEDESTRIAN, MOTClass.STATIC_PERSON, MOTClass.CAR
    classes = np.array([ped, ped, ped, static, car])
    flags = np.array([1, 1, 0, 0, 0])
    vis = np.array([0.9, 0.1, 0.9, 1.0, 1.0])
    cat = filtering.categorize(classes, flags, vis, min_visibility=0.25)
    assert cat.tolist() == [
        Category.POSITIVE,
        Category.LOW_VISIBILITY,
        Category.IGNORE,
        Category.IGNORE,
        Category.DROPPED,
    ]
    reasons = filtering.ignore_reason(classes, cat).tolist()
    assert reasons == ["", "low_visibility", "zero_flag", "distractor_class", ""]


def test_visibility_threshold_is_inclusive():
    cat = filtering.categorize(np.array([1]), np.array([1]), np.array([0.25]), 0.25)
    assert cat[0] == Category.POSITIVE
