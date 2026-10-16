"""MOT writer: 1-based frames and coordinates, exact round trip from gt.txt."""

from __future__ import annotations

import numpy as np
import pytest

from cvdt.data.config import REPO_ROOT, dataset_config
from cvdt.data.mot import load_gt, mot_to_xyxy
from cvdt.pipeline.mot_writer import format_rows, read_mot, write_mot
from cvdt.types import Tracks

CANON = REPO_ROOT / dataset_config()["canonical_root"]


def test_format_is_one_based():
    t = Tracks(
        boxes=np.array([[0.0, 0.0, 10.0, 20.0]], np.float32),
        ids=np.array([7]),
        scores=np.array([0.5], np.float32),
        classes=np.array([0]),
    )
    assert format_rows(1, t) == ["1,7,1.00,1.00,10.00,20.00,0.5000,-1,-1,-1"]
    with pytest.raises(ValueError):
        format_rows(0, t)


@pytest.mark.parametrize("seq", ["MOT17-09", "MOT17-05"])
def test_gt_round_trip_is_exact(seq, tmp_path):
    gt_path = CANON / seq / "gt" / "gt.txt"
    if not gt_path.exists():
        pytest.skip("canonical MOT17 not built")
    df = load_gt(gt_path)
    df = df[(df["cls"] == 1) & (df["conf"] == 1)]
    per_frame = []
    for f, g in df.groupby("frame"):
        per_frame.append(
            (
                int(f),
                Tracks(
                    boxes=mot_to_xyxy(g[["x", "y", "w", "h"]].to_numpy()).astype(np.float32),
                    ids=g["id"].to_numpy(),
                    scores=np.ones(len(g), np.float32),
                    classes=np.zeros(len(g), np.int64),
                ),
            )
        )
    out = tmp_path / f"{seq}.txt"
    assert write_mot(out, per_frame) == len(df)
    back = read_mot(out)
    # gt.txt is ordered by track, the writer by frame: compare in (frame, id) order.
    expected = df.sort_values(["frame", "id"])[["frame", "id", "x", "y", "w", "h"]].to_numpy(float)
    back = back[np.lexsort((back[:, 1], back[:, 0]))]
    np.testing.assert_array_equal(back[:, :6], expected)
    assert np.all(back[:, 7:] == -1)
