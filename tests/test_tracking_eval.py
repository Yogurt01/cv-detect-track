"""TrackEval wrapper sanity: GT as tracker output, and two controlled perturbations.

Integration test: needs the canonical MOT17 data (skips otherwise). Runs the
real chain internal boxes -> MOT writer -> TrackEval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvdt.data.config import REPO_ROOT, dataset_config
from cvdt.data.mot import load_gt, mot_to_xyxy
from cvdt.eval.tracking import evaluate
from cvdt.pipeline.mot_writer import write_mot
from cvdt.types import Tracks

GT_ROOT = REPO_ROOT / dataset_config()["canonical_root"]
SEQS = ["MOT17-09", "MOT17-05"]

pytestmark = pytest.mark.skipif(
    not all((GT_ROOT / s / "gt" / "gt.txt").exists() for s in SEQS),
    reason="canonical MOT17 not built",
)


def gt_pedestrians(seq: str) -> pd.DataFrame:
    df = load_gt(GT_ROOT / seq / "gt" / "gt.txt")
    return df[(df["cls"] == 1) & (df["conf"] == 1)].copy()


def drop_fraction(df: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    keep = np.random.default_rng(seed).random(len(df)) >= frac
    return df[keep]


def swap_two_ids_halfway(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    """Swap the ids of the two tracks with the longest common time span, from its middle on."""
    spans = df.groupby("id")["frame"].agg(["min", "max"])
    ids = spans.index.to_numpy()
    lo = np.maximum.outer(spans["min"].to_numpy(), spans["min"].to_numpy())
    hi = np.minimum.outer(spans["max"].to_numpy(), spans["max"].to_numpy())
    overlap = np.triu(hi - lo, k=1)
    i, j = np.unravel_index(np.argmax(overlap), overlap.shape)
    assert overlap[i, j] > 0, "no two tracks overlap in time"
    a, b = ids[i], ids[j]
    mid = (lo[i, j] + hi[i, j]) // 2
    out = df.copy()
    late = out["frame"] > mid
    ia, ib = late & (out["id"] == a), late & (out["id"] == b)
    out.loc[ia, "id"], out.loc[ib, "id"] = b, a
    return out, (int(a), int(b), int(mid))


def write_run(root, run: str, frames: dict[str, pd.DataFrame]) -> None:
    for seq, df in frames.items():
        per_frame = [
            (
                int(f),
                Tracks(
                    boxes=mot_to_xyxy(g[["x", "y", "w", "h"]].to_numpy()).astype(np.float32),
                    ids=g["id"].to_numpy(),
                    scores=np.ones(len(g), np.float32),
                    classes=np.zeros(len(g), np.int64),
                ),
            )
            for f, g in df.sort_values(["frame", "id"]).groupby("frame")
        ]
        write_mot(root / run / "data" / f"{seq}.txt", per_frame)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("trackers")
    gt = {s: gt_pedestrians(s) for s in SEQS}
    write_run(root, "gt", gt)
    write_run(root, "drop10", {s: drop_fraction(d, 0.10) for s, d in gt.items()})
    write_run(root, "swap", {s: swap_two_ids_halfway(d)[0] for s, d in gt.items()})
    return {run: evaluate(root, run, SEQS, GT_ROOT) for run in ("gt", "drop10", "swap")}


@pytest.mark.parametrize("seq", SEQS)
def test_gt_as_tracker_scores_100(results, seq):
    r = results["gt"][seq]
    assert r["HOTA"] == pytest.approx(100.0)
    assert r["IDF1"] == pytest.approx(100.0)
    assert r["MOTA"] == pytest.approx(100.0)
    assert r["IDSW"] == 0


@pytest.mark.parametrize("seq", SEQS)
def test_dropping_boxes_lowers_deta(results, seq):
    r = results["drop10"][seq]
    assert r["DetA"] < 95.0
    assert r["MOTA"] == pytest.approx(90.0, abs=1.0)


@pytest.mark.parametrize("seq", SEQS)
def test_swapping_ids_lowers_assa_and_idf1(results, seq):
    r = results["swap"][seq]
    # HOTA matching is association-aware: where the swapped tracks are side by side
    # (MOT17-05), a few matches flip at strict localisation thresholds, so DetA may
    # dip slightly below 100 (observed 99.55); the drop is in AssA.
    assert r["DetA"] >= 99.0
    assert r["AssA"] < 100.0 and r["IDF1"] < 100.0
    assert r["IDSW"] == 2
