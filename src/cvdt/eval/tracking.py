"""Tracking evaluation with TrackEval (standard MOT17 handling, DECISIONS.md D10).

Layout expected by :func:`evaluate`::

    <gt_root>/<seq>/gt/gt.txt              canonical MOT17 ground truth
    <trackers_root>/<run>/data/<seq>.txt   tracker output (MOT format, 1-based)

HOTA, DetA, AssA (averaged over TrackEval's localisation thresholds), IDF1 and
MOTA are reported in percentage points (D18); IDSW is a count. ``COMBINED`` is
TrackEval's combination over the evaluated sequences.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cvdt.eval.trackeval_compat import import_trackeval

PCT_METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA")


def _seq_lengths(gt_root: Path, seqs: list[str]) -> dict[str, int]:
    from cvdt.data.mot import read_seqinfo

    return {s: int(read_seqinfo(gt_root / s / "seqinfo.ini")["seqLength"]) for s in seqs}


def evaluate(
    trackers_root: Path,
    run: str,
    seqs: list[str],
    gt_root: Path,
    out_json: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate ``<trackers_root>/<run>/data/<seq>.txt`` for ``seqs``.

    Returns ``{seq | "COMBINED": {metric: value}}`` and optionally writes it as JSON.
    """
    te = import_trackeval()
    ds_cfg = te.datasets.MotChallenge2DBox.get_default_dataset_config()
    with tempfile.TemporaryDirectory() as tmp:
        ds_cfg.update(
            GT_FOLDER=str(gt_root),
            TRACKERS_FOLDER=str(trackers_root),
            TRACKERS_TO_EVAL=[run],
            BENCHMARK="MOT17",
            SPLIT_TO_EVAL="train",
            SEQ_INFO=_seq_lengths(gt_root, seqs),
            SKIP_SPLIT_FOL=True,
            PRINT_CONFIG=False,
            OUTPUT_FOLDER=tmp,
        )
        ev_cfg = te.Evaluator.get_default_eval_config()
        # LOG_ON_ERROR defaults to a file inside the installed trackeval package (D7).
        ev_cfg.update(
            USE_PARALLEL=False,
            PRINT_RESULTS=False,
            PRINT_CONFIG=False,
            LOG_ON_ERROR=None,
            TIME_PROGRESS=False,
            OUTPUT_SUMMARY=False,
            OUTPUT_DETAILED=False,
            PLOT_CURVES=False,
            DISPLAY_LESS_PROGRESS=True,
        )
        quiet = {"PRINT_CONFIG": False}
        metrics = [te.metrics.HOTA(quiet), te.metrics.CLEAR(quiet), te.metrics.Identity(quiet)]
        res, _ = te.Evaluator(ev_cfg).evaluate([te.datasets.MotChallenge2DBox(ds_cfg)], metrics)

    out: dict[str, dict[str, float]] = {}
    for seq, r in res["MotChallenge2DBox"][run].items():
        p = r["pedestrian"]
        out[seq] = {
            "HOTA": 100 * float(np.mean(p["HOTA"]["HOTA"])),
            "DetA": 100 * float(np.mean(p["HOTA"]["DetA"])),
            "AssA": 100 * float(np.mean(p["HOTA"]["AssA"])),
            "IDF1": 100 * float(p["Identity"]["IDF1"]),
            "MOTA": 100 * float(p["CLEAR"]["MOTA"]),
            "IDSW": int(p["CLEAR"]["IDSW"]),
        }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, indent=2) + "\n")
    return out
