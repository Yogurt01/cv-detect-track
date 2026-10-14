"""Self-test of the TrackEval install on a tiny synthetic MOT17-format sequence.

Runs TrackEval's MOT17 code path (``MotChallenge2DBox`` + HOTA/CLEAR/Identity)
on hand-made ground truth with known answers:

* ``perfect``: tracker output equals the pedestrian GT -> HOTA = IDF1 = MOTA = 1.
* ``idswitch``: one pedestrian changes id half-way -> 1 ID switch, IDF1 < 1.
* ``distractor``: perfect output plus a box on a static person (class 7) -> the
  extra box is removed by preprocessing, so MOTA stays 1 (no FP).
* ``empty``: no output at all -> HOTA = 0; exercises the code paths that use
  the removed NumPy aliases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cvdt.eval.trackeval_compat import import_trackeval

SEQ = "SYN-01"
N_FRAMES = 20


def _gt_rows() -> list[tuple]:
    rows = []
    for f in range(1, N_FRAMES + 1):
        rows.append((f, 1, 100 + 5 * f, 200, 50, 120, 1, 1, 1.0))  # pedestrian, flag 1
        rows.append((f, 2, 900 - 4 * f, 300, 60, 150, 1, 1, 0.8))  # pedestrian, flag 1
        rows.append((f, 3, 1500, 500, 55, 130, 0, 7, 1.0))  # static person (distractor)
        rows.append((f, 4, 400, 700, 200, 100, 0, 3, 1.0))  # car (non-person)
    return rows


def _tracker_rows(case: str) -> list[tuple]:
    peds = [r for r in _gt_rows() if r[7] == 1]
    if case == "empty":
        return []
    out = []
    for f, i, x, y, w, h, *_ in peds:
        tid = 3 if (case == "idswitch" and i == 1 and f > N_FRAMES // 2) else i
        out.append((f, tid, x, y, w, h, 0.9, -1, -1, -1))
    if case == "distractor":
        out += [(f, 99, 1500, 500, 55, 130, 0.9, -1, -1, -1) for f in range(1, N_FRAMES + 1)]
    return out


def _write(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(",".join(str(v) for v in r) + "\n" for r in rows))


def run(cases: tuple[str, ...] = ("perfect", "idswitch", "distractor", "empty")) -> dict:
    """Return {case: {HOTA, DetA, AssA, IDF1, MOTA, IDSW, FP}} computed by TrackEval."""
    te = import_trackeval()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write(tmp / "gt" / SEQ / "gt" / "gt.txt", _gt_rows())
        for c in cases:
            _write(tmp / "trackers" / c / "data" / f"{SEQ}.txt", _tracker_rows(c))
        ds_cfg = te.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update(
            GT_FOLDER=str(tmp / "gt"),
            TRACKERS_FOLDER=str(tmp / "trackers"),
            BENCHMARK="MOT17",
            SPLIT_TO_EVAL="train",
            TRACKERS_TO_EVAL=list(cases),
            SEQ_INFO={SEQ: N_FRAMES},
            SKIP_SPLIT_FOL=True,
            PRINT_CONFIG=False,
            OUTPUT_FOLDER=str(tmp / "out"),
        )
        ev_cfg = te.Evaluator.get_default_eval_config()
        # LOG_ON_ERROR defaults to a file inside the installed trackeval package.
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
    out = {}
    for c in cases:
        r = res["MotChallenge2DBox"][c][SEQ]["pedestrian"]
        out[c] = {
            "HOTA": float(r["HOTA"]["HOTA"].mean()),
            "DetA": float(r["HOTA"]["DetA"].mean()),
            "AssA": float(r["HOTA"]["AssA"].mean()),
            "IDF1": float(r["Identity"]["IDF1"]),
            "MOTA": float(r["CLEAR"]["MOTA"]),
            "IDSW": int(r["CLEAR"]["IDSW"]),
            "FP": int(r["CLEAR"]["CLR_FP"]),
        }
    return out


def check(results: dict) -> list[str]:
    """Return a list of failed expectations (empty if all pass)."""
    fails = []
    p, s, d, e = (results[k] for k in ("perfect", "idswitch", "distractor", "empty"))
    if not all(abs(p[k] - 1.0) < 1e-9 for k in ("HOTA", "IDF1", "MOTA")):
        fails.append(f"perfect: expected HOTA=IDF1=MOTA=1, got {p}")
    if s["IDSW"] != 1 or not s["IDF1"] < 1.0 or abs(s["DetA"] - 1.0) > 1e-9:
        fails.append(f"idswitch: expected IDSW=1, IDF1<1, DetA=1, got {s}")
    if d["FP"] != 0 or abs(d["MOTA"] - 1.0) > 1e-9:
        fails.append(f"distractor: expected FP=0, MOTA=1, got {d}")
    if e["HOTA"] != 0.0:
        fails.append(f"empty: expected HOTA=0, got {e}")
    return fails
