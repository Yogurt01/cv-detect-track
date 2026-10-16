"""Measure how low-visibility ignore regions affect false positives (D13 vs D19).

Low-visibility pedestrians are ignore regions with a full-body (amodal) box,
while a detector boxes the visible part. A detection is "on" a low-visibility
pedestrian if it is not matched to a positive (IoU 0.5) and at least half of its
area lies inside that pedestrian's box (IoA >= 0.5).

For each ignore rule the script reports, at IoU 0.5 and the given confidence:
how many such detections end up as false positives, their share of all false
positives, and (D19) how many detections the duplicate guard turned into false
positives although an ignore region would otherwise have removed them. The
labels come from :func:`cvdt.eval.detection.match_image`, i.e. the same code
that computes AP. This is a diagnostic; it does not change any decision.

    uv run python scripts/measure_ignore_match.py --cache outputs/detections/<key_dir> \\
        --seqs MOT17-09 MOT17-05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cvdt.data.config import REPO_ROOT, dataset_config
from cvdt.detect.cache import CachedSequence
from cvdt.eval.detection import (
    FP,
    IGNORE_IOA,
    IGNORE_RULES,
    R_FP_GUARD,
    R_FP_GUARD_EXCUSABLE,
    TP,
    ioa_matrix,
    iou_matrix,
    load_eval_gt,
    match_image,
)


def measure(seq: str, cache: CachedSequence, thr: float, min_conf: float, rule: str) -> dict:
    root = REPO_ROOT / dataset_config()["canonical_root"] / seq
    gts = load_eval_gt(root, thr, "primary")
    n_fp = n_on = n_on_fp = n_guard = n_guard_overrode = 0
    on_ious: list[float] = []
    for f, gt in gts.items():
        d = cache.tight(f, min_conf)
        _, labels, reasons = match_image(d.boxes, d.scores, gt, 0.5, rule, return_reasons=True)
        boxes = d.boxes[np.argsort(-d.scores, kind="mergesort")]
        n_fp += int((labels == FP).sum())
        n_guard += int(((reasons == R_FP_GUARD) | (reasons == R_FP_GUARD_EXCUSABLE)).sum())
        n_guard_overrode += int((reasons == R_FP_GUARD_EXCUSABLE).sum())
        unmatched = labels != TP
        if not unmatched.any() or len(gt.low_vis) == 0:
            continue
        ioa = ioa_matrix(boxes[unmatched], gt.low_vis)
        on = ioa.max(axis=1) >= IGNORE_IOA
        best = ioa.argmax(axis=1)
        iou_best = iou_matrix(boxes[unmatched], gt.low_vis)[np.arange(unmatched.sum()), best]
        n_on += int(on.sum())
        n_on_fp += int((on & (labels[unmatched] == FP)).sum())
        on_ious += iou_best[on].tolist()
    ious = np.array(on_ious)
    return {
        "false_positives": n_fp,
        "on_low_visibility_pedestrian": n_on,
        "iou_below_0.5_with_that_box": int((ious < 0.5).sum()),
        "on_low_vis_counted_as_FP": n_on_fp,
        "share_of_all_FP": n_on_fp / n_fp if n_fp else float("nan"),
        "duplicate_guard_FP": n_guard,
        "duplicate_guard_overrode_ignore": n_guard_overrode,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--seqs", nargs="+", required=True)
    p.add_argument("--min-conf", type=float, nargs="+", default=[0.25, 0.1])
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    thr = float(dataset_config()["eval"]["detection"]["ignore_visibility_below"])
    out = {}
    for rule in IGNORE_RULES:
        for c in args.min_conf:
            for seq in args.seqs:
                r = measure(seq, CachedSequence(args.cache / f"{seq}.npz"), thr, c, rule)
                out[f"{rule}/{seq}@conf{c:g}"] = r
                print(
                    f"{rule} {seq} conf>={c:g}: "
                    + ", ".join(
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                    )
                )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
