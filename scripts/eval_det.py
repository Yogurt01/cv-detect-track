"""Detection evaluation from cached detections (DECISIONS.md D10, D13, D19).

    uv run python scripts/eval_det.py --cache outputs/detections/<key_dir> \\
        --seqs MOT17-09 MOT17-05 --protocol primary

Evaluates under the D19 ignore rule (primary) and under the D13 rule
(sensitivity analysis), per sequence and pooled. Prints AP50, AP50:95,
precision / recall at the operating confidence and the ignore / duplicate-guard
counts; ``--out`` writes everything as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cvdt.data.config import REPO_ROOT, dataset_config
from cvdt.detect.cache import CachedSequence
from cvdt.eval.detection import IGNORE_RULES, OPERATING_CONF, evaluate_images, load_eval_gt

TIER2_TEST = {"MOT17-02", "MOT17-10"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--seqs", nargs="+", required=True)
    p.add_argument("--protocol", choices=("primary", "secondary"), default="primary")
    p.add_argument("--operating-conf", type=float, default=OPERATING_CONF)
    p.add_argument("--out", type=Path)
    p.add_argument("--allow-tier2-test", action="store_true")
    args = p.parse_args()
    if TIER2_TEST & set(args.seqs) and not args.allow_tier2_test:
        sys.exit(f"refusing to run on Tier 2 test sequences {sorted(TIER2_TEST & set(args.seqs))}")

    cfg = dataset_config()
    thr = float(cfg["eval"]["detection"]["ignore_visibility_below"])
    root = REPO_ROOT / cfg["canonical_root"]
    dets_by_seq, gts_by_seq = {}, {}
    for seq in args.seqs:
        gts_by_seq[seq] = load_eval_gt(root / seq, thr, args.protocol)
        cache = CachedSequence(args.cache / f"{seq}.npz")
        dets_by_seq[seq] = {
            f: (d.boxes, d.scores) for f in gts_by_seq[seq] for d in [cache.tight(f)]
        }
    pooled_dets = {(s, f): v for s, d in dets_by_seq.items() for f, v in d.items()}
    pooled_gts = {(s, f): v for s, g in gts_by_seq.items() for f, v in g.items()}

    results: dict[str, dict] = {}
    for rule in IGNORE_RULES:  # "d19" first (primary), then "d13" (sensitivity)
        results[rule] = {
            seq: evaluate_images(dets_by_seq[seq], gts_by_seq[seq], args.operating_conf, rule)
            for seq in args.seqs
        }
        results[rule]["POOLED"] = evaluate_images(
            pooled_dets, pooled_gts, args.operating_conf, rule
        )
        for k, r in results[rule].items():
            print(
                f"{rule} {k:9s} AP50={100 * r['AP50']:.2f} AP50:95={100 * r['AP50_95']:.2f} "
                f"P={100 * r['precision']:.2f} R={100 * r['recall']:.2f} "
                f"TP={r['TP']} FP={r['FP']} FN={r['FN']} ignored={r['ignored']} "
                f"(distractor={r['ignored_distractor']} low_vis={r['ignored_low_vis']}) "
                f"guard_FP={r['fp_duplicate_guard']} "
                f"guard_overrode_ignore={r['fp_guard_overrode_ignore']}"
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "protocol": args.protocol,
            "ignore_visibility_below": thr,
            "primary_rule": "d19",
            "sensitivity_rule": "d13",
            "results": results,
        }
        args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
