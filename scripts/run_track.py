"""Track sequences from cached or live detections, write MOT files, optionally evaluate.

    # cached detections (accuracy runs; D15)
    uv run python scripts/run_track.py --cache outputs/detections/<key_dir> \\
        --tracker bytetrack --seqs MOT17-09 MOT17-05 --run-name smoke01 --eval

    # live detector (speed / consistency checks)
    uv run python scripts/run_track.py --weights weights/yolo26s.pt --imgsz 640 \\
        --tracker bytetrack --seqs MOT17-09 --run-name live01 --timing

Outputs go to ``outputs/tracks/<run-name>/``: ``data/<seq>.txt`` (MOT format),
``run.json`` (configuration, versions, timings) and, with ``--eval``,
``trackeval.json``. ``--video`` writes an annotated MP4 per sequence (local only).
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from cvdt.data.config import REPO_ROOT, dataset_config, open_sequence
from cvdt.detect.cache import CachedSequence, read_meta
from cvdt.pipeline.run import TRACKER_INPUT_CONF, run_sequence
from cvdt.track.ultralytics_trackers import TRACKER_TYPES, UltralyticsTracker
from cvdt.utils import gpu

OUT_ROOT = REPO_ROOT / "outputs" / "tracks"
TIER2_TEST = {"MOT17-02", "MOT17-10"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache", type=Path, help="cache key directory (outputs/detections/...)")
    src.add_argument("--weights", help="run the detector live")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--tracker", choices=TRACKER_TYPES, required=True)
    p.add_argument("--no-fps-scaling", action="store_true", help="disable D17 buffer scaling")
    p.add_argument("--clip-boxes", action="store_true", help="clip track boxes like model.track()")
    p.add_argument("--tracker-input-conf", type=float, default=TRACKER_INPUT_CONF)
    p.add_argument("--seqs", nargs="+", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--eval", action="store_true", help="run TrackEval on the outputs")
    p.add_argument("--video", action="store_true", help="write annotated MP4s (local only)")
    p.add_argument("--timing", action="store_true", help="record stage timings and GPU state")
    p.add_argument("--max-frames", type=int)
    p.add_argument(
        "--allow-tier2-test",
        action="store_true",
        help="required to run on Tier 2 test sequences (D2)",
    )
    args = p.parse_args()

    if TIER2_TEST & set(args.seqs) and not args.allow_tier2_test:
        sys.exit(f"refusing to run on Tier 2 test sequences {sorted(TIER2_TEST & set(args.seqs))}")
    device = int(args.device) if args.device.isdigit() else args.device
    run_dir = OUT_ROOT / args.run_name
    detector = None
    meta = None
    if args.cache:
        meta = read_meta(args.cache)
    else:
        from cvdt.detect.yolo import YOLODetector

        detector = YOLODetector(
            args.weights, imgsz=args.imgsz, conf=args.tracker_input_conf, device=device
        )

    record = {
        "tracker": args.tracker,
        "fps_scaling": not args.no_fps_scaling,
        "clip_boxes": args.clip_boxes,
        "tracker_input_conf": args.tracker_input_conf,
        "source": "cache" if args.cache else "live",
        "cache_dir": str(args.cache) if args.cache else None,
        "cache_key": meta["key"] if meta else None,
        "weights": args.weights,
        "imgsz": args.imgsz if detector else meta["key"]["imgsz"],
        "ultralytics": version("ultralytics"),
        "gpu_before": gpu.query() if args.timing else None,
        "sequences": {},
    }
    for seq in args.seqs:
        ds = open_sequence(seq)
        tracker = UltralyticsTracker(
            args.tracker,
            fps=ds.info.fps,
            frame_shape=(ds.info.height, ds.info.width),
            scale_buffer_by_fps=not args.no_fps_scaling,
            clip_boxes=args.clip_boxes,
        )
        if detector is not None:
            detector.warmup((ds.info.height, ds.info.width, 3), n=5)
        res = run_sequence(
            ds,
            tracker,
            run_dir / "data" / f"{seq}.txt",
            detector=detector,
            cache=CachedSequence(args.cache / f"{seq}.npz") if args.cache else None,
            tracker_input_conf=args.tracker_input_conf,
            loose_iou=0.95 if (detector is not None and args.tracker == "tracktrack") else None,
            video_path=(run_dir / "video" / f"{seq}.mp4") if args.video else None,
            max_frames=args.max_frames,
        )
        record["sequences"][seq] = {
            "frames": res.frames,
            "mot_lines": res.mot_lines,
            "fps": ds.info.fps,
            "track_buffer_frames": tracker.cfg.track_buffer,
            "timing": res.timer.summary(skip_first=10) if args.timing else None,
        }
        print(
            f"{seq}: {res.frames} frames, {res.mot_lines} MOT lines, "
            f"buffer={tracker.cfg.track_buffer} frames"
        )
    if args.timing:
        record["gpu_after"] = gpu.query()
    if args.eval:
        from cvdt.eval.tracking import evaluate

        gt_root = REPO_ROOT / dataset_config()["canonical_root"]
        record["trackeval"] = evaluate(
            OUT_ROOT, args.run_name, args.seqs, gt_root, out_json=run_dir / "trackeval.json"
        )
        for s, m in record["trackeval"].items():
            print(
                f"  {s:13s} "
                + " ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()
                )
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
