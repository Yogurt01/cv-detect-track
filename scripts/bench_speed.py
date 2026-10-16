"""Speed benchmark (DECISIONS.md D20): perception and end-to-end latency, live, never cached.

    uv run python scripts/bench_speed.py --weights weights/yolo26s.pt --imgsz 640 \\
        --tracker bytetrack --name smoke

Variants: ``--weights`` (n / s), ``--imgsz`` (640 / 960), ``--half`` (FP16),
``--nms-free`` (NMS-free one-to-one head, ``nms=False``), ``--tracker`` (any of
the six). Default sequences: MOT17-04 and MOT17-05. Writes
``outputs/bench/<name>.json`` with results, environment and the hardware state
before and after. Stops if the machine is not on AC power; changes no settings.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version

from cvdt.data.config import REPO_ROOT, open_sequence
from cvdt.detect.yolo import YOLODetector
from cvdt.eval.speed import bench_sequence, summarize
from cvdt.pipeline.run import TRACKER_INPUT_CONF
from cvdt.track.ultralytics_trackers import TRACKER_TYPES, UltralyticsTracker
from cvdt.utils import gpu, system

OUT_ROOT = REPO_ROOT / "outputs" / "bench"
TIER2_TEST = {"MOT17-02", "MOT17-10"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 (default FP32)")
    p.add_argument("--nms-free", action="store_true", help="NMS-free one-to-one head (nms=False)")
    p.add_argument("--tracker", choices=TRACKER_TYPES, default="bytetrack")
    p.add_argument("--seqs", nargs="+", default=["MOT17-04", "MOT17-05"])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--min-timed", type=int, default=500)
    p.add_argument("--device", default="0")
    p.add_argument("--name", required=True)
    args = p.parse_args()

    if TIER2_TEST & set(args.seqs):
        sys.exit(
            f"refusing to benchmark on Tier 2 test sequences {sorted(TIER2_TEST & set(args.seqs))}"
        )
    before = system.snapshot()
    if before["power"]["on_ac"] is not True:
        sys.exit(f"not on AC power ({before['power']}); stopping as D20 requires")

    import torch

    device = int(args.device) if args.device.isdigit() else args.device
    detector = YOLODetector(
        args.weights,
        imgsz=args.imgsz,
        conf=TRACKER_INPUT_CONF,
        half=args.half,
        device=device,
        nms=False if args.nms_free else None,
    )
    loose_iou = 0.95 if args.tracker == "tracktrack" and not args.nms_free else None
    record = {
        "protocol": "D20",
        "config": {
            "weights": args.weights,
            "weights_sha256": detector.weights_sha256,
            "imgsz": args.imgsz,
            "precision": "fp16" if args.half else "fp32",
            "head": "nms-free (nms=False)" if args.nms_free else "one-to-many + NMS (nms=None)",
            "tracker": args.tracker,
            "tracker_input_conf": TRACKER_INPUT_CONF,
            "loose_nms_iou": loose_iou,
            "batch": 1,
            "warmup_frames": args.warmup,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": version("ultralytics"),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": _git_commit(),
        },
        "state_before": before,
        "sequences": {},
    }
    for seq in args.seqs:
        ds = open_sequence(seq)
        if len(ds) < args.min_timed:
            sys.exit(f"{seq} has {len(ds)} frames < --min-timed {args.min_timed}")
        tracker = UltralyticsTracker(
            args.tracker, fps=ds.info.fps, frame_shape=(ds.info.height, ds.info.width)
        )
        gpu_before = gpu.query()
        res = summarize(
            bench_sequence(ds, detector, tracker, TRACKER_INPUT_CONF, loose_iou, args.warmup)
        )
        res["gpu_before"], res["gpu_after"] = gpu_before, gpu.query()
        res["frames_timed"] = len(ds)
        res["resolution"] = f"{ds.info.width}x{ds.info.height}"
        record["sequences"][seq] = res
        pc, ee = res["perception"], res["end_to_end"]
        print(
            f"{seq}: perception p50/p95/p99 = {pc['p50_ms']}/{pc['p95_ms']}/{pc['p99_ms']} ms, "
            f"end-to-end p50/p95/p99 = {ee['p50_ms']}/{ee['p95_ms']}/{ee['p99_ms']} ms, "
            f"throughput {res['throughput_fps']} FPS, "
            f"peak mem {res['peak_gpu_mem_reserved_mb']} MB reserved, real-time={res['real_time']}"
        )
    record["state_after"] = system.snapshot()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / f"{args.name}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out}")


def _git_commit() -> str:
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()
