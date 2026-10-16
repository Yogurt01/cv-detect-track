"""Cache person detections per (detector configuration, sequence) (DECISIONS.md D15, D16).

    uv run python scripts/cache_detections.py --weights weights/yolo26s.pt --imgsz 640 \\
        --seqs MOT17-09 MOT17-05

Writes ``outputs/detections/<key_dir>/<seq>.npz`` and ``meta.json``. Existing
sequence files are skipped unless ``--overwrite``. Runs one forward pass per frame;
the loose-NMS pass (TrackTrack TAI) re-runs only NMS on its raw predictions.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from cvdt.data.config import REPO_ROOT, open_sequence
from cvdt.detect.cache import CacheKey, key_dir, write_meta, write_sequence
from cvdt.detect.yolo import YOLODetector
from cvdt.utils import gpu


def git_commit() -> str:
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf-floor", type=float, default=0.01)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--loose-iou", type=float, default=0.95)
    p.add_argument("--half", action="store_true", help="FP16 (default FP32)")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="0")
    p.add_argument("--seqs", nargs="+", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    device = int(args.device) if args.device.isdigit() else args.device

    import torch

    det = YOLODetector(
        args.weights,
        imgsz=args.imgsz,
        conf=args.conf_floor,
        iou=args.iou,
        half=args.half,
        device=device,
        max_det=args.max_det,
    )
    key = CacheKey(
        weights_sha256=det.weights_sha256,
        weights_stem=Path(args.weights).stem,
        imgsz=args.imgsz,
        precision="fp16" if args.half else "fp32",
        conf_floor=args.conf_floor,
        nms_iou=args.iou,
        loose_nms_iou=args.loose_iou,
        classes=(0,),
        max_det=args.max_det,
    )
    out_dir = key_dir(key)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        key={**key.__dict__, "classes": list(key.classes)},
        key_digest=key.digest(),
        weights_path=str(Path(args.weights)),
        ultralytics=version("ultralytics"),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        python=platform.python_version(),
        device=str(device),
        gpu=gpu.query(device if isinstance(device, int) else 0),
        git_commit=git_commit(),
        nms_mode="external NMS (ultralytics nms=None, one-to-many head)",
        batch=1,
        coordinates="0-based xyxy pixels of the original frame",
    )
    meta.setdefault("sequences", {})
    for seq in args.seqs:
        ds = open_sequence(seq)
        path = out_dir / f"{seq}.npz"
        if path.exists() and not args.overwrite:
            print(f"{seq}: cached, skipping ({path})")
            continue
        det.warmup((ds.info.height, ds.info.width, 3), n=3)
        tight, loose = [], []
        t0 = time.perf_counter()
        for fr in ds:
            img = fr.load()
            t, lo = det.detect_with_loose(img, args.loose_iou)  # one forward pass (D16)
            tight.append(t)
            loose.append(lo)
        dt = time.perf_counter() - t0
        write_sequence(path, tight, loose, len(ds))
        counts = np.array([len(d) for d in tight])
        lcounts = np.array([len(d) for d in loose])
        meta["sequences"][seq] = {
            "frames": len(ds),
            "detections_tight": int(counts.sum()),
            "detections_loose": int(lcounts.sum()),
            "max_per_frame_tight": int(counts.max(initial=0)),
            "max_per_frame_loose": int(lcounts.max(initial=0)),
            "frames_hitting_max_det": int(
                (counts >= args.max_det).sum() + (lcounts >= args.max_det).sum()
            ),
            "file_bytes": path.stat().st_size,
            "seconds": round(dt, 2),
        }
        print(
            f"{seq}: {len(ds)} frames, {int(counts.sum())} tight / "
            f"{int(lcounts.sum())} loose dets, "
            f"max/frame {counts.max(initial=0)}, {dt:.1f}s -> {path}"
        )
        write_meta(out_dir, meta)
    write_meta(out_dir, meta)
    print(f"cache dir: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
