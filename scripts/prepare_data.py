"""Prepare MOT17: download, dedupe, stats, YOLO conversion and visual check.

Thin CLI over ``cvdt.data``. Each step is idempotent; ``all`` runs them in order.

    uv run python scripts/prepare_data.py all
    uv run python scripts/prepare_data.py download
    uv run python scripts/prepare_data.py dedupe
    uv run python scripts/prepare_data.py stats
    uv run python scripts/prepare_data.py convert
    uv run python scripts/prepare_data.py render --n 20 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CANON = DATA / "mot17" / "train"
YOLO = DATA / "yolo" / "mot17"

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
# SHA256 of MOT17.zip as downloaded on 2026-10-14 (Last-Modified 2020-12-18).
# motchallenge.net does not publish a checksum; this pins the copy we used.
MOT17_SHA256 = "7b486bb2af5cfedbfabc020511ff933a3edbf12182ce4985eec1a210f20f9e06"


def cmd_download(args: argparse.Namespace) -> None:
    from cvdt.data import download as dl

    archive = RAW / "MOT17.zip"
    size = dl.remote_size(MOT17_URL)
    print(f"remote size: {size} bytes ({(size or 0) / 1e9:.2f} GB)")
    free = dl.free_bytes(RAW)
    print(f"free disk space before: {free / 1e9:.1f} GB")
    # Archive plus extracted train split, with a safety margin.
    need = 3 * (size or 6e9)
    if free < need:
        sys.exit(f"not enough disk space: need ~{need / 1e9:.1f} GB, have {free / 1e9:.1f} GB")
    dl.download(MOT17_URL, archive, expected_size=size)
    digest = dl.sha256sum(archive)
    print(f"sha256: {digest}")
    if digest != MOT17_SHA256:
        sys.exit(f"checksum mismatch: expected {MOT17_SHA256}")
    prefixes = ("MOT17/train/",) + (("MOT17/test/",) if args.include_test else ())
    n = dl.safe_extract(archive, RAW, prefixes)
    print(f"extracted {n} new files under {RAW}")
    print(f"free disk space after: {dl.free_bytes(RAW) / 1e9:.1f} GB")


def cmd_dedupe(args: argparse.Namespace) -> None:
    from cvdt.data import dedupe

    report = dedupe.dedupe_mot17(RAW / "MOT17" / "train", CANON, variant=args.variant)
    out = ROOT / "results" / "dedupe_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {out}")
    if not report["summary"]["all_variants_identical"]:
        sys.exit("variants differ; no canonical copy built. See the report.")


def cmd_stats(args: argparse.Namespace) -> None:
    from cvdt.data import stats

    df = stats.compute_stats(CANON, min_visibility=args.min_visibility)
    out = ROOT / "results" / "data_stats.csv"
    df.to_csv(out, index=False)
    print(stats.to_markdown(df))
    print(f"wrote {out}")


def cmd_convert(args: argparse.Namespace) -> None:
    from cvdt.data import yolo

    summary = yolo.convert_all(CANON, YOLO, min_visibility=args.min_visibility)
    print(json.dumps(summary, indent=2))


def cmd_render(args: argparse.Namespace) -> None:
    from cvdt.data import render

    out = ROOT / "reports" / "figures" / "data_check"
    paths = render.render_check_frames(
        CANON, out, n=args.n, seed=args.seed, min_visibility=args.min_visibility
    )
    print(f"rendered {len(paths)} frames to {out}")
    for seq in sorted(p for p in CANON.iterdir() if p.is_dir()):
        render.camera_montage(seq, out / "camera" / f"{seq.name}.jpg")
    print(f"wrote camera montages to {out / 'camera'}")


def default_min_visibility() -> float:
    from cvdt.data.config import dataset_config

    return float(dataset_config()["labels"]["min_visibility_train"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download, checksum and extract MOT17")
    d.add_argument("--include-test", action="store_true", help="also extract the test split")
    dd = sub.add_parser("dedupe", help="hash the 3 variants per sequence, keep one copy")
    dd.add_argument("--variant", default="FRCNN", help="variant used as the canonical copy")
    for name in ("stats", "convert", "render", "all"):
        sp = sub.add_parser(name)
        sp.add_argument(
            "--min-visibility",
            type=float,
            default=default_min_visibility(),
            help="training-label visibility threshold (configs/data/mot17.yaml)",
        )
        if name in ("render", "all"):
            sp.add_argument("--n", type=int, default=20)
            sp.add_argument("--seed", type=int, default=0)
        if name == "all":
            sp.add_argument("--include-test", action="store_true")
            sp.add_argument("--variant", default="FRCNN")
    args = p.parse_args()
    steps = {
        "download": [cmd_download],
        "dedupe": [cmd_dedupe],
        "stats": [cmd_stats],
        "convert": [cmd_convert],
        "render": [cmd_render],
        "all": [cmd_download, cmd_dedupe, cmd_stats, cmd_convert, cmd_render],
    }[args.cmd]
    for step in steps:
        step(args)


if __name__ == "__main__":
    main()
