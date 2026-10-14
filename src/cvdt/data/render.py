"""Render ground-truth boxes for a visual check of the filtering rule.

Colours (BGR): kept positive = green, ignore region = orange,
dropped by visibility = red. Non-person classes (DROPPED) are not drawn.
Outputs contain dataset frames and are git-ignored.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cvdt.data import filtering
from cvdt.data.filtering import Category
from cvdt.data.mot import load_gt, mot_to_xyxy, read_seqinfo

COLORS = {
    Category.POSITIVE: (0, 200, 0),
    Category.IGNORE: (0, 140, 255),
    Category.LOW_VISIBILITY: (0, 0, 255),
}
LEGEND = [
    ("kept positive", Category.POSITIVE),
    ("ignore region", Category.IGNORE),
    ("dropped: visibility", Category.LOW_VISIBILITY),
]


def sample_frames(seq_lengths: dict[str, int], n: int, seed: int) -> list[tuple[str, int]]:
    """``n`` (sequence, frame) pairs spread over all sequences: an even share
    each, the remainder assigned to randomly chosen sequences."""
    rng = np.random.default_rng(seed)
    seqs = sorted(seq_lengths)
    per = {s: n // len(seqs) for s in seqs}
    for s in rng.choice(seqs, size=n % len(seqs), replace=False):
        per[str(s)] += 1
    out = []
    for s in seqs:
        frames = rng.choice(np.arange(1, seq_lengths[s] + 1), size=per[s], replace=False)
        out += [(s, int(f)) for f in sorted(frames)]
    return out


def draw_frame(
    seq_dir: Path, frame: int, min_visibility: float, max_width: int = 1600
) -> np.ndarray:
    si = read_seqinfo(seq_dir / "seqinfo.ini")
    img = cv2.imread(
        str(seq_dir / si.get("imDir", "img1") / f"{frame:06d}{si.get('imExt', '.jpg')}")
    )
    df = load_gt(seq_dir / "gt" / "gt.txt")
    g = df[df["frame"] == frame]
    cat = filtering.categorize(
        g["cls"].to_numpy(), g["conf"].to_numpy(), g["vis"].to_numpy(), min_visibility
    )
    boxes = mot_to_xyxy(g[["x", "y", "w", "h"]].to_numpy())
    scale = min(1.0, max_width / img.shape[1])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    thick = max(1, round(2 * scale + 0.5))
    # Draw ignore and low-visibility boxes first so positives stay on top.
    for c in (Category.IGNORE, Category.LOW_VISIBILITY, Category.POSITIVE):
        for b, v, cls in zip(
            boxes[cat == c],
            g["vis"].to_numpy()[cat == c],
            g["cls"].to_numpy()[cat == c],
            strict=True,
        ):
            p0 = tuple(int(round(x * scale)) for x in b[:2])
            p1 = tuple(int(round(x * scale)) for x in b[2:])
            cv2.rectangle(img, p0, p1, COLORS[c], thick)
            if c != Category.POSITIVE:
                tag = f"c{cls} v{v:.2f}"
                cv2.putText(
                    img,
                    tag,
                    (p0[0], max(p0[1] - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLORS[c],
                    1,
                    cv2.LINE_AA,
                )
    counts = {c: int((cat == c).sum()) for c in COLORS}
    y = 22
    header = f"{seq_dir.name} frame {frame}  min_vis={min_visibility}"
    cv2.rectangle(img, (0, 0), (430, 22 + 20 * len(LEGEND)), (0, 0, 0), -1)
    cv2.putText(
        img, header, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    for label, c in LEGEND:
        y += 20
        cv2.putText(
            img,
            f"{label}: {counts[c]}",
            (6, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS[c],
            1,
            cv2.LINE_AA,
        )
    return img


def render_check_frames(
    canon_root: Path, out_dir: Path, n: int, seed: int, min_visibility: float
) -> list[Path]:
    seq_dirs = {
        p.name: p
        for p in sorted(canon_root.iterdir())
        if p.is_dir() and p.name.startswith("MOT17-")
    }
    lengths = {s: int(read_seqinfo(p / "seqinfo.ini")["seqLength"]) for s, p in seq_dirs.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (s, f) in enumerate(sample_frames(lengths, n, seed)):
        path = out_dir / f"{i:02d}_{s}_{f:06d}.jpg"
        cv2.imwrite(
            str(path), draw_frame(seq_dirs[s], f, min_visibility), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        paths.append(path)
    return paths


def camera_montage(seq_dir: Path, out_path: Path, n: int = 4, width: int = 480) -> Path:
    """Evenly spaced frames side by side, to judge camera motion by eye."""
    si = read_seqinfo(seq_dir / "seqinfo.ini")
    length = int(si["seqLength"])
    tiles = []
    for f in np.linspace(1, length, n).round().astype(int):
        img = cv2.imread(str(seq_dir / "img1" / f"{f:06d}.jpg"))
        img = cv2.resize(img, (width, round(width * img.shape[0] / img.shape[1])))
        cv2.putText(
            img,
            f"{seq_dir.name} f{f}",
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(img)
    h = min(t.shape[0] for t in tiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack([t[:h] for t in tiles]))
    return out_path
