"""Split validation and per-split summaries.

A split config (``configs/data/tier*.yaml``) assigns whole sequences to named
splits. Splitting is by sequence, never by frame: consecutive frames are near
duplicates, so a frame-level split leaks scenes and identities.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from cvdt.data.config import SplitConfig


def validate(split: SplitConfig, known_sequences: set[str]) -> list[str]:
    """Return a list of problems (empty if the split is valid)."""
    problems = []
    counts = Counter(split.sequences)
    for seq, n in counts.items():
        if n > 1:
            where = [k for k, v in split.splits.items() if seq in v]
            problems.append(f"{seq} appears {n} times (splits: {where})")
    unknown = set(counts) - known_sequences
    if unknown:
        problems.append(f"unknown sequences: {sorted(unknown)}")
    return problems


def hash_overlap(split: SplitConfig, manifest: Path) -> dict[tuple[str, str], int]:
    """Number of identical image hashes shared by each pair of splits."""
    df = pd.read_csv(manifest)
    hashes = {k: set(df.loc[df["sequence"].isin(v), "sha256"]) for k, v in split.splits.items()}
    names = sorted(hashes)
    return {(a, b): len(hashes[a] & hashes[b]) for i, a in enumerate(names) for b in names[i + 1 :]}


def summarize(split: SplitConfig, stats: pd.DataFrame) -> pd.DataFrame:
    """Per-split totals from ``results/data_stats.csv``."""
    s = stats.set_index("sequence")
    rows = []
    for name, seqs in split.splits.items():
        sub = s.loc[seqs]
        rows.append(
            {
                "split": name,
                "sequences": " ".join(seqs),
                "static": int((sub["camera"] == "static").sum()),
                "moving": int((sub["camera"] == "moving").sum()),
                "frames": int(sub["frames"].sum()),
                "gt_tracks": int(sub["gt_tracks"].sum()),
                "eval_gt_boxes": int(sub["ped_flag1"].sum()),
                "kept_boxes": int(sub["kept_positive"].sum()),
                "kept_pct": round(100 * sub["kept_positive"].sum() / s["kept_positive"].sum(), 1),
                "kept_small_lt50px": int(sub["kept_h_small"].sum()),
            }
        )
    return pd.DataFrame(rows)
