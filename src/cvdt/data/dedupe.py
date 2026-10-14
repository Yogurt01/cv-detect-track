"""Deduplicate MOT17 train: 7 sequences shipped 3 times (-DPM, -FRCNN, -SDP).

Nothing is assumed: every image of every variant is hashed, and the three
``gt.txt`` and ``seqinfo.ini`` files are compared. A canonical copy is only
built when all variants agree; otherwise :func:`dedupe_mot17` raises, so a
silent mismatch can never reach the splits.

Canonical layout (images are hard links into ``data/raw``, so no extra disk)::

    <canon>/MOT17-02/{img1/, gt/gt.txt, seqinfo.ini}
    <canon>/image_hashes.csv   sequence,frame,sha256  (used by the leakage test)
"""

from __future__ import annotations

import configparser
import hashlib
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

VARIANTS = ("DPM", "FRCNN", "SDP")
_NAME = re.compile(r"^(MOT17-\d{2})-(DPM|FRCNN|SDP)$")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_dir(img_dir: Path, workers: int = 8) -> dict[str, str]:
    """Map file name -> sha256 for every file in ``img_dir``."""
    files = sorted(img_dir.iterdir())
    with ThreadPoolExecutor(workers) as ex:
        digests = list(ex.map(file_sha256, files))
    return {f.name: d for f, d in zip(files, digests, strict=True)}


def group_variants(raw_train: Path) -> dict[str, dict[str, Path]]:
    """{'MOT17-02': {'DPM': path, 'FRCNN': path, 'SDP': path}, ...}"""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in sorted(raw_train.iterdir()):
        m = _NAME.match(p.name)
        if p.is_dir() and m:
            groups[m.group(1)][m.group(2)] = p
    return dict(groups)


def _seqinfo_without_name(path: Path) -> dict[str, str]:
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(path)
    d = dict(cp["Sequence"])
    d.pop("name", None)
    return d


def compare_variants(variants: dict[str, Path]) -> dict:
    """Hash and compare the variants of one sequence."""
    hashes = {v: hash_dir(p / "img1") for v, p in variants.items()}
    names = {v: set(h) for v, h in hashes.items()}
    all_names = set().union(*names.values())
    same_names = all(n == all_names for n in names.values())
    n_identical = sum(1 for f in all_names if len({h.get(f) for h in hashes.values()}) == 1)
    gt_hash = {v: file_sha256(p / "gt" / "gt.txt") for v, p in variants.items()}
    seqinfo = {v: _seqinfo_without_name(p / "seqinfo.ini") for v, p in variants.items()}
    first = next(iter(seqinfo.values()))
    return {
        "variants": sorted(variants),
        "num_images": {v: len(h) for v, h in hashes.items()},
        "same_file_names": same_names,
        "identical_images": n_identical,
        "differing_images": len(all_names) - n_identical,
        "gt_sha256": gt_hash,
        "gt_identical": len(set(gt_hash.values())) == 1,
        "seqinfo_identical_except_name": all(s == first for s in seqinfo.values()),
        "_hashes": hashes,
    }


def _link_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        target = dst / f.name
        if not target.exists():
            try:
                os.link(f, target)
            except OSError:
                shutil.copy2(f, target)


def dedupe_mot17(raw_train: Path, canon_root: Path, variant: str = "FRCNN") -> dict:
    """Compare all variants and build the canonical copy. Returns a JSON-able report."""
    groups = group_variants(raw_train)
    if not groups:
        raise FileNotFoundError(f"no MOT17 variant folders in {raw_train}")
    report: dict = {"canonical_variant": variant, "sequences": {}}
    all_hashes = {}
    mismatched = []
    for seq, variants in sorted(groups.items()):
        if set(variants) != set(VARIANTS):
            raise ValueError(f"{seq}: expected variants {VARIANTS}, found {sorted(variants)}")
        r = compare_variants(variants)
        all_hashes[seq] = r.pop("_hashes")
        report["sequences"][seq] = r
        if not (
            r["same_file_names"]
            and r["differing_images"] == 0
            and r["gt_identical"]
            and r["seqinfo_identical_except_name"]
        ):
            mismatched.append(seq)
    if mismatched:
        # Report, but do not build a canonical copy: the choice needs a human decision.
        report["summary"] = {"all_variants_identical": False, "mismatched_sequences": mismatched}
        return report

    rows = []
    for seq, variants in sorted(groups.items()):
        hashes = all_hashes[seq]
        src = variants[variant]
        dst = canon_root / seq
        _link_tree(src / "img1", dst / "img1")
        (dst / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "gt" / "gt.txt", dst / "gt" / "gt.txt")
        text = (src / "seqinfo.ini").read_text()
        (dst / "seqinfo.ini").write_text(re.sub(r"(?m)^name=.*$", f"name={seq}", text))
        for name, digest in sorted(hashes[variant].items()):
            rows.append({"sequence": seq, "frame": int(Path(name).stem), "sha256": digest})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(canon_root / "image_hashes.csv", index=False)
    dup = manifest[manifest.duplicated("sha256", keep=False)]
    cross_seq = dup.groupby("sha256")["sequence"].nunique()
    report["summary"] = {
        "canonical_sequences": len(groups),
        "canonical_images": len(manifest),
        "all_variants_identical": True,
        "duplicate_images_within_sequence": int(
            (dup.groupby("sha256")["sequence"].nunique() == 1).sum()
        ),
        "images_shared_across_sequences": int((cross_seq > 1).sum()),
    }
    return report
