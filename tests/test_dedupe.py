"""Deduplication: logic on synthetic variants, and the real canonical set."""

from __future__ import annotations

import pandas as pd
import pytest

from cvdt.data import dedupe
from cvdt.data.config import REPO_ROOT, dataset_config

CANON = REPO_ROOT / dataset_config()["canonical_root"]

SEQINFO = (
    "[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength=2\n"
    "imWidth=4\nimHeight=4\nimExt=.jpg\n"
)


def _make_variant(root, seq, variant, images, gt="1,1,1,1,2,2,1,1,1.0\n"):
    d = root / f"{seq}-{variant}"
    (d / "img1").mkdir(parents=True)
    (d / "gt").mkdir()
    for name, data in images.items():
        (d / "img1" / name).write_bytes(data)
    (d / "gt" / "gt.txt").write_text(gt)
    (d / "seqinfo.ini").write_text(SEQINFO.format(name=f"{seq}-{variant}"))


def test_identical_variants_give_one_canonical_copy(tmp_path):
    raw, canon = tmp_path / "raw", tmp_path / "canon"
    for v in dedupe.VARIANTS:
        _make_variant(raw, "MOT17-02", v, {"000001.jpg": b"a", "000002.jpg": b"b"})
    report = dedupe.dedupe_mot17(raw, canon)
    assert report["summary"]["all_variants_identical"]
    assert report["summary"]["canonical_sequences"] == 1
    assert sorted(p.name for p in (canon / "MOT17-02" / "img1").iterdir()) == [
        "000001.jpg",
        "000002.jpg",
    ]
    assert "name=MOT17-02\n" in (canon / "MOT17-02" / "seqinfo.ini").read_text()


def test_differing_image_blocks_canonical_copy(tmp_path):
    raw, canon = tmp_path / "raw", tmp_path / "canon"
    for v in dedupe.VARIANTS:
        second = b"DIFFERENT" if v == "SDP" else b"b"
        _make_variant(raw, "MOT17-02", v, {"000001.jpg": b"a", "000002.jpg": second})
    report = dedupe.dedupe_mot17(raw, canon)
    assert not report["summary"]["all_variants_identical"]
    assert report["sequences"]["MOT17-02"]["differing_images"] == 1
    assert not canon.exists()


def test_differing_gt_blocks_canonical_copy(tmp_path):
    raw, canon = tmp_path / "raw", tmp_path / "canon"
    for v in dedupe.VARIANTS:
        gt = "1,1,1,1,2,2,1,1,0.5\n" if v == "DPM" else "1,1,1,1,2,2,1,1,1.0\n"
        _make_variant(raw, "MOT17-02", v, {"000001.jpg": b"a"}, gt=gt)
    report = dedupe.dedupe_mot17(raw, canon)
    assert report["sequences"]["MOT17-02"]["gt_identical"] is False
    assert not report["summary"]["all_variants_identical"]


def test_exactly_seven_canonical_sequences():
    manifest = CANON / "image_hashes.csv"
    if not manifest.exists():
        pytest.skip("canonical MOT17 not built; run scripts/prepare_data.py dedupe")
    seq_dirs = sorted(p.name for p in CANON.iterdir() if p.is_dir())
    assert seq_dirs == sorted(dataset_config()["sequences"])
    assert len(seq_dirs) == 7
    df = pd.read_csv(manifest)
    assert df["sequence"].nunique() == 7
    assert not df["sha256"].duplicated().any()
    for seq in seq_dirs:
        assert len(list((CANON / seq / "img1").glob("*.jpg"))) == (df["sequence"] == seq).sum()
