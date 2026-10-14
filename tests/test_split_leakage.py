"""Leakage checks that run on every split config in configs/data."""

from __future__ import annotations

import pytest

from cvdt.data import splits
from cvdt.data.config import REPO_ROOT, dataset_config, load_split, split_config_paths

MANIFEST = REPO_ROOT / dataset_config()["canonical_root"] / "image_hashes.csv"
SPLIT_PATHS = split_config_paths()


def test_split_configs_found():
    names = {p.stem for p in SPLIT_PATHS}
    assert "tier1_all7" in names
    assert any(n.startswith("tier2_") for n in names)


@pytest.mark.parametrize("path", SPLIT_PATHS, ids=lambda p: p.stem)
def test_no_sequence_in_more_than_one_split(path):
    split = load_split(path)
    assert splits.validate(split, set(dataset_config()["sequences"])) == []


@pytest.mark.parametrize("path", SPLIT_PATHS, ids=lambda p: p.stem)
def test_no_image_hash_shared_across_splits(path):
    if not MANIFEST.exists():
        pytest.skip("image hash manifest missing; run scripts/prepare_data.py dedupe")
    overlap = splits.hash_overlap(load_split(path), MANIFEST)
    assert all(n == 0 for n in overlap.values()), overlap


@pytest.mark.parametrize("path", SPLIT_PATHS, ids=lambda p: p.stem)
def test_tier2_test_has_one_static_and_one_moving(path):
    split = load_split(path)
    if split.tier != 2:
        pytest.skip("rule applies to Tier 2 splits only")
    cameras = sorted(dataset_config()["sequences"][s]["camera"] for s in split.splits["test"])
    assert cameras == ["moving", "static"]


def test_validate_flags_overlap():
    bad = splits.SplitConfig(
        name="bad",
        tier=2,
        dataset="MOT17",
        path=REPO_ROOT,
        splits={"train": ["MOT17-02", "MOT17-04"], "test": ["MOT17-04"]},
    )
    assert any("MOT17-04" in p for p in splits.validate(bad, {"MOT17-02", "MOT17-04"}))
