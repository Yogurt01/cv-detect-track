"""MOT17Sequence implements SequenceDataset and matches the official counts."""

from __future__ import annotations

import pytest

from cvdt.data.base import SequenceDataset
from cvdt.data.config import REPO_ROOT, dataset_config, open_sequence

CANON = REPO_ROOT / dataset_config()["canonical_root"]

# MOT17 data page (https://motchallenge.net/data/MOT17/): frames, pedestrian boxes, tracks.
OFFICIAL = {"MOT17-02": (600, 18581, 62), "MOT17-05": (837, 6917, 133)}


@pytest.mark.parametrize("seq", sorted(OFFICIAL))
def test_sequence_matches_official_counts(seq):
    if not (CANON / seq).exists():
        pytest.skip("canonical MOT17 not built; run scripts/prepare_data.py dedupe")
    ds = open_sequence(seq)
    assert isinstance(ds, SequenceDataset)
    frames, boxes, tracks = OFFICIAL[seq]
    assert len(ds) == frames
    n_ped, ids = 0, set()
    for fr in ds:
        mask = (fr.gt.classes == 1) & (fr.gt.attrs["conf_flag"] == 1)
        n_ped += int(mask.sum())
        ids.update(fr.gt.ids[mask].tolist())
    assert (n_ped, len(ids)) == (boxes, tracks)
    first = ds.frame(1)
    assert first.image_path.exists() and first.load().shape == (ds.info.height, ds.info.width, 3)
