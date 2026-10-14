"""Load dataset metadata and split configs from ``configs/data``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cvdt.data.mot import MOT17Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "data"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dataset_config(name: str = "mot17") -> dict:
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def open_sequence(seq: str, cfg: dict | None = None) -> MOT17Sequence:
    """Open a canonical MOT17 sequence with its configured camera type."""
    cfg = cfg or dataset_config()
    meta = dict(cfg["sequences"][seq])
    camera = meta.pop("camera")
    return MOT17Sequence(REPO_ROOT / cfg["canonical_root"] / seq, camera=camera,
                         extra={k: str(v) for k, v in meta.items()})


@dataclass(frozen=True)
class SplitConfig:
    """A named assignment of sequences to splits (e.g. train / val / test)."""

    name: str
    tier: int
    dataset: str
    splits: dict[str, list[str]]
    path: Path

    @property
    def sequences(self) -> list[str]:
        return [s for seqs in self.splits.values() for s in seqs]


def load_split(path: Path) -> SplitConfig:
    d = load_yaml(path)
    return SplitConfig(
        name=d["name"], tier=int(d["tier"]), dataset=d["dataset"],
        splits={k: list(v) for k, v in d["splits"].items()}, path=Path(path),
    )


def split_config_paths() -> list[Path]:
    """All split configs: every YAML in ``configs/data`` with a ``splits`` key."""
    return sorted(p for p in CONFIG_DIR.glob("*.yaml") if "splits" in (load_yaml(p) or {}))
