"""Repository hygiene: ignore rules must not hide source files or tracked files.

An unanchored pattern such as ``data/`` matches every directory named ``data``,
including source packages; these checks catch that kind of mistake.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "tests", "scripts", "configs")


def _git(*args: str) -> list[str]:
    out = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


@pytest.fixture(scope="module", autouse=True)
def _require_git_repo():
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    try:
        inside = _git("rev-parse", "--is-inside-work-tree")
    except subprocess.CalledProcessError:
        pytest.skip("not a git repository")
    if inside != ["true"]:
        pytest.skip("not a git work tree")


def _is_bytecode(path: str) -> bool:
    return "__pycache__/" in path or path.endswith(".pyc")


def test_no_source_file_is_ignored():
    ignored = _git("ls-files", "--others", "--ignored", "--exclude-standard", "--", *SOURCE_DIRS)
    offenders = [p for p in ignored if not _is_bytecode(p)]
    assert offenders == [], f"files under {SOURCE_DIRS} are hidden by an ignore rule: {offenders}"


def test_no_tracked_file_matches_an_ignore_rule():
    offenders = _git("ls-files", "-ci", "--exclude-standard")
    assert offenders == [], f"tracked files match an ignore rule: {offenders}"
