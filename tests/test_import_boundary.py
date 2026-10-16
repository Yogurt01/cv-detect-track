"""Ultralytics (AGPL-3.0) may only be imported in cvdt.detect and in the one
tracker adapter module (decisions D5, D15). The project's own tracker code and
everything else must not import it."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_DIR = REPO_ROOT / "src" / "cvdt" / "detect"
ALLOWED_FILES = {REPO_ROOT / "src" / "cvdt" / "track" / "ultralytics_trackers.py"}


def _imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _offenders():
    offenders = []
    for folder in ("src", "scripts", "tests"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if ALLOWED_DIR in path.parents or path in ALLOWED_FILES:
                continue
            if any(m.split(".")[0] == "ultralytics" for m in _imports(path)):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    return offenders


def test_ultralytics_only_imported_in_allowed_modules():
    assert _offenders() == []


def test_allowed_adapter_module_exists():
    # Guards against the allow-list silently pointing at a renamed file.
    assert all(p.is_file() for p in ALLOWED_FILES)
