"""Ultralytics (AGPL-3.0) may only be imported inside cvdt.detect (decision D5)."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED = REPO_ROOT / "src" / "cvdt" / "detect"


def _imports(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_ultralytics_only_imported_in_detect():
    offenders = []
    for folder in ("src", "scripts", "tests"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if ALLOWED in path.parents:
                continue
            if any(m.split(".")[0] == "ultralytics" for m in _imports(path)):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []
