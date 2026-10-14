"""Import upstream TrackEval on NumPy >= 1.24.

Upstream TrackEval (last commit 12c8791, 2022-11-29) still uses ``np.float``,
``np.int`` and ``np.bool``, which NumPy deprecated in 1.20 and removed in 1.24.
They are hit on every MOT evaluation (``metrics/identity.py`` computes IDF1
with ``.astype(np.int)``; ``datasets/mot_challenge_2d_box.py`` uses ``np.int``
in preprocessing) and in HOTA's empty-sequence branches.

The removed names were plain aliases of the Python builtins (``np.float is
float`` in NumPy < 1.24), so restoring exactly those aliases reproduces the
original behaviour; no metric code is changed. The alternatives considered were
pinning NumPy < 1.24 (incompatible with the current torch/OpenCV stack) and the
PyPI ``trackeval`` fork (refactors the HOTA code; we prefer the reference
implementation and use the fork only as a numerical cross-check).
"""

from __future__ import annotations

import types

import numpy as np

_REMOVED_ALIASES = {"float": float, "int": int, "bool": bool, "object": object}


def restore_numpy_aliases() -> list[str]:
    """Add the removed builtin aliases to ``numpy`` if missing. Returns names added."""
    added = []
    for name, value in _REMOVED_ALIASES.items():
        if name not in np.__dict__:
            setattr(np, name, value)
            added.append(name)
    return added


def import_trackeval() -> types.ModuleType:
    """Return the ``trackeval`` module, importable and usable on modern NumPy."""
    restore_numpy_aliases()
    import trackeval

    return trackeval
