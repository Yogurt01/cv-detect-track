"""GPU state logging via ``nvidia-smi`` queries.

Logs clocks, performance state, temperature and utilisation. Power readings are
deliberately not used: on the development laptop they are unreliable (749 W
reported against a 60 W cap).
"""

from __future__ import annotations

import subprocess

FIELDS = (
    "name",
    "driver_version",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "clocks.gr",
    "clocks.max.sm",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
)


def query(index: int = 0) -> dict[str, str]:
    """One snapshot of :data:`FIELDS` for GPU ``index``; empty dict if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                f"--query-gpu={','.join(FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(zip(FIELDS, (v.strip() for v in out.split(",")), strict=False))
