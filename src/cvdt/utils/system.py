"""Read-only snapshot of power and CPU / GPU state for speed benchmarks (DECISIONS.md D20).

Everything here only reads system state; nothing changes settings.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from cvdt.utils import gpu

POWER_SUPPLY = Path("/sys/class/power_supply")
CPU_ROOT = Path("/sys/devices/system/cpu")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def ac_power() -> dict:
    """Mains adapters and whether any is online (``None`` if unknown, e.g. not Linux)."""
    supplies = {}
    if POWER_SUPPLY.exists():
        for d in sorted(POWER_SUPPLY.iterdir()):
            if _read(d / "type") == "Mains":
                supplies[d.name] = _read(d / "online")
    online = any(v == "1" for v in supplies.values()) if supplies else None
    return {"on_ac": online, "mains": supplies}


def power_profile() -> dict:
    out = {"platform_profile": _read(Path("/sys/firmware/acpi/platform_profile"))}
    if shutil.which("powerprofilesctl"):
        try:
            out["powerprofilesctl"] = subprocess.run(
                ["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out["powerprofilesctl"] = None
    return out


def cpu_state() -> dict:
    governors = Counter(
        g for p in CPU_ROOT.glob("cpu[0-9]*/cpufreq/scaling_governor") if (g := _read(p))
    )
    epp = Counter(
        e
        for p in CPU_ROOT.glob("cpu[0-9]*/cpufreq/energy_performance_preference")
        if (e := _read(p))
    )
    model = None
    cpuinfo = _read(Path("/proc/cpuinfo")) or ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    return {
        "model": model,
        "governors": dict(governors),
        "energy_performance_preference": dict(epp),
    }


def snapshot(gpu_index: int = 0) -> dict:
    return {
        "power": ac_power(),
        "power_profile": power_profile(),
        "cpu": cpu_state(),
        "gpu": gpu.query(gpu_index),
        "os": platform.platform(),
    }
