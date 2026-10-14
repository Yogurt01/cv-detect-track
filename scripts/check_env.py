"""Print the software/hardware environment and self-test TrackEval.

uv run python scripts/check_env.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys


def nvidia_smi() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f"unavailable ({e})"


def main() -> None:
    import cv2
    import numpy as np
    import torch

    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": nvidia_smi(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    try:  # read installed versions without importing the packages
        from importlib.metadata import version

        info["ultralytics"] = version("ultralytics")
        info["trackeval"] = version("trackeval")
    except Exception as e:  # noqa: BLE001
        info["versions_error"] = str(e)
    print(json.dumps(info, indent=2))

    from cvdt.eval import trackeval_selftest

    results = trackeval_selftest.run()
    print(json.dumps(results, indent=2))
    fails = trackeval_selftest.check(results)
    print("TrackEval self-test:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  ", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
