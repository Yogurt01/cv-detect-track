"""Download, checksum and extract dataset archives.

Only the standard library is used so that the download step has no
dependency beyond Python itself.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

CHUNK = 8 * 1024 * 1024


def free_bytes(path: Path) -> int:
    """Free disk space on the filesystem holding ``path`` (or its nearest existing parent)."""
    p = path.resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def remote_size(url: str, timeout: float = 30.0) -> int | None:
    """Content-Length of ``url`` from a HEAD request, or None if the server does not send it."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
    return int(length) if length is not None else None


def download(url: str, dest: Path, expected_size: int | None = None, timeout: float = 60.0) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial ``dest.part`` file if present.

    The final file is only moved into place once the byte count matches
    ``expected_size`` (when known), so an interrupted run never leaves a
    truncated archive under the final name.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if offset and resp.status != 206:
            offset = 0  # server ignored the Range header; start over
        mode = "ab" if offset else "wb"
        total = expected_size or 0
        done, last_report, t0 = offset, 0.0, time.monotonic()
        with open(part, mode) as f:
            while chunk := resp.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last_report > 15:
                    rate = (done - offset) / max(now - t0, 1e-6) / 1e6
                    pct = f"{100 * done / total:5.1f}%" if total else "?"
                    print(f"  {done / 1e9:6.2f} GB  {pct}  {rate:5.1f} MB/s", file=sys.stderr)
                    last_report = now
    size = part.stat().st_size
    if expected_size is not None and size != expected_size:
        raise OSError(f"incomplete download: {size} of {expected_size} bytes; rerun to resume")
    part.rename(dest)
    return dest


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, out_dir: Path, prefixes: tuple[str, ...] = ()) -> int:
    """Extract members of a zip archive whose path starts with one of ``prefixes``.

    Rejects absolute paths and ``..`` components (zip-slip). Returns the number
    of files written. Existing files with the right size are skipped, so the
    call is idempotent.
    """
    out_dir = out_dir.resolve()
    n = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            if prefixes and not any(info.filename.startswith(p) for p in prefixes):
                continue
            target = out_dir.joinpath(*name.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            n += 1
    return n
