"""A cache for large files an op needs but a repository should not carry.

Model weights are fetched on first use and kept in a user cache directory,
shared across environments and ops. This module runs inside worker processes,
so it stays on the standard library.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from ._progress import progress


def cache_dir() -> Path:
    """Where skop keeps downloaded assets.

    Honors ``OPKIT_CACHE_DIR``, then ``XDG_CACHE_HOME``, then ``~/.cache``.
    """
    override = os.environ.get("OPKIT_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "skop"


def unzip_from_url(url: str, name: str, marker: str | None = None) -> Path:
    """Download a zip archive and extract it into the asset cache.

    Args:
        url: Where to fetch the archive from.
        name: Cache-relative directory to extract into, e.g.
            ``"starfun3d/confocal"``.
        marker: A file expected inside the extracted directory. When it is
            already present, the download is skipped. Defaults to treating any
            non-empty directory as complete.

    Returns:
        The directory the archive was extracted into.
    """
    dest = cache_dir() / name
    if _complete(dest, marker):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Downloading {url}")

    # Extract to a sibling temp directory, then move it into place, so an
    # interrupted download can never look like a complete one.
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}-"))
    try:
        archive = staging / "archive.zip"
        _download(url, archive)

        progress(f"Extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging / "content")
        archive.unlink()

        if dest.exists():
            shutil.rmtree(dest)
        (staging / "content").rename(dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return dest


def file_from_url(url: str, name: str) -> Path:
    """Download a single file into the asset cache.

    The counterpart to :func:`unzip_from_url` for weights that are shipped
    bare rather than zipped. Downloading them here, and handing the library
    an absolute path, is what keeps a model from deciding for itself where
    to put a 145 MB file -- which for several of them means the worker's
    working directory.

    Args:
        url: Where to fetch the file from.
        name: Cache-relative path to store it at, e.g. ``"fastsam/FastSAM-s.pt"``.

    Returns:
        The path of the downloaded file.
    """
    dest = cache_dir() / name
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Downloading {url}")

    # Download beside the target and rename, so an interrupted download is
    # never mistaken for a complete one on the next call.
    fd, staging = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}-")
    os.close(fd)
    staging = Path(staging)
    try:
        _download(url, staging)
        staging.replace(dest)
    finally:
        staging.unlink(missing_ok=True)

    return dest


def _complete(dest: Path, marker: str | None) -> bool:
    if marker is not None:
        return (dest / marker).exists()
    return dest.is_dir() and any(dest.iterdir())


def _download(url: str, target: Path) -> None:
    with urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(target, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    progress(
                        f"Downloaded {done >> 20} of {total >> 20} MiB", done, total
                    )
