"""Keep appose's own pixi new enough to build environments.

Appose downloads a pixi of its own and builds every environment with it.
Released appose (0.12.0) pins pixi v0.58.0, and that pixi now fails fresh
builds with ``io error: unexpected end of file`` while solving: every recipe
under ``envs/`` that has PyPI packages, which is all of them. Environments
built earlier keep working, because an up-to-date environment is never
installed again; a new user cannot build any.

Appose's main fixed this upstream (apposed/appose-python#10, pin v0.81.0, and
then a self-update), but no release carries it yet. Until one does, this
does the same from here:

* Before appose has downloaded pixi, point its download at ``MINIMUM``, so a
  new user gets that version and no other.
* If appose already has an older pixi, ``pixi self-update`` it once, before
  the first build that needs to install anything.

Remove this module once scikit-ops requires an appose release whose pinned
pixi is at least ``MINIMUM``.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

#: The pixi appose's main pins; the version apposed/appose-python#10 tested.
MINIMUM = "0.81.0"

_checked = False


def _parse(text: str) -> tuple[int, ...] | None:
    found = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(part) for part in found.groups()) if found else None


def _installed_version(command: Path) -> tuple[int, ...] | None:
    out = subprocess.run(
        [str(command), "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return _parse(out.stdout)


def ensure_recent_pixi(report: Callable[[str], None] = print) -> None:
    """Make appose build with a pixi at least ``MINIMUM``. Once per process.

    Never raises: a failure is reported, and the build goes ahead with
    whatever pixi there is, so this can only ever help.
    """
    global _checked
    if _checked:
        return
    _checked = True
    try:
        from appose.tool.pixi import Pixi
    except ImportError:
        return

    minimum = _parse(MINIMUM)
    pinned = _parse(Pixi.PIXI_VERSION)
    if pinned is not None and pinned >= minimum:
        return  # an appose that is new enough already

    # Before appose's first download: fetch MINIMUM instead of its pin.
    Pixi.PIXI_VERSION = f"v{MINIMUM}"
    if Pixi.PIXI_BINARY:
        Pixi.DOWNLOAD_URL = (
            "https://github.com/prefix-dev/pixi/releases/download/"
            f"{Pixi.PIXI_VERSION}/{Pixi.PIXI_BINARY}"
        )

    # After it: an older pixi already installed is updated in place.
    try:
        command = Path(Pixi().command)
        if not command.exists():
            return
        installed = _installed_version(command)
        if installed is None or installed >= minimum:
            return
        old = ".".join(map(str, installed))
        report(
            f"Updating appose's pixi from {old} to {MINIMUM} ({command}): "
            f"pixi {old} can no longer build environments"
        )
        subprocess.run(
            [str(command), "self-update", "--version", MINIMUM],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        now = _installed_version(command)
        if now is None or now < minimum:
            report(f"pixi still reports {now}; building with it anyway")
    except Exception as e:  # noqa: BLE001 - never block a build over this
        report(f"Could not update appose's pixi ({e}); building with it anyway")
