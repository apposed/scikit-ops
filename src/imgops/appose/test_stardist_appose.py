"""
Test running StarDist remotely via appose.

Run this from the *naplari-hacking* environment (which has appose). It will
launch StarDist in another pixi environment and return a label image.

Usage:
    # from the naplari-hacking pixi env
    python appose/test_stardist_appose.py /path/to/stardist/pixi/env

If no env path is given, it tries to resolve one pinned in the registry for
``StardistCommand``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sibling packages importable when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "magicgui"))
sys.path.insert(0, str(_ROOT / "appose"))

REPO_ROOT = Path(__file__).resolve().parents[1]
STARDIST_ENV = REPO_ROOT / "pixi" / "stardist"

from execute_appose import execute_appose, run_segmenter_remotely  # noqa: E402
from StardistMagicGuiCommand import StardistCommand, StardistModel  # noqa: E402


def _make_test_image(size: int = 256, n_blobs: int = 12) -> np.ndarray:
    """Synthetic grayscale image with a few bright gaussian blobs."""
    rng = np.random.default_rng(0)
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_blobs):
        cy, cx = rng.integers(20, size - 20, size=2)
        r = rng.integers(6, 14)
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
    img += 0.02 * rng.standard_normal(img.shape)
    return img.astype(np.float32)


def main() -> int:
    image = _make_test_image()

    command = StardistCommand()
    command.model = StardistModel.fluo
    command.prob_thresh = 0.5
    command.nms_thresh = 0.4
    command.normalize = True

    print("Flattened execution inputs (primitives sent to worker):")
    for k, v in command.get_execution_inputs().items():
        print(f"  {k} = {v!r}")

    env_path = sys.argv[1] if len(sys.argv) > 1 else STARDIST_ENV

    if env_path:
        print(f"\nRunning StarDist remotely in env: {env_path}")
        mask = execute_appose(image, command, env_path)
    else:
        print("\nNo env path given; resolving from registry...")
        mask = run_segmenter_remotely(command, image, env='')

    n = int(mask.max())
    print(f"\nRemote StarDist done. dtype={mask.dtype}, "
          f"shape={mask.shape}, objects={n}")

    assert mask.shape == image.shape, "mask shape must match input"
    assert n > 0, "expected StarDist to find at least one object"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
