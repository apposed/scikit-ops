"""Running an op tile by tile, on the host.

opspec plans the tiles (``opspec.tiling``); this carries a plan out. Each
tile is read from the input as numpy -- the input itself may be a zarr or
dask array that never fits in memory whole -- handed to the op through the
ordinary call path, and the part of its result that is the tile's own is
written into the output. The op sees plain numpy and knows nothing of tiles,
and the host never holds more than one tile's input and result at a time.

The loop runs on the host, so each tile is one round trip to the worker.
That is negligible for a handful of large tiles, which is what a memory
budget produces; an op cut into thousands of small ones tiles itself
(docs/design/0017-memory-and-tiled-processing/tiling.md).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from opspec.tiling import TilePlan

__all__ = ["DEFAULT_FRACTION", "default_budget", "run_tiles"]

#: The share of the memory free right now that an op may have by default.
DEFAULT_FRACTION = 0.85


def default_budget(fraction: float = DEFAULT_FRACTION) -> int:
    """The memory an op may use, when the caller names no budget.

    *fraction* of what is available right now: what the machine has free,
    or, when this process runs inside a memory-limited cgroup -- under
    ``systemd-run -p MemoryMax=``, a Slurm job, a container -- the room left
    in it, whichever is less. The machine's own figure knows nothing of the
    cgroup, and the cgroup's limit is the one that kills.
    """
    import psutil

    available = psutil.virtual_memory().available
    room = _cgroup_headroom()
    if room is not None:
        available = min(available, room)
    return max(0, int(fraction * available))


def _cgroup_headroom(
    root: Path = Path("/sys/fs/cgroup"), proc: Path = Path("/proc/self/cgroup")
) -> int | None:
    """Bytes left before this process's cgroup, or one above it, is full.

    cgroup v2 only. A limit can sit on any ancestor -- Slurm sets it on the
    job, one level above the step a program runs in -- so every level is
    read, and the tightest wins. None when there is no limit, or no cgroup
    v2 to read.
    """
    try:
        line = next(
            entry for entry in proc.read_text().splitlines() if entry.startswith("0::")
        )
    except (OSError, StopIteration):
        return None
    group = root / line[3:].strip().lstrip("/")
    room = None
    while True:
        try:
            limit = (group / "memory.max").read_text().strip()
            if limit != "max":
                left = int(limit) - int((group / "memory.current").read_text())
                room = left if room is None else min(room, left)
        except (OSError, ValueError):
            pass  # No memory controller at this level, or no file to read.
        if group == root or group.parent == group:
            return room
        group = group.parent


def run_tiles(call: Callable[[np.ndarray], Any], image: Any, plan: TilePlan) -> Any:
    """Call *call* on each tile of *image*, and put the results together.

    Args:
        call: Runs the op on one tile, given as numpy, and returns its result.
        image: The whole input: numpy, or anything that slices into numpy.
        plan: From ``opspec.tiling.plan_tiles``.

    Returns:
        The whole result: a numpy array when it fits the plan's budget, and
        otherwise a ``numpy.memmap`` backed by a temporary ``.npy`` file.
    """
    output = None
    for number, tile in enumerate(plan.tiles, 1):
        piece = np.asarray(image[tile.read])
        result = np.asarray(call(piece))
        if result.shape != piece.shape:
            raise ValueError(
                f"Tile {number} of {plan.calls}: the op returned shape "
                f"{result.shape} for an input of shape {piece.shape}. Cropping "
                "tiles back together needs a result the shape of its input."
            )
        if output is None:
            # The tiles' own cost is already planned into the budget; the
            # whole output stays in memory only if it fits beside that.
            output = _allocate(plan.shape, result.dtype, plan.budget - plan.peak)
        output[tile.write] = result[tile.keep]
    if isinstance(output, np.memmap):
        output.flush()
    return output


def _allocate(shape: tuple[int, ...], dtype: Any, room: int) -> np.ndarray:
    """The whole output: in memory if it fits in *room*, on disk if not."""
    if np.dtype(dtype).itemsize * int(np.prod(shape)) <= room:
        return np.empty(shape, dtype=dtype)
    folder = Path(tempfile.mkdtemp(prefix="skop-tiled-"))
    return np.lib.format.open_memmap(
        folder / "result.npy", mode="w+", dtype=dtype, shape=shape
    )
