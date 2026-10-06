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

__all__ = ["run_tiles"]


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
            output = _allocate(plan.shape, result.dtype, plan.budget)
        output[tile.write] = result[tile.keep]
    if isinstance(output, np.memmap):
        output.flush()
    return output


def _allocate(shape: tuple[int, ...], dtype: Any, budget: int) -> np.ndarray:
    """The whole output: in memory if it fits the budget, on disk if not."""
    if np.dtype(dtype).itemsize * int(np.prod(shape)) <= budget:
        return np.empty(shape, dtype=dtype)
    folder = Path(tempfile.mkdtemp(prefix="skop-tiled-"))
    return np.lib.format.open_memmap(
        folder / "result.npy", mode="w+", dtype=dtype, shape=shape
    )
