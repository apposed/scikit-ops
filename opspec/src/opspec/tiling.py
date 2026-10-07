"""Cutting one call into tiles that fit a memory budget.

Arithmetic only -- shapes, overlaps and bytes, from an op's ``PeakMemory``
and ``Overlap``. Reading the tiles, calling the op on each and putting the
results back touches pixels, and is a runner's job.

A tile has three regions. Its *core* is its share of the image: the cores
partition the image exactly. It *reads* its core plus the overlap on every
side, except where that side is the real edge of the image -- there it adds
nothing, and the op handles the boundary as it would untiled. Of the result
it *keeps* the core, and *writes* it where the core came from.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .op import PeakMemory

__all__ = ["Tile", "TilePlan", "parse_bytes", "plan_tiles"]

_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_bytes(size: int | str) -> int:
    """A number of bytes, from ``"1G"``, ``"512M"``, ``"1.5GiB"`` or an int.

    Binary units, as ``systemd-run -p MemoryMax=`` and ``free`` use them.
    """
    if isinstance(size, int):
        return size
    match = re.fullmatch(
        r"\s*([\d.]+)\s*([KMGT]?)(?:i?B)?\s*", str(size), re.IGNORECASE
    )
    if not match:
        raise ValueError(f"Not a size: {size!r}; try '512M' or '2G'")
    number, unit = match.groups()
    return int(float(number) * _UNITS[unit.upper()])


@dataclass(frozen=True)
class Tile:
    """One piece of a tiled call."""

    #: What to read from the input: the core, plus overlap.
    read: tuple[slice, ...]
    #: Where the kept part of the result goes in the output.
    write: tuple[slice, ...]
    #: Which part of the tile's result to keep: its core.
    keep: tuple[slice, ...]


@dataclass(frozen=True)
class TilePlan:
    """How a call is cut up, and what each piece costs."""

    shape: tuple[int, ...]
    #: The core of a full tile; tiles at the far edges may be smaller.
    tile_shape: tuple[int, ...]
    overlap: int
    tiles: tuple[Tile, ...]
    #: Peak bytes for the costliest tile, overlap included.
    peak: int
    budget: int

    @property
    def calls(self) -> int:
        return len(self.tiles)

    @property
    def summary(self) -> str:
        """One line for a front end to show."""
        if self.calls == 1:
            return "1 tile: the whole input fits"
        size = " x ".join(str(n) for n in self.tile_shape)
        return f"{self.calls} tiles of {size}, overlap {self.overlap}"


def plan_tiles(
    shape: Sequence[int],
    dtype: Any,
    peak_memory: PeakMemory,
    budget: int | str,
    overlap: int = 0,
    axes: Sequence[int] | None = None,
    extra: int = 0,
) -> TilePlan:
    """Cut an input of *shape* into tiles whose peak fits *budget*.

    Args:
        shape: The input's shape.
        dtype: The input's dtype, by name or as numpy has it; what
            ``peak_memory`` counts in when it names no dtype of its own.
        peak_memory: The op's declaration.
        budget: Bytes, or a size such as ``"1G"``.
        overlap: Pixels each tile reaches past its edges, on cut axes.
        axes: The axes that may be cut; all of them by default. An axis the
            op must see whole -- colour, say -- is left out.
        extra: Bytes per element the caller holds on top of the op's own
            peak while a tile runs -- its copies of the tile and of the
            result, on their way to a worker and back.

    Raises:
        ValueError: if even the smallest tile does not fit.
    """
    shape = tuple(int(n) for n in shape)
    budget = parse_bytes(budget)
    cuttable = range(len(shape)) if axes is None else tuple(axes)
    core = list(shape)

    def extent(axis: int, size: int) -> int:
        # An uncut axis is read whole; a cut one, its core plus both sides.
        return (
            shape[axis] if size >= shape[axis] else min(size + 2 * overlap, shape[axis])
        )

    def cost(sizes: list[int]) -> int:
        elements = math.prod(extent(axis, size) for axis, size in enumerate(sizes))
        return peak_memory.bytes_for(elements, dtype) + extra * elements

    while cost(core) > budget:
        # Halve the longest side still worth halving.
        candidates = [axis for axis in cuttable if core[axis] > 1]
        if not candidates:
            raise ValueError(
                f"Cannot fit {budget} bytes: the smallest tile, with overlap "
                f"{overlap}, needs {cost(core)}"
            )
        axis = max(candidates, key=lambda a: core[a])
        core[axis] = math.ceil(core[axis] / 2)

    starts = [range(0, n, size) for n, size in zip(shape, core)]
    tiles = []
    for origin in itertools.product(*starts):
        read, write, keep = [], [], []
        for axis, start in enumerate(origin):
            end = min(start + core[axis], shape[axis])
            pad = overlap if core[axis] < shape[axis] else 0
            first, last = max(0, start - pad), min(shape[axis], end + pad)
            read.append(slice(first, last))
            write.append(slice(start, end))
            keep.append(slice(start - first, end - first))
        tiles.append(Tile(tuple(read), tuple(write), tuple(keep)))

    return TilePlan(
        shape=shape,
        tile_shape=tuple(core),
        overlap=overlap,
        tiles=tuple(tiles),
        peak=cost(core),
        budget=budget,
    )
