"""Divides large inputs into tiles that fit a memory budget.

Ops can declare the overlap they need, and their peak memory as a multiple of
the input size. ``plan_tiles`` uses these to calculate a tile size that fits
the memory available. Reading the tiles, calling the op on each one and
writing the results to the output is the runner's job.

A tile has three regions. Its *core* is its share of the image; the cores
cover the image exactly, with no gaps or overlap. It *reads* its core plus the
overlap on every side, except at the image's real edges, where the op handles
the boundary as it would untiled. It *keeps* the core of its result, and
*writes* it where the core came from.
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
    """Bytes, from an int or a size such as ``"1G"``, ``"512M"``, ``"1.5GiB"``.

    Units are binary (1K is 1024), as in ``systemd-run -p MemoryMax=`` and
    ``free``.
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

    #: What to read from the input: the core plus overlap.
    read: tuple[slice, ...]
    #: Where the kept part of the result goes in the output.
    write: tuple[slice, ...]
    #: Which part of the tile's result to keep: its core.
    keep: tuple[slice, ...]


@dataclass(frozen=True)
class TilePlan:
    """How an input is cut into tiles, and the peak memory of the largest."""

    #: The input's shape.
    shape: tuple[int, ...]
    #: The core of a full tile; tiles at the far edges may be smaller.
    tile_shape: tuple[int, ...]
    #: One number for every axis, or one per axis.
    overlap: int | tuple[int, ...]
    tiles: tuple[Tile, ...]
    #: Peak bytes of the largest tile, overlap and padding included.
    peak: int
    #: The budget, in bytes.
    budget: int

    @property
    def calls(self) -> int:
        return len(self.tiles)

    @property
    def summary(self) -> str:
        """One line for a GUI to show."""
        if self.calls == 1:
            return "1 tile: the whole input fits"
        size = " x ".join(str(n) for n in self.tile_shape)
        overlap = (
            " x ".join(str(n) for n in self.overlap)
            if isinstance(self.overlap, tuple)
            else self.overlap
        )
        return f"{self.calls} tiles of {size}, overlap {overlap}"


def _per_axis(value: int | Sequence[int], ndim: int, what: str) -> tuple[int, ...]:
    """One number per axis, from one number or a sequence of them."""
    if isinstance(value, int):
        return (value,) * ndim
    value = tuple(int(n) for n in value)
    if len(value) != ndim:
        raise ValueError(f"{what} has {len(value)} values for {ndim} axes")
    return value


def plan_tiles(
    shape: Sequence[int],
    dtype: Any,
    peak_memory: PeakMemory,
    budget: int | str,
    overlap: int | Sequence[int] = 0,
    axes: Sequence[int] | None = None,
    extra: int = 0,
    pad: int | Sequence[int] = 0,
    tile_shape: Sequence[int] | None = None,
) -> TilePlan:
    """Cut an input of *shape* into tiles whose peak memory fits *budget*.

    Starts from the whole input and halves the longest cuttable axis until a
    tile fits.

    Args:
        shape: The input's shape.
        dtype: The input's dtype, as a name or a numpy dtype. Used by
            ``peak_memory`` when it names no dtype of its own.
        peak_memory: The op's ``PeakMemory``.
        budget: Bytes, or a size such as ``"1G"``.
        overlap: Pixels a tile reads past its core on each side, on cut axes.
            One number, or one per axis.
        axes: The axes that may be cut; all by default. Leave out any the op
            must see whole, such as colour.
        extra: Bytes per element the caller holds while a tile runs, on top
            of the op's peak: its copies of the tile and the result, sent to
            a worker and back.
        pad: Pixels the op pads its input by on each side, one number or one
            per axis. Memory is counted on the padded size.
        tile_shape: A tile size chosen by the caller, used instead of the
            largest that fits. Its peak is still computed, but a tile over
            the budget is not refused.

    Raises:
        ValueError: If even the smallest tile doesn't fit.
    """
    shape = tuple(int(n) for n in shape)
    budget = parse_bytes(budget)
    cuttable = range(len(shape)) if axes is None else tuple(axes)
    reach = _per_axis(overlap, len(shape), "overlap")
    padding = _per_axis(pad, len(shape), "pad")
    core = (
        [min(int(n), size) for n, size in zip(tile_shape, shape)]
        if tile_shape is not None
        else list(shape)
    )

    def extent(axis: int, size: int) -> int:
        # An uncut axis is read whole; a cut one, its core plus overlap.
        if size >= shape[axis]:
            return shape[axis]
        return min(size + 2 * reach[axis], shape[axis])

    def cost(sizes: list[int]) -> int:
        read = math.prod(extent(axis, size) for axis, size in enumerate(sizes))
        # The op works on what it read, padded on every side.
        worked = math.prod(
            extent(axis, size) + 2 * padding[axis] for axis, size in enumerate(sizes)
        )
        return peak_memory.bytes_for(worked, dtype) + extra * read

    while tile_shape is None and cost(core) > budget:
        # Halve the longest axis that can still be cut.
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
            side = reach[axis] if core[axis] < shape[axis] else 0
            first, last = max(0, start - side), min(shape[axis], end + side)
            read.append(slice(first, last))
            write.append(slice(start, end))
            keep.append(slice(start - first, end - first))
        tiles.append(Tile(tuple(read), tuple(write), tuple(keep)))

    return TilePlan(
        shape=shape,
        tile_shape=tuple(core),
        overlap=overlap if isinstance(overlap, int) else reach,
        tiles=tuple(tiles),
        peak=cost(core),
        budget=budget,
    )
