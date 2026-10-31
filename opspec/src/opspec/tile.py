"""Tiling hints: which inputs may be cut into tiles, and how.

Hints, not rules: whoever does the tiling uses them for its defaults, and a
caller can override any of them. The tiling itself is not here; opspec's is
in tiling.py. Design: scikit-ops'
docs/design/0017-memory-and-tiled-processing/tiling.md.

Standard library only, so any project can use it without extra dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Overlap:
    """How far a tile reads past its core on each side, in pixels::

        Overlap(10)                                   # always 10 pixels
        Overlap(param="sigma", scale=4)               # 4 x sigma
        Overlap(param="psf", of="shape", scale=0.5)   # half the PSF, per axis
        Overlap(param="psf", of="shape", scale=0.5,
                only_if="noncirc", otherwise=10)      # 10 when noncirc is off

    A formula is data, not code, so any front end can evaluate it.
    ``pixels`` is added to it. With ``only_if``, the formula applies when that
    parameter is true, and ``otherwise`` pixels when it is false.

    With ``of="shape"`` the parameter is an array, and the overlap is per
    axis, from its shape, rounded down: half a kernel of 31 is 15, exactly
    how far it reaches. Otherwise the parameter is a number, or one per axis
    (the largest is used), and the overlap is rounded up.
    """

    pixels: int = 0
    param: str | None = None
    scale: float = 1.0
    of: str | None = None
    only_if: str | None = None
    otherwise: int = 0

    def resolve(self, values: dict) -> int | tuple[int, ...]:
        """The overlap for a call with these argument *values*.

        One number, the same on every axis, or one per axis for ``of="shape"``.
        """
        if self.only_if is not None and not values.get(self.only_if):
            return self.otherwise
        if self.param is None:
            return self.pixels
        value = values.get(self.param)
        if value is None:
            raise ValueError(f"Overlap needs {self.param}, and the call has none")
        if self.of == "shape":
            return tuple(self.pixels + math.floor(self.scale * n) for n in value.shape)
        if isinstance(value, (list, tuple)):
            value = max(value)
        return self.pixels + math.ceil(self.scale * float(value))

    def __post_init__(self) -> None:
        if self.of not in (None, "shape"):
            raise ValueError(f"Overlap of={self.of!r}: only 'shape' is known")


@dataclass(frozen=True, init=False)
class Tiling:
    """Which inputs may be cut into tiles, and how::

        Tiling(tile="image", overlap=16, merge="crop")
        Tiling(tile=("image", "mask"), split=("y", "x"),
               overlap=Overlap(param="psf", of="shape", scale=0.5), merge="blend")

    - ``tile``: the inputs that may be tiled, all cut the same way.
    - ``split``: the axes a tile may be cut along; empty means any axis.
    - ``overlap``: how far each tile reads past its edges, as an ``Overlap``
      or a number of pixels.
    - ``merge``: how the tiles are put back together, ``"crop"`` (keep each
      tile's core, drop the overlap) or ``"blend"`` (neighbours fade together).
    """

    tile: tuple[str, ...]
    split: tuple[str, ...]
    overlap: Overlap | None
    merge: str | None

    def __init__(
        self,
        tile: str | tuple[str, ...] = (),
        split: tuple[str, ...] = (),
        overlap: Overlap | int | None = None,
        merge: str | None = None,
    ) -> None:
        # frozen blocks normal assignment, so set fields as dataclass does.
        object.__setattr__(
            self, "tile", (tile,) if isinstance(tile, str) else tuple(tile)
        )
        object.__setattr__(self, "split", tuple(split))
        object.__setattr__(
            self, "overlap", Overlap(overlap) if isinstance(overlap, int) else overlap
        )
        object.__setattr__(self, "merge", merge)
