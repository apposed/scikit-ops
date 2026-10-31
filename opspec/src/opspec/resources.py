"""Resource hints: what an op needs to run, such as its peak memory.

Hints, not rules. A runner can use them to size tiles, and a host to pick a
machine, a VM or an HPC node with enough memory.

Standard library only, so any project can use it without extra dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .tile import Overlap

#: Bytes per element, by dtype name. opspec cannot ask numpy.
_ITEMSIZE = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def _dtype_name(dtype: Any) -> str:
    """``"float32"``, from a name, a numpy dtype or a numpy scalar type."""
    name = getattr(dtype, "name", None) if not isinstance(dtype, str) else dtype
    if not isinstance(name, str):
        name = getattr(dtype, "__name__", str(dtype))
    if name not in _ITEMSIZE:
        raise ValueError(f"Unknown dtype for memory accounting: {dtype!r}")
    return name


@dataclass(frozen=True)
class PeakMemory:
    """An op's peak memory, as a multiple of its input's size::

        PeakMemory(scale=2, dtype="float32")   # two float32 copies at once
        PeakMemory(16, "float32", device="gpu")

    - ``dtype``: what the op's buffers hold; None means the input's dtype.
    - ``fixed``: bytes that don't grow with the input, such as a model's
      weights.
    - ``device``: ``"cpu"`` (RAM) or ``"gpu"`` (GPU memory). The budget is
      taken from that device.
    """

    scale: float
    dtype: str | None = None
    fixed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.dtype is not None:
            object.__setattr__(self, "dtype", _dtype_name(self.dtype))
        if self.device not in ("cpu", "gpu"):
            raise ValueError(f"PeakMemory device={self.device!r}: 'cpu' or 'gpu'")

    def bytes_for(self, n_elements: int, input_dtype: Any = "uint8") -> int:
        """Peak bytes for an input of *n_elements* elements."""
        itemsize = _ITEMSIZE[self.dtype or _dtype_name(input_dtype)]
        return math.ceil(self.scale * n_elements * itemsize) + self.fixed


@dataclass(frozen=True, init=False)
class Resources:
    """What an op needs to run::

        Resources(peak_memory=PeakMemory(16, "float32", device="gpu"),
                  pad=Overlap(param="psf", of="shape", scale=0.5))

    - ``peak_memory``: its peak memory, as a ``PeakMemory``.
    - ``pad``: how far it pads its input on each side, as an ``Overlap`` or
      a number of pixels. ``peak_memory`` then counts the padded size. An
      FFT op pads by part of its kernel; with ``Overlap(only_if=...)`` the
      padding, and so the memory, can depend on a parameter.
    """

    peak_memory: PeakMemory | None
    pad: Overlap | None

    def __init__(
        self,
        peak_memory: PeakMemory | None = None,
        pad: Overlap | int | None = None,
    ) -> None:
        # frozen blocks normal assignment, so set fields as dataclass does.
        object.__setattr__(self, "peak_memory", peak_memory)
        object.__setattr__(self, "pad", Overlap(pad) if isinstance(pad, int) else pad)
