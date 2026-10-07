"""Running an op tile by tile, on the host.

opspec plans the tiles (``opspec.tiling``); this carries a plan out. Each
tile is read from the input as numpy -- the input itself may be a zarr or
dask array that never fits in memory whole -- handed to the op through the
ordinary call path, and its result is written into the output: cropped to the
tile's core, or blended into its neighbours across the overlap. The op sees plain numpy and knows nothing of tiles,
and the host never holds more than one tile's input and result at a time.

The loop runs on the host, so each tile is one round trip to the worker.
That is negligible for a handful of large tiles, which is what a memory
budget produces; an op cut into thousands of small ones tiles itself
(docs/design/0017-memory-and-tiled-processing/tiling.md).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opspec.op import OpSpec
from opspec.tiling import TilePlan, parse_bytes, plan_tiles

__all__ = [
    "DEFAULT_FRACTION",
    "Tiler",
    "default_budget",
    "gpu_free",
    "prepare_out",
    "run_tiles",
]

#: How tiles can be put back together. None means crop.
MERGES = (None, "crop", "blend")

#: The share of the memory free right now that an op may have by default.
DEFAULT_FRACTION = 0.85


def default_budget(
    fraction: float = DEFAULT_FRACTION, device: str = "cpu"
) -> int | None:
    """The memory an op may use, when the caller names no budget.

    For ``device="cpu"``, *fraction* of what is available right now: what the
    machine has free, or, when this process runs inside a memory-limited
    cgroup -- under ``systemd-run -p MemoryMax=``, a Slurm job, a container --
    the room left in it, whichever is less. The machine's own figure knows
    nothing of the cgroup, and the cgroup's limit is the one that kills.

    For ``device="gpu"``, *fraction* of the GPU's free memory (``gpu_free``),
    or None when that cannot be found out -- and then nothing is tiled.
    """
    if device == "gpu":
        free = gpu_free()
        return None if free is None else max(0, int(fraction * free))

    import psutil

    available = psutil.virtual_memory().available
    room = _cgroup_headroom()
    if room is not None:
        available = min(available, room)
    return max(0, int(fraction * available))


def gpu_free() -> int | None:
    """Bytes free on the first NVIDIA GPU, or None if there is none to ask.

    Asked of ``nvidia-smi``, so the host needs no CUDA library of its own:
    the op's environment has cupy, and the host does not. The first GPU is
    the one an op uses unless told otherwise.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    free = _parse_nvidia_smi(text)
    return free[0] if free else None


def _parse_nvidia_smi(text: str) -> list[int]:
    """Bytes free per GPU, from ``nvidia-smi --query-gpu=memory.free``'s MiB."""
    return [int(line) * 1024**2 for line in text.split() if line.strip().isdigit()]


@dataclass(frozen=True)
class Tiler:
    """The caller's tiling choices, in one object; see tiling.md in 0017.

    The op declares hints (``@op(tile=..., peak_memory=...)``); these are
    what a caller, or a user in a GUI, decides on top of them::

        Tiler()                         # the defaults
        Tiler(memory=0.5)               # half the free memory
        Tiler(memory="2G", overlap=16)  # a set budget, and a smaller overlap
        Tiler(tile_shape=(64, 512, 512))
        Tiler.off()                     # never tile

    ``memory`` is a fraction of the free memory of the op's device -- RAM,
    or the GPU's for an op whose ``PeakMemory`` says ``device="gpu"`` -- or
    a size, in bytes or as ``"2G"``. ``overlap`` and ``tile_shape`` override
    the op's hint and the budget's choice. ``plan`` turns all of it into the
    tiles for one call, which a front end shows and the runner then follows.
    """

    memory: float | int | str | None = None
    overlap: int | tuple[int, ...] | None = None
    tile_shape: tuple[int, ...] | None = None
    enabled: bool = True

    @classmethod
    def off(cls) -> Tiler:
        """A Tiler that never tiles."""
        return cls(enabled=False)

    def budget(self, device: str = "cpu") -> int | None:
        """The memory budget, in bytes, for an op on *device*; None if unknown."""
        if self.memory is None:
            return default_budget(DEFAULT_FRACTION, device)
        if isinstance(self.memory, float) and 0 < self.memory <= 1:
            return default_budget(self.memory, device)
        return parse_bytes(self.memory)

    def plan(self, op: Any, args: dict) -> TilePlan | None:
        """How this call is cut up, or None when it is not tiled at all.

        None when tiling is off, the op declares no hints, or the budget for
        its device cannot be found out. A plan of one tile means it fits.
        """
        spec = op if isinstance(op, OpSpec) else OpSpec.from_op(op)
        if not (self.enabled and spec.tile and spec.peak_memory):
            return None
        if spec.merge not in MERGES:
            raise NotImplementedError(
                f"Op {spec.name}: merging tiles by {spec.merge!r} is not "
                f"implemented; only {' and '.join(MERGES[1:])} are"
            )
        peak = spec.peak_memory
        budget = self.budget(peak.device)
        if budget is None:
            return None
        # The first tiled input sets the shape; any others are tiled the same.
        image = args[spec.tile[0]]
        values = {p.name: p.default for p in spec.params if not p.required}
        values.update(args)
        if self.overlap is not None:
            overlap = self.overlap
        else:
            overlap = spec.overlap.resolve(values) if spec.overlap else 0
        if peak.pad is None:
            pad = 0
        elif peak.pad.param and values.get(peak.pad.param) is None and self.overlap:
            # Not made yet -- a workflow makes its PSF partway through -- so
            # assume the op pads by the overlap. The run plans again with the
            # real one.
            pad = self.overlap
        else:
            pad = peak.pad.resolve(values)
        # The copies a tile makes on its way: read on the host, into shared
        # memory for the worker, and its result back the same way. They are
        # in RAM, so they count against a RAM budget only; a GPU op's budget
        # is the GPU's. The result's dtype is not known until it exists; the
        # op's working dtype is the best guess.
        item_in = np.dtype(image.dtype).itemsize
        item_out = np.dtype(peak.dtype or image.dtype).itemsize
        extra = 0 if peak.device == "gpu" else 2 * item_in + 2 * item_out
        return plan_tiles(
            image.shape,
            image.dtype,
            peak,
            budget,
            overlap,
            axes=_split_axes(spec, image.ndim),
            extra=extra,
            pad=pad,
            tile_shape=self.tile_shape,
        )


def _split_axes(spec: OpSpec, ndim: int) -> list[int] | None:
    """Which of the input's axes ``@op(split=...)`` lets a tile be cut along.

    The names are the tiled input's Axes, matched to the input from the right,
    so Axes("z?", "y", "x") reads a volume as z, y, x and a plane as y, x. An
    input with no axis names to match against may be cut along any axis.
    """
    if not spec.split:
        return None
    param = next(p for p in spec.params if p.name == spec.tile[0])
    if param.axes is None or param.axes.variadic or ndim > len(param.axes.names):
        return None
    names = param.axes.names[len(param.axes.names) - ndim :]
    return [axis for axis, name in enumerate(names) if name in spec.split]


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


def run_tiles(
    call: Callable[[Any], Any],
    image: Any,
    plan: TilePlan,
    out: Any = None,
    room: int | None = None,
    merge: str | None = "crop",
) -> Any:
    """Call *call* on each tile of *image*, and put the results together.

    Args:
        call: Runs the op on one tile and returns its result. It is given the
            tile as numpy, or, when *image* is a dict, a dict of tiles.
        image: The whole input: numpy, or anything that slices into numpy.
            Or a dict of inputs, all tiled the same way -- an image and
            its mask, say. A ``None`` in the dict is passed through as
            ``None``: an optional input nobody gave.
        plan: From ``opspec.tiling.plan_tiles``.
        out: Where the result goes; see ``prepare_out``.
        room: Bytes the whole result may take in memory, when it goes to no
            *out*. By default what the plan's budget leaves beside its
            tiles; a GPU op's budget is the GPU's, so its caller says.
        merge: ``"crop"`` keeps each tile's core. ``"blend"`` fades each
            tile out across the overlap and adds them, so neighbours mix
            smoothly where they meet; it needs a float result.

    Returns:
        The whole result: *out*, if given. Otherwise a numpy array when it
        fits beside the tiles in the budget, and a ``numpy.memmap`` backed by
        a temporary ``.npy`` file when it does not.
    """
    several = isinstance(image, dict)
    inputs = image if several else {"input": image}
    for name, value in inputs.items():
        if value is not None and tuple(value.shape) != plan.shape:
            raise ValueError(
                f"{name} has shape {tuple(value.shape)}, but the tiles are "
                f"planned for {plan.shape}. Inputs tiled together must match in shape."
            )

    blend = merge == "blend"
    weights = _blend_weights(plan) if blend else None
    output = None
    for number, tile in enumerate(plan.tiles, 1):
        pieces = {
            name: None if value is None else np.asarray(value[tile.read])
            for name, value in inputs.items()
        }
        result = np.asarray(call(pieces if several else pieces["input"]))
        shape = next(p.shape for p in pieces.values() if p is not None)
        if result.shape != shape:
            raise ValueError(
                f"Tile {number} of {plan.calls}: the op returned shape "
                f"{result.shape} for an input of shape {shape}. Cropping "
                "tiles back together needs a result the shape of its input."
            )
        if blend and not np.issubdtype(result.dtype, np.floating):
            raise ValueError(
                f"Blending tiles needs a float result; the op returned {result.dtype}"
            )
        if output is None and out is not None:
            output = prepare_out(out, plan.shape, result.dtype)
            if blend:
                output[...] = 0  # tiles are added into it
        elif output is None:
            # The tiles' own cost is already planned into the budget; the
            # whole output stays in memory only if it fits beside that.
            if room is None:
                room = plan.budget - plan.peak
            output = _allocate(plan.shape, result.dtype, room, zeros=blend)
        if blend:
            for axis, (part, by_range) in enumerate(zip(tile.read, weights)):
                shape = [1] * result.ndim
                shape[axis] = -1
                result = result * by_range[part.start, part.stop].reshape(shape)
            output[tile.read] = output[tile.read] + result
        else:
            output[tile.write] = result[tile.keep]
    if isinstance(output, np.memmap):
        output.flush()
    return output


def prepare_out(out: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    """The array a result is written into, from what a caller passed as *out*.

    *out* is anything that takes ``out[region] = values`` -- numpy, a zarr
    array, an HDF5 dataset -- or a function ``(shape, dtype) -> array`` that
    makes one, for a caller that cannot know the result's dtype until there
    is one. Which kind of array is the caller's business; this only checks
    that the shape is the result's.
    """
    target = out(shape, dtype) if callable(out) else out
    if tuple(target.shape) != tuple(shape):
        raise ValueError(
            f"out has shape {tuple(target.shape)}, but the result's is {tuple(shape)}"
        )
    return target


def _blend_weights(plan: TilePlan) -> list[dict[tuple[int, int], np.ndarray]]:
    """For each axis, the weight of each tile along it, by the tile's read range.

    A tile's weight is 1 except across an overlap, where it ramps linearly
    from 1 to 0 over the middle of the overlap, centred on the cut. The outer
    part, nearest the edge of what the tile read, gets no weight: an op sees
    that edge as the image's edge, so its result there is the least like the
    untiled one. At a real image edge there is no overlap and no ramp. Each
    weight is divided by the sum of all weights at that pixel, so the tiles'
    weights add up to exactly 1 everywhere. Tiles are a grid, so a tile's
    weight is the product of its weight along each axis.
    """
    weights = []
    for axis, size in enumerate(plan.shape):
        parts = {
            (t.read[axis].start, t.read[axis].stop): (
                t.read[axis].start + t.keep[axis].start,  # core start
                t.read[axis].start + t.keep[axis].stop,  # core stop
            )
            for t in plan.tiles
        }
        raw = {}
        total = np.zeros(size, np.float64)
        for (first, last), (start, stop) in parts.items():
            x = np.arange(first, last) + 0.5
            weight = np.ones(last - first)
            if start > first:  # overlap on the left: ramp up across the cut
                side = start - first
                ramp = (x - first - side / 2) / side
                weight = np.minimum(weight, np.clip(ramp, 0, 1))
            if last > stop:  # and on the right: ramp down
                side = last - stop
                ramp = (last - x - side / 2) / side
                weight = np.minimum(weight, np.clip(ramp, 0, 1))
            raw[first, last] = weight
            total[first:last] += weight
        weights.append(
            {
                key: (weight / total[key[0] : key[1]]).astype(np.float32)
                for key, weight in raw.items()
            }
        )
    return weights


def _allocate(
    shape: tuple[int, ...], dtype: Any, room: int, zeros: bool = False
) -> np.ndarray:
    """The whole output: in memory if it fits in *room*, on disk if not.

    A new memmap file is zeros already; one in memory is only if *zeros*.
    """
    if np.dtype(dtype).itemsize * int(np.prod(shape)) <= room:
        return np.zeros(shape, dtype=dtype) if zeros else np.empty(shape, dtype=dtype)
    folder = Path(tempfile.mkdtemp(prefix="skop-tiled-"))
    return np.lib.format.open_memmap(
        folder / "result.npy", mode="w+", dtype=dtype, shape=shape
    )
