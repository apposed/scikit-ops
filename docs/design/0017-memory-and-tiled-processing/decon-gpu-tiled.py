# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-ops", "numpy"]
#
# [tool.uv.sources]
# scikit-ops = { path = "../../..", editable = true }
# ///
"""Tiled Richardson-Lucy on the GPU, on a volume made here

Makes a synthetic volume -- random points blurred by a Gaussian PSF, plus
noise -- and deconvolves it with richardson_lucy_cupy, tiled to fit the
GPU. Prints what the GPU has free, the budget, and how the run will be cut
before it starts; then the time, and how many tiles ran.

The budget is a Tiler's (tiling.md, "The Tiler"): by default 85% of the
GPU's free memory, from nvidia-smi. --memory takes a fraction (0.5) or a
size (1G), and a small one forces tiles on a volume that would fit.
--overlap and --tile override the op's overlap and the budget's tile size.

--compare runs it untiled as well and reports the difference: everywhere,
and away from the seams. Decon tiles are never exactly the untiled result
-- every iteration spreads a pixel further -- so the seams are where to
look. Needs the volume to fit untiled.

Non-circulant by default, as clij2-fft and tnia-python tile decon.
--circulant tiles too, and matches the untiled result away from the image's
own edges; at them, a tile wraps differently from the whole image.

--cpu runs numpy's richardson_lucy instead, in RAM: the same tiling, for a
machine whose GPU is not working.

If CUDA reports "unknown error" after the laptop has slept:
    sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm

D=docs/design/0017-memory-and-tiled-processing
uv run $D/decon-gpu-tiled.py                          # fits: one tile
uv run $D/decon-gpu-tiled.py --memory 1G              # force tiles
uv run $D/decon-gpu-tiled.py --memory 1G --compare    # and check the seams
uv run $D/decon-gpu-tiled.py --shape 128 2048 2048    # bigger than the GPU
uv run $D/decon-gpu-tiled.py --memory 1G --overlap 4  # less than half the PSF
uv run $D/decon-gpu-tiled.py --memory 1G --compare --circulant   # the wrong way
uv run $D/decon-gpu-tiled.py --cpu --shape 32 256 256 --memory 64M --compare
"""

import argparse
import time

import numpy as np

import skop
from skop._tiling import gpu_free
from skop.ops.deconvolve.richardson_lucy import richardson_lucy
from skop.ops.deconvolve.richardson_lucy_cupy import richardson_lucy_cupy

parser = argparse.ArgumentParser()
parser.add_argument("--shape", type=int, nargs=3, default=(64, 1024, 1024))
parser.add_argument("--psf", type=int, nargs=3, default=(31, 15, 15))
parser.add_argument("--sigma", type=float, nargs=3, default=(4.0, 1.5, 1.5))
parser.add_argument("--iters", type=int, default=20)
parser.add_argument(
    "--circulant", action="store_true", help="wrap each tile; wrong when tiled"
)
parser.add_argument("--memory", default=None, help="a fraction, 0.5, or a size, 1G")
parser.add_argument("--overlap", type=int, default=None, help="pixels, every axis")
parser.add_argument("--tile", type=int, nargs=3, default=None, help="z y x")
parser.add_argument("--compare", action="store_true", help="also run untiled")
parser.add_argument("--cpu", action="store_true", help="numpy, in RAM")
args = parser.parse_args()


def gauss_1d(n: int, sigma: float) -> np.ndarray:
    x = np.arange(n) - (n - 1) / 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def blur(volume: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    """Separable blur by shifted adds: three float32 arrays, no FFT."""
    for axis, kernel in enumerate(kernels):
        out = np.zeros_like(volume)
        half = len(kernel) // 2
        for i, weight in enumerate(kernel):
            out += weight * np.roll(volume, i - half, axis=axis)
        volume = out
    return volume


def gb(n: int | None) -> str:
    return "unknown" if n is None else f"{n / 1024**3:.2f} GB"


# -- the data -----------------------------------------------------------------

started = time.perf_counter()
rng = np.random.default_rng(0)
kernels = [gauss_1d(n, s) for n, s in zip(args.psf, args.sigma)]
psf = np.einsum("i,j,k->ijk", *kernels).astype(np.float32)
truth = np.zeros(args.shape, np.float32)
points = rng.integers(0, args.shape, size=(truth.size // 2000, 3))
truth[tuple(points.T)] = 1000.0
image = blur(truth, kernels) + 10.0
image += rng.normal(0.0, 2.0, image.shape).astype(np.float32)
del truth
print(
    f"image {image.shape} float32, {image.nbytes / 1024**3:.2f} GB; "
    f"PSF {psf.shape}; made in {time.perf_counter() - started:.0f} s"
)

# -- the plan -----------------------------------------------------------------

op = richardson_lucy if args.cpu else richardson_lucy_cupy
memory = args.memory
if memory is not None:
    try:
        memory = float(memory)  # 0.5: a fraction
    except ValueError:
        pass  # 1G: a size
tiler = skop.Tiler(
    memory=memory,
    overlap=args.overlap,
    tile_shape=tuple(args.tile) if args.tile else None,
)
call = {
    "image": image,
    "psf": psf,
    "num_iters": args.iters,
    "noncirc": not args.circulant,
}
device = "cpu" if args.cpu else "gpu"
if not args.cpu:
    print(f"GPU free: {gb(gpu_free())}")
print(f"budget ({device}): {gb(tiler.budget(device))}")
plan = tiler.plan(op, call)
print(f"plan: {plan.summary if plan else 'not tiled (no budget known)'}")

# -- the run ------------------------------------------------------------------

tasks = []
with skop.Runner() as runner:
    started = time.perf_counter()
    result = runner.run(op, call, tiler=tiler, on_start=tasks.append)
    took = time.perf_counter() - started
    print(
        f"tiled: {len(tasks)} task(s) in {took:.1f} s -> "
        f"{type(result).__name__} {result.shape} {result.dtype}"
    )

    if args.compare:
        started = time.perf_counter()
        whole = runner.run(op, call, tiler=skop.Tiler.off())
        print(f"untiled: {time.perf_counter() - started:.1f} s")
        diff = np.abs(np.asarray(result, np.float64) - whole)
        scale = float(np.abs(whole).max())
        # Away from the seams: further than the overlap from any cut.
        far = np.ones(whole.shape, bool)
        if plan is not None and plan.calls > 1:
            reach = plan.overlap
            reach = reach if isinstance(reach, tuple) else (reach,) * whole.ndim
            for tile in plan.tiles:
                for axis, cut in enumerate(tile.write):
                    for edge in (cut.start, cut.stop):
                        if 0 < edge < whole.shape[axis]:
                            near = [slice(None)] * whole.ndim
                            near[axis] = slice(
                                max(0, edge - reach[axis]), edge + reach[axis]
                            )
                            far[tuple(near)] = False
        print(f"largest difference, everywhere: {diff.max() / scale:.2e} of the peak")
        if far.any():
            away = diff[far].max() / scale
            print(f"largest difference, away from the seams: {away:.2e} of the peak")
