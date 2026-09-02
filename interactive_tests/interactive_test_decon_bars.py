"""Deconvolution of the Bars phantom, on real data, in napari.

The companion to interactive_test_decon.py, which makes its own synthetic
volume. This one loads the Bars stack from test_images/deconvolution: a
blurred and Poisson-noised acquisition, the PSF it was blurred with, and the
ground truth to score against -- which a real experiment never has.

Run it with napari available:

    uv run --with napari --with pyqt5 python interactive_tests/interactive_test_decon_bars.py

The first run builds the 'skimage' environment, and the cupy backend needs an
NVIDIA GPU -- it is skipped, with a message, when there is not one.
"""

import time
from pathlib import Path

import numpy as np
import tifffile  # NB: napari depends on it, so it is here whenever napari is.

import skop
from skop.ops.deconvolve import richardson_lucy_cupy

ITERATIONS = 200

# NB: circulant. These stacks are 128x256x256, and non-circulant handling
# extends them to 256x512x512 -- eight times the voxels, which turns a CPU run
# from slow into a coffee break. Flip it on to see the edges improve.
NONCIRC = False

IMAGES = Path(__file__).resolve().parent.parent / "test_images" / "deconvolution"
BLURRED = IMAGES / "Bars-G10-P30-stack.tif"  # also available: Bars-G10-P15
TRUTH = IMAGES / "Bars-stack.tif"
PSF = IMAGES / "PSF-Bars-stack.tif"

for path in (BLURRED, TRUTH, PSF):
    if not path.exists():
        raise SystemExit(f"Missing test image: {path}")

print("Loading the Bars stack...")
blurred = tifffile.imread(BLURRED).astype(np.float64)
truth = tifffile.imread(TRUTH).astype(np.float64)
psf = tifffile.imread(PSF).astype(np.float64)
for name, array in (("blurred", blurred), ("truth", truth), ("psf", psf)):
    print(
        f"  {name:8s} {array.shape} {array.dtype}  range {array.min():g}-{array.max():g}"
    )

# A PSF has to sum to 1, or every iteration rescales the estimate. The file is
# 16-bit counts, so it does not.
psf = psf / psf.sum()
print(f"  psf normalized, now sums to {psf.sum():.6f}")


def rmse_vs_truth(array):
    """Compare on a common scale: the estimate's brightness is arbitrary."""
    return float(np.sqrt(np.mean((array / array.max() - truth / truth.max()) ** 2)))


baseline = rmse_vs_truth(blurred)
print(f"\nBlurred input scores {baseline:.5f} against the truth. Lower is better.")


def show_progress(event):
    """Overwrite one line, and ignore the message-less events around the run."""
    if event.message:
        print(f"    {event.message:40s}", end="\r")


runner = skop.Runner()

print(f"\nDeconvolving on the GPU, {ITERATIONS} iterations...")

gpu_seconds = None
try:
    start = time.perf_counter()
    gpu = runner.run(
        richardson_lucy_cupy,
        image=blurred,
        psf=psf,
        num_iters=ITERATIONS,
        noncirc=NONCIRC,
        on_progress=show_progress,
    )
    gpu_seconds = time.perf_counter() - start
    print()
except Exception as exc:  # noqa: BLE001 -- a missing GPU must not end the script
    gpu = None
    print(f"  skipped: {type(exc).__name__}: {exc}")
    print("  (expected when there is no NVIDIA GPU or the 'cupy' env is unbuilt)")

print(f"\nTimings ({ITERATIONS} iterations, {blurred.shape} volume):")
print(f"  cupy  (GPU): {gpu_seconds:.2f} s")

runner.close()

print("\nOpening napari...")
# Imported late, so the ops still run on a machine with no GUI.
import napari

viewer = napari.Viewer()
viewer.add_image(truth, name="truth", colormap="gray", visible=False)
viewer.add_image(blurred, name="blurred + noise (P30)", colormap="gray")
viewer.add_image(gpu, name=f"deconvolved (cupy, {ITERATIONS})", colormap="magma")
viewer.add_image(psf, name="psf", colormap="viridis", visible=False)

napari.run()
