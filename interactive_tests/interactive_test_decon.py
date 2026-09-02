"""Deconvolution, end to end, in napari.

Informal and meant to be played with -- change the sigmas, the iteration count
or the noise level and watch what happens. Napari rather than matplotlib
because the interesting data here is 3D.

Run it with napari available:

    uv run --with napari --with pyqt5 python interactive_tests/interactive_test_decon.py

The first run builds the 'skimage' environment, and the cupy backend needs an
NVIDIA GPU -- it is skipped, with a message, when there is not one.
"""

import time

import numpy as np

import skop
from skop.ops.deconvolve import richardson_lucy, richardson_lucy_cupy
from skop.ops.generate import synthetic_nuclei
from skop.ops.kernels import gaussian_psf

ITERATIONS = 50
PHOTONS = 200.0  # lower means noisier, and a harder time for the algorithm

runner = skop.Runner()

print("Generating a synthetic volume...")
truth = runner.run(
    synthetic_nuclei,
    size_z=32,
    size_y=128,
    size_x=128,
    n_nuclei=6,
    seed=3,
).astype(np.float64)
print(f"  truth: {truth.shape} {truth.dtype}, range {truth.min()}-{truth.max()}")

print("\nGenerating a 3D Gaussian PSF...")
psf = runner.run(
    gaussian_psf,
    xy_dim=truth.shape[1],
    xy_sigma=4.0,
    z_dim=truth.shape[0],
    z_sigma=2.0,
)
print(f"  psf: {psf.shape}, sums to {psf.sum():.6f}")

# Blur, then add shot noise -- what a microscope does to a specimen. Done here
# with plain numpy rather than through an op; porting tnia's forward.py is
# future work.
print("\nBlurring and adding Poisson noise...")
blurred = np.real(np.fft.ifftn(np.fft.fftn(truth) * np.fft.fftn(np.fft.ifftshift(psf))))
scale = PHOTONS / blurred.max()
rng = np.random.default_rng(0)
blurred = rng.poisson(np.clip(blurred, 0, None) * scale) / scale


def report(name, result):
    """How close did it get, and how much sharper is it than the input?"""

    def rmse(a):
        return np.sqrt(np.mean((a / a.max() - truth / truth.max()) ** 2))

    print(f"  {name}: {result.shape} {result.dtype}")
    print(
        f"    rmse vs truth: {rmse(result):.5f}  (blurred input: {rmse(blurred):.5f})"
    )


print(f"\nDeconvolving on the CPU, {ITERATIONS} iterations...")
cpu_start = time.perf_counter()
cpu = runner.run(
    richardson_lucy,
    image=blurred,
    psf=psf,
    num_iters=ITERATIONS,
    noncirc=True,
    on_progress=lambda event: print(f"    {event.message}", end="\r"),
)
cpu_seconds = time.perf_counter() - cpu_start
print()
report("numpy", cpu)
print(
    f"    elapsed: {cpu_seconds:.3f} s  ({cpu_seconds / ITERATIONS * 1e3:.1f} ms/iter)"
)

print(f"\nDeconvolving on the GPU, {ITERATIONS} iterations...")
gpu_seconds = None
try:
    gpu_start = time.perf_counter()
    gpu = runner.run(
        richardson_lucy_cupy,
        image=blurred,
        psf=psf,
        num_iters=ITERATIONS,
        noncirc=True,
        on_progress=lambda event: print(f"    {event.message}", end="\r"),
    )
    gpu_seconds = time.perf_counter() - gpu_start
    print()
    report("cupy", gpu)
    print(
        f"    elapsed: {gpu_seconds:.3f} s  "
        f"({gpu_seconds / ITERATIONS * 1e3:.1f} ms/iter)"
    )
    print(
        f"    agreement with numpy: {np.abs(gpu - cpu).max() / cpu.max():.2e} of peak"
    )
except Exception as exc:  # noqa: BLE001 -- a missing GPU must not end the script
    gpu = None
    print(f"  skipped: {type(exc).__name__}: {exc}")
    print("  (expected when there is no NVIDIA GPU or the 'cupy' env is unbuilt)")

print(f"\nTimings ({ITERATIONS} iterations, {truth.shape} volume):")
print(f"  numpy (CPU): {cpu_seconds:.3f} s")
if gpu_seconds is None:
    print("  cupy  (GPU): not run")
else:
    print(f"  cupy  (GPU): {gpu_seconds:.3f} s")
    print(f"  speedup:     {cpu_seconds / gpu_seconds:.2f}x")

runner.close()

print("\nOpening napari...")
import napari

viewer = napari.Viewer()
viewer.add_image(truth, name="truth", colormap="gray")
viewer.add_image(blurred, name="blurred + noise", colormap="gray")
viewer.add_image(cpu, name=f"deconvolved (numpy, {ITERATIONS})", colormap="magma")
if gpu is not None:
    viewer.add_image(gpu, name=f"deconvolved (cupy, {ITERATIONS})", colormap="magma")
viewer.add_image(psf, name="psf", colormap="viridis", visible=False)
# viewer.dims.ndisplay = 3

napari.run()
