"""Richardson-Lucy deconvolution on an NVIDIA GPU, with cupy.

Ported from tnia-python's ``tnia.deconvolution.richardson_lucy``. The
iteration is the same one ``richardson_lucy`` runs, and the two are held
numerically equivalent by a test; what differs is where the arrays live.

Dropped in the port: the RMSE-against-truth tracking (a benchmarking concern,
not an op's), ``do_unpad`` (an op returns something the caller can use), and
the memory-pool diagnostics.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import cancel_requested, op, progress

from ._edges import pad_and_mask
from .richardson_lucy import DELTA


@op(env="cupy")
def richardson_lucy_cupy(
    image: np.ndarray,
    psf: np.ndarray,
    num_iters: Annotated[int, {"min": 1, "max": 1000}] = 10,
    noncirc: bool = False,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Deconvolve an image by the Richardson-Lucy algorithm, on the GPU.

    Args:
        image: The image to deconvolve.
        psf: The point spread function. Need not match the image's shape.
        num_iters: How many iterations to run.
        noncirc: Whether to use non-circulant edge handling, which pads the
            image out rather than treating it as wrapping around.
        mask: Optional array the shape of the image, zero at pixels that
            should not be considered -- saturated ones, say. They are held out
            of the deconvolution and their original values restored at the end.

    Returns:
        The deconvolved image, as float32, back in host memory.

    Note:
        cupy's FFT works in single precision, so this runs in float32 where
        ``richardson_lucy`` runs in float64. Expect the two to agree to
        single-precision tolerance, not exactly.
    """
    import cupy as cp

    # Masking and padding happen on the host, so only what the iteration needs
    # is uploaded -- and in float32, which is all cupy's FFT will give back.
    padded = pad_and_mask(image, psf, noncirc, mask, dtype=np.float32)
    image = cp.asarray(padded.image)
    psf = cp.asarray(padded.psf)
    htones = cp.asarray(padded.htones)

    otf = cp.fft.fftn(cp.fft.ifftshift(psf))
    otf_conj = cp.conjugate(otf)

    htones = cp.real(cp.fft.ifftn(cp.fft.fftn(htones) * otf_conj))
    htones[htones < DELTA] = 1.0

    estimate = cp.full_like(image, float(image.mean()))

    for i in range(num_iters):
        if cancel_requested():
            break
        progress(f"Iteration {i + 1} of {num_iters}", i, num_iters)

        reblurred = cp.real(cp.fft.ifftn(cp.fft.fftn(estimate) * otf))
        reblurred[reblurred < DELTA] = DELTA
        ratio = image / reblurred
        correction = cp.real(cp.fft.ifftn(cp.fft.fftn(ratio) * otf_conj))
        correction[correction < 0] = DELTA
        estimate = estimate * correction / htones
        estimate[estimate < 0] = DELTA

        # Wait for the iteration actually to finish. Every cupy call above
        # queues work and returns at once, so without this the loop issues all
        # num_iters iterations in a few milliseconds -- reporting each -- and
        # then blocks on the transfer at the end. The bar would sweep to 100%
        # and sit there while the GPU did the work, and Cancel would arrive
        # long after every iteration had been queued and so do nothing.
        #
        # The cost is that the queue can no longer run ahead. For FFTs this
        # size that is small, and a progress bar that lies is worse.
        cp.cuda.runtime.deviceSynchronize()

    progress("Deconvolution complete", num_iters, num_iters)
    return padded.crop_and_restore(cp.asnumpy(estimate))
