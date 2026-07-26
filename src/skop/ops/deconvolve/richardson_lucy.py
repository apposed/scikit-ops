"""Richardson-Lucy deconvolution on the CPU, with numpy.

Ported from tnia-python's ``tnia.deconvolution.richardson_lucy_np``. Where it
differed from the cupy version, the cupy behaviour won -- see the NB comments
below and ``richardson_lucy_cupy``, which this op is kept numerically
equivalent to.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import cancel_requested, op, progress

from ._edges import pad_and_mask

# Floor for the divisions and the estimate: an iteration must not divide by
# zero, and intensities are non-negative by construction.
DELTA = 1e-6


@op(env="skimage")
def richardson_lucy(
    image: np.ndarray,
    psf: np.ndarray,
    num_iters: Annotated[int, {"min": 1, "max": 1000}] = 10,
    noncirc: bool = False,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Deconvolve an image by the Richardson-Lucy algorithm.

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
        The deconvolved image, as float64.

    Note:
        numpy's FFT promotes float32 to float64, so passing 32-bit input to
        save memory does not work here; the result is float64 regardless.
    """
    fftn, ifftn, ifftshift = np.fft.fftn, np.fft.ifftn, np.fft.ifftshift

    padded = pad_and_mask(image, psf, noncirc, mask)
    image, psf, htones = padded.image, padded.psf, padded.htones

    otf = fftn(ifftshift(psf))
    otf_conj = np.conjugate(otf)

    htones = np.real(ifftn(fftn(htones) * otf_conj))
    htones[htones < DELTA] = 1.0

    # NB: the numpy original seeded the circulant case with the image itself.
    # Both backends now start from a flat sheet at the image's mean, which is
    # what the cupy version did and what the non-circulant case always needed.
    estimate = np.full_like(image, image.mean())

    for i in range(num_iters):
        if cancel_requested():
            break
        progress(f"Iteration {i + 1} of {num_iters}", i, num_iters)

        reblurred = np.real(ifftn(fftn(estimate) * otf))
        reblurred[reblurred < DELTA] = DELTA
        ratio = image / reblurred
        correction = np.real(ifftn(fftn(ratio) * otf_conj))
        correction[correction < 0] = DELTA
        estimate = estimate * correction / htones
        estimate[estimate < 0] = DELTA

    progress("Deconvolution complete", num_iters, num_iters)
    return padded.crop_and_restore(estimate)
