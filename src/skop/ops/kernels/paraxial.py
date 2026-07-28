"""Paraxial (scalar, in-focus) optical transfer function and point spread function.

The cheapest useful optical model there is: a 2-D diffraction-limited OTF that
follows from wavelength, numerical aperture and pixel size alone, with no
immersion medium, no sample refractive index, and no focal depth. Where
Gibson-Lanni models a real objective looking into a real sample, this models
the textbook. It costs no environment of its own and takes a millisecond, which
makes it the right kernel for a quick simulation or a sanity check.

Ported from tnia-python's ``tnia.deconvolution.psfs``, which credits
https://github.com/jdmanton/rl_positivity_sim by James Manton.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import op


@op(env="skimage")
def paraxial_otf(
    size: Annotated[int, {"min": 4, "max": 4096}] = 128,
    wavelength: Annotated[float, {"min": 0.1, "max": 2.0, "step": 0.01}] = 0.53,
    numerical_aperture: Annotated[float, {"min": 0.1, "max": 2.0, "step": 0.05}] = 1.4,
    pixel_size: Annotated[float, {"min": 0.001, "step": 0.01}] = 0.1,
) -> np.ndarray:
    """Generate a paraxial optical transfer function.

    Args:
        size: Extent of the OTF, in pixels. Square.
        wavelength: Emission wavelength, in microns. ``Fluorophore`` members
            can be passed here by name.
        numerical_aperture: Numerical aperture of the objective.
        pixel_size: Pixel size in the sample plane, in microns.

    Returns:
        A ``(size, size)`` OTF, 1 at DC and falling to 0 at the cutoff
        frequency, beyond which it is exactly 0.
    """
    return _otf(size, wavelength, numerical_aperture, pixel_size)


@op(env="skimage")
def paraxial_psf(
    size: Annotated[int, {"min": 4, "max": 4096}] = 128,
    wavelength: Annotated[float, {"min": 0.1, "max": 2.0, "step": 0.01}] = 0.53,
    numerical_aperture: Annotated[float, {"min": 0.1, "max": 2.0, "step": 0.05}] = 1.4,
    pixel_size: Annotated[float, {"min": 0.001, "step": 0.01}] = 0.1,
) -> np.ndarray:
    """Generate a paraxial point spread function.

    The inverse transform of :func:`paraxial_otf`, which is what a
    deconvolution op wants as a kernel.

    Args:
        size: Extent of the PSF, in pixels. Square.
        wavelength: Emission wavelength, in microns. ``Fluorophore`` members
            can be passed here by name.
        numerical_aperture: Numerical aperture of the objective.
        pixel_size: Pixel size in the sample plane, in microns.

    Returns:
        A ``(size, size)`` float32 PSF summing to 1.

    Note:
        **This PSF has negative side lobes, and Richardson-Lucy will diverge
        on it.** An ideal paraxial PSF is the Airy intensity pattern and is
        non-negative; this one is sampled on a grid and inverted with a DFT,
        which leaves ringing, some of it below zero. No single negative value
        is large -- each is around 1e-4 of the peak -- but together they hold
        about 1% of the PSF's mass. That is invisible for simulation, and
        fatal for deconvolution, which is built on ratios and has no meaning
        for a negative kernel: fifty iterations of ``richardson_lucy`` on an
        unclipped one reaches 1e53.

        Clip and renormalize before deconvolving::

            psf = np.clip(psf, 0.0, None)
            psf /= psf.sum()

        The op does not do this itself. At 1% of the mass the correction is
        too large to apply silently -- it would move every number the
        tnia-python original produced by more than a rounding error -- and the
        simulation callers who are the reason this exists do not need it.
    """
    otf = _otf(size, wavelength, numerical_aperture, pixel_size)

    # NB: the original cast the complex result of ifftn straight to float32,
    # which discards the imaginary part behind a ComplexWarning. Taking .real
    # is the same number, said out loud. The imaginary part is numerical noise
    # either way -- the OTF is real and symmetric, so its transform is real.
    psf = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(otf)).real).astype(np.float32)
    return psf / psf.sum()


def _otf(
    size: int, wavelength: float, numerical_aperture: float, pixel_size: float
) -> np.ndarray:
    """The OTF itself, shared by both ops so neither has to call the other.

    An op calling an op would mean a round trip through the worker for what is
    twenty lines of numpy.
    """
    resolution = 0.5 * wavelength / numerical_aperture

    # NB: the centre is size/2 + 1, not size/2, and the radial coordinate is
    # normalized by the largest coordinate *after* that offset is subtracted --
    # so the normalization runs to size/2 - 2 rather than size/2. Both are
    # inherited verbatim. They amount to a sub-pixel shift of the centre and a
    # few-percent scaling of the cutoff, which matters to nobody using this as
    # a simulation kernel, and changing either would silently move every result
    # this has ever produced.
    centre = size / 2 + 1
    axis = np.linspace(0, size - 1, size) - centre
    x, y = np.meshgrid(axis, axis)

    filter_radius = 2 * pixel_size / resolution
    radius = np.sqrt(x * x + y * y) / axis.max()

    inside = radius <= filter_radius
    # Zeroed outside the passband before the arccos, so the square root never
    # sees a negative and the arccos never sees an argument above 1.
    v = np.where(inside, radius / filter_radius, 0.0)

    return 2 / np.pi * (np.arccos(v) - v * np.sqrt(1 - v * v)) * inside
