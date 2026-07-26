"""Point spread functions.

Ported from tnia-python's ``tnia.deconvolution.gaussian_psf``, whose separate
``gaussian_2d`` and ``gaussian_3d`` are collapsed into one op here: they differ
only in whether there is a Z axis.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import op


@op(env="skimage")
def gaussian_psf(
    xy_dim: Annotated[int, {"min": 1, "max": 4096}] = 64,
    xy_sigma: Annotated[float, {"min": 0.0, "step": 0.1}] = 2.0,
    z_dim: Annotated[int, {"min": 0, "max": 4096}] = 0,
    z_sigma: Annotated[float, {"min": 0.0, "step": 0.1}] = 2.0,
) -> np.ndarray:
    """Generate a Gaussian point spread function, in 2D or 3D.

    Args:
        xy_dim: Extent of the PSF in XY, in pixels.
        xy_sigma: Standard deviation in XY, in pixels.
        z_dim: Extent along Z, in pixels. 0 or 1 gives a 2D PSF.
        z_sigma: Standard deviation along Z, in pixels. Ignored in 2D.

    Returns:
        A float64 PSF summing to 1, shaped (Z, Y, X) or (Y, X).
    """
    # NB: the original built this with a Python loop per voxel. The Gaussian is
    # separable, so the outer product below is the same array for a 64^3 PSF's
    # 260k iterations of nothing.
    profile_xy = _profile(xy_dim, xy_sigma)

    if z_dim > 1:
        profile_z = _profile(z_dim, z_sigma)
        psf = (
            profile_z[:, np.newaxis, np.newaxis]
            * profile_xy[np.newaxis, :, np.newaxis]
            * profile_xy[np.newaxis, np.newaxis, :]
        )
    else:
        psf = profile_xy[:, np.newaxis] * profile_xy[np.newaxis, :]

    # NB: gaussian_3d added 1e-12 before normalizing and gaussian_2d did not.
    # Neither does now -- after normalization the floor is ~1e-12 of the peak,
    # which no caller can act on, and having the two paths agree is worth more.
    return psf / psf.sum()


def _profile(dim: int, sigma: float) -> np.ndarray:
    """One axis of a separable Gaussian, centered on the array."""
    coords = np.linspace(-(dim // 2), dim // 2, dim)
    return np.exp(-(coords**2) / (2.0 * sigma**2))
