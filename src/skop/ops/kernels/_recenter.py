"""Putting an off-centre PSF back on centre.

A theoretical PSF computed at a non-zero focal depth is not centred on the
middle plane -- that asymmetry is the spherical aberration being modelled, and
it is the point. But a deconvolution kernel has to be centred, or the result
comes out shifted along Z by however far the peak sat from the middle.

sdeconv in particular does not centre its output at all, which is the reason
this exists. Compute taller than needed, find the peak, crop back around it.

numpy and the standard library only, so any environment holding a PSF op can
import it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["recenter_psf_axial"]


def recenter_psf_axial(psf: np.ndarray, new_z: int) -> np.ndarray:
    """Crop a 3-D PSF to ``new_z`` planes, centred on its brightest plane.

    Args:
        psf: A ``(Z, Y, X)`` PSF, generally taller than ``new_z``.
        new_z: Number of planes to return.

    Returns:
        A ``(new_z, Y, X)`` view-derived copy whose peak sits on the centre
        plane, at index ``new_z // 2``.

    Raises:
        ValueError: If the crop would run off either end of ``psf`` -- which
            means the PSF was not computed with enough headroom for this
            ``pz``, and the honest answer is to compute a taller one.
    """
    if psf.ndim != 3:
        raise ValueError(f"expected a 3-D PSF, got {psf.ndim}-D")
    if new_z > psf.shape[0]:
        raise ValueError(
            f"cannot crop {psf.shape[0]} planes down to {new_z}: nothing to crop"
        )

    peak_z = int(np.unravel_index(int(psf.argmax()), psf.shape)[0])
    start = peak_z - new_z // 2

    # NB: the original sliced without checking, and Python's slicing is
    # forgiving in exactly the wrong way here -- a negative start counts from
    # the far end and an overlong stop silently truncates, so a PSF whose peak
    # sits near an edge came back with the wrong number of planes and no
    # complaint. The caller cannot recover from that; it can from an error.
    if start < 0 or start + new_z > psf.shape[0]:
        raise ValueError(
            f"peak at plane {peak_z} is too close to the edge to crop "
            f"{new_z} planes around it, from {psf.shape[0]}: compute the PSF "
            f"with more Z headroom, or reduce pz"
        )

    return psf[start : start + new_z].copy()
