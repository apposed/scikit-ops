"""The Gibson-Lanni point spread function, via sdeconv.

A scalar diffraction model of a real objective looking into a real sample: it
takes the immersion medium and the sample's refractive index, and the depth
below the coverslip, so it produces the axially asymmetric PSF that spherical
aberration actually gives you. That asymmetry is the reason to use this over
:mod:`~skop.ops.kernels.paraxial`.

Ported from tnia-python's ``tnia.deconvolution.psfs``, whose two functions --
``gibson_lanni_3D`` and ``gibson_lanni_3D_partial_confocal`` -- collapse into
one op here, the same way the two Gaussians did. The partial-confocal variant
was the same computation plus recentring, and its ``confocal`` flag was just
``confocal_factor=2``.

**psfmodels is deliberately absent.** The original could compute this either
through sdeconv or through psfmodels, chosen by a ``use_psfm`` flag. psfmodels
is GPL-3.0 and this project is BSD, so that branch is not defaulted off, it is
gone -- there is no import path here that could reach it.
"""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np

from skop import op, progress

from ._fluorophore import Fluorophore
from ._recenter import recenter_psf_axial

__all__ = ["Fluorophore", "gibson_lanni"]


@op(env="sdeconv")
def gibson_lanni(
    xy_size: Annotated[int, {"min": 4, "max": 2048}] = 128,
    z_size: Annotated[int, {"min": 1, "max": 1024}] = 64,
    voxel_size_xy: Annotated[float, {"min": 0.001, "step": 0.01}] = 0.1,
    voxel_size_z: Annotated[float, {"min": 0.001, "step": 0.01}] = 0.2,
    numerical_aperture: Annotated[float, {"min": 0.1, "max": 2.0, "step": 0.05}] = 1.4,
    ni: Annotated[float, {"min": 1.0, "max": 2.0, "step": 0.001}] = 1.518,
    ns: Annotated[float, {"min": 1.0, "max": 2.0, "step": 0.001}] = 1.33,
    pz: Annotated[float, {"min": 0.0, "step": 0.1}] = 0.0,
    wavelength: Annotated[float, {"min": 0.1, "max": 2.0, "step": 0.01}] = 0.53,
    confocal_factor: Annotated[float, {"min": 1.0, "max": 2.0, "step": 0.1}] = 1.0,
    recenter: bool = True,
) -> np.ndarray:
    """Generate a 3-D Gibson-Lanni point spread function.

    Args:
        xy_size: Extent of the PSF in XY, in voxels. Square.
        z_size: Number of Z planes to return.
        voxel_size_xy: Lateral voxel size, in microns.
        voxel_size_z: Axial voxel size, in microns.
        numerical_aperture: Numerical aperture of the objective.
        ni: Refractive index of the immersion medium. 1.518 is oil, 1.33
            water, 1.0 air.
        ns: Refractive index of the sample.
        pz: Depth of the point below the coverslip, in microns. Non-zero is
            what makes the PSF axially asymmetric, and is the whole reason to
            model spherical aberration.
        wavelength: Emission wavelength, in microns. ``Fluorophore`` members
            can be passed here by name.
        confocal_factor: 1.0 is widefield. 2.0 is confocal -- the PSF squared,
            since a confocal detects through the same aperture it illuminates
            through. Values in between are an ad-hoc approximation of a
            partially closed pinhole and should be used carefully: it is a
            convenient interpolation, not a model of one.
        recenter: Whether to put the peak on the centre plane. sdeconv does
            not centre its output, so a PSF used as a deconvolution kernel
            wants this on, or the result comes out shifted along Z.

    Returns:
        A ``(z_size, xy_size, xy_size)`` float32 PSF summing to 1.
    """
    from sdeconv.psfs import SPSFGibsonLanni

    # NB: only sdeconv 1.x is supported, and the environment pins it. The
    # original sniffed sdeconv.__version__ and branched to `PSFGibsonLanni`
    # with a different constructor for 0.x. 0.x is the only version on
    # conda-forge but is years old and takes wavelength in nanometres with no
    # ni/ns at all, so supporting both would mean two different models behind
    # one signature.
    compute_z = _compute_z(z_size, voxel_size_z, pz) if recenter else z_size

    progress(f"Computing a {compute_z}x{xy_size}x{xy_size} Gibson-Lanni PSF")
    model = SPSFGibsonLanni(
        (compute_z, xy_size, xy_size),
        NA=numerical_aperture,
        ni=ni,
        ni0=ni,
        ns=ns,
        res_lateral=voxel_size_xy,
        res_axial=voxel_size_z,
        wavelength=wavelength,
        pZ=pz,
    )

    # sdeconv computes in torch and hands back a tensor, on the CPU here since
    # nothing above asked for a device.
    psf = model().cpu().numpy().astype(np.float32)

    if recenter:
        progress("Recentring")
        psf = recenter_psf_axial(psf, z_size)

    # NB: the power comes after the crop and before the normalization, which
    # is the order the partial-confocal original used. It matters: squaring
    # first and cropping second would normalize over planes that get thrown
    # away.
    if confocal_factor != 1.0:
        psf = psf**confocal_factor

    return psf / psf.sum()


def _compute_z(z_size: int, voxel_size_z: float, pz: float) -> int:
    """How many planes to compute so that ``z_size`` can be cropped out centred.

    The original used a flat ``z_size + z_size // 2``, and the spec described
    that as what "makes a non-zero ``pz`` usable". Measured against sdeconv
    1.0.4, it is not: the PSF's peak marches toward plane zero as the point
    goes deeper -- about 6.5 planes per micron of ``pz`` at a 0.2 um axial
    voxel, or 1.3 um of apparent focal shift per micron of depth -- and the
    flat headroom is used up somewhere around ``pz`` = 1 um. Past that the
    crop runs off the front of the volume.

    Which the original did not notice, because it sliced without checking and
    got a silently short array back. :func:`~skop.ops.kernels.recenter_psf_axial`
    raises instead, so the same PSFs that used to come back quietly wrong
    would now come back as errors -- hence this.

    The headroom therefore scales with the shift. The factor of 3 is empirical
    and deliberately generous: the measured shift is ~1.3 um per micron of
    depth, and doubling that (the peak has to clear ``z_size // 2``, not just
    stay in the volume) gives ~2.6. At 3 the margin grows rather than shrinks
    with depth -- 7 planes to spare at ``pz`` = 0, 31 at ``pz`` = 16 um, for
    ``z_size`` = 32 -- and the cost is only a taller FFT.

    The shift depends on the refractive index mismatch, so a strongly
    mismatched ``ni``/``ns`` could still outrun this. That case is what the
    error from ``recenter_psf_axial`` is for; it says to raise ``z_size``.
    """
    return z_size + z_size // 2 + math.ceil(3.0 * pz / voxel_size_z)
