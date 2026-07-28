"""Helpers shared between ops, tested on the host.

``to_rgb`` in particular: it is the first thing every detector does to its
input, so when it is wrong the symptom is a model that silently finds nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from skop.ops._util import to_gray, to_rgb


def test_to_rgb_promotes_a_2d_image():
    result = to_rgb(np.linspace(0, 4000, 64 * 64).reshape(64, 64).astype(np.uint16))
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result[..., 0], result[..., 2])


def test_to_rgb_stretches_low_contrast_input():
    # 16-bit data using a fraction of its range is the normal microscopy case,
    # and casting it straight to uint8 would leave the model a black image.
    image = np.zeros((32, 32), dtype=np.uint16)
    image[8:24, 8:24] = 300
    result = to_rgb(image)
    assert result.max() == 255
    assert result.min() == 0


def test_to_rgb_survives_a_blank_image():
    result = to_rgb(np.zeros((16, 16), dtype=np.uint16))
    assert result.shape == (16, 16, 3)
    assert result.max() == 0


def test_to_rgb_leaves_uint8_rgb_alone():
    image = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    assert to_rgb(image) is image


def test_to_rgb_drops_an_alpha_channel():
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    assert to_rgb(image).shape == (8, 8, 3)


def test_to_rgb_rejects_a_volume():
    with pytest.raises(ValueError):
        to_rgb(np.zeros((4, 8, 8), dtype=np.uint16))


def test_to_gray_and_to_rgb_round_trip_shape():
    image = np.zeros((8, 8), dtype=np.uint8)
    assert to_gray(to_rgb(image)).shape == (8, 8)


def offset_psf(z: int = 24, peak_z: int = 15) -> np.ndarray:
    """A crude PSF whose brightest plane is deliberately not the middle one."""
    psf = np.zeros((z, 8, 8), dtype=np.float32)
    for plane in range(z):
        psf[plane, 4, 4] = np.exp(-((plane - peak_z) ** 2) / 8.0)
    return psf


def test_recenter_puts_the_peak_on_the_centre_plane():
    from skop.ops.kernels import recenter_psf_axial

    result = recenter_psf_axial(offset_psf(), 8)
    assert result.shape == (8, 8, 8)
    assert np.unravel_index(result.argmax(), result.shape)[0] == 4


def test_recenter_refuses_a_crop_that_runs_off_the_end():
    # The original sliced without checking, and Python obliged: a negative
    # start counts from the far end, an overlong stop truncates, and either
    # way a differently-shaped PSF came back with no complaint.
    from skop.ops.kernels import recenter_psf_axial

    with pytest.raises(ValueError, match="too close to the edge"):
        recenter_psf_axial(offset_psf(z=24, peak_z=1), 16)


def test_recenter_refuses_to_grow_a_psf():
    from skop.ops.kernels import recenter_psf_axial

    with pytest.raises(ValueError, match="nothing to crop"):
        recenter_psf_axial(offset_psf(z=8), 16)


def test_recenter_wants_a_volume():
    from skop.ops.kernels import recenter_psf_axial

    with pytest.raises(ValueError, match="3-D"):
        recenter_psf_axial(np.zeros((8, 8), dtype=np.float32), 4)


def test_a_fluorophore_is_its_own_wavelength():
    # The float mixin is what lets `wavelength=Fluorophore.DAPI` be passed to
    # a parameter annotated as a plain float, so the enum can be a convenience
    # without becoming the only way to say it.
    from skop.ops.kernels import Fluorophore

    assert float(Fluorophore.DAPI) == 0.461
    assert Fluorophore.Cy5 > Fluorophore.FITC
    # Microns, not nanometres -- every PSF op here works in microns, and a
    # wavelength off by 1000x produces a PSF that looks plausible and is wrong.
    assert all(0.2 < float(f) < 1.0 for f in Fluorophore)
