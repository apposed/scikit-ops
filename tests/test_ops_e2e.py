"""Real ops, running in their real environments.

Each test declares the environment it needs with ``@pytest.mark.env``. By
default a test whose environment is not built yet is skipped, so the suite
stays runnable without waiting on a TensorFlow or PyTorch install. To run
them all, building whatever is missing:

    uv run pytest --build-envs

That is what CI runs, and it is the only way these ops get exercised against
the stacks they actually target. Expect it to take a while the first time.
The skipping logic lives in the root conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import skop


@pytest.fixture(scope="module")
def runner():
    with skop.Runner() as r:
        yield r


def blobs_2d(size: int = 128, sigma: float = 7.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=np.float32)
    quarter, half, three = size // 4, size // 2, 3 * size // 4
    for cy, cx in [
        (quarter, quarter),
        (quarter, three),
        (three, quarter),
        (three, three),
        (half, half),
    ]:
        image += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    return (image / image.max() * 255).astype(np.uint16)


def blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Circularly convolve an image with a psf of the same shape, via FFT."""
    return np.real(
        np.fft.ifftn(np.fft.fftn(image) * np.fft.fftn(np.fft.ifftshift(psf)))
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def points_2d(size: int = 64) -> np.ndarray:
    """Point sources, the hardest thing to deconvolve and the clearest."""
    truth = np.zeros((size, size))
    quarter, half, three = size // 4, size // 2, 3 * size // 4
    for cy, cx in [
        (quarter, quarter),
        (quarter, three),
        (three, quarter),
        (three, three),
        (half, half),
    ]:
        truth[cy, cx] = 100.0
    return truth


@pytest.mark.env("skimage")
def test_otsu(runner):
    from skop.ops.threshold import otsu

    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:25, 10:25] = 200
    image[40:55, 40:55] = 220

    labels = runner.run(otsu, image=image)
    assert labels.dtype == np.uint16
    assert labels.max() == 2

    mask = runner.run(otsu, image=image, label_objects=False)
    assert set(np.unique(mask)) == {0, 1}


@pytest.mark.env("skimage")
def test_synthetic_nuclei(runner):
    from skop.ops.generate import synthetic_nuclei

    volume = runner.run(
        synthetic_nuclei,
        size_z=16,
        size_y=64,
        size_x=64,
        n_nuclei=3,
        radius_xy_min=6.0,
        radius_xy_max=8.0,
    )
    assert volume.shape == (16, 64, 64)
    assert volume.dtype == np.uint16
    assert volume.max() > volume.min()


@pytest.mark.env("stardist-tf")
def test_stardist2d(runner):
    from skop.ops.segment import stardist2d_fluo

    labels = runner.run(stardist2d_fluo, image=blobs_2d())
    assert labels.dtype == np.uint16
    assert labels.max() == 5


@pytest.mark.env("stardist-tf")
def test_stardist2d_reports_progress(runner):
    from skop.ops.segment import stardist2d_fluo

    messages = []
    runner.run(
        stardist2d_fluo,
        image=blobs_2d(),
        on_progress=lambda event: messages.append(event.message),
    )
    assert any(m and "Predicting" in m for m in messages)


@pytest.mark.env("stardist-tf")
def test_starfun3d_segments_what_was_generated(runner):
    from skop.ops.generate import synthetic_nuclei
    from skop.ops.segment import segment_nuclei

    # NB: this crosses two environments -- generated under skimage, segmented
    # under stardist-tf -- in two worker processes.
    volume = runner.run(
        synthetic_nuclei,
        size_z=32,
        size_y=128,
        size_x=128,
        n_nuclei=5,
        seed=3,
    )
    result = runner.run(segment_nuclei, image=volume)

    # NB: not an exact count. Randomly placed nuclei can overlap, and how many
    # remain separable is a property of the model, not of this plumbing.
    assert result.labels.shape == volume.shape
    assert 1 <= result.labels.max() <= 5
    assert result.points.shape == (result.labels.max(), 3)


@pytest.mark.env("stardist-tf")
def test_starfun3d_honors_model_choice(runner):
    from skop.ops.segment import segment_nuclei
    from skop.ops.segment.starfun3d import Model

    # The original loaded 'confocal' whichever model was requested; each of
    # these now resolves to its own weights.
    volume = np.zeros((16, 64, 64), dtype=np.uint16)
    for model in (Model.sospim, Model.confocal):
        result = runner.run(segment_nuclei, image=volume, model=model)
        assert result.labels.shape == volume.shape


@pytest.mark.env("skimage")
def test_gaussian_psf(runner):
    from skop.ops.kernels import gaussian_psf

    psf = runner.run(gaussian_psf, xy_dim=15, xy_sigma=2.0)
    assert psf.shape == (15, 15)
    assert psf.sum() == pytest.approx(1.0)
    assert psf.argmax() == np.ravel_multi_index((7, 7), psf.shape)

    volume_psf = runner.run(gaussian_psf, xy_dim=15, xy_sigma=4.0, z_dim=9, z_sigma=1.0)
    assert volume_psf.shape == (9, 15, 15)
    assert volume_psf.sum() == pytest.approx(1.0)
    # Anisotropy is the whole point of a 3D psf: tight in Z, broad in XY. This
    # is also what catches the axes being transposed.
    above_half = volume_psf > volume_psf.max() / 2
    assert above_half[:, 7, 7].sum() < above_half[4, 7, :].sum()


@pytest.mark.env("skimage")
def test_richardson_lucy_recovers_a_blurred_image(runner):
    from skop.ops.deconvolve import richardson_lucy
    from skop.ops.kernels import gaussian_psf

    truth = points_2d()
    psf = runner.run(gaussian_psf, xy_dim=64, xy_sigma=2.0)
    blurred = blur(truth, psf)

    result = runner.run(richardson_lucy, image=blurred, psf=psf, num_iters=50)

    assert result.shape == truth.shape
    # The assertion that actually means something: closer to the truth than
    # what went in. A shape check would pass on an op that returned its input.
    assert rmse(result, truth) < rmse(blurred, truth)
    assert result.max() > blurred.max()


@pytest.mark.env("skimage")
def test_richardson_lucy_takes_a_psf_smaller_than_the_image(runner):
    from skop.ops.deconvolve import richardson_lucy
    from skop.ops.kernels import gaussian_psf

    truth = points_2d()
    psf = runner.run(gaussian_psf, xy_dim=64, xy_sigma=2.0)
    small = runner.run(gaussian_psf, xy_dim=15, xy_sigma=2.0)
    blurred = blur(truth, psf)

    result = runner.run(richardson_lucy, image=blurred, psf=small, num_iters=30)
    assert result.shape == truth.shape
    assert rmse(result, truth) < rmse(blurred, truth)


@pytest.mark.env("skimage")
def test_richardson_lucy_noncirc_handles_signal_at_the_edge(runner):
    from skop.ops.deconvolve import richardson_lucy
    from skop.ops.kernels import gaussian_psf

    # An object against the border is the case non-circulant handling exists
    # for; away from the edges the two modes agree to nine decimal places.
    truth = np.zeros((64, 64))
    truth[1:6, 1:6] = 100.0
    truth[30:35, 30:35] = 100.0
    psf = runner.run(gaussian_psf, xy_dim=64, xy_sigma=3.0)
    small = runner.run(gaussian_psf, xy_dim=21, xy_sigma=3.0)
    blurred = blur(truth, psf)

    circ = runner.run(richardson_lucy, image=blurred, psf=small, num_iters=50)
    noncirc = runner.run(
        richardson_lucy, image=blurred, psf=small, num_iters=50, noncirc=True
    )

    assert noncirc.shape == truth.shape
    assert rmse(noncirc, truth) < rmse(circ, truth)


@pytest.mark.env("skimage")
def test_richardson_lucy_restores_masked_pixels(runner):
    from skop.ops.deconvolve import richardson_lucy
    from skop.ops.kernels import gaussian_psf

    psf = runner.run(gaussian_psf, xy_dim=64, xy_sigma=2.0)
    blurred = blur(points_2d(), psf)

    mask = np.ones_like(blurred)
    mask[0:5, 0:5] = 0.0
    damaged = blurred.copy()
    damaged[0:5, 0:5] = 9999.0  # a saturated corner

    result = runner.run(
        richardson_lucy, image=damaged, psf=psf, num_iters=20, mask=mask
    )
    # Masked pixels come back untouched, and do not smear into their surroundings.
    assert np.allclose(result[0:5, 0:5], 9999.0)
    assert result[10:, 10:].max() < 9999.0


@pytest.mark.env("skimage")
def test_richardson_lucy_reports_progress(runner):
    from skop.ops.deconvolve import richardson_lucy
    from skop.ops.kernels import gaussian_psf

    psf = runner.run(gaussian_psf, xy_dim=32, xy_sigma=2.0)
    image = blur(points_2d(32), psf)

    messages = []
    runner.run(
        richardson_lucy,
        image=image,
        psf=psf,
        num_iters=5,
        on_progress=lambda event: messages.append(event.message),
    )
    assert any(m and "Iteration" in m for m in messages)


@pytest.mark.gpu
@pytest.mark.env("cupy")
def test_richardson_lucy_cupy_recovers_a_blurred_image(runner):
    from skop.ops.deconvolve import richardson_lucy_cupy

    truth = points_2d()
    # NB: generated here rather than through the skimage env, so this test
    # needs only the one environment it declares.
    coords = np.linspace(-32, 32, 64)
    profile = np.exp(-(coords**2) / (2.0 * 2.0**2))
    psf = profile[:, np.newaxis] * profile[np.newaxis, :]
    psf /= psf.sum()
    blurred = blur(truth, psf)

    result = runner.run(richardson_lucy_cupy, image=blurred, psf=psf, num_iters=50)

    assert result.shape == truth.shape
    assert rmse(result, truth) < rmse(blurred, truth)


@pytest.mark.gpu
@pytest.mark.env("cupy")
@pytest.mark.env("skimage")
def test_both_backends_agree(runner):
    from skop.ops.deconvolve import richardson_lucy, richardson_lucy_cupy
    from skop.ops.kernels import gaussian_psf

    psf = runner.run(gaussian_psf, xy_dim=64, xy_sigma=2.0)
    blurred = blur(points_2d(), psf)

    cpu = runner.run(richardson_lucy, image=blurred, psf=psf, num_iters=25)
    gpu = runner.run(richardson_lucy_cupy, image=blurred, psf=psf, num_iters=25)

    # Loosely: the two run the same iteration, but cupy's FFT is single
    # precision and numpy's is double, and 25 iterations compound the gap.
    assert gpu.shape == cpu.shape
    assert rmse(gpu, cpu) < 0.01 * cpu.max()


@pytest.mark.env("unseg-cv")
def test_unseg(runner):
    from skop.ops.segment import unseg

    rng = np.random.default_rng(11)
    size = 224
    yy, xx = np.mgrid[0:size, 0:size]

    centers = [
        (y + rng.integers(-5, 6), x + rng.integers(-5, 6))
        for y in range(28, size - 20, 48)
        for x in range(28, size - 20, 48)
    ]
    nuclei = np.zeros((size, size))
    for cy, cx in centers:
        # NB: heterogeneous sizes on purpose. UNSEG multi-Otsu-thresholds the
        # small-object areas, which needs them to actually differ.
        sigma = rng.uniform(5.0, 11.0)
        nuclei += rng.uniform(0.6, 1.0) * np.exp(
            -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        )

    membrane = np.zeros((size, size))
    for edge in range(0, size, 48):
        membrane[max(edge - 2, 0) : edge + 3, :] = 1.0
        membrane[:, max(edge - 2, 0) : edge + 3] = 1.0

    image = np.zeros((3, size, size), dtype=np.uint8)
    image[0] = (
        np.clip(membrane + rng.normal(0, 0.10, (size, size)), 0, 1) * 220
    ).astype(np.uint8)
    nuclei = np.clip(nuclei + rng.normal(0, 0.06, (size, size)), 0, 1)
    image[2] = (nuclei / nuclei.max() * 255).astype(np.uint8)

    result = runner.run(unseg, image=image)
    assert result.n_nuclei == len(centers)
    assert result.n_cells == len(centers)
    assert result.nuclei.shape == (size, size)
    assert result.cells.shape == (size, size)


def check_boxes(result, shape: tuple[int, int]) -> None:
    """Assertions every box detector's output has to satisfy."""
    boxes = result.boxes

    assert boxes.ndim == 2 and boxes.shape[1] == 4
    # Both detectors are class-agnostic, so finding nothing in an image full
    # of obvious blobs is a real failure -- most likely to_rgb handing the
    # model a black image, or a threshold set wrong.
    assert len(boxes) > 0

    assert np.all(boxes[:, 0] < boxes[:, 2]), "min_y must be below max_y"
    assert np.all(boxes[:, 1] < boxes[:, 3]), "min_x must be left of max_x"
    assert np.all(boxes >= 0)
    assert np.all(boxes[:, [0, 2]] <= shape[0])
    assert np.all(boxes[:, [1, 3]] <= shape[1])
    # Canonical order is row-major. If a detector forgot to convert, its
    # boxes would sit transposed -- which on a non-square image shows up as
    # coordinates outside the frame, and on a square one would not. Hence
    # the non-square image below.


def coins_like(height: int = 200, width: int = 320) -> np.ndarray:
    """Bright discs on a dark background: obviously objects, obviously no class.

    Non-square on purpose, so a missed x/y transposition fails here.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width), dtype=np.float32)
    for cy, cx, r in [
        (60, 60, 28),
        (60, 160, 24),
        (60, 260, 30),
        (140, 100, 26),
        (140, 220, 22),
    ]:
        image[(yy - cy) ** 2 + (xx - cx) ** 2 < r**2] = 1.0
    return (image * 200 + 20).astype(np.uint8)


@pytest.mark.env("pytorch")
def test_fastsam_finds_objects_of_no_particular_class(runner):
    from skop.ops.detect import fastsam

    image = coins_like()
    check_boxes(runner.run(fastsam, image=image), image.shape)


@pytest.mark.env("segment-everything")
def test_object_aware_yolo_finds_objects_of_no_particular_class(runner):
    from skop.ops.detect import object_aware_yolo

    image = coins_like()
    check_boxes(runner.run(object_aware_yolo, image=image), image.shape)


@pytest.mark.env("pytorch")
@pytest.mark.env("segment-everything")
def test_both_detectors_agree_there_are_objects(runner):
    # Needs both environments, so it skips on most machines. When it does
    # run, it is the check that the two are interchangeable in practice and
    # not only in signature.
    from skop.ops.detect import fastsam, object_aware_yolo

    image = coins_like()
    a = runner.run(fastsam, image=image)
    b = runner.run(object_aware_yolo, image=image)

    assert len(a.boxes) > 0 and len(b.boxes) > 0
    assert a.boxes.dtype == b.boxes.dtype

    # Both should land somewhere near the middle of the frame, where the
    # discs are -- a weak assertion on purpose, since the models are
    # different and their box counts will not match.
    for result in (a, b):
        centers = (result.boxes[:, :2] + result.boxes[:, 2:]) / 2
        assert np.all(centers[:, 0] < image.shape[0])
        assert np.all(centers[:, 1] < image.shape[1])


@pytest.mark.env("skimage")
def test_paraxial_otf_is_a_diffraction_limited_passband(runner):
    from skop.ops.kernels import paraxial_otf

    otf = runner.run(paraxial_otf, size=128, wavelength=0.53, pixel_size=0.1)

    assert otf.shape == (128, 128)
    # An OTF passes DC untouched and attenuates everything else, so its
    # maximum is 1 and sits at the centre.
    assert otf.max() == pytest.approx(1.0)
    # Past the cutoff it is exactly zero, not merely small: that hard edge is
    # what makes it diffraction-limited rather than just low-pass. The corners
    # are the farthest points from the centre, so they are always outside.
    assert otf[0, 0] == 0.0
    assert otf[-1, -1] == 0.0
    assert np.all(otf >= 0.0)


@pytest.mark.env("skimage")
def test_a_wider_aperture_passes_more_frequencies(runner):
    # The physics the op exists to express: resolution goes as wavelength over
    # NA, so opening the aperture widens the passband. If the two got divided
    # the wrong way round this is what would catch it.
    from skop.ops.kernels import paraxial_otf

    narrow = runner.run(paraxial_otf, size=128, numerical_aperture=0.5)
    wide = runner.run(paraxial_otf, size=128, numerical_aperture=1.4)

    assert (wide > 0).sum() > (narrow > 0).sum()


@pytest.mark.env("skimage")
def test_paraxial_psf_is_centred_symmetric_and_normalized(runner):
    from skop.ops.kernels import paraxial_psf

    psf = runner.run(paraxial_psf, size=128, wavelength=0.53, pixel_size=0.1)

    assert psf.shape == (128, 128)
    assert psf.dtype == np.float32
    assert psf.sum() == pytest.approx(1.0)

    # The peak sits on the fftshift centre, which is what a deconvolution op
    # assumes about a kernel -- a PSF centred anywhere else shifts the result.
    peak = np.unravel_index(psf.argmax(), psf.shape)
    assert peak == (64, 64)

    # Exactly symmetric, not approximately: the OTF is real and even, so its
    # transform is too, and any asymmetry here would mean an indexing slip.
    # Symmetry is i <-> size - i, so index 0 has no partner and is excluded.
    assert np.array_equal(psf, psf.T)
    assert np.array_equal(psf[64, 65:], psf[64, 63::-1][:63])


@pytest.mark.env("skimage")
def test_a_paraxial_psf_has_negative_side_lobes(runner):
    """The property that stops this being a drop-in Richardson-Lucy kernel.

    An ideal paraxial PSF is the Airy intensity pattern, which is
    non-negative everywhere. This one is not: sampling the OTF on a grid and
    inverting it with a DFT leaves ringing of about 1e-4 of the peak, some of
    it below zero.

    That is small enough to ignore for simulation -- blur an image with it and
    nothing looks wrong -- and fatal for deconvolution. Richardson-Lucy is a
    Poisson maximum-likelihood iteration built on ratios of the forward
    projection, so a kernel with negative entries does not converge slowly, it
    diverges: feeding this straight to ``richardson_lucy`` reaches 1e53 in
    fifty iterations.

    Recorded as a test rather than fixed in the op, because clipping would
    silently change every number the original tnia-python function produced.
    A caller that wants to deconvolve with this should clip and renormalize,
    which is what the assertion below shows costs essentially nothing.
    """
    from skop.ops.kernels import paraxial_psf

    psf = runner.run(paraxial_psf, size=128, wavelength=0.53, pixel_size=0.1)

    # No single negative value is large -- each is around 1e-4 of the peak,
    # which is why this is a trap rather than an obvious bug.
    assert psf.min() < 0.0
    assert abs(psf.min()) < 1e-3 * psf.max()

    # In aggregate they are not negligible: about 1% of the PSF's mass sits
    # below zero. That is the number that says the op must not quietly clip,
    # since doing so would move every result by more than a rounding error.
    negative_mass = float(psf[psf < 0].sum())
    assert -0.02 < negative_mass < -0.005

    clipped = np.clip(psf, 0.0, None)
    clipped /= clipped.sum()
    assert clipped.sum() == pytest.approx(1.0)
    assert abs(clipped.max() - psf.max()) / psf.max() < 0.02


@pytest.mark.env("skimage")
def test_a_clipped_paraxial_psf_can_be_deconvolved_with(runner):
    # With the negative lobes removed it behaves like any other kernel, which
    # is what makes the diagnosis above the whole story rather than a guess.
    from skop.ops.deconvolve import richardson_lucy
    from skop.ops.kernels import paraxial_psf

    truth = points_2d(64)
    psf = runner.run(paraxial_psf, size=64, numerical_aperture=0.8, pixel_size=0.1)
    psf = np.clip(psf, 0.0, None)
    psf /= psf.sum()

    blurred = blur(truth, psf)
    restored = runner.run(richardson_lucy, image=blurred, psf=psf, num_iters=50)
    assert rmse(restored, truth) < rmse(blurred, truth)


@pytest.mark.env("sdeconv")
def test_gibson_lanni_is_a_normalized_centred_volume(runner):
    # Small on purpose: this is a CPU torch computation and the shape and the
    # normalization do not need a big one.
    from skop.ops.kernels import gibson_lanni

    psf = runner.run(gibson_lanni, xy_size=64, z_size=32)

    assert psf.shape == (32, 64, 64)
    assert psf.dtype == np.float32
    assert psf.sum() == pytest.approx(1.0, rel=1e-5)
    assert np.all(psf >= 0.0)

    # In focus and on axis, so the peak belongs on the centre plane. That is
    # what recenter=True buys, and this is the assertion that says so.
    peak_z, peak_y, peak_x = np.unravel_index(psf.argmax(), psf.shape)
    assert peak_z == 32 // 2
    # Laterally the peak sits at (size - 1) // 2, not size // 2: sdeconv builds
    # its grid so that an even-sized PSF centres on the lower of the two middle
    # pixels. Worth pinning, since it is the kind of half-pixel convention that
    # shows up later as a deconvolution result drifting by one pixel.
    assert (peak_y, peak_x) == (31, 31)


@pytest.mark.env("sdeconv")
def test_a_confocal_psf_is_tighter_than_a_widefield_one(runner):
    # confocal_factor squares the PSF, which is the whole physical claim being
    # made: detecting through the illumination aperture narrows it.
    from skop.ops.kernels import gibson_lanni

    widefield = runner.run(gibson_lanni, xy_size=64, z_size=32, confocal_factor=1.0)
    confocal = runner.run(gibson_lanni, xy_size=64, z_size=32, confocal_factor=2.0)

    def above_half(psf: np.ndarray) -> int:
        return int((psf > psf.max() / 2).sum())

    assert above_half(confocal) < above_half(widefield)


@pytest.mark.env("sdeconv")
def test_depth_below_the_coverslip_makes_the_psf_axially_asymmetric(runner):
    # Spherical aberration is the reason to reach for Gibson-Lanni over the
    # paraxial model, and a non-zero pz is how you ask for it. At pz=0 the two
    # halves of the axial profile match; deep in the sample they must not.
    from skop.ops.kernels import gibson_lanni

    def axial_asymmetry(pz: float) -> float:
        psf = runner.run(gibson_lanni, xy_size=64, z_size=32, pz=pz, ns=1.33)
        profile = psf.sum(axis=(1, 2))
        return float(np.abs(profile - profile[::-1]).max() / profile.max())

    assert axial_asymmetry(0.0) < axial_asymmetry(2.0)


@pytest.mark.env("sdeconv")
def test_recentring_survives_a_point_deep_in_the_sample(runner):
    """The headroom is computed from pz, and this is why.

    sdeconv's PSF peak marches toward plane zero as the point goes deeper, so
    a fixed crop window -- which is what the original used, and what design
    0010 originally specified -- runs off the front of the volume somewhere
    around one micron. These depths are all ordinary; none of them should be
    an error, and every one of them should come back centred.
    """
    from skop.ops.kernels import gibson_lanni

    for pz in (0.0, 1.0, 4.0, 8.0):
        psf = runner.run(gibson_lanni, xy_size=32, z_size=32, pz=pz)
        assert psf.shape == (32, 32, 32)
        assert psf.sum() == pytest.approx(1.0, rel=1e-5)
        assert int(np.unravel_index(psf.argmax(), psf.shape)[0]) == 16, (
            f"pz={pz} came back off-centre"
        )


@pytest.mark.env("sdeconv")
def test_a_fluorophore_can_stand_in_for_a_wavelength(runner):
    # The enum is a float, so it crosses into the worker as one -- there is no
    # codec for an enum, and this is the test that would fail if that changed.
    from skop.ops.kernels import Fluorophore, gibson_lanni

    psf = runner.run(gibson_lanni, xy_size=32, z_size=16, wavelength=Fluorophore.DAPI)
    assert psf.shape == (16, 32, 32)
    assert psf.sum() == pytest.approx(1.0, rel=1e-5)
