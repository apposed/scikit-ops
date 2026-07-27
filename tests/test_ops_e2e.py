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
