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


def rainbow_pancakes() -> np.ndarray:
    """A slicewise segmentation of two touching columns, renumbered per slice.

    What any 2-D op iterated over Z produces: each slice is correct on its own
    terms, and no two slices agree on what to call anything.
    """
    stack = np.zeros((6, 24, 24), dtype=np.uint16)
    for z in range(6):
        left, right = (7, 3) if z % 2 else (1, 2)
        stack[z, 4:16, 4:12] = left
        stack[z, 4:16, 12:20] = right  # Shares an edge with the left column.
    stack[3:, 18:22, 2:8] = 9  # A third object, starting partway up.
    return stack


@pytest.mark.env("skimage")
def test_connect(runner):
    from skop.ops.labels import connect

    stack = rainbow_pancakes()
    out = runner.run(connect, labels=stack)

    assert out.shape == stack.shape
    # Three objects, numbered consecutively however the slices were numbered.
    assert set(np.unique(out)) == {0, 1, 2, 3}
    # Every slice agrees, and the two touching columns stay two objects.
    left = {int(out[z, 8, 8]) for z in range(6)}
    right = {int(out[z, 8, 16]) for z in range(6)}
    assert len(left) == len(right) == 1
    assert left != right
    # The late object exists only where it existed before connecting.
    late = int(out[4, 20, 4])
    assert late not in left | right
    assert np.array_equal(out == late, stack == 9)


@pytest.mark.env("skimage")
def test_connect_threshold_and_criterion(runner):
    from skop.ops.labels import connect

    # An object that barely grazes its successor: linked on any overlap, but
    # not at the default threshold.
    grazing = np.zeros((2, 16, 16), dtype=np.uint16)
    grazing[0, 0:9, 0:9] = 1
    grazing[1, 8:16, 8:16] = 1
    assert runner.run(connect, labels=grazing).max() == 2
    assert runner.run(connect, labels=grazing, threshold=0.0).max() == 1

    # An object that shrinks sharply: too small a union for IoU, but fully
    # contained, which is what 'iop' scores on.
    shrinking = np.zeros((2, 16, 16), dtype=np.uint16)
    shrinking[0, 0:12, 0:12] = 1
    shrinking[1, 0:3, 0:3] = 1
    assert runner.run(connect, labels=shrinking).max() == 2
    assert runner.run(connect, labels=shrinking, criterion="iop").max() == 1


@pytest.mark.env("skimage")
def test_connect_gaps_are_not_closed(runner):
    from skop.ops.labels import connect

    # An object missing from the middle slice starts over rather than
    # reclaiming its old label, which is the documented behaviour.
    stack = np.zeros((3, 16, 16), dtype=np.uint16)
    stack[0, 2:8, 2:8] = 1
    stack[2, 2:8, 2:8] = 1
    out = runner.run(connect, labels=stack)
    assert out[0, 4, 4] != out[2, 4, 4]
    assert set(np.unique(out)) == {0, 1, 2}


@pytest.mark.env("skimage")
def test_otsu_then_connect(runner):
    from skop.ops.labels import connect
    from skop.ops.threshold import otsu

    # The pairing this op exists for: threshold each plane on its own, which
    # numbers each plane on its own, then reconcile the numbering.
    volume = np.zeros((5, 64, 64), dtype=np.uint8)
    volume[:, 8:24, 8:24] = 200
    volume[:, 40:56, 40:56] = 220
    sliced = np.stack([runner.run(otsu, image=plane) for plane in volume])
    out = runner.run(connect, labels=sliced)

    assert set(np.unique(out)) == {0, 1, 2}
    assert len({int(out[z, 16, 16]) for z in range(5)}) == 1


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


def check_cellpose(result, image, expected: int) -> None:
    """Assertions either Cellpose op has to satisfy."""
    assert result.shape == image.shape
    assert result.dtype == np.uint16
    # Round blobs on a flat background are the easiest thing Cellpose does.
    # An off-by-a-few count is a model difference; zero is a broken op.
    assert result.max() == expected, f"found {result.max()} objects, wanted {expected}"


@pytest.mark.env("cellpose3")
def test_cellpose3_segments_round_cells(runner):
    # The v3 model zoo, kept because a model finetuned on your organism can
    # still beat the generalist that replaced it.
    #
    # NB: an explicit diameter, unlike the CellposeSAM test above. cyto3's
    # size model reads these synthetic discs as far smaller than they are and
    # then finds nothing at all -- at diameter=0 or 30 it returns an empty
    # label image, at 50 it finds all five. That is a real difference between
    # the two models rather than a broken op, and it is the reason the size
    # estimate is worth overriding on anything that does not look like cells.
    from skop.ops.segment import cellpose3

    image = coins_like()
    check_cellpose(runner.run(cellpose3, image=image, diameter=50.0), image, 5)


@pytest.mark.env("cellpose3")
def test_cellpose3_estimates_diameter_on_softer_objects(runner):
    # Where the size model does work, so that the test above is understood as
    # a property of that image and not as "auto never works".
    from skop.ops.segment import cellpose3

    image = blobs_2d()
    check_cellpose(runner.run(cellpose3, image=image, diameter=0.0), image, 5)


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


def check_masks(result, image, prompts) -> None:
    """Assertions every mask detector's output has to satisfy."""
    import numpy as np

    from skop import boxes as skop_boxes
    from skop import masks as skop_masks

    assert len(result.masks) == len(prompts)
    assert result.masks.shape[1:] == image.shape
    assert result.masks.dtype == np.uint8
    assert set(np.unique(result.masks)) <= {0, 1}
    # Empty prompts are dropped from both, so these stay parallel.
    assert len(result.boxes) == len(result.masks)
    assert np.array_equal(result.boxes, prompts)

    for i, box in enumerate(prompts):
        mask = result.masks[i]
        # A disc inscribed in its prompt fills about pi/4 of it. Well under
        # that means SAM found an edge instead of the object; at 1.0 it
        # returned the box it was given.
        min_y, min_x, max_y, max_x = box.astype(int)
        inside = mask[min_y:max_y, min_x:max_x].sum()
        assert inside == mask.sum(), "mask escaped the box it was prompted with"
        fraction = inside / ((max_y - min_y) * (max_x - min_x))
        assert 0.3 < fraction < 0.95, f"mask {i} fills {fraction:.2f} of its prompt"

    # The whole reason these are not a label image: the projection is lossy,
    # and going through it is what a front end will do.
    labels = skop_masks.to_labels_2d(result.masks)
    assert labels.shape == image.shape
    assert len(np.unique(labels)) == len(prompts) + 1  # the discs, plus 0
    assert skop_boxes.from_labels(labels).shape == (len(prompts), 4)


def coin_prompts() -> np.ndarray:
    """Boxes around the discs coins_like() draws, generous enough to be hints.

    Loose on purpose: if a detector returned its prompt unchanged the fill
    fraction in check_masks would be 1.0 and the test would catch it.
    """
    return np.array(
        [[28, 28, 92, 92], [30, 130, 90, 190], [26, 226, 94, 294]],
        dtype=np.float32,
    )


@pytest.mark.env("pytorch")
def test_microsam_segments_what_it_is_prompted_with(runner):
    from skop.ops.mask import microsam_masks

    image = coins_like()
    prompts = coin_prompts()
    check_masks(runner.run(microsam_masks, image=image, boxes=prompts), image, prompts)


@pytest.mark.env("segment-everything")
def test_mobilesam_segments_what_it_is_prompted_with(runner):
    from skop.ops.mask import mobilesam_masks

    image = coins_like()
    prompts = coin_prompts()
    check_masks(runner.run(mobilesam_masks, image=image, boxes=prompts), image, prompts)


@pytest.mark.env("segment-everything")
def test_mobilesam_batches_without_changing_the_answer(runner):
    from skop.ops.mask import mobilesam_masks

    # batch_size trades GPU memory for decoder calls and must not touch the
    # result. Two prompts in one batch, then in two -- the path where the
    # original narrowed its cached embeddings in place.
    image = coins_like()
    prompts = coin_prompts()

    one = runner.run(mobilesam_masks, image=image, boxes=prompts, batch_size=100)
    many = runner.run(mobilesam_masks, image=image, boxes=prompts, batch_size=2)
    assert np.array_equal(one.masks, many.masks)


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


def noisy_step(size: int = 64, seed: int = 0) -> np.ndarray:
    """A step edge under Gaussian noise: the thing edge preservation is about."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), dtype=np.float32)
    image[:, size // 2 :] = 100.0
    return image + rng.normal(0.0, 5.0, image.shape).astype(np.float32)


def salt_and_pepper(image: np.ndarray, fraction: float = 0.05, seed: int = 0):
    """An image with a fraction of its pixels shot to the extremes."""
    rng = np.random.default_rng(seed)
    out = image.copy()
    hit = rng.random(image.shape) < fraction
    out[hit] = np.where(rng.random(image.shape)[hit] < 0.5, image.min(), image.max())
    return out


@pytest.mark.env("skimage")
@pytest.mark.parametrize(
    "method", ["otsu", "isodata", "li", "mean", "minimum", "triangle", "yen"]
)
def test_every_global_threshold_finds_two_squares(runner, method):
    # The point of having seven: on an easy image they must all agree, so
    # the choice only starts to matter once the histogram gets hard.
    from skop.ops import threshold

    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:25, 10:25] = 200
    image[40:55, 40:55] = 220

    labels = runner.run(getattr(threshold, method), image=image)
    assert labels.dtype == np.uint16
    assert labels.max() == 2, f"{method} did not find both squares"


@pytest.mark.env("skimage")
def test_a_threshold_can_be_inverted_and_left_unlabelled(runner):
    from skop.ops.threshold import triangle

    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:25, 10:25] = 200

    mask = runner.run(triangle, image=image, label_objects=False)
    assert set(np.unique(mask)) == {0, 1}
    assert mask[10:25, 10:25].all()

    inverted = runner.run(triangle, image=image, invert=True, label_objects=False)
    assert not inverted[10:25, 10:25].any()
    assert (mask.astype(bool) ^ inverted.astype(bool)).all()


@pytest.mark.env("skimage")
def test_multiotsu_returns_classes_rather_than_objects(runner):
    # The one threshold op that is not a foreground/background question.
    from skop.ops.threshold import multiotsu

    image = np.zeros((60, 60), dtype=np.uint8)
    image[10:25, 10:25] = 120
    image[35:50, 35:50] = 240

    classes = runner.run(multiotsu, image=image, classes=3)
    assert set(np.unique(classes)) == {0, 1, 2}
    assert (classes[10:25, 10:25] == 1).all()
    assert (classes[35:50, 35:50] == 2).all()

    # ...and with label_objects it becomes one, cutting at the top threshold.
    objects = runner.run(multiotsu, image=image, classes=3, label_objects=True)
    assert objects.max() == 1


@pytest.mark.env("skimage")
def test_median_beats_the_mean_on_salt_and_pepper(runner):
    # Why there are both: an outlier is discarded by one and spread by the
    # other, which is the whole argument for a rank filter.
    from skop.ops.smooth import mean, median

    truth = np.zeros((64, 64), dtype=np.float32)
    truth[:, 32:] = 100.0
    corrupted = salt_and_pepper(truth)

    by_median = runner.run(median, image=corrupted, radius=2)
    by_mean = runner.run(mean, image=corrupted, radius=2)
    assert rmse(by_median, truth) < rmse(by_mean, truth)
    assert rmse(by_median, truth) < rmse(corrupted, truth)


@pytest.mark.env("skimage")
@pytest.mark.parametrize("name", ["kuwahara", "bilateral", "tv_chambolle", "median"])
def test_the_edge_preserving_filters_keep_the_edge(runner, name):
    # Each of these denoises about as hard as the Gaussian below, and the
    # difference is what happens at the boundary: they must smooth the flat
    # regions at least as well while leaving the step nearly intact.
    from skop.ops import smooth

    image = noisy_step()
    settings = {
        "kuwahara": {"radius": 3},
        "bilateral": {"sigma_color": 0.05, "sigma_spatial": 3.0},
        "tv_chambolle": {"weight": 0.1},
        "median": {"radius": 3},
    }[name]

    filtered = runner.run(getattr(smooth, name), image=image, **settings)
    blurred = runner.run(smooth.gaussian, image=image, sigma=3.0)

    def jump(x):
        return float(x[:, 32:].mean(axis=0)[0] - x[:, :32].mean(axis=0)[-1])

    def flat_noise(x):
        return float(x[:, :24].std())

    assert flat_noise(filtered) < 0.7 * flat_noise(image)
    assert jump(filtered) > 0.8 * jump(image), f"{name} smeared the edge"
    assert jump(blurred) < 0.5 * jump(image), "the Gaussian was supposed to smear it"


@pytest.mark.env("skimage")
def test_the_denoisers_preserve_the_intensity_range(runner):
    # They work internally on a [0, 1] copy so their strengths are portable
    # between dtypes; this is the test that they scale the result back.
    from skop.ops import smooth

    image = noisy_step().astype(np.uint16) * 10
    for name, settings in [
        ("bilateral", {}),
        ("tv_chambolle", {}),
        ("wavelet", {}),
        ("nl_means", {"patch_size": 3, "patch_distance": 3}),
    ]:
        out = runner.run(getattr(smooth, name), image=image, **settings)
        assert out.dtype == np.float32
        assert out.mean() == pytest.approx(image.mean(), rel=0.1), name


@pytest.mark.env("skimage")
def test_smoothing_a_volume_and_an_rgb_image(runner):
    # A neighborhood is spatial, so the channels of an RGB image are filtered
    # apart -- but the planes of a volume are not.
    from skop.ops.smooth import kuwahara, median

    volume = np.zeros((8, 32, 32), dtype=np.uint8)
    volume[2:6, 8:24, 8:24] = 200
    assert runner.run(median, image=volume, radius=1).shape == volume.shape

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24, 0] = 200
    filtered = runner.run(kuwahara, image=rgb, radius=2)
    assert filtered.shape == rgb.shape
    assert filtered[..., 1].max() == 0, "red leaked into green"


@pytest.mark.env("skimage")
@pytest.mark.parametrize("name", ["sobel", "scharr", "prewitt", "farid", "roberts"])
def test_the_gradient_filters_respond_at_the_edge(runner, name):
    from skop.ops import edges

    image = np.zeros((64, 64), dtype=np.float32)
    image[:, 32:] = 100.0

    response = runner.run(getattr(edges, name), image=image)
    assert response.dtype == np.float32
    # Large within two pixels of the boundary, near zero everywhere else.
    assert response[:, 30:34].max() > 10 * response[:, :28].max()


@pytest.mark.env("skimage")
def test_difference_of_gaussians_picks_out_blobs_of_a_size(runner):
    # A band-pass keeps what falls between its two scales, which is what
    # makes it a blob detector rather than a smoother.
    from skop.ops.edges import difference_of_gaussians

    image = blobs_2d(size=128, sigma=7.0).astype(np.float32)
    matched = runner.run(
        difference_of_gaussians, image=image, low_sigma=5.0, high_sigma=12.0
    )
    mismatched = runner.run(
        difference_of_gaussians, image=image, low_sigma=0.5, high_sigma=1.0
    )
    assert matched.max() > 5 * mismatched.max()


@pytest.mark.env("skimage")
def test_unsharp_mask_steepens_an_edge(runner):
    from skop.ops.edges import unsharp_mask
    from skop.ops.smooth import gaussian

    image = np.zeros((64, 64), dtype=np.float32)
    image[:, 32:] = 100.0
    blurred = runner.run(gaussian, image=image, sigma=3.0)
    sharpened = runner.run(unsharp_mask, image=blurred, radius=3.0, amount=2.0)

    def slope(x):
        return float(np.diff(x[32]).max())

    assert slope(sharpened) > slope(blurred)


@pytest.mark.env("skimage")
@pytest.mark.parametrize("name", ["frangi", "sato", "meijering"])
def test_the_ridge_filters_prefer_a_line_to_a_blob(runner, name):
    # What separates them from the gradient filters: they answer "is this a
    # filament of about this thickness", not "is this a boundary".
    from skop.ops import edges

    image = np.zeros((96, 96), dtype=np.float32)
    image[46:50, 8:88] = 100.0  # A line, four pixels thick.

    response = runner.run(
        getattr(edges, name),
        image=image,
        sigma_min=1.0,
        sigma_max=3.0,
        sigma_step=1.0,
        black_ridges=False,
    )
    assert response.dtype == np.float32
    assert response[44:52, 20:76].mean() > 5 * response[:20, :20].mean() + 1e-6


@pytest.mark.env("skimage")
def test_erosion_and_dilation_bound_the_image(runner):
    # Every other morphological op is these two composed, so this is the
    # invariant the rest inherit.
    from skop.ops.morphology import dilation, erosion

    image = blobs_2d(size=64).astype(np.uint16)
    eroded = runner.run(erosion, image=image, radius=2)
    dilated = runner.run(dilation, image=image, radius=2)
    assert eroded.dtype == image.dtype
    assert (eroded <= image).all()
    assert (image <= dilated).all()


@pytest.mark.env("skimage")
def test_white_tophat_is_the_image_minus_its_opening(runner):
    # Not an implementation detail: it is what makes the radius mean "the
    # size of the things to keep", and why this is background subtraction.
    from skop.ops.morphology import opening, white_tophat

    image = blobs_2d(size=64).astype(np.uint16)
    opened = runner.run(opening, image=image, radius=5)
    tophat = runner.run(white_tophat, image=image, radius=5)
    assert np.array_equal(tophat, image - opened)


@pytest.mark.env("skimage")
def test_opening_drops_what_is_smaller_than_the_footprint(runner):
    # The one thing the radius controls, stated as directly as it can be.
    from skop.ops.morphology import opening

    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:30, 10:30] = 200  # Large: survives.
    image[50:52, 50:52] = 200  # Small: does not.

    opened = runner.run(opening, image=image, radius=4)
    assert opened[15:25, 15:25].min() == 200, "the large square did not survive"
    assert opened[48:56, 48:56].max() == 0, "the speck did"
    # The corners are rounded off, which is what a ball footprint means.
    assert opened[10, 10] == 0


@pytest.mark.env("skimage")
def test_footprint_shape_reaches_the_op(runner):
    # The enum crosses into the worker as its value, like Fluorophore does.
    from skop.ops.morphology import Footprint, dilation

    image = np.zeros((33, 33), dtype=np.uint8)
    image[16, 16] = 255

    counts = {
        shape: int((runner.run(dilation, image=image, radius=4, shape=shape) > 0).sum())
        for shape in Footprint
    }
    assert counts[Footprint.diamond] < counts[Footprint.ball] < counts[Footprint.box]
    assert counts[Footprint.box] == 81


@pytest.mark.env("skimage")
def test_smoothing_before_thresholding(runner):
    # The workflow these ops exist for: a noisy image Otsu shatters, made
    # whole by an edge-preserving smooth first.
    from skop.ops.smooth import kuwahara
    from skop.ops.threshold import otsu

    truth = np.zeros((96, 96), dtype=np.float32)
    truth[20:70, 20:70] = 200.0
    noisy = salt_and_pepper(truth, fraction=0.08, seed=1)

    assert runner.run(otsu, image=noisy).max() > 5, "the test image was too easy"
    smoothed = runner.run(kuwahara, image=noisy, radius=3)
    assert runner.run(otsu, image=smoothed).max() == 1
