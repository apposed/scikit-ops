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
    from skop.ops.segment import stardist2d

    labels = runner.run(stardist2d, image=blobs_2d())
    assert labels.dtype == np.uint16
    assert labels.max() == 5


@pytest.mark.env("stardist-tf")
def test_stardist2d_reports_progress(runner):
    from skop.ops.segment import stardist2d

    messages = []
    runner.run(
        stardist2d,
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
