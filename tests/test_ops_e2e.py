"""Real ops, running in their real environments.

Each test skips unless its environment has already been built, so the suite
stays runnable without waiting on a TensorFlow or PyTorch install. To opt in,
build the environment first:

    uv run python -c "import opkit; opkit.Runner().environment('stardist-tf')"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from appose.util.filepath import appose_envs_dir

import opkit


def requires_env(env_id: str):
    built = (Path(appose_envs_dir()) / f"opkit-{env_id}").is_dir()
    return pytest.mark.skipif(not built, reason=f"env '{env_id}' is not built")


@pytest.fixture(scope="module")
def runner():
    with opkit.Runner() as r:
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


@requires_env("skimage")
def test_otsu(runner):
    from ops import otsu

    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:25, 10:25] = 200
    image[40:55, 40:55] = 220

    labels = runner.run(otsu.otsu, image=image)
    assert labels.dtype == np.uint16
    assert labels.max() == 2

    mask = runner.run(otsu.otsu, image=image, label_objects=False)
    assert set(np.unique(mask)) == {0, 1}


@requires_env("skimage")
def test_synthetic_nuclei(runner):
    from ops import starfun3d

    volume = runner.run(
        starfun3d.synthetic_nuclei,
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


@requires_env("stardist-tf")
def test_stardist2d(runner):
    from ops import stardist2d

    labels = runner.run(stardist2d.stardist2d, image=blobs_2d())
    assert labels.dtype == np.uint16
    assert labels.max() == 5


@requires_env("stardist-tf")
def test_stardist2d_reports_progress(runner):
    from ops import stardist2d

    messages = []
    runner.run(
        stardist2d.stardist2d,
        image=blobs_2d(),
        on_progress=lambda event: messages.append(event.message),
    )
    assert any(m and "Predicting" in m for m in messages)


@requires_env("stardist-tf")
def test_starfun3d_segments_what_was_generated(runner):
    from ops import starfun3d

    # NB: this crosses two environments -- generated under skimage, segmented
    # under stardist-tf -- in two worker processes.
    volume = runner.run(
        starfun3d.synthetic_nuclei,
        size_z=32,
        size_y=128,
        size_x=128,
        n_nuclei=5,
        seed=3,
    )
    result = runner.run(starfun3d.segment_nuclei, image=volume)

    # NB: not an exact count. Randomly placed nuclei can overlap, and how many
    # remain separable is a property of the model, not of this plumbing.
    assert result.labels.shape == volume.shape
    assert 1 <= result.labels.max() <= 5
    assert result.points.shape == (result.labels.max(), 3)


@requires_env("stardist-tf")
def test_starfun3d_honors_model_choice(runner):
    from ops import starfun3d

    # The original loaded 'confocal' whichever model was requested; each of
    # these now resolves to its own weights.
    volume = np.zeros((16, 64, 64), dtype=np.uint16)
    for model in (starfun3d.Model.sospim, starfun3d.Model.confocal):
        result = runner.run(starfun3d.segment_nuclei, image=volume, model=model)
        assert result.labels.shape == volume.shape


@requires_env("unseg-cv")
def test_unseg(runner):
    from ops import unseg

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

    result = runner.run(unseg.unseg, image=image)
    assert result.n_nuclei == len(centers)
    assert result.n_cells == len(centers)
    assert result.nuclei.shape == (size, size)
    assert result.cells.shape == (size, size)
