"""InstanSeg, run for real in the pytorch environment.

Skipped unless that environment is built; see test_ops_e2e.py for how these
run. The image is synthetic nuclei, disks about 10 um across at InstanSeg's
0.5 um pixels, so the right count is known.
"""

from __future__ import annotations

import numpy as np
import pytest

import skop

pytestmark = pytest.mark.env("pytorch")

RADIUS = 10


@pytest.fixture(scope="module")
def runner():
    with skop.Runner() as r:
        yield r


def nuclei_image(size: int, spacing: int = 60, seed: int = 0) -> tuple[np.ndarray, int]:
    """Bright disks on a noisy dark background, and how many there are."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size))
    image = np.zeros((size, size), np.float32)
    count = 0
    for cy in range(spacing // 2, size - RADIUS, spacing):
        for cx in range(spacing // 2, size - RADIUS, spacing):
            cy2 = cy + rng.integers(-8, 9)
            cx2 = cx + rng.integers(-8, 9)
            image[(y - cy2) ** 2 + (x - cx2) ** 2 <= RADIUS**2] = 1.0
            count += 1
    image += rng.normal(0, 0.05, image.shape).astype(np.float32)
    return (image * 1000 + 100).clip(0).astype(np.uint16), count


def found(labels) -> int:
    return len(np.unique(labels)) - 1


def test_instanseg_finds_nuclei_and_cells(runner):
    from skop.ops.segment.instanseg import instanseg

    image, count = nuclei_image(512)
    result = runner.run(instanseg, image=image)
    assert result.nuclei.shape == result.cells.shape == image.shape
    assert abs(found(result.nuclei) - count) <= 0.2 * count


def test_instanseg_nuclei_only_gives_no_cells(runner):
    from skop.ops.segment.instanseg import Target, instanseg

    image, count = nuclei_image(256)
    result = runner.run(instanseg, image=image, target=Target.nuclei)
    assert result.cells is None
    assert abs(found(result.nuclei) - count) <= 0.2 * count


def test_instanseg_takes_several_channels(runner):
    from skop.ops.segment.instanseg import instanseg

    image, count = nuclei_image(256)
    two = np.stack([image, image // 2], axis=-1)
    result = runner.run(instanseg, image=two)
    assert result.nuclei.shape == image.shape
    assert abs(found(result.nuclei) - count) <= 0.2 * count


def test_instanseg_tiles_a_big_image(runner):
    from skop.ops.segment.instanseg import instanseg

    image, count = nuclei_image(2600)
    result = runner.run(instanseg, image=image)
    assert result.nuclei.shape == image.shape
    assert abs(found(result.nuclei) - count) <= 0.2 * count


def test_instanseg_takes_the_channel_it_is_told_is_nuclei(runner):
    from skop.ops.segment.instanseg import Target, instanseg

    nuclei, count = nuclei_image(256)
    noise = np.random.default_rng(1).integers(100, 200, nuclei.shape, np.uint16)
    image = np.stack([noise, nuclei], axis=-1)  # y, x, c: nuclei second
    result = runner.run(instanseg, image=image, nuclei_channel=2, target=Target.nuclei)
    assert abs(found(result.nuclei) - count) <= 0.2 * count

    with pytest.raises(Exception, match="has 2"):
        runner.run(instanseg, image=image, nuclei_channel=3)


def test_instanseg_takes_the_cells_as_a_layer_of_their_own(runner):
    from skop.ops.segment.instanseg import instanseg

    nuclei, count = nuclei_image(256)
    cells = np.random.default_rng(2).integers(100, 200, nuclei.shape, np.uint16)
    result = runner.run(instanseg, image=nuclei, cells=cells)
    assert result.nuclei.shape == result.cells.shape == nuclei.shape
    assert abs(found(result.nuclei) - count) <= 0.2 * count
