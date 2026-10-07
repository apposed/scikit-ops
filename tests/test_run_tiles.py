"""Running an op tile by tile on the host: skop._tiling, and Runner(memory=).

The planner itself is opspec's, tested in opspec/tests/test_tiling.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import skop
from opspec.op import PeakMemory
from opspec.tiling import plan_tiles
from skop._tiling import run_tiles

RADIUS = 2


def box_filter(image: np.ndarray) -> np.ndarray:
    """A mean over a (2r+1)^n neighbourhood, edges repeated: a stand-in op."""
    padded = np.pad(image.astype(np.float64), RADIUS, mode="edge")
    out = np.zeros(image.shape)
    for offset in np.ndindex(*(2 * RADIUS + 1,) * image.ndim):
        out += padded[tuple(slice(o, o + n) for o, n in zip(offset, image.shape))]
    return out / (2 * RADIUS + 1) ** image.ndim


class Lazy:
    """Slices like a zarr array, and remembers how much each read took."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.shape, self.dtype = data.shape, data.dtype
        self.ndim, self.size = data.ndim, data.size
        self.reads: list[int] = []

    def __getitem__(self, key):
        piece = self.data[key]
        self.reads.append(piece.size)
        return piece


def image(shape=(12, 30, 40)) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, shape).astype(np.uint8)


def test_tiles_put_back_together_equal_the_whole():
    # The overlap is the op's reach, so cropping loses nothing: exactly equal.
    data = image()
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K", overlap=RADIUS)
    assert plan.calls > 1
    assert np.array_equal(run_tiles(box_filter, data, plan), box_filter(data))


def test_the_input_is_only_ever_read_a_tile_at_a_time():
    data = Lazy(image())
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K", overlap=RADIUS)
    run_tiles(box_filter, data, plan)
    assert len(data.reads) == plan.calls
    assert max(data.reads) < data.size


def test_a_result_bigger_than_the_budget_goes_to_disk():
    data = image()
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K", overlap=RADIUS)
    result = run_tiles(box_filter, data, plan)
    # float64 output, 8x the input: more than the budget, so a memmap.
    assert isinstance(result, np.memmap)
    assert str(result.filename).endswith("result.npy")


@pytest.mark.env("skimage")
def test_gaussian_tiled_through_a_runner_equals_gaussian_whole():
    from skop.ops.smooth import gaussian

    data = image((24, 60, 80))
    lazy = Lazy(data)
    with skop.Runner() as runner:
        whole = runner.run(gaussian, image=data, sigma=1.5)
        # 8 bytes a pixel, ~0.9 MB whole: a 200K budget means several tiles.
        tiled = runner.run(gaussian, image=lazy, sigma=1.5, memory="200K")
    assert len(lazy.reads) > 1
    assert max(lazy.reads) < lazy.size
    np.testing.assert_allclose(tiled, whole, rtol=0, atol=1e-5)


@pytest.mark.env("skimage")
def test_a_plan_that_changes_nothing_does_not_stop_tiling():
    # A front end passes a plan for every input with declared axes; for a
    # variadic op on a volume it passes every axis through, unsliced and in
    # order, and tiling goes ahead.
    from skop.ops.smooth import gaussian

    data = image((24, 60, 80))
    plan = skop.plan(gaussian, "image", data, list("zyx"))
    assert not plan.iterate and not plan.select
    with skop.Runner() as runner:
        tiled = runner.run(
            gaussian, image=data, sigma=1.5, memory="200K", plans={"image": plan}
        )
    assert tiled.shape == data.shape
