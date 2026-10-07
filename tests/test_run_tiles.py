"""Running an op tile by tile on the host: skop._tiling, and Runner(memory=).

The planner itself is opspec's, tested in opspec/tests/test_tiling.py.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pytest

import skop
from opspec.op import Axes, Overlap, PeakMemory, op
from opspec.tiling import plan_tiles
from skop._tiling import run_tiles
from skop.runner import tile_plan

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


def test_tiles_are_written_into_the_callers_array():
    data = image()
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K", overlap=RADIUS)
    out = np.zeros(data.shape)
    assert run_tiles(box_filter, data, plan, out) is out
    assert np.array_equal(out, box_filter(data))


def test_out_may_be_made_once_the_dtype_is_known():
    data = image()
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K", overlap=RADIUS)
    made = []

    def make(shape, dtype):
        made.append((shape, np.dtype(dtype)))
        return np.empty(shape, dtype)

    run_tiles(box_filter, data, plan, make)
    assert made == [(data.shape, np.dtype(np.float64))]


def test_an_out_of_the_wrong_shape_is_refused():
    data = image()
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K", overlap=RADIUS)
    with pytest.raises(ValueError, match="out has shape"):
        run_tiles(box_filter, data, plan, np.zeros((3, 3)))


@pytest.mark.env("skimage")
def test_out_is_filled_whether_or_not_the_run_is_tiled():
    from skop.ops.smooth import gaussian

    data = image((24, 60, 80))
    with skop.Runner() as runner:
        for memory in ("200K", "off"):
            out = np.zeros(data.shape, np.float32)
            result = runner.run(gaussian, image=data, memory=memory, out=out)
            assert result is out
            assert out.any()


def test_inputs_given_together_are_tiled_the_same_way():
    data = image()
    mask = (data > 128).astype(np.uint8)

    def masked(pieces):
        return box_filter(pieces["image"]) * pieces["mask"]

    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K", overlap=RADIUS)
    result = run_tiles(masked, {"image": data, "mask": mask}, plan)
    assert np.array_equal(result, box_filter(data) * mask)


def test_an_input_nobody_gave_is_passed_through_as_none():
    data = image()
    seen = []

    def remember(pieces):
        seen.append(pieces["mask"])
        return pieces["image"]

    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K")
    run_tiles(remember, {"image": data, "mask": None}, plan)
    assert seen and all(mask is None for mask in seen)


def test_inputs_tiled_together_must_match_in_shape():
    data = image()
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K")
    with pytest.raises(ValueError, match="must match"):
        run_tiles(lambda p: p["image"], {"image": data, "mask": data[:5]}, plan)


def test_the_runner_keeps_axes_it_may_not_split_whole():
    half_psf = Overlap(param="psf", of="shape", scale=0.5)

    @op(
        tile="image",
        overlap=half_psf,
        peak_memory=PeakMemory(10, "float64", pad=half_psf),
        split=("y", "x"),
    )
    def deconvolve(
        image: Annotated[np.ndarray, Axes("z?", "y", "x")], psf: np.ndarray
    ) -> np.ndarray:
        return image

    spec = skop.OpSpec.from_op(deconvolve)
    volume, psf = np.zeros((32, 256, 256), np.uint16), np.zeros((15, 9, 9))
    plan = tile_plan(spec, {"image": volume, "psf": psf}, "20M")
    assert plan.calls > 1
    assert all(t.read[0] == slice(0, 32) for t in plan.tiles)  # z never cut
    assert plan.overlap == (7, 4, 4)

    # A plane reads its axes as y, x: both may be cut.
    plane = tile_plan(
        spec, {"image": np.zeros((256, 256), np.uint16), "psf": psf[0]}, "200K"
    )
    assert plane.calls > 1


# -- the Tiler ------------------------------------------------------------------


@op(
    tile="image",
    overlap=Overlap(param="sigma", scale=4),
    peak_memory=PeakMemory(2, "float32"),
)
def smooth(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return image


@op(tile="image", peak_memory=PeakMemory(2, "float32", device="gpu"))
def on_gpu(image: np.ndarray) -> np.ndarray:
    return image


VOLUME = {"image": np.zeros((16, 512, 512), np.uint8)}


def test_a_tiler_can_override_the_overlap_and_the_tile_size():
    plan = skop.Tiler(memory="1G", overlap=3, tile_shape=(16, 128, 128)).plan(
        smooth, VOLUME
    )
    assert plan.overlap == 3
    assert plan.tile_shape == (16, 128, 128)
    assert plan.calls == 16


def test_off_and_an_op_without_hints_plan_nothing():
    assert skop.Tiler.off().plan(smooth, VOLUME) is None

    @op
    def plain(image: np.ndarray) -> np.ndarray:
        return image

    assert skop.Tiler().plan(plain, VOLUME) is None


def test_a_gpu_op_budgets_the_gpus_free_memory(monkeypatch):
    from skop import _tiling

    monkeypatch.setattr(_tiling, "gpu_free", lambda: 1024**3)
    assert skop.Tiler(memory=0.5).budget("gpu") == 1024**3 // 2
    # 16 x 512 x 512 x 8 bytes is 32M: a GPU budget of 4M cuts it up.
    assert skop.Tiler(memory=4 / 1024).plan(on_gpu, VOLUME).calls > 1


def test_a_gpu_that_cannot_be_asked_tiles_nothing(monkeypatch):
    from skop import _tiling

    monkeypatch.setattr(_tiling, "gpu_free", lambda: None)
    assert skop.Tiler().plan(on_gpu, VOLUME) is None


def test_nvidia_smi_is_read_in_mib():
    from skop._tiling import _parse_nvidia_smi

    assert _parse_nvidia_smi("7800\n512\n") == [7800 * 1024**2, 512 * 1024**2]


# -- blending, and tiling inside a workflow ----------------------------------


def test_blended_tiles_add_up_to_the_whole():
    # The weights add to 1 at every pixel, so an op that changes nothing
    # comes back unchanged, small tiles and overlapping ramps included.
    data = np.random.default_rng(0).random((12, 30, 40)).astype(np.float32)
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(4), "40K", overlap=3)
    assert plan.calls > 1
    result = run_tiles(lambda tile: tile, data, plan, merge="blend")
    np.testing.assert_allclose(result, data, rtol=0, atol=1e-6)


def test_blending_needs_a_float_result():
    data = image()
    plan = plan_tiles(data.shape, data.dtype, PeakMemory(8), "40K", overlap=RADIUS)
    with pytest.raises(ValueError, match="float"):
        run_tiles(lambda tile: tile, data, plan, merge="blend")


def test_a_tiler_guesses_the_padding_of_a_psf_not_made_yet():
    from skop.ops.deconvolve.richardson_lucy_cupy import richardson_lucy_cupy

    volume = {"image": np.zeros((64, 512, 512), np.float32), "noncirc": True}
    plan = skop.Tiler(memory="1G", overlap=10).plan(richardson_lucy_cupy, volume)
    assert plan.overlap == 10 and plan.calls > 1


SEEN: list[tuple[int, ...]] = []


@op(tile="image", peak_memory=PeakMemory(8))
def recorded(image: np.ndarray) -> np.ndarray:
    SEEN.append(image.shape)
    return image


@op
def calls_recorded(image: np.ndarray) -> np.ndarray:
    return skop.run(recorded, image=image)


def test_a_workflow_passes_its_tiler_to_the_ops_it_runs():
    SEEN.clear()
    data = image()
    with skop.Runner() as runner:
        result = runner.run(calls_recorded, image=data, tiler=skop.Tiler(memory="40K"))
    assert len(SEEN) > 1 and all(shape != data.shape for shape in SEEN)
    assert np.array_equal(result, data)


def test_a_tiled_run_says_which_tile_it_is_on():
    events = []
    data = image()
    with skop.Runner() as runner:
        runner.run(
            recorded,
            image=data,
            tiler=skop.Tiler(memory="40K"),
            on_progress=events.append,
        )
    tiles = [event.tile for event in events if getattr(event, "tile", None)]
    count = tiles[0][1]
    assert count > 1
    assert tiles == [(m, count) for m in range(1, count + 1)]
