"""Tiling hints on @op, and the planner that turns them into tiles."""

from __future__ import annotations

import json
import math
from typing import Annotated

import numpy as np
import pytest

from opspec.op import Axes, OpSpec, Overlap, PeakMemory, op
from opspec.tiling import parse_bytes, plan_tiles

# -- the hints --------------------------------------------------------------


@op(
    env="skimage",
    tile="image",
    overlap=Overlap(param="sigma", scale=4),
    peak_memory=PeakMemory(scale=2, dtype=np.float32),
    merge="crop",
)
def blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return image


def test_hints_are_read_off_the_decorator():
    spec = OpSpec.from_op(blur)
    assert spec.tile == ("image",)
    assert spec.overlap == Overlap(param="sigma", scale=4)
    # A numpy dtype is kept by name: opspec cannot import numpy.
    assert spec.peak_memory == PeakMemory(scale=2, dtype="float32")
    assert spec.merge == "crop"


def test_an_op_without_hints_has_none():
    @op
    def plain(image: np.ndarray) -> np.ndarray:
        return image

    spec = OpSpec.from_op(plain)
    assert (spec.tile, spec.overlap, spec.peak_memory, spec.merge) == (
        (),
        None,
        None,
        None,
    )
    # And its wire form is what it was before tiling hints existed.
    assert not {"tile", "overlap", "peak_memory", "merge"} & set(spec.to_dict())


def test_overlap_resolves_against_the_call():
    overlap = OpSpec.from_op(blur).overlap
    assert overlap.resolve({"sigma": 1.5}) == 6
    assert overlap.resolve({"sigma": (1.0, 2.5)}) == 10  # the widest axis
    assert Overlap(10).resolve({}) == 10


def test_a_number_of_pixels_is_an_overlap():
    @op(tile="image", overlap=10)
    def fixed(image: np.ndarray) -> np.ndarray:
        return image

    assert OpSpec.from_op(fixed).overlap == Overlap(10)


def test_hints_must_name_real_parameters():
    @op(tile="picture", overlap=Overlap(param="radius"))
    def wrong(image: np.ndarray) -> np.ndarray:
        return image

    with pytest.raises(TypeError, match="picture, radius"):
        OpSpec.from_op(wrong)


def test_peak_memory_counts_bytes():
    assert PeakMemory(scale=2, dtype="float32").bytes_for(1000) == 8000
    # No dtype: the input's own.
    assert PeakMemory(scale=3).bytes_for(1000, "uint16") == 6000
    assert PeakMemory(scale=1, fixed=500).bytes_for(1000, "uint8") == 1500


def test_hints_survive_the_wire():
    spec = OpSpec.from_op(blur)
    back = OpSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert (back.tile, back.overlap, back.peak_memory, back.merge) == (
        spec.tile,
        spec.overlap,
        spec.peak_memory,
        spec.merge,
    )


# -- the planner ------------------------------------------------------------

FLOAT_PAIR = PeakMemory(scale=2, dtype="float32")  # 8 bytes a pixel


def coverage(plan):
    """How many tiles write each pixel."""
    counts = np.zeros(plan.shape, dtype=int)
    for tile in plan.tiles:
        counts[tile.write] += 1
    return counts


def test_parse_bytes():
    assert parse_bytes("1G") == 1024**3
    assert parse_bytes("512M") == 512 * 1024**2
    assert parse_bytes("1.5GiB") == int(1.5 * 1024**3)
    assert parse_bytes(4096) == 4096
    with pytest.raises(ValueError):
        parse_bytes("lots")


def test_an_input_that_fits_is_one_tile():
    plan = plan_tiles((10, 20, 30), "uint8", FLOAT_PAIR, "1M", overlap=3)
    assert plan.calls == 1
    assert plan.tiles[0].read == (slice(0, 10), slice(0, 20), slice(0, 30))
    assert plan.summary == "1 tile: the whole input fits"


def test_tiles_cover_the_input_exactly_once_and_fit():
    plan = plan_tiles((142, 394, 792), "uint8", FLOAT_PAIR, "64M", overlap=6)
    assert plan.calls > 1
    assert (coverage(plan) == 1).all()
    for tile in plan.tiles:
        elements = math.prod(s.stop - s.start for s in tile.read)
        assert FLOAT_PAIR.bytes_for(elements) <= plan.budget
    assert plan.peak <= plan.budget


def test_overlap_stops_at_the_real_edge():
    # Two tiles along x: each reaches 3 pixels into the other, never outside.
    plan = plan_tiles((4, 100), "uint8", PeakMemory(scale=1), 300, overlap=3)
    left, right = plan.tiles
    assert left.read[1] == slice(0, 53) and left.keep[1] == slice(0, 50)
    assert right.read[1] == slice(47, 100) and right.keep[1] == slice(3, 53)
    assert left.write[1] == slice(0, 50) and right.write[1] == slice(50, 100)


def test_an_axis_left_out_is_never_cut():
    plan = plan_tiles((8, 64, 3), "uint8", PeakMemory(scale=1), 600, axes=(0, 1))
    assert all(tile.read[2] == slice(0, 3) for tile in plan.tiles)
    assert (coverage(plan) == 1).all()


def test_an_impossible_budget_says_so():
    with pytest.raises(ValueError, match="Cannot fit"):
        plan_tiles((100, 100), "uint8", FLOAT_PAIR, 1000, overlap=20)


def test_a_callers_own_copies_count_against_the_budget():
    alone = plan_tiles((64, 64), "uint8", PeakMemory(scale=1), 2048)
    with_copies = plan_tiles((64, 64), "uint8", PeakMemory(scale=1), 2048, extra=3)
    assert with_copies.calls > alone.calls
    assert with_copies.peak <= with_copies.budget


# -- per-axis overlap, padding and split ------------------------------------

HALF_PSF = Overlap(param="psf", of="shape", scale=0.5)


def test_an_overlap_can_be_half_an_arrays_shape():
    # Rounded down: half a kernel of 31 is 15, exactly how far it reaches.
    assert HALF_PSF.resolve({"psf": np.zeros((31, 64, 65))}) == (15, 32, 32)
    assert Overlap.from_dict(json.loads(json.dumps(HALF_PSF.to_dict()))) == HALF_PSF


def test_each_axis_reaches_its_own_overlap():
    plan = plan_tiles((8, 100, 100), "uint8", PeakMemory(1), 20_000, overlap=(0, 2, 5))
    assert plan.calls > 1
    interior = [t for t in plan.tiles if t.read[1].start > 0 and t.read[2].start > 0]
    assert all(t.write[1].start - t.read[1].start == 2 for t in interior)
    assert all(t.write[2].start - t.read[2].start == 5 for t in interior)
    assert plan.summary.endswith("overlap 0 x 2 x 5")


def test_padding_counts_against_the_budget():
    peak = PeakMemory(1)
    unpadded = plan_tiles((64, 64), "uint8", peak, 2048)
    padded = plan_tiles((64, 64), "uint8", peak, 2048, pad=8)
    assert padded.calls > unpadded.calls
    assert padded.peak <= padded.budget


def test_padding_survives_the_wire():
    peak = PeakMemory(10, "float64", pad=HALF_PSF)
    assert PeakMemory.from_dict(json.loads(json.dumps(peak.to_dict()))) == peak


def test_split_names_the_axes_a_tile_may_be_cut_along():
    @op(tile="image", peak_memory=PeakMemory(1), split=("y", "x"))
    def deconvolve(
        image: Annotated[np.ndarray, Axes("z?", "y", "x")], psf: np.ndarray
    ) -> np.ndarray:
        return image

    spec = OpSpec.from_op(deconvolve)
    assert spec.split == ("y", "x")
    assert OpSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).split == ("y", "x")


def test_split_must_name_axes_the_input_has():
    @op(tile="image", peak_memory=PeakMemory(1), split=("t",))
    def wrong(image: Annotated[np.ndarray, Axes("z", "y", "x")]) -> np.ndarray:
        return image

    with pytest.raises(TypeError, match="splits along t"):
        OpSpec.from_op(wrong)


def test_a_gpu_peak_says_so_on_the_wire():
    peak = PeakMemory(16, "float32", device="gpu")
    assert PeakMemory.from_dict(json.loads(json.dumps(peak.to_dict()))) == peak
    # A CPU peak's wire form is as it was before devices existed.
    assert "device" not in PeakMemory(2).to_dict()


def test_a_tile_size_set_by_the_caller_is_kept():
    plan = plan_tiles(
        (10, 100, 100), "uint8", PeakMemory(1), 10, tile_shape=(10, 40, 40)
    )
    assert plan.tile_shape == (10, 40, 40)
    assert (coverage(plan) == 1).all()


def test_an_overlap_can_depend_on_a_flag():
    rule = Overlap(param="psf", of="shape", scale=0.5, only_if="noncirc", otherwise=10)

    class Psf:
        shape = (31, 15, 15)

    assert rule.resolve({"psf": Psf(), "noncirc": True}) == (15, 7, 7)
    assert rule.resolve({"psf": Psf(), "noncirc": False}) == 10
    # Off, the PSF isn't needed at all.
    assert rule.resolve({"noncirc": False}) == 10
    assert Overlap.from_dict(rule.to_dict()) == rule
