"""Tiling hints on @op, and the planner that turns them into tiles."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from opspec.op import OpSpec, Overlap, PeakMemory, op
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
