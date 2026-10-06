"""Tiling hints on @op, and the planner that turns them into tiles."""

from __future__ import annotations

import json

import numpy as np
import pytest

from opspec.op import OpSpec, Overlap, PeakMemory, op

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
