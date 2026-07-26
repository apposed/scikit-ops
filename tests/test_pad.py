"""The padding helpers the deconvolution ops share.

Pure numpy, so these run in the host environment with no op environment built
and no ``@pytest.mark.env`` marker.
"""

from __future__ import annotations

import numpy as np
import pytest

from skop.ops.deconvolve._pad import (
    get_next_smooth,
    next_smooth,
    pad,
    pad_to_largest,
    unpad,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (7, 7),  # already smooth
        (11, 12),  # 12 = 2^2 * 3
        (13, 14),  # 14 = 2 * 7
        (127, 128),
        (1000, 1000),  # 2^3 * 5^3
    ],
)
def test_next_smooth(value, expected):
    assert next_smooth(value) == expected


def test_next_smooth_result_has_no_large_prime_factors():
    for value in range(2, 200):
        remainder = next_smooth(value)
        assert remainder >= value
        for prime in (2, 3, 5, 7):
            while remainder % prime == 0:
                remainder //= prime
        assert remainder == 1, f"next_smooth({value}) is not 7-smooth"


def test_get_next_smooth_maps_every_axis():
    assert get_next_smooth((11, 13, 127)) == (12, 14, 128)


@pytest.mark.parametrize("shape", [(8, 8), (7, 9), (4, 6, 8), (5, 5, 5)])
@pytest.mark.parametrize("grow", [2, 3])
def test_pad_then_unpad_round_trips(shape, grow):
    rng = np.random.default_rng(0)
    image = rng.random(shape)
    target = tuple(s + grow for s in shape)

    padded, _ = pad(image, target, "constant")
    assert padded.shape == target
    # An odd size delta puts the extra row on one particular side; unpad has
    # to make the same choice, which is the whole point of testing the pair.
    assert np.array_equal(unpad(padded, shape), image)


def test_pad_centers_the_image():
    image = np.ones((2, 2))
    padded, padding = pad(image, (4, 4), "constant")
    assert padding == ((1, 1), (1, 1))
    assert np.array_equal(padded[1:3, 1:3], image)
    assert padded.sum() == 4


def test_pad_to_largest_takes_the_max_of_each_axis():
    image = np.ones((16, 4))
    psf = np.ones((8, 12))

    image, psf = pad_to_largest(image, psf, "constant")
    assert image.shape == psf.shape == (16, 12)
    assert image.sum() == 16 * 4
    assert psf.sum() == 8 * 12
