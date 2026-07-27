"""Helpers shared between ops, tested on the host.

``to_rgb`` in particular: it is the first thing every detector does to its
input, so when it is wrong the symptom is a model that silently finds nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from skop.ops._util import to_gray, to_rgb


def test_to_rgb_promotes_a_2d_image():
    result = to_rgb(np.linspace(0, 4000, 64 * 64).reshape(64, 64).astype(np.uint16))
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result[..., 0], result[..., 2])


def test_to_rgb_stretches_low_contrast_input():
    # 16-bit data using a fraction of its range is the normal microscopy case,
    # and casting it straight to uint8 would leave the model a black image.
    image = np.zeros((32, 32), dtype=np.uint16)
    image[8:24, 8:24] = 300
    result = to_rgb(image)
    assert result.max() == 255
    assert result.min() == 0


def test_to_rgb_survives_a_blank_image():
    result = to_rgb(np.zeros((16, 16), dtype=np.uint16))
    assert result.shape == (16, 16, 3)
    assert result.max() == 0


def test_to_rgb_leaves_uint8_rgb_alone():
    image = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    assert to_rgb(image) is image


def test_to_rgb_drops_an_alpha_channel():
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    assert to_rgb(image).shape == (8, 8, 3)


def test_to_rgb_rejects_a_volume():
    with pytest.raises(ValueError):
        to_rgb(np.zeros((4, 8, 8), dtype=np.uint16))


def test_to_gray_and_to_rgb_round_trip_shape():
    image = np.zeros((8, 8), dtype=np.uint8)
    assert to_gray(to_rgb(image)).shape == (8, 8)
