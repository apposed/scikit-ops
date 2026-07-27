"""The one box format, and the converters at its edges.

Host-only: no environment, no model, no download. These run everywhere and
they are the cheapest place to catch the bug this module exists to prevent --
a transposed box, which looks plausible right up until it is drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

from skop import boxes


def test_from_xyxy_transposes():
    # The assertion worth writing out by hand: a detector's
    # [x1=10, y1=20, x2=30, y2=40] is our [min_y=20, min_x=10, ...].
    assert boxes.from_xyxy([[10, 20, 30, 40]]).tolist() == [[20, 10, 40, 30]]


def test_xyxy_round_trip():
    canonical = np.array([[20, 10, 40, 30], [1, 2, 3, 4]], dtype=np.float32)
    assert np.array_equal(boxes.from_xyxy(boxes.to_xyxy(canonical)), canonical)


def test_napari_round_trip():
    canonical = np.array([[20, 10, 40, 30], [0, 0, 5, 5]], dtype=np.float32)
    corners = boxes.to_napari(canonical)
    assert corners.shape == (2, 2, 2)
    # The corner pair is (upper-left, lower-right) in row-major order.
    assert corners[0].tolist() == [[20, 10], [40, 30]]
    assert np.array_equal(boxes.from_napari(corners), canonical)


def test_from_napari_accepts_four_corners():
    # What napari hands back for a rectangle the user drew.
    rect = np.array([[[20, 10], [20, 30], [40, 30], [40, 10]]], dtype=np.float32)
    assert boxes.from_napari(rect).tolist() == [[20, 10, 40, 30]]


def test_from_napari_collapses_a_rotated_rectangle_to_its_extent():
    rotated = np.array([[[0, 5], [5, 10], [10, 5], [5, 0]]], dtype=np.float32)
    assert boxes.from_napari(rotated).tolist() == [[0, 0, 10, 10]]


def test_from_labels():
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[2:5, 3:7] = 1
    labels[7, 8] = 2

    result = boxes.from_labels(labels)
    assert result.tolist() == [[2, 3, 5, 7], [7, 8, 8, 9]]


def test_from_labels_ignores_background():
    assert boxes.from_labels(np.zeros((4, 4), dtype=np.uint16)).shape == (0, 4)


@pytest.mark.parametrize(
    "convert",
    [boxes.from_xyxy, boxes.to_xyxy, boxes.to_napari],
    ids=lambda f: f.__name__,
)
def test_empty_input_keeps_its_columns(convert):
    # A detector finding nothing is ordinary, and everything downstream
    # indexes columns -- so the empty case must stay (N, 4)-shaped.
    result = convert([])
    assert result.shape in {(0, 4), (0, 2, 2)}
    assert result.dtype == np.float32


def test_empty_napari_input():
    assert boxes.from_napari([]).shape == (0, 4)


def test_conversions_do_not_alias_their_input():
    canonical = np.array([[20, 10, 40, 30]], dtype=np.float32)
    converted = boxes.to_xyxy(canonical)
    converted[0, 0] = 999
    assert canonical[0, 0] == 20


@pytest.mark.parametrize("bad", [np.zeros((3, 5)), np.zeros(4), np.zeros((2, 2, 2))])
def test_malformed_input_is_rejected(bad):
    with pytest.raises(ValueError):
        boxes.from_xyxy(bad)
