"""Mask collections, and the projections a front end shows them through.

Host-only: no environment, no model, no download. The thing worth catching
cheaply here is an overlap resolved the wrong way -- which, like a transposed
box, produces a picture that looks entirely plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from skop import masks


def _two_overlapping():
    """A big mask, and a small one drawn entirely inside it."""
    stack = np.zeros((2, 10, 10), dtype=np.uint8)
    stack[0, 0:8, 0:8] = 1  # area 64
    stack[1, 2:5, 2:5] = 1  # area 9, wholly within the first
    return stack


def test_to_labels_3d_gives_each_mask_its_own_plane_and_label():
    labels = masks.to_labels_3d(_two_overlapping())
    assert labels.shape == (2, 10, 10)
    assert labels[0].max() == 1
    assert labels[1].max() == 2
    # The overlapping pixel is present on both planes -- that is the point.
    assert labels[0][3, 3] == 1
    assert labels[1][3, 3] == 2


def test_to_labels_2d_min_keeps_the_smaller_object_when_sorted_largest_first():
    # The pairing the default exists for: largest-first order plus "min"
    # means the small mask survives being drawn inside the big one.
    labels = masks.to_labels_2d(_two_overlapping(), strategy="min")
    assert labels[3, 3] == 1  # contested: the lower-numbered, larger mask
    assert labels[0, 0] == 1
    assert labels[9, 9] == 0


def test_to_labels_2d_max_lets_the_later_mask_win():
    labels = masks.to_labels_2d(_two_overlapping(), strategy="max")
    assert labels[3, 3] == 2  # contested: the higher-numbered mask
    assert labels[0, 0] == 1


def test_to_labels_2d_matches_a_projection_of_the_3d_stack():
    # to_labels_2d avoids materializing the 3-D array; it must still agree
    # with what projecting it would have given.
    stack = _two_overlapping()
    labels_3d = masks.to_labels_3d(stack)

    expected_min = np.ma.masked_equal(labels_3d, 0).min(axis=0).filled(0)
    assert np.array_equal(masks.to_labels_2d(stack, "min"), expected_min)
    assert np.array_equal(masks.to_labels_2d(stack, "max"), labels_3d.max(axis=0))


def test_to_labels_2d_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        masks.to_labels_2d(_two_overlapping(), strategy="mean")


def test_order_by_area_is_a_permutation_that_reorders_boxes_too():
    stack = _two_overlapping()  # plane 0 is the larger
    small_first = stack[::-1]

    order = masks.order_by_area(small_first)
    assert order.tolist() == [1, 0]
    # The reason it returns indices: anything indexed alike travels with it.
    boxes = np.array([[2, 2, 5, 5], [0, 0, 8, 8]], dtype=np.float32)
    assert boxes[order].tolist() == [[0, 0, 8, 8], [2, 2, 5, 5]]


def test_order_by_area_keeps_ties_in_detector_order():
    stack = np.zeros((3, 4, 4), dtype=np.uint8)
    stack[:, 0, 0] = 1  # all three have area 1
    assert masks.order_by_area(stack).tolist() == [0, 1, 2]


def test_from_labels_round_trips_through_to_labels_2d():
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[2:5, 3:7] = 1
    labels[7, 8] = 2

    stack = masks.from_labels(labels)
    assert stack.shape == (2, 10, 10)
    assert stack.dtype == np.uint8
    # Non-overlapping labels survive the round trip exactly.
    assert np.array_equal(masks.to_labels_2d(stack), labels)


def test_from_labels_ignores_background_and_label_gaps():
    labels = np.zeros((5, 5), dtype=np.uint16)
    labels[0, 0] = 4
    labels[1, 1] = 9
    stack = masks.from_labels(labels)
    assert stack.shape == (2, 5, 5)
    # Planes come out in ascending label order, renumbered from 1.
    assert stack[0][0, 0] == 1
    assert stack[1][1, 1] == 1


def test_bool_masks_are_accepted_and_cast():
    # SAM returns bool, and bool cannot cross the Appose boundary -- the cast
    # has to happen somewhere, and here is somewhere.
    stack = np.zeros((1, 4, 4), dtype=bool)
    stack[0, 1, 1] = True
    assert masks.to_labels_2d(stack).tolist()[1][1] == 1


def test_float_masks_above_one_are_treated_as_set():
    stack = np.zeros((1, 4, 4), dtype=np.float32)
    stack[0, 1, 1] = 255.0
    assert masks.to_labels_2d(stack)[1, 1] == 1


def test_empty_collection_projects_to_a_blank_image_of_the_right_size():
    nothing = masks.empty((6, 7))
    assert nothing.shape == (0, 6, 7)
    assert masks.to_labels_2d(nothing).shape == (6, 7)
    assert masks.to_labels_2d(nothing).max() == 0
    assert masks.to_labels_3d(nothing).shape == (0, 6, 7)
    assert masks.order_by_area(nothing).tolist() == []


def test_from_labels_of_an_empty_label_image_is_an_empty_collection():
    assert masks.from_labels(np.zeros((6, 7), dtype=np.uint16)).shape == (0, 6, 7)


def test_wrong_rank_is_rejected():
    with pytest.raises(ValueError, match=r"\(N, Y, X\)"):
        masks.to_labels_2d(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="2-D"):
        masks.from_labels(np.zeros((2, 10, 10), dtype=np.uint16))


def test_label_dtype_grows_past_uint16():
    # Not a stack anyone will build, but the width is picked from N rather
    # than hardcoded, and that is the assertion.
    assert masks.to_labels_3d(masks.empty((2, 2))).dtype == np.uint16
    assert masks._label_dtype(70_000) == np.uint32
