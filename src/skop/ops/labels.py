"""Ops that operate on a label image, rather than producing one.

Where a segmentation op turns intensities into objects, these take the objects
as given and fix up their numbering. They share the ``skimage`` environment
because none of them needs more than numpy and scipy.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import Axes, op, progress
from skop.types import LabelsData


@op(env="skimage")
def connect(
    labels: Annotated[LabelsData, Axes("z", "y", "x")],
    threshold: float = 0.1,
    criterion: Annotated[str, {"choices": ["iou", "iop"]}] = "iou",
) -> LabelsData:
    """Give one object one label along an axis, by linking overlaps.

    Segmenting a volume slice by slice numbers each slice from scratch, so one
    cell becomes a stack of unrelated labels -- rainbow pancakes. This walks
    the stack and relabels each slice so that an object inherits the label of
    the object it overlaps in the slice before, turning the pancakes back into
    solid objects.

    The link is one-to-one per adjacent pair: each object may claim at most one
    predecessor, chosen by maximizing total overlap across the whole pair
    (:func:`scipy.optimize.linear_sum_assignment`). That is what keeps two
    labels that merely *touch* within a slice from being fused, which is why
    this cannot be done by running connected components over the binary union.
    An object matching nothing starts a new label, so a label that disappears
    for a slice does not reclaim its old number when it comes back.

    The same algorithm is tracking, with Z where T usually goes; it is
    StarDist's ``group_matching_labels`` and Cellpose's ``stitch3D``, and
    scikit-image has no equivalent.

    Args:
        labels: Label image to connect, one object numbering per slice along
            the first declared axis. The other axes are the slice, and are
            never mixed into the matching.
        threshold: Minimum overlap for two objects to be considered the same,
            in [0, 1]. StarDist links on any overlap at all, Cellpose defaults
            to 0.25; low is safe here because the one-to-one constraint already
            rules out the pathological links.
        criterion: How overlap is scored. ``"iou"`` is intersection over union,
            the usual choice. ``"iop"`` is intersection over the *smaller* of
            the two areas, which is more forgiving when an object grows or
            shrinks sharply from slice to slice -- as it does at the top and
            bottom of a cell, or in an anisotropic stack.

    Returns:
        A label image of the same shape, numbered consecutively from 1.
    """
    from scipy.optimize import linear_sum_assignment

    if criterion not in ("iou", "iop"):
        raise ValueError(f"criterion must be 'iou' or 'iop', got {criterion!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    out = np.zeros(labels.shape, dtype=np.int32)
    if out.size == 0:
        return out
    # The first slice sets the numbering everything after it inherits, so it
    # is compacted rather than copied: that alone makes the result sequential.
    ids, inverse = np.unique(labels[0].ravel(), return_inverse=True)
    lut = np.arange(len(ids), dtype=np.int32)
    if ids[0] != 0:
        lut += 1  # No background in this slice, so nothing may map to 0.
    out[0] = lut[inverse].reshape(labels[0].shape)
    next_id = int(out[0].max()) + 1

    for z in range(1, len(labels)):
        progress(f"Connecting slice {z + 1} of {len(labels)}", z, len(labels))
        prev, cur = out[z - 1], labels[z]
        prev_ids, cur_ids, inter, prev_area, cur_area = _overlaps(prev, cur)

        # Every object gets a fresh label until a match says otherwise.
        lut = np.zeros(len(cur_ids) + 1, dtype=np.int32)
        matches: dict[int, int] = {}
        if len(prev_ids) and len(cur_ids):
            if criterion == "iou":
                score = inter / (prev_area[:, None] + cur_area[None, :] - inter)
            else:
                score = inter / np.minimum(prev_area[:, None], cur_area[None, :])
            rows, cols = linear_sum_assignment(score, maximize=True)
            keep = (score[rows, cols] >= threshold) & (score[rows, cols] > 0)
            matches = {int(j): int(i) for j, i in zip(cols[keep], prev_ids[rows[keep]])}
        for j in range(len(cur_ids)):
            if j in matches:
                lut[j + 1] = matches[j]
            else:
                lut[j + 1] = next_id
                next_id += 1

        # Index 0 of the lut is background; object j sits at j + 1.
        flat = cur.ravel()
        compact = np.searchsorted(cur_ids, flat) + 1
        compact[flat == 0] = 0
        out[z] = lut[compact].reshape(cur.shape)

    return out


def _overlaps(
    prev: np.ndarray, cur: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pixel counts shared by each pair of objects across two slices.

    Returns the object labels of each slice (background excluded, sorted), the
    dense intersection matrix between them, and each object's area. Dense
    because the matrix is objects-by-objects, not pixels-by-pixels: a slice
    with a thousand cells still only costs a megabyte.
    """
    prev_ids, prev_idx = np.unique(prev.ravel(), return_inverse=True)
    cur_ids, cur_idx = np.unique(cur.ravel(), return_inverse=True)
    prev_area = np.bincount(prev_idx, minlength=len(prev_ids)).astype(np.float64)
    cur_area = np.bincount(cur_idx, minlength=len(cur_ids)).astype(np.float64)

    both = (prev.ravel() != 0) & (cur.ravel() != 0)
    pairs = prev_idx[both] * len(cur_ids) + cur_idx[both]
    inter = np.bincount(pairs, minlength=len(prev_ids) * len(cur_ids)).reshape(
        len(prev_ids), len(cur_ids)
    )

    # Drop the background row and column, which are only there because 0 sorts
    # first among the labels np.unique found.
    prev_from = 1 if len(prev_ids) and prev_ids[0] == 0 else 0
    cur_from = 1 if len(cur_ids) and cur_ids[0] == 0 else 0
    return (
        prev_ids[prev_from:],
        cur_ids[cur_from:],
        inter[prev_from:, cur_from:].astype(np.float64),
        prev_area[prev_from:],
        cur_area[cur_from:],
    )
