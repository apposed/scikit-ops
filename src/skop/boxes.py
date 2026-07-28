"""Bounding boxes, in one format, with converters at the edges.

Every detector states boxes differently -- YOLO gives ``[x1, y1, x2, y2]``,
micro_sam wants ``[y1, x1, y2, x2]``, napari draws corner pairs, skimage's
``regionprops`` reports ``(min_row, min_col, max_row, max_col)``. Converting
ad hoc, at each call site, is how a transposed box ends up somewhere nobody
looks.

So skop has one format and converts only at its edges:

    (N, 4) float32, each row [min_y, min_x, max_y, max_x]

Row-major, matching ``PointsData``'s rule that coordinates are in the axis
order of the image they came from -- and matching numpy indexing, napari's
axis order, and micro_sam's box prompts. A detector op converts on the way
out, a front end on the way in, and nothing in between converts at all.

This module is numpy and the standard library only, like ``skop.assets``, so
that ops can import it inside their own environments as freely as the host
does.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EMPTY",
    "as_boxes",
    "from_labels",
    "from_napari",
    "from_xyxy",
    "to_napari",
    "to_xyxy",
]

#: What a detector that found nothing returns. Shaped so that indexing a
#: column still works: ``EMPTY[:, 0]`` is empty, not an IndexError.
EMPTY = np.zeros((0, 4), dtype=np.float32)


def as_boxes(boxes: object, name: str = "boxes") -> np.ndarray:
    """Coerce to a well-formed ``(N, 4)`` float32 array.

    Public because every op *taking* boxes needs it: an op is called from a
    front end, a workflow and a test, and only one of those is guaranteed to
    have gone through a converter here first.
    """
    array = np.asarray(boxes, dtype=np.float32)
    if array.size == 0:
        return EMPTY.copy()
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {array.shape}")
    return array


def from_xyxy(boxes: object) -> np.ndarray:
    """Convert detector-order ``[x1, y1, x2, y2]`` boxes to canonical order.

    This is the swap that every YOLO-family detector needs on the way out.
    """
    array = as_boxes(boxes)
    return array[:, [1, 0, 3, 2]].copy()


def to_xyxy(boxes: object) -> np.ndarray:
    """Convert canonical boxes to detector-order ``[x1, y1, x2, y2]``.

    The same permutation as :func:`from_xyxy` -- it is its own inverse -- but
    named for the direction, so call sites read correctly.
    """
    array = as_boxes(boxes)
    return array[:, [1, 0, 3, 2]].copy()


def to_napari(boxes: object) -> np.ndarray:
    """Convert canonical boxes to ``(N, 2, 2)`` corner pairs for a Shapes layer.

    napari reads a two-vertex rectangle as a pair of opposite corners, in the
    same row-major order used here, so this is a reshape and nothing else::

        viewer.add_shapes(to_napari(result.boxes), shape_type="rectangle")
    """
    array = as_boxes(boxes)
    return array.reshape(-1, 2, 2).copy()


def from_napari(boxes: object) -> np.ndarray:
    """Convert napari shapes to canonical boxes.

    Accepts what a Shapes layer actually hands over, which is a *list* of
    ``(V, 2)`` vertex arrays, one per shape -- and which goes ragged the
    moment a polygon sits beside a rectangle, so it cannot be treated as one
    array. A uniform ``(N, V, 2)`` array works too: ``(N, 2, 2)`` opposite
    corners, or ``(N, 4, 2)`` from a rectangle the user drew.

    Every shape collapses to its axis-aligned extent, which is all a box can
    say. That is exact for an upright rectangle and a bounding box for
    anything else -- a rotated rectangle, an ellipse, a hand-drawn polygon --
    which is the useful answer in each case.
    """
    # A list of arrays is the common case and cannot go through np.asarray:
    # ragged input raises, and uniform input silently works, so the two would
    # behave differently for the same layer on different days.
    if isinstance(boxes, (list, tuple)):
        shapes = [np.asarray(shape, dtype=np.float32) for shape in boxes]
        if not shapes:
            return EMPTY.copy()
        for shape in shapes:
            if shape.ndim != 2 or shape.shape[1] != 2:
                raise ValueError(f"each napari shape must be (V, 2), got {shape.shape}")
        lower = np.array([shape.min(axis=0) for shape in shapes], dtype=np.float32)
        upper = np.array([shape.max(axis=0) for shape in shapes], dtype=np.float32)
        return np.concatenate([lower, upper], axis=1)

    array = np.asarray(boxes, dtype=np.float32)
    if array.size == 0:
        return EMPTY.copy()
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"napari boxes must be (N, V, 2), got {array.shape}")
    lower = array.min(axis=1)
    upper = array.max(axis=1)
    return np.concatenate([lower, upper], axis=1).astype(np.float32)


def from_labels(labels: np.ndarray) -> np.ndarray:
    """Bounding boxes of every object in a label image.

    Background (0) is not an object. Boxes are inclusive of the last pixel,
    so a single-pixel object at ``(3, 4)`` gives ``[3, 4, 4, 5]`` -- the same
    half-open convention as ``regionprops`` and as slicing.

    Pure numpy so it needs no environment: this is how a Cellpose or StarDist
    result becomes prompts for a mask detector.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2-D label image, got {labels.ndim}-D")

    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return EMPTY.copy()

    out = np.empty((ids.size, 4), dtype=np.float32)
    for i, label in enumerate(ids):
        rows, cols = np.nonzero(labels == label)
        out[i] = (rows.min(), cols.min(), rows.max() + 1, cols.max() + 1)
    return out
