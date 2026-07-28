"""Collections of masks that are allowed to overlap, and how to look at them.

A mask detector returns one mask per object, and SAM-family masks overlap --
a cell and the debris on top of it, two touching nuclei, a coin behind a coin.
A label image cannot say that: one pixel, one integer. So the collection is a
stack instead,

    (N, Y, X) uint8, 0 or 1, one object per plane

which keeps every overlap, and is what ``MasksData`` annotates.

``uint8`` rather than ``bool`` is forced, not chosen. Appose derives an
element size from the dtype name, and ``bool`` has no digit in it, so a
boolean array cannot cross the process boundary at all (``skop._codec``).
The functions here accept ``bool`` and cast, since that is what SAM hands
back and the cast has to happen somewhere.

Nothing displays a stack directly, so this module holds the projections onto
things that do -- and they are utilities rather than ops, like ``skop.boxes``,
because making them ops would buy a shared-memory round trip and a worker
process to transpose an array. numpy and the standard library only, so a
worker, the host and a front end can all import it freely.

Which projection to show is the front end's decision, not the op's: the op
returns the collection every time, and re-projecting is a numpy call on a
result the host already has. See docs/design/0008-mask-detector-ops.md.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "empty",
    "from_labels",
    "order_by_area",
    "to_labels_2d",
    "to_labels_3d",
]


def empty(shape: tuple[int, int]) -> np.ndarray:
    """What a mask detector that found nothing returns.

    Takes the image shape, because unlike ``boxes.EMPTY`` an empty stack still
    has to say how big the planes it does not have would have been -- a front
    end asked to project it should get a blank image of the right size, not an
    error.
    """
    return np.zeros((0, *shape), dtype=np.uint8)


def _as_masks(masks: object, name: str = "masks") -> np.ndarray:
    """Coerce to a well-formed ``(N, Y, X)`` uint8 stack of 0s and 1s."""
    array = np.asarray(masks)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N, Y, X), got {array.shape}")
    if array.dtype == np.uint8:
        return array
    # != 0 rather than astype: SAM sometimes returns float 0.0/1.0 masks, and
    # a detector's threshold may leave values above 1.
    return (array != 0).astype(np.uint8)


def _label_dtype(count: int) -> np.dtype:
    """The smallest unsigned type that can hold labels 1..count.

    ``StackedLabels`` hardcoded uint16, which silently wraps somewhere past
    65,535 objects. Unlikely, but the check is one line.
    """
    return np.dtype(np.uint16 if count < np.iinfo(np.uint16).max else np.uint32)


def to_labels_3d(masks: object) -> np.ndarray:
    """Give every mask its own plane and its own label.

    Returns ``(N, Y, X)``, with plane *i* holding the value ``i + 1`` wherever
    mask *i* is set. Nothing is lost -- overlapping objects sit on separate
    planes -- and napari will show it as a Labels layer you can rotate.

    Be aware that the first axis is an object index and not a spatial one.
    It has no pixel size and nothing measured along it means anything.
    """
    array = _as_masks(masks)
    labels = array.astype(_label_dtype(len(array)))
    # Broadcasting the label down each plane, rather than looping.
    ids = np.arange(1, len(array) + 1, dtype=labels.dtype)
    return labels * ids[:, None, None]


def to_labels_2d(masks: object, strategy: str = "min") -> np.ndarray:
    """Flatten the stack into one label image, resolving overlap by strategy.

    This is the projection that loses information, and ``strategy`` decides
    what it loses where two masks claim a pixel:

    - ``"min"`` gives the pixel to the lower-numbered mask.
    - ``"max"`` gives it to the higher-numbered one.

    Neither means anything on its own -- it is the *order* of the stack that
    gives them meaning. Sorted largest-first with :func:`order_by_area`, the
    order ``StackedLabels`` imposed, the largest object holds label 1, and so:

    - ``"min"`` gives a contested pixel to the **largest** object. An object
      wholly inside another disappears from the projection entirely.
    - ``"max"`` gives it to the **smallest**, so nested objects are drawn on
      top of the ones containing them and every mask still appears somewhere.

    ``"min"`` is the default because it is what ``StackedLabels`` did, not
    because it is the better answer. It is the right one when the big mask is
    the object and the small ones are fragments of it; ``"max"`` is right when
    the nested masks are the objects. Neither is safe to assume, which is why
    this is a parameter and why a front end should expose it.

    Computed a plane at a time rather than by projecting :func:`to_labels_3d`,
    which would allocate an ``(N, Y, X)`` integer array to produce a single
    ``(Y, X)`` one -- 800 MB, for the 400 objects a detector on a 1024x1024
    image can easily return.
    """
    if strategy not in ("min", "max"):
        raise ValueError(f"strategy must be 'min' or 'max', got {strategy!r}")

    array = _as_masks(masks)
    out = np.zeros(array.shape[1:], dtype=_label_dtype(len(array)))
    # Whichever mask is written last wins a contested pixel, so "max" writes
    # in order and "min" writes in reverse.
    order = range(len(array)) if strategy == "max" else reversed(range(len(array)))
    for i in order:
        out[array[i] != 0] = i + 1
    return out


def order_by_area(masks: object, descending: bool = True) -> np.ndarray:
    """The permutation that sorts the stack by area.

    Returns indices rather than sorting, so that everything indexed the same
    way travels together::

        order = masks.order_by_area(result.masks)
        result.masks[order], result.boxes[order]

    ``StackedLabels`` sorted its mask list inside ``__init__``, which quietly
    desynchronised it from any box array the caller was still holding. An
    index array reorders both, or neither.

    Ties keep their original order, so a detector's own ranking survives where
    area cannot distinguish two masks.
    """
    array = _as_masks(masks)
    areas = array.sum(axis=(1, 2))
    if descending:
        areas = -areas
    return np.argsort(areas, kind="stable")


def from_labels(labels: np.ndarray) -> np.ndarray:
    """Split a label image into a stack, one plane per object.

    The inverse of :func:`to_labels_2d`, and the way a label image from
    Cellpose or StarDist becomes something the rest of this module handles.
    Planes come out in ascending label order; background (0) is not an object.

    Objects sharing a label become one plane, wherever they are in the image.
    Separating them is connected-component labelling, which needs skimage and
    so belongs to the caller, not here.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"expected a 2-D label image, got {labels.ndim}-D")

    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return empty(labels.shape)
    return (labels[None] == ids[:, None, None]).astype(np.uint8)
