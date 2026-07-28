"""Helpers shared between ops.

Kept as light as an op module, since it is imported alongside them during
discovery.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import numpy as np


class Footprint(Enum):
    """Shape of a structuring element, as the metric its radius measures in.

    ``ball`` is Euclidean and the usual choice: it is isotropic, so an
    operation does not depend on how the object happens to be oriented.
    ``box`` is Chebyshev and the fastest, at the cost of favouring the
    diagonals. ``diamond`` is Manhattan, the tightest of the three.
    """

    ball = "ball"
    box = "box"
    diamond = "diamond"


def footprint(ndim: int, radius: int, shape: Footprint = Footprint.ball) -> np.ndarray:
    """A structuring element of the given radius, in the given dimension.

    Generated rather than taken from ``skimage.morphology``, whose generators
    are one function per dimension -- ``disk`` for 2-D, ``ball`` for 3-D and
    nothing beyond. The three shapes are the three usual metrics, so each is a
    distance from the centre thresholded at ``radius``.
    """
    grid = np.indices((2 * radius + 1,) * ndim) - radius
    if shape is Footprint.ball:
        distance = np.sqrt((grid.astype(np.float64) ** 2).sum(axis=0))
    elif shape is Footprint.box:
        distance = np.abs(grid).max(axis=0)
    else:
        distance = np.abs(grid).sum(axis=0)
    return (distance <= radius).astype(np.uint8)


def channel_axis(image: np.ndarray) -> int | None:
    """``-1`` when a trailing axis looks like RGB(A), else ``None``.

    The same guess :func:`to_gray` makes, for ops that keep the channels
    rather than collapsing them.
    """
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        return -1
    return None


def per_channel(
    image: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply a spatial filter to each RGB(A) channel, or to the whole array.

    A neighborhood is a spatial notion, and mixing red into green is never
    what a filter meant. Arrays without a trailing channel axis go through
    untouched, so an op needs no branch of its own.
    """
    if channel_axis(image) is None:
        return fn(image)
    return np.stack(
        [fn(image[..., c]) for c in range(image.shape[-1])],
        axis=-1,
    )


def to_gray(image: np.ndarray) -> np.ndarray:
    """Collapse a small trailing channel axis, leaving other axes alone.

    A trailing extent of 3 or 4 is taken to be RGB(A); anything else is
    assumed to be spatial and returned untouched.
    """
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    return image


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Promote an image to the ``(H, W, 3)`` uint8 an RGB model expects.

    The other direction from :func:`to_gray`, and the thing every detector
    ported from natural-image work needs: a 16-bit microscopy plane means
    nothing to a model trained on photographs until it is contrast-stretched
    into 8-bit and given three channels.

    Percentile-stretched rather than min-max scaled, so a single hot pixel
    cannot flatten everything else to black. Already-uint8 RGB is returned
    untouched, since that is exactly what the model wants.
    """
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8:
        return image
    if image.ndim > 2 and not (image.ndim == 3 and image.shape[-1] == 3):
        raise ValueError(f"expected a 2-D image or (H, W, 3), got shape {image.shape}")

    x = image.astype(np.float32)
    low, high = np.percentile(x, (1.0, 99.8))
    if high <= low:
        # A blank or single-valued image: stretching it would divide by zero.
        high = low + 1.0
    x = np.clip((x - low) / (high - low), 0.0, 1.0)
    x = (x * 255).astype(np.uint8)

    if x.ndim == 2:
        x = np.repeat(x[:, :, None], 3, axis=2)
    return np.ascontiguousarray(x)
