"""Helpers shared between ops.

Kept as light as an op module, since it is imported alongside them during
discovery.
"""

from __future__ import annotations

import numpy as np


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
