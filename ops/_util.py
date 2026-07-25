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
