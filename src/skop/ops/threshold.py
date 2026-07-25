"""Otsu thresholding, via scikit-image.

Ported from src/imgops/implementations/skimagessegmenter.py.
"""

from __future__ import annotations

import numpy as np

from skop import op

from ._util import to_gray


@op(env="skimage")
def otsu(
    image: np.ndarray,
    invert: bool = False,
    label_objects: bool = True,
) -> np.ndarray:
    """Threshold an image by Otsu's method.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        invert: Whether to keep what falls below the threshold instead.
        label_objects: Whether to label connected components, rather than
            returning a plain binary mask.

    Returns:
        A label image, or a 0/1 mask when ``label_objects`` is off.
    """
    from skimage import filters, measure

    gray = to_gray(image)
    mask = gray > filters.threshold_otsu(gray)
    if invert:
        mask = ~mask
    if label_objects:
        return measure.label(mask).astype(np.uint16)
    return mask.astype(np.uint16)
