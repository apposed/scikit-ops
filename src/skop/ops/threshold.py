"""Otsu thresholding, via scikit-image.

Ported from src/imgops/implementations/skimagessegmenter.py.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import Axes, op
from skop.types import ImageData, LabelsData

from ._util import to_gray


@op(env="skimage")
def otsu(
    image: Annotated[ImageData, Axes(variadic=True)],
    invert: bool = False,
    label_objects: bool = True,
) -> LabelsData:
    """Threshold an image by Otsu's method.

    A global threshold consumes no particular axis -- it is a histogram over
    whatever it is handed -- so this declares no slots at all, and works on a
    line, a plane or a whole volume. Whether one threshold for a stack or one
    per plane is the right answer depends on the experiment, so it is left to
    the caller: by default the stack arrives whole, and a caller wanting a
    threshold per plane iterates instead.

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
