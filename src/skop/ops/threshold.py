"""Global thresholding, via scikit-image.

Every op here is the same shape: pick one number from the image's histogram,
keep what lies above it. They differ only in how that number is chosen, and
which choice is right is a property of the image rather than of the method --
so they are variations on one call, sharing one module and one environment.

A global threshold consumes no particular axis -- it is a histogram over
whatever it is handed -- so none of these declare slots at all, and each works
on a line, a plane or a whole volume. Whether one threshold for a stack or one
per plane is the right answer depends on the experiment, so it is left to the
caller: by default the stack arrives whole, and a caller wanting a threshold
per plane iterates instead.

Otsu is the default worth reaching for first. Beyond it:
``li`` minimizes cross-entropy and copes well with a small foreground;
``triangle`` and ``yen`` suit a skewed histogram with no clear valley;
``isodata`` and ``mean`` are the cheap classical answers; ``minimum``
insists on a genuinely bimodal histogram and raises when it cannot find one.
``multiotsu`` is the odd one out -- it cuts into several classes rather than
two, so it returns those classes rather than a foreground.

Ported from src/imgops/implementations/skimagessegmenter.py.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import Axes, op
from skop.types import ImageData, LabelsData

from ._util import to_gray

#: What every op here takes: any number of axes, an RGB(A) axis collapsed.
_Image = Annotated[ImageData, Axes(variadic=True)]


def _mask(gray: np.ndarray, value: float, invert: bool, label_objects: bool):
    """Turn one threshold into the mask, or labels, an op returns."""
    from skimage import measure

    mask = gray <= value if invert else gray > value
    if label_objects:
        return measure.label(mask).astype(np.uint16)
    return mask.astype(np.uint16)


@op(env="skimage")
def otsu(
    image: _Image,
    invert: bool = False,
    label_objects: bool = True,
) -> LabelsData:
    """Threshold an image by Otsu's method.

    Chooses the threshold that minimizes the variance within the two classes
    it creates, which is the same as maximizing the variance between them.
    The standard first thing to try, and what it assumes -- two comparably
    sized modes -- is also what it fails on.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        invert: Whether to keep what falls below the threshold instead.
        label_objects: Whether to label connected components, rather than
            returning a plain binary mask.

    Returns:
        A label image, or a 0/1 mask when ``label_objects`` is off.
    """
    from skimage import filters

    gray = to_gray(image)
    return _mask(gray, filters.threshold_otsu(gray), invert, label_objects)


@op(env="skimage")
def isodata(
    image: _Image,
    invert: bool = False,
    label_objects: bool = True,
) -> LabelsData:
    """Threshold an image by the isodata (Ridler-Calvard) method.

    Iterates until the threshold sits exactly halfway between the mean of
    what falls below it and the mean of what falls above. Usually lands very
    near Otsu, and is the method ImageJ's "Default" is descended from.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        invert: Whether to keep what falls below the threshold instead.
        label_objects: Whether to label connected components, rather than
            returning a plain binary mask.

    Returns:
        A label image, or a 0/1 mask when ``label_objects`` is off.
    """
    from skimage import filters

    gray = to_gray(image)
    return _mask(gray, filters.threshold_isodata(gray), invert, label_objects)


@op(env="skimage")
def li(
    image: _Image,
    invert: bool = False,
    label_objects: bool = True,
) -> LabelsData:
    """Threshold an image by Li's minimum cross-entropy method.

    Minimizes the information lost by replacing each class with its mean,
    which does not require the two classes to be of similar size. The one to
    try when Otsu swallows a sparse foreground.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        invert: Whether to keep what falls below the threshold instead.
        label_objects: Whether to label connected components, rather than
            returning a plain binary mask.

    Returns:
        A label image, or a 0/1 mask when ``label_objects`` is off.
    """
    from skimage import filters

    gray = to_gray(image)
    return _mask(gray, filters.threshold_li(gray), invert, label_objects)


@op(env="skimage")
def mean(
    image: _Image,
    invert: bool = False,
    label_objects: bool = True,
) -> LabelsData:
    """Threshold an image at its mean intensity.

    The simplest global threshold there is, and a reasonable baseline on an
    image whose background is genuinely uniform.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        invert: Whether to keep what falls below the threshold instead.
        label_objects: Whether to label connected components, rather than
            returning a plain binary mask.

    Returns:
        A label image, or a 0/1 mask when ``label_objects`` is off.
    """
    from skimage import filters

    gray = to_gray(image)
    return _mask(gray, filters.threshold_mean(gray), invert, label_objects)


@op(env="skimage")
def minimum(
    image: _Image,
    invert: bool = False,
    label_objects: bool = True,
) -> LabelsData:
    """Threshold an image at the valley between its two histogram peaks.

    Smooths the histogram until exactly two maxima remain, then cuts at the
    minimum between them. The most literal reading of "bimodal", and the
    least forgiving: an image whose histogram will not smooth down to two
    peaks raises rather than guessing.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        invert: Whether to keep what falls below the threshold instead.
        label_objects: Whether to label connected components, rather than
            returning a plain binary mask.

    Returns:
        A label image, or a 0/1 mask when ``label_objects`` is off.
    """
    from skimage import filters

    gray = to_gray(image)
    return _mask(gray, filters.threshold_minimum(gray), invert, label_objects)


@op(env="skimage")
def triangle(
    image: _Image,
    invert: bool = False,
    label_objects: bool = True,
) -> LabelsData:
    """Threshold an image by the triangle method.

    Draws a line from the histogram's peak to its far end and cuts where the
    histogram is furthest from that line. Geometric rather than statistical,
    which is why it holds up on the heavily skewed histogram of a mostly
    empty fluorescence image.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        invert: Whether to keep what falls below the threshold instead.
        label_objects: Whether to label connected components, rather than
            returning a plain binary mask.

    Returns:
        A label image, or a 0/1 mask when ``label_objects`` is off.
    """
    from skimage import filters

    gray = to_gray(image)
    return _mask(gray, filters.threshold_triangle(gray), invert, label_objects)


@op(env="skimage")
def yen(
    image: _Image,
    invert: bool = False,
    label_objects: bool = True,
) -> LabelsData:
    """Threshold an image by Yen's maximum correlation criterion.

    An entropy method like Li's, but maximizing a correlation between the two
    classes rather than minimizing a divergence. Tends to cut higher than
    Otsu, keeping less.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        invert: Whether to keep what falls below the threshold instead.
        label_objects: Whether to label connected components, rather than
            returning a plain binary mask.

    Returns:
        A label image, or a 0/1 mask when ``label_objects`` is off.
    """
    from skimage import filters

    gray = to_gray(image)
    return _mask(gray, filters.threshold_yen(gray), invert, label_objects)


@op(env="skimage")
def multiotsu(
    image: _Image,
    classes: int = 3,
    label_objects: bool = False,
) -> LabelsData:
    """Split an image into several intensity classes, by Otsu's criterion.

    Otsu generalized: instead of one threshold there are ``classes - 1`` of
    them, chosen together to minimize the variance within the classes they
    make. Where the other ops here answer "foreground or not", this one
    answers "which layer" -- background, dim, bright -- so it returns those
    classes as a label image, numbered 0 upward by intensity.

    Args:
        image: Image to threshold. A trailing RGB(A) axis is collapsed.
        classes: How many intensity classes to split into, at least 2.
        label_objects: Whether to label the connected components of the
            brightest class instead, discarding the rest. That makes this a
            drop-in for the two-class ops, cutting nearer the bright end.

    Returns:
        A label image: one label per intensity class, or per object in the
        brightest class when ``label_objects`` is on.
    """
    from skimage import filters, measure

    if classes < 2:
        raise ValueError(f"classes must be at least 2, got {classes}")

    gray = to_gray(image)
    thresholds = filters.threshold_multiotsu(gray, classes=classes)
    if label_objects:
        return measure.label(gray > thresholds[-1]).astype(np.uint16)
    # Note: right=True so a pixel *at* a threshold falls below it, matching
    # the strict `gray > value` the two-class ops here use.
    return np.digitize(gray, bins=thresholds, right=True).astype(np.uint16)
