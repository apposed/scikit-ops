"""Grayscale morphology, via scikit-image.

Morphology asks a different question from the filters in
:mod:`skop.ops.smooth`: not "what is the average around here" but "does a
shape of this size fit here". Every op below slides one structuring element
-- a footprint of a given radius -- over the image and takes an extremum
under it, and the whole family follows from the two that do:
:func:`erosion` takes the minimum, :func:`dilation` the maximum, and the
rest are those two composed.

Because they choose an existing value rather than computing a new one, they
introduce no intermediate intensities and preserve the input's dtype -- which
is why they are the safe thing to run on a label image or a mask, and why
:func:`white_tophat` is the classical background subtraction.

The radius is the whole parameter: it says what counts as small. Anything
smaller than the footprint does not survive an opening, and that is the
knob to turn. Its shape is the second, and :class:`Footprint` -- re-exported
here, since this is the namespace it belongs to -- is what names the three
choices. :func:`skop.ops.smooth.median` takes the same enum.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import Axes, op
from skop.types import ImageData

from ._util import Footprint, channel_axis, footprint, per_channel

#: Every op here works on any number of axes, RGB(A) channel by channel.
_Image = Annotated[ImageData, Axes(variadic=True)]


def _apply(image: np.ndarray, radius: int, shape: Footprint, fn) -> np.ndarray:
    """Run a morphological op with a footprint matching the image's rank."""
    if radius < 1:
        raise ValueError(f"radius must be at least 1, got {radius}")
    spatial = image.ndim - (0 if channel_axis(image) is None else 1)
    fp = footprint(spatial, radius, shape)
    return per_channel(image, lambda plane: fn(plane, footprint=fp))


@op(env="skimage")
def erosion(
    image: _Image,
    radius: int = 1,
    shape: Footprint = Footprint.ball,
) -> ImageData:
    """Shrink bright regions, by taking the minimum under the footprint.

    Bright objects lose a border of ``radius`` and dark ones grow by the
    same; anything bright and thinner than the footprint disappears. On a
    binary mask this is how touching objects are pulled apart before a
    connected-components pass.

    Args:
        image: Image to erode. Any number of axes; a trailing RGB(A) axis is
            eroded channel by channel.
        radius: Radius of the footprint, in pixels.
        shape: Shape of the footprint.

    Returns:
        The eroded image, in the input's dtype.
    """
    from skimage import morphology

    return _apply(image, radius, shape, morphology.erosion)


@op(env="skimage")
def dilation(
    image: _Image,
    radius: int = 1,
    shape: Footprint = Footprint.ball,
) -> ImageData:
    """Grow bright regions, by taking the maximum under the footprint.

    The dual of :func:`erosion`: bright objects gain a border of ``radius``,
    and gaps narrower than the footprint close up. The usual use is joining a
    structure that noise or thresholding has broken into pieces.

    Args:
        image: Image to dilate. Any number of axes; a trailing RGB(A) axis is
            dilated channel by channel.
        radius: Radius of the footprint, in pixels.
        shape: Shape of the footprint.

    Returns:
        The dilated image, in the input's dtype.
    """
    from skimage import morphology

    return _apply(image, radius, shape, morphology.dilation)


@op(env="skimage")
def opening(
    image: _Image,
    radius: int = 1,
    shape: Footprint = Footprint.ball,
) -> ImageData:
    """Remove bright detail smaller than the footprint.

    An erosion followed by a dilation. The erosion deletes whatever the
    footprint does not fit inside, and the dilation restores what survived to
    roughly its original size -- so unlike a plain erosion this removes small
    bright specks without shrinking everything else. Sizes and shapes larger
    than the footprint come through essentially untouched.

    Args:
        image: Image to open. Any number of axes; a trailing RGB(A) axis is
            opened channel by channel.
        radius: Radius of the footprint, in pixels. Bright structure thinner
            than this is what gets removed.
        shape: Shape of the footprint.

    Returns:
        The opened image, in the input's dtype.
    """
    from skimage import morphology

    return _apply(image, radius, shape, morphology.opening)


@op(env="skimage")
def closing(
    image: _Image,
    radius: int = 1,
    shape: Footprint = Footprint.ball,
) -> ImageData:
    """Remove dark detail smaller than the footprint.

    A dilation followed by an erosion: the mirror of :func:`opening`, filling
    small dark holes and thin dark gaps without growing the bright regions
    around them. Between the two, most mask cleanup is a closing to mend
    objects and then an opening to drop debris.

    Args:
        image: Image to close. Any number of axes; a trailing RGB(A) axis is
            closed channel by channel.
        radius: Radius of the footprint, in pixels. Dark structure thinner
            than this is what gets filled.
        shape: Shape of the footprint.

    Returns:
        The closed image, in the input's dtype.
    """
    from skimage import morphology

    return _apply(image, radius, shape, morphology.closing)


@op(env="skimage")
def white_tophat(
    image: _Image,
    radius: int = 1,
    shape: Footprint = Footprint.ball,
) -> ImageData:
    """Keep bright detail smaller than the footprint, dropping the background.

    The image minus its :func:`opening` -- that is, exactly what the opening
    removed. Since an opening keeps whatever is larger than the footprint,
    the difference is the small bright structure sitting on top of it, with
    slow background variation subtracted away.

    This is background subtraction, done morphologically: a radius comfortably
    larger than the objects and smaller than the illumination gradient
    flattens the field without touching the objects.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.
        radius: Radius of the footprint, in pixels. Bright structure smaller
            than this is what is kept.
        shape: Shape of the footprint.

    Returns:
        The bright detail, in the input's dtype.
    """
    from skimage import morphology

    return _apply(image, radius, shape, morphology.white_tophat)


@op(env="skimage")
def black_tophat(
    image: _Image,
    radius: int = 1,
    shape: Footprint = Footprint.ball,
) -> ImageData:
    """Keep dark detail smaller than the footprint, dropping the background.

    The :func:`closing` minus the image, and the dual of
    :func:`white_tophat`: it isolates small dark structure on a bright,
    slowly varying background. What to reach for on a brightfield image,
    where the objects are the dark things.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.
        radius: Radius of the footprint, in pixels. Dark structure smaller
            than this is what is kept.
        shape: Shape of the footprint.

    Returns:
        The dark detail, in the input's dtype.
    """
    from skimage import morphology

    return _apply(image, radius, shape, morphology.black_tophat)
