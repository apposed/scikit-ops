"""Derivative filters: edges, ridges, and the sharpening they enable.

Where :mod:`skop.ops.smooth` averages a neighborhood, everything here
*differences* one. The first group -- :func:`sobel` and its relatives -- are
first-derivative operators, and answer "how fast is intensity changing here",
which is large at a boundary and near zero inside an object. They differ only
in the small kernel they use to estimate that derivative, and the differences
between them matter mainly on noisy or anisotropic data.

The second group works on the second derivative, which is what separates a
*ridge* -- a line, a filament, a vessel -- from a step edge. :func:`frangi`
and friends examine the Hessian's eigenvalues over a range of scales, so they
respond to structures of a given thickness rather than to contrast alone.
scikit-image's fourth ridge filter, ``hessian``, is deliberately not wrapped:
it is ``frangi`` with every non-positive response replaced by 1, so the
background comes back reading as a maximal detection, and nothing sensible
happens when a front end hands that to a threshold.

:func:`unsharp_mask` is the third thing derivatives are good for: adding an
image's own detail back to it.

Every op takes an image and returns a response map of the same shape, as
float32 -- these are filters, and their output is meant to be thresholded or
fed onward, not read as intensities. A trailing RGB(A) axis is filtered
channel by channel.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from skop import Axes, op
from skop.types import ImageData

from ._util import channel_axis, per_channel, to_gray

#: An op happy with any number of axes, RGB(A) handled channel by channel.
_Image = Annotated[ImageData, Axes(variadic=True)]

#: An op that only knows about planes, with or without colour.
_Plane = Annotated[ImageData, Axes("y", "x", "c?")]

#: An op that handles a plane or a volume, and nothing else.
_Volume = Annotated[ImageData, Axes("z?", "y", "x")]


def _sigmas(sigma_min: float, sigma_max: float, sigma_step: float) -> np.ndarray:
    """The scales a ridge filter sweeps, from three numbers a GUI can offer."""
    if sigma_step <= 0:
        raise ValueError(f"sigma_step must be positive, got {sigma_step}")
    if sigma_max < sigma_min:
        raise ValueError(f"sigma_max {sigma_max} is below sigma_min {sigma_min}")
    return np.arange(sigma_min, sigma_max + sigma_step / 2, sigma_step)


def _apply(image: np.ndarray, fn) -> np.ndarray:
    return per_channel(np.asarray(image, dtype=np.float32), fn).astype(np.float32)


@op(env="skimage")
def sobel(image: _Image) -> ImageData:
    """Find edges with the Sobel operator.

    A central difference along each axis, smoothed across the others, with
    the per-axis responses combined into one magnitude. The default edge
    filter, and the one to reach for absent a reason not to: it is cheap,
    and the smoothing built into its kernel makes it far steadier on noisy
    data than a bare difference.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.

    Returns:
        Edge magnitude, as float32.
    """
    from skimage import filters

    return _apply(image, filters.sobel)


@op(env="skimage")
def scharr(image: _Image) -> ImageData:
    """Find edges with the Scharr operator.

    Sobel's kernel, with the weights rebalanced so that the operator's
    response is as close to rotationally symmetric as a 3x3 kernel can be. A
    diagonal edge therefore reads at the same strength as an axis-aligned
    one, which matters when the magnitude is going to be compared against a
    threshold or an orientation read off it.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.

    Returns:
        Edge magnitude, as float32.
    """
    from skimage import filters

    return _apply(image, filters.scharr)


@op(env="skimage")
def prewitt(image: _Image) -> ImageData:
    """Find edges with the Prewitt operator.

    Sobel without the extra weight on the centre row: the smoothing across
    the perpendicular axis is a plain box mean rather than a triangular one.
    Marginally cheaper, marginally noisier, and included because a good deal
    of older literature specifies it.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.

    Returns:
        Edge magnitude, as float32.
    """
    from skimage import filters

    return _apply(image, filters.prewitt)


@op(env="skimage")
def farid(image: _Plane) -> ImageData:
    """Find edges with the Farid-Simoncelli operator.

    A larger kernel, fitted so that the derivative it computes is as accurate
    as possible rather than merely symmetric. Worth its extra cost when the
    gradient's *value* is being used -- for orientation, or optical flow --
    rather than just thresholded.

    Args:
        image: Image to filter, as a plane with an optional RGB(A) axis.

    Returns:
        Edge magnitude, as float32.
    """
    from skimage import filters

    return _apply(image, filters.farid)


@op(env="skimage")
def roberts(image: _Plane) -> ImageData:
    """Find edges with the Roberts cross operator.

    The smallest edge filter there is: two 2x2 diagonal differences. It
    localizes an edge to within a pixel, which no larger kernel does, and it
    has no noise rejection whatsoever -- so it belongs on clean, high
    contrast images and nowhere else.

    Args:
        image: Image to filter, as a plane. A trailing RGB(A) axis is
            filtered channel by channel.

    Returns:
        Edge magnitude, as float32.
    """
    from skimage import filters

    return _apply(image, filters.roberts)


@op(env="skimage")
def laplace(image: _Image, ksize: int = 3) -> ImageData:
    """Find edges with the Laplacian, the sum of the second derivatives.

    Unlike the gradient filters, this is not directional: it responds to
    curvature, so it peaks on *either* side of an edge and crosses zero at
    the edge itself. Those zero crossings, not the peaks, are the edge --
    which is why the Laplacian is usually a step toward something else
    rather than an answer. It amplifies noise sharply; smooth first.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.
        ksize: Width of the discrete Laplacian kernel, in pixels.

    Returns:
        The Laplacian, as float32. Signed, unlike the gradient filters.
    """
    from skimage import filters

    return _apply(image, lambda plane: filters.laplace(plane, ksize=ksize))


@op(env="skimage")
def difference_of_gaussians(
    image: _Image,
    low_sigma: float = 1.0,
    high_sigma: float = 4.0,
) -> ImageData:
    """Band-pass an image, by subtracting one Gaussian blur from another.

    Blurring at two scales and taking the difference keeps exactly the
    structure that falls between them: finer detail is in both blurs and
    cancels, coarser shading is in neither. That makes this the standard way
    to find blobs of a known size, and equally the standard way to flatten
    uneven illumination while keeping the objects.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.
        low_sigma: Standard deviation of the finer blur, in pixels. Sets the
            smallest detail kept.
        high_sigma: Standard deviation of the coarser blur, in pixels. Sets
            the largest. Must exceed ``low_sigma``.

    Returns:
        The band-passed image, as float32. Signed.
    """
    from skimage import filters

    if high_sigma <= low_sigma:
        raise ValueError(f"high_sigma {high_sigma} must exceed low_sigma {low_sigma}")
    filtered = filters.difference_of_gaussians(
        np.asarray(image, dtype=np.float32),
        low_sigma=low_sigma,
        high_sigma=high_sigma,
        channel_axis=channel_axis(image),
    )
    return filtered.astype(np.float32)


@op(env="skimage")
def unsharp_mask(
    image: _Image,
    radius: float = 1.0,
    amount: float = 1.0,
) -> ImageData:
    """Sharpen an image by adding back what blurring it removes.

    Blur the image, subtract that from the original to get the detail, then
    add a multiple of the detail back. Older than digital imaging -- the name
    is from the darkroom -- and still the sharpening every tool implements.
    It creates no information, so overdoing it produces halos around edges
    rather than more detail.

    Args:
        image: Image to sharpen. Any number of axes; a trailing RGB(A) axis
            is sharpened channel by channel.
        radius: Standard deviation of the blur that defines "detail", in
            pixels. Larger radii sharpen coarser structure.
        amount: How much of the detail to add back. 0 is a no-op, 1 is the
            usual starting point, and much above 2 halos.

    Returns:
        The sharpened image, as float32.
    """
    from skimage import filters

    sharpened = filters.unsharp_mask(
        np.asarray(image, dtype=np.float32),
        radius=radius,
        amount=amount,
        preserve_range=True,
        channel_axis=channel_axis(image),
    )
    return sharpened.astype(np.float32)


@op(env="skimage")
def frangi(
    image: _Volume,
    sigma_min: float = 1.0,
    sigma_max: float = 10.0,
    sigma_step: float = 2.0,
    black_ridges: bool = True,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> ImageData:
    """Enhance tubular structures, by Frangi's vesselness measure.

    Written for angiograms and used ever since for anything filamentous --
    vessels, neurites, cytoskeleton. At each scale it takes the Hessian's
    eigenvalues and asks whether they look like a tube: one small eigenvalue
    along the structure, large ones across it. The strongest response over
    all the scales is kept, so a range of ``sigmas`` covering the thicknesses
    present is the parameter that matters.

    Args:
        image: Image to filter, as a plane or a volume. RGB(A) is collapsed
            to grayscale, since a vesselness has no per-channel meaning.
        sigma_min: Smallest structure thickness to look for, as a Gaussian
            sigma in pixels.
        sigma_max: Largest, inclusive.
        sigma_step: Spacing between the scales swept.
        black_ridges: Whether to find dark structures on a light background,
            as in a brightfield image, rather than the reverse.
        alpha: Sensitivity to plate-like rather than tube-like structure.
        beta: Sensitivity to blob-like rather than tube-like structure.

    Returns:
        Vesselness in [0, 1], as float32.
    """
    from skimage import filters

    return filters.frangi(
        to_gray(np.asarray(image, dtype=np.float32)),
        sigmas=_sigmas(sigma_min, sigma_max, sigma_step),
        alpha=alpha,
        beta=beta,
        black_ridges=black_ridges,
    ).astype(np.float32)


@op(env="skimage")
def sato(
    image: _Volume,
    sigma_min: float = 1.0,
    sigma_max: float = 10.0,
    sigma_step: float = 2.0,
    black_ridges: bool = True,
) -> ImageData:
    """Enhance tubular structures, by Sato's tubeness measure.

    The same Hessian sweep as :func:`frangi` with a simpler combination of
    the eigenvalues, and no free sensitivity parameters. That makes it the
    easier of the two to use, and the less selective: it responds more
    readily at junctions and to structures that are not quite tubes.

    Args:
        image: Image to filter, as a plane or a volume. RGB(A) is collapsed
            to grayscale.
        sigma_min: Smallest structure thickness to look for, as a Gaussian
            sigma in pixels.
        sigma_max: Largest, inclusive.
        sigma_step: Spacing between the scales swept.
        black_ridges: Whether to find dark structures on a light background
            rather than the reverse.

    Returns:
        Tubeness, as float32.
    """
    from skimage import filters

    return filters.sato(
        to_gray(np.asarray(image, dtype=np.float32)),
        sigmas=_sigmas(sigma_min, sigma_max, sigma_step),
        black_ridges=black_ridges,
    ).astype(np.float32)


@op(env="skimage")
def meijering(
    image: _Volume,
    sigma_min: float = 1.0,
    sigma_max: float = 10.0,
    sigma_step: float = 2.0,
    black_ridges: bool = True,
) -> ImageData:
    """Enhance neurite-like structures, by Meijering's neuriteness measure.

    A Hessian filter tuned for thin, low-contrast, branching structures --
    it was written for tracing neurites in noisy microscopy. It holds up
    better than :func:`frangi` where the structure is barely above the
    background, and is correspondingly noisier where it is not.

    Args:
        image: Image to filter, as a plane or a volume. RGB(A) is collapsed
            to grayscale.
        sigma_min: Smallest structure thickness to look for, as a Gaussian
            sigma in pixels.
        sigma_max: Largest, inclusive.
        sigma_step: Spacing between the scales swept.
        black_ridges: Whether to find dark structures on a light background
            rather than the reverse.

    Returns:
        Neuriteness, as float32.
    """
    from skimage import filters

    return filters.meijering(
        to_gray(np.asarray(image, dtype=np.float32)),
        sigmas=_sigmas(sigma_min, sigma_max, sigma_step),
        black_ridges=black_ridges,
    ).astype(np.float32)
