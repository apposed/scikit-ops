"""Smoothing and denoising, via scikit-image.

Two families live here, and the difference between them is what they do to an
edge. A linear filter -- :func:`gaussian`, :func:`mean` -- averages a
neighborhood without asking what is in it, so it blurs the boundaries along
with the noise. The rest are edge-preserving: they each decide, per pixel,
which neighbors are worth averaging with. :func:`median` throws out the
outliers, :func:`kuwahara` averages only the most uniform corner of its
window, :func:`bilateral` weights by intensity as well as distance, and
:func:`nl_means` looks for similar *patches* anywhere nearby rather than
similar pixels.

Every op takes an image and returns one of the same shape, as float32 --
these are filters, not segmenters, and their output is meant to be fed to
something else. A trailing RGB(A) axis is filtered channel by channel rather
than across, since a neighborhood is a spatial notion.

The denoisers -- :func:`bilateral`, :func:`tv_chambolle`, :func:`wavelet`,
:func:`nl_means` -- each carry a strength parameter measured in intensity
units, which would otherwise mean something different for a uint8 image than
for a uint16 one. They run on a copy scaled into [0, 1] and scale the result
back, so their parameters are fractions of the image's own range and a
setting that worked on one image transfers to the next.
"""

from __future__ import annotations

import itertools
from typing import Annotated

import numpy as np

from skop import Axes, op
from skop.types import ImageData

from ._util import Footprint, channel_axis, footprint, per_channel

#: An op happy with any number of axes, RGB(A) handled channel by channel.
_Image = Annotated[ImageData, Axes(variadic=True)]

#: An op that only knows about planes, with or without colour.
_Plane = Annotated[ImageData, Axes("y", "x", "c?")]


def _unit(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """An image scaled into [0, 1], with what it takes to put it back."""
    x = np.asarray(image, dtype=np.float32)
    low = float(x.min())
    span = float(x.max()) - low
    if span <= 0:
        # A blank image: any scaling is the identity, and 0 would divide.
        return np.zeros_like(x), low, 1.0
    return (x - low) / span, low, span


@op(env="skimage")
def gaussian(image: _Image, sigma: float = 1.0) -> ImageData:
    """Blur an image with a Gaussian kernel.

    The linear smoother, and the one everything else is compared against: it
    is the only kernel that neither invents structure at coarser scales nor
    rings, which is what makes it the right thing to put in front of a scale
    space, a gradient or a threshold. It is also indiscriminate -- an edge is
    a high frequency like any other, so it goes too.

    Args:
        image: Image to blur. Any number of axes; a trailing RGB(A) axis is
            blurred channel by channel.
        sigma: Standard deviation of the kernel, in pixels, along every axis.
            An anisotropic volume wants this small enough for its thinnest
            axis, or a smoothing pass of its own per axis.

    Returns:
        The blurred image, as float32.
    """
    from skimage import filters

    smoothed = filters.gaussian(
        np.asarray(image, dtype=np.float32),
        sigma=sigma,
        preserve_range=True,
        channel_axis=channel_axis(image),
    )
    return smoothed.astype(np.float32)


@op(env="skimage")
def median(
    image: _Image,
    radius: int = 1,
    shape: Footprint = Footprint.ball,
) -> ImageData:
    """Replace each pixel with the median of its neighborhood.

    The classic answer to salt-and-pepper noise, and the reason to prefer it
    over a mean: a median is decided by the middle of its neighborhood rather
    than pulled by the extremes, so a hot pixel is discarded outright instead
    of being smeared across everything near it. Edges survive because a
    neighborhood straddling one still has a majority on one side.

    Cost grows with the neighborhood, so large radii are slow.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.
        radius: Radius of the neighborhood, in pixels.
        shape: Shape of that neighborhood.

    Returns:
        The filtered image, as float32.
    """
    from skimage import filters

    fp = footprint(
        image.ndim - (0 if channel_axis(image) is None else 1), radius, shape
    )
    return per_channel(
        np.asarray(image, dtype=np.float32),
        lambda plane: filters.median(plane, footprint=fp),
    ).astype(np.float32)


@op(env="skimage")
def mean(image: _Image, radius: int = 1) -> ImageData:
    """Replace each pixel with the mean of a box around it.

    The cheapest smoother there is -- the cost of a box mean does not depend
    on the size of the box -- and the bluntest. Its kernel has hard edges, so
    it rings where a Gaussian does not; reach for it when speed is the point,
    and for :func:`gaussian` otherwise.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.
        radius: Half-width of the box, in pixels, along every axis. The box
            is ``2 * radius + 1`` across.

    Returns:
        The filtered image, as float32.
    """
    from scipy.ndimage import uniform_filter

    return per_channel(
        np.asarray(image, dtype=np.float32),
        lambda plane: uniform_filter(plane, size=2 * radius + 1, mode="reflect"),
    ).astype(np.float32)


@op(env="skimage")
def kuwahara(image: _Image, radius: int = 2) -> ImageData:
    """Smooth an image while keeping its edges sharp, by Kuwahara's method.

    Each pixel's window is split into overlapping corners -- four quadrants in
    2-D, eight octants in 3-D -- and the pixel takes the mean of whichever
    corner has the lowest variance. On a pixel inside a region every corner
    looks alike and the result is an ordinary local mean. On a pixel astride
    an edge, the corner lying wholly on one side wins, so the average never
    crosses the boundary and the edge stays where it was.

    The trade is that edges are not merely preserved but sharpened, and
    gentle gradients turn painterly -- flattened into broad patches with
    abrupt steps between them, since a pixel's value comes from whichever
    corner won rather than from its neighborhood as a whole. That is a help
    before a threshold and a hindrance before a measurement, and it is why
    this is also a favourite for stylizing photographs.

    scikit-image has no Kuwahara filter, so this is implemented here, over
    ``scipy.ndimage``.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.
        radius: Radius of the window, in pixels. Each corner is
            ``radius + 1`` across, and the whole window ``2 * radius + 1``.

    Returns:
        The filtered image, as float32.
    """
    if radius < 0:
        raise ValueError(f"radius must not be negative, got {radius}")
    return per_channel(np.asarray(image, dtype=np.float32), _kuwahara(radius)).astype(
        np.float32
    )


def _kuwahara(radius: int):
    """Kuwahara over one channel, for any number of axes."""
    from scipy.ndimage import uniform_filter

    def filter_one(x: np.ndarray) -> np.ndarray:
        size = radius + 1
        pad = size - 1
        # Padded so every corner of every pixel's window is in bounds, and
        # offset so the filter at index j covers [j - radius, j] -- the
        # "lowest" corner. Every other corner is then that same array read at
        # a shifted index, which is why only one pass is needed.
        padded = np.pad(x.astype(np.float64), pad, mode="reflect")
        origin = (size - 1) // 2
        means = uniform_filter(padded, size=size, origin=origin, mode="nearest")
        squares = uniform_filter(padded**2, size=size, origin=origin, mode="nearest")
        variances = np.clip(squares - means**2, 0.0, None)

        best_mean = best_var = None
        for offsets in itertools.product((0, size - 1), repeat=x.ndim):
            window = tuple(
                slice(pad + off, pad + off + extent)
                for off, extent in zip(offsets, x.shape)
            )
            corner_mean, corner_var = means[window], variances[window]
            if best_var is None:
                best_mean, best_var = corner_mean.copy(), corner_var.copy()
                continue
            better = corner_var < best_var
            best_mean[better] = corner_mean[better]
            best_var[better] = corner_var[better]
        return best_mean

    return filter_one


@op(env="skimage")
def bilateral(
    image: _Plane,
    sigma_color: float = 0.1,
    sigma_spatial: float = 1.0,
) -> ImageData:
    """Smooth an image, weighting neighbors by intensity as well as distance.

    A Gaussian blur whose kernel is multiplied by a second Gaussian over the
    *difference in value* between the two pixels, so a neighbor across an
    edge contributes almost nothing however close it is. The result is a
    smoother that respects boundaries without the blockiness
    :func:`kuwahara` introduces.

    Planes only, which is scikit-image's limit rather than the algorithm's.

    Args:
        image: Image to filter, as a plane with an optional RGB(A) axis.
        sigma_color: How different in value a neighbor may be and still
            count, as a fraction of the image's range. Large values converge
            on a plain Gaussian blur.
        sigma_spatial: Standard deviation of the spatial term, in pixels.

    Returns:
        The filtered image, as float32.
    """
    from skimage import restoration

    unit, low, span = _unit(image)
    smoothed = restoration.denoise_bilateral(
        unit,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=channel_axis(image),
    )
    return (smoothed * span + low).astype(np.float32)


@op(env="skimage")
def tv_chambolle(
    image: _Image,
    weight: float = 0.1,
    max_num_iter: int = 200,
) -> ImageData:
    """Denoise an image by total-variation minimization.

    Finds the image closest to this one whose total variation -- the summed
    magnitude of its gradient -- is smallest. Because a step edge costs the
    same total variation whether it is sharp or spread out, there is no
    incentive to blur it, and edges come through intact while noise, which is
    expensive in gradient per unit of signal, does not. The characteristic
    artifact is the flip side: flat cartoon-like patches, "staircasing".

    Args:
        image: Image to denoise. Any number of axes; a trailing RGB(A) axis
            is denoised channel by channel.
        weight: How hard to smooth, as a fraction of the image's range.
            Larger means more denoising and more staircasing.
        max_num_iter: Cap on the number of optimization steps.

    Returns:
        The denoised image, as float32.
    """
    from skimage import restoration

    unit, low, span = _unit(image)
    denoised = restoration.denoise_tv_chambolle(
        unit,
        weight=weight,
        max_num_iter=max_num_iter,
        channel_axis=channel_axis(image),
    )
    return (denoised * span + low).astype(np.float32)


@op(env="skimage")
def wavelet(
    image: _Image,
    method: Annotated[str, {"choices": ["BayesShrink", "VisuShrink"]}] = "BayesShrink",
    mode: Annotated[str, {"choices": ["soft", "hard"]}] = "soft",
    wavelet_name: Annotated[str, {"choices": ["db1", "db2", "haar", "sym2"]}] = "db1",
    sigma: float | None = None,
) -> ImageData:
    """Denoise an image by thresholding its wavelet coefficients.

    Noise is spread evenly across a wavelet decomposition while structure
    concentrates into a few large coefficients, so zeroing the small ones
    removes much of the former and little of the latter. Fast, and good at
    fine texture; it can leave faint ripples around sharp edges, which is
    what the wavelet choice trades against.

    Args:
        image: Image to denoise. Any number of axes; a trailing RGB(A) axis
            is denoised channel by channel.
        method: How the threshold is chosen. ``BayesShrink`` adapts it per
            subband and is usually the better of the two; ``VisuShrink``
            applies one threshold everywhere, and errs toward oversmoothing.
        mode: ``soft`` shrinks every coefficient toward zero, ``hard`` zeroes
            those below the threshold and leaves the rest. Soft is smoother,
            hard keeps more contrast.
        wavelet_name: Which wavelet to decompose with. ``db1`` -- the Haar
            wavelet -- is the usual default.
        sigma: Noise standard deviation, as a fraction of the image's range.
            None estimates it from the image.

    Returns:
        The denoised image, as float32.
    """
    from skimage import restoration

    unit, low, span = _unit(image)
    denoised = restoration.denoise_wavelet(
        unit,
        sigma=sigma,
        wavelet=wavelet_name,
        mode=mode,
        method=method,
        channel_axis=channel_axis(image),
    )
    return (denoised * span + low).astype(np.float32)


@op(env="skimage")
def nl_means(
    image: _Image,
    patch_size: int = 7,
    patch_distance: int = 11,
    h: float = 0.1,
    sigma: float | None = None,
    fast_mode: bool = True,
) -> ImageData:
    """Denoise an image by averaging over similar patches nearby.

    Every other filter here decides what to average by where a pixel is; this
    one decides by what its surroundings look like. Each pixel is replaced by
    a weighted mean of the pixels whose surrounding *patch* resembles its
    own, anywhere within a search window. Repeated structure -- texture, a
    row of similar objects, an edge running through the field -- therefore
    reinforces itself, and this preserves fine detail better than anything
    else here.

    It is also by far the slowest, growing with both the patch and the search
    window.

    Args:
        image: Image to denoise. Any number of axes; a trailing RGB(A) axis
            is denoised channel by channel.
        patch_size: Width of the patches being compared, in pixels.
        patch_distance: How far from a pixel to search for similar patches,
            in pixels. The dominant cost.
        h: How much a patch may differ and still be averaged in, as a
            fraction of the image's range. Larger means smoother.
        sigma: Noise standard deviation, as a fraction of the image's range,
            subtracted from the patch distances so that noise is not mistaken
            for dissimilarity. None estimates it from the image.
        fast_mode: Use the faster approximation, which weights patches
            slightly differently. Rarely worth turning off.

    Returns:
        The denoised image, as float32.
    """
    from skimage import restoration

    unit, low, span = _unit(image)
    axis = channel_axis(image)
    if sigma is None:
        sigma = float(
            restoration.estimate_sigma(unit, channel_axis=axis, average_sigmas=True)
        )
    denoised = restoration.denoise_nl_means(
        unit,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        sigma=sigma,
        fast_mode=fast_mode,
        preserve_range=True,
        channel_axis=axis,
    )
    return (denoised * span + low).astype(np.float32)


@op(env="skimage")
def butterworth(
    image: _Image,
    cutoff_frequency_ratio: float = 0.05,
    order: float = 2.0,
    high_pass: bool = False,
) -> ImageData:
    """Smooth an image by attenuating its high frequencies.

    The only filter here that works in the frequency domain: it multiplies
    the image's spectrum by a Butterworth window, whose defining property is
    that it is as flat as possible in the band it keeps. That flatness is why
    it rings far less than an abrupt cutoff, and the order is the knob
    between the two -- a low order rolls off gently, a high one approaches a
    brick wall and the ringing that comes with it.

    Inverted, the same filter is the standard way to flatten uneven
    illumination, which is what ``high_pass`` is for.

    Args:
        image: Image to filter. Any number of axes; a trailing RGB(A) axis is
            filtered channel by channel.
        cutoff_frequency_ratio: Where the cutoff sits, as a fraction of the
            sampling frequency, in [0, 0.5]. Smaller means more smoothing.
        order: Steepness of the roll-off.
        high_pass: Keep the high frequencies instead, subtracting the smooth
            background rather than the detail.

    Returns:
        The filtered image, as float32.
    """
    from skimage import filters

    filtered = filters.butterworth(
        np.asarray(image, dtype=np.float32),
        cutoff_frequency_ratio=cutoff_frequency_ratio,
        high_pass=high_pass,
        order=order,
        channel_axis=channel_axis(image),
    )
    return filtered.astype(np.float32)
