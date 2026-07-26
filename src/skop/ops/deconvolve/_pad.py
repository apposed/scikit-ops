"""Padding helpers for FFT-based deconvolution.

Vendored from ``tnia.deconvolution.pad`` (tnia-python), reformatted but
semantically unchanged. Only what the ops here need came across.

Private, so discovery never imports it as an op module. Pure numpy and
``math``, which is what lets both backends share it: ``unpad`` only slices, so
it works on a cupy array as readily as a numpy one.
"""

from __future__ import annotations

import math

import numpy as np


def _handle_prime(p: int, x: int, a: np.ndarray) -> None:
    """Accumulate log(p) into every index reachable as a multiple of p."""
    log = math.log(p)
    power = p

    while power <= x + a.shape[0]:
        j = x % power
        if j > 0:
            j = power - j

        while j < a.shape[0]:
            a[j] += log
            j += power

        power *= p


def next_smooth(x: int) -> int:
    """Return the next number >= x divisible only by primes up to 7.

    FFTs are dramatically faster on smooth sizes, so a padded extent is
    rounded up to one. Based on A. Granville, *Finding smooth numbers
    computationally*.

    Authors: Johannes Schindelin, Brian Northan.
    """
    if x == 1:
        return 1

    z = int(16 * math.log2(x))
    delta = 0.000001

    a = np.zeros(z)

    _handle_prime(2, x, a)
    _handle_prime(3, x, a)
    _handle_prime(5, x, a)
    _handle_prime(7, x, a)

    log = math.log(x)
    for i in range(a.shape[0]):
        if a[i] >= log - delta:
            return x + i

    return -1


def get_next_smooth(size) -> tuple[int, ...]:
    """Apply ``next_smooth`` to every element of an n-dimensional size."""
    return tuple(next_smooth(i) for i in size)


def pad(img: np.ndarray, paddedsize, mode: str):
    """Pad an image up to ``paddedsize``, centered.

    Returns:
        The padded array, and the per-axis padding that was applied.
    """
    padding = tuple(
        (math.ceil((i - j) / 2), math.floor((i - j) / 2))
        for i, j in zip(paddedsize, img.shape)
    )
    return np.pad(img, padding, mode), padding


def pad_to_largest(img: np.ndarray, psf: np.ndarray, mode: str):
    """Pad both arrays so each axis matches the larger of the two."""
    largest = tuple(max(i, p) for i, p in zip(img.shape, psf.shape))
    img, _ = pad(img, largest, mode)
    psf, _ = pad(psf, largest, mode)
    return img, psf


def unpad(padded, imgsize):
    """Crop a padded array back to ``imgsize``, undoing ``pad``.

    Slicing only, so this accepts a cupy array as well as a numpy one.
    """
    padding = tuple(
        (math.ceil((i - j) / 2), math.floor((i - j) / 2))
        for i, j in zip(padded.shape, imgsize)
    )
    slices = tuple(slice(p[0], p[0] + s) for p, s in zip(padding, imgsize))
    return padded[slices]
