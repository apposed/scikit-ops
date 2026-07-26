"""Kernels: generate something to convolve an image with.

A PSF is as much a forward-simulation tool as a deconvolution one, so it lives
here rather than under ``skop.ops.deconvolve``. Later PSF models join
``psf.py``; other kernel families get a module of their own beside it.
"""

from __future__ import annotations

from .psf import gaussian_psf

__all__ = ["gaussian_psf"]
