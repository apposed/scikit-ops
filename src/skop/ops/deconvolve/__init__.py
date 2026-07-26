"""Deconvolution: undo the blur a microscope's PSF imposed.

One op per backend, because ``@op(env=...)`` is fixed per function and the
backends need different environments -- ``richardson_lucy`` runs anywhere,
``richardson_lucy_cupy`` needs an NVIDIA GPU. They implement the same
iteration and are held numerically equivalent by a test, so a caller that can
choose is choosing on speed alone.

PSFs to feed them live in ``skop.ops.kernels``.
"""

from __future__ import annotations

from .richardson_lucy import richardson_lucy
from .richardson_lucy_cupy import richardson_lucy_cupy

__all__ = ["richardson_lucy", "richardson_lucy_cupy"]
