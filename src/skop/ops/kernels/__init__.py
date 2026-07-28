"""Kernels: generate something to convolve an image with.

A PSF is as much a forward-simulation tool as a deconvolution one, so it lives
here rather than under ``skop.ops.deconvolve``.

Three models, in increasing order of what they know about the microscope:

``gaussian_psf``    no optics at all, just a width you pick
``paraxial_psf``    wavelength, NA and pixel size -- the textbook, in focus
``gibson_lanni``    the above plus immersion medium, sample index and depth,
                    so it can be axially asymmetric

Everything here is a *theoretical* PSF, computed from optical parameters with
no measurement involved. Extracting a PSF from a bead image is deliberately
elsewhere: it composes a deconvolution op whose backend depends on the
machine, which makes it a workflow op rather than a plain one. See
docs/spec/workflow-ops.md.

This namespace spans two environments -- ``gibson_lanni`` needs sdeconv, the
rest are content with ``skimage`` -- which is the README's rule for a
namespace being a package rather than a module.
"""

from __future__ import annotations

from ._fluorophore import Fluorophore
from ._recenter import recenter_psf_axial
from .gibson_lanni import gibson_lanni
from .paraxial import paraxial_otf, paraxial_psf
from .psf import gaussian_psf

__all__ = [
    "Fluorophore",
    "gaussian_psf",
    "gibson_lanni",
    "paraxial_otf",
    "paraxial_psf",
    "recenter_psf_axial",
]
