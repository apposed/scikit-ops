"""Emission wavelengths for common fluorophores.

Every PSF model here takes a wavelength in microns, and in practice nobody
knows one off the top of their head -- they know they imaged DAPI. This is
that lookup, ported from tnia-python's ``wave_dictionary``.

A ``float`` mixin rather than a plain ``Enum``, so a member *is* its
wavelength: ``gibson_lanni(wavelength=Fluorophore.DAPI)`` works with no
conversion, and the parameter stays a plain ``float`` that also accepts the
647.1 nm line of whatever laser is actually on the bench.

Stdlib only, so any environment holding a PSF op can import it.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Fluorophore"]


class Fluorophore(float, Enum):
    """Peak emission wavelength, in microns.

    The source dictionary paired an excitation with each emission. Only
    emission is kept: it is what forms the image, and so what every PSF model
    here asks for. Excitation is noted in the comments rather than carried,
    since nothing computes with it.
    """

    DAPI = 0.461  # excitation 358 nm, blue
    Cy2 = 0.506  # excitation 489 nm, green
    FITC = 0.519  # excitation 495 nm, green
    TexasRed = 0.610  # excitation 561 nm, orange-red
    AF594 = 0.617  # excitation 590 nm, red
    Cy5 = 0.670  # excitation 649 nm, far-red / NIR

    # NB: the original spelled Texas Red "Texa", which reads as a typo rather
    # than an abbreviation -- these were dictionary keys, so nothing depended
    # on the spelling. Ordered by emission here rather than alphabetically,
    # because a wavelength list that runs blue to red is the one a microscopist
    # can scan.
