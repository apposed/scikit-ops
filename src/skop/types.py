"""Annotated array types that say what an array *is*.

An op that returns a bare ``np.ndarray`` has told a front end nothing about
how to show it. Annotating the same array as ``LabelsData`` says it is a
segmentation, so napari gives it a Labels layer rather than a grayscale
Image, and Fiji gives it an ``ImgLabeling``.

    from skop.types import ImageData, LabelsData

    @op(env="skimage")
    def threshold(image: ImageData) -> LabelsData: ...

These are plain ``Annotated`` aliases, so they remain ``np.ndarray`` to every
other reader -- to the type checker, to the codec that ships them over the
Appose boundary, and to whoever calls the op directly. They compose with
skop's other annotations in either order: ``Out[LabelsData]`` and
``Annotated[LabelsData, {"label": "Nuclei"}]`` both work.

The names mirror ``napari.types`` on purpose, but nothing here imports napari.
Roles lacking an alias attach directly, as ``Annotated[T, Role.surface]``.

The ``…Of`` aliases leave the array type open, so an op can say which it
takes (docs/design/0018)::

    ImageOf[np.ndarray]    numpy only -- exactly ImageData
    ImageOf[cp.ndarray]    cupy only
    ImageOf[Array]         anything array-shaped
"""

from __future__ import annotations

import numpy as np

from opspec.types import (
    Array,
    BoxesOf,
    ImageOf,
    LabelsOf,
    MasksOf,
    PointsOf,
    ShapesOf,
    SurfaceOf,
    TracksOf,
    VectorsOf,
)

from ._spec import Role

# The role aliases and the Array protocol are opspec's; these are the numpy
# spellings skop's ops use.

ImageData = ImageOf[np.ndarray]
LabelsData = LabelsOf[np.ndarray]
#: See ``skop.masks`` for the projections a front end shows these through.
MasksData = MasksOf[np.ndarray]
PointsData = PointsOf[np.ndarray]
VectorsData = VectorsOf[np.ndarray]
TracksData = TracksOf[np.ndarray]

#: Axis-aligned bounding boxes, as (N, 4): [min_y, min_x, max_y, max_x].
#: See ``skop.boxes`` for the converters between this and everyone else's order.
BoxesData = BoxesOf[np.ndarray]

__all__ = [
    "Array",
    "BoxesData",
    "BoxesOf",
    "ImageData",
    "ImageOf",
    "LabelsData",
    "LabelsOf",
    "MasksData",
    "MasksOf",
    "PointsData",
    "PointsOf",
    "Role",
    "ShapesOf",
    "SurfaceOf",
    "TracksData",
    "TracksOf",
    "VectorsData",
    "VectorsOf",
]
