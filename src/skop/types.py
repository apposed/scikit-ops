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
"""

from __future__ import annotations

from typing import Annotated

import numpy as np

from ._spec import Role

#: A picture: intensities to be displayed as such.
ImageData = Annotated[np.ndarray, Role.image]

#: A label image: integer object IDs, 0 for background.
LabelsData = Annotated[np.ndarray, Role.labels]

#: A stack of binary masks, as (N, Y, X) uint8, one object per plane. Unlike
#: a label image these may overlap, which is why they are not one. See
#: ``skop.masks`` for the projections a front end shows them through.
MasksData = Annotated[np.ndarray, Role.masks]

#: Coordinates, as (N, D) in axis order matching the image they came from.
PointsData = Annotated[np.ndarray, Role.points]

#: Displacements, as (N, 2, D): a start position and a projection.
VectorsData = Annotated[np.ndarray, Role.vectors]

#: Trajectories, as (N, D+2): track ID, time, then coordinates.
TracksData = Annotated[np.ndarray, Role.tracks]

#: Axis-aligned bounding boxes, as (N, 4): [min_y, min_x, max_y, max_x].
#: See ``skop.boxes`` for the converters between this and everyone else's order.
BoxesData = Annotated[np.ndarray, Role.shapes]

__all__ = [
    "BoxesData",
    "ImageData",
    "LabelsData",
    "MasksData",
    "PointsData",
    "Role",
    "TracksData",
    "VectorsData",
]
