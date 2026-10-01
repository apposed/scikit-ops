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

from typing import Annotated, Any, Protocol, TypeVar, runtime_checkable

import numpy as np

from ._spec import Role


@runtime_checkable
class Array(Protocol):
    """Anything array-shaped: numpy, cupy, dask, zarr and xarray all qualify.

    ``__array__`` is absent on purpose: cupy defines it and raises, so testing
    for it passes and the conversion then fails.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def ndim(self) -> int: ...

    @property
    def size(self) -> int: ...

    def __getitem__(self, key: Any) -> Any: ...


A = TypeVar("A")

#: A picture: intensities to be displayed as such.
ImageOf = Annotated[A, Role.image]

#: A label image: integer object IDs, 0 for background.
LabelsOf = Annotated[A, Role.labels]

#: A stack of binary masks, as (N, Y, X) uint8, one object per plane. Unlike
#: a label image these may overlap, which is why they are not one. See
#: ``skop.masks`` for the projections a front end shows them through.
MasksOf = Annotated[A, Role.masks]

#: Coordinates, as (N, D) in axis order matching the image they came from.
PointsOf = Annotated[A, Role.points]

#: Displacements, as (N, 2, D): a start position and a projection.
VectorsOf = Annotated[A, Role.vectors]

#: Trajectories, as (N, D+2): track ID, time, then coordinates.
TracksOf = Annotated[A, Role.tracks]

#: Freeform shapes: polygons, lines, paths.
ShapesOf = Annotated[A, Role.shapes]

#: A mesh: vertices, faces and values.
SurfaceOf = Annotated[A, Role.surface]

ImageData = ImageOf[np.ndarray]
LabelsData = LabelsOf[np.ndarray]
MasksData = MasksOf[np.ndarray]
PointsData = PointsOf[np.ndarray]
VectorsData = VectorsOf[np.ndarray]
TracksData = TracksOf[np.ndarray]

#: Axis-aligned bounding boxes, as (N, 4): [min_y, min_x, max_y, max_x].
#: See ``skop.boxes`` for the converters between this and everyone else's order.
BoxesData = Annotated[np.ndarray, Role.shapes]

__all__ = [
    "Array",
    "BoxesData",
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
