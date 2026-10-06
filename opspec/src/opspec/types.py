"""What an op's data is: the array it takes, and what that array means.

Standard library only, like the rest of opspec. numpy is named in op
signatures, never imported here.
"""

from __future__ import annotations

from typing import Annotated, Any, Protocol, TypeVar, runtime_checkable

from .op import Role

__all__ = [
    "Array",
    "BoxesOf",
    "ImageOf",
    "LabelsOf",
    "MasksOf",
    "PointsOf",
    "ShapesOf",
    "SurfaceOf",
    "TracksOf",
    "VectorsOf",
]


@runtime_checkable
class Array(Protocol):
    """

    A subset of the Python array API standard, numpy,
    cupy, dask, zarr and xarray all satisfy it. napari's
    ``LayerDataProtocol`` is similar.

    ``__array__`` is absent: cupy defines it and raises, so
    testing for it passes and the conversion then fails. 
    
    Potentially converting between array libraries belongs to a runner.

    This is what a plugin uses, A host can define it's own (ie napari define LayerDataProtocol).
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


# The aliases below are templates: the role is fixed, the array type is
# filled in by the op. `A` is the blank.
#
#     ImageOf[np.ndarray]    numpy only
#     ImageOf[cp.ndarray]    cupy only
#     ImageOf[Array]         anything array-shaped
#
# `ImageOf[np.ndarray]` is exactly `Annotated[np.ndarray, Role.image]`, so
# to anything not reading roles it is still a plain ndarray.
A = TypeVar("A")

#: A picture: intensities to be displayed as such.
ImageOf = Annotated[A, Role.image]

#: A label image: integer object IDs, 0 for background.
LabelsOf = Annotated[A, Role.labels]

#: A stack of binary masks, one object per plane. Unlike a label image
#: these may overlap, which is why they are not one.
MasksOf = Annotated[A, Role.masks]

#: Axis-aligned bounding boxes, as (N, 4): [min_y, min_x, max_y, max_x].
BoxesOf = Annotated[A, Role.boxes]

#: Coordinates, as (N, D) in axis order matching the image they came from.
PointsOf = Annotated[A, Role.points]

#: Freeform shapes: polygons, lines, paths.
ShapesOf = Annotated[A, Role.shapes]

#: A mesh: vertices, faces and values.
SurfaceOf = Annotated[A, Role.surface]

#: Trajectories, as (N, D+2): track ID, time, then coordinates.
TracksOf = Annotated[A, Role.tracks]

#: Displacements, as (N, 2, D): a start position and a projection.
VectorsOf = Annotated[A, Role.vectors]
