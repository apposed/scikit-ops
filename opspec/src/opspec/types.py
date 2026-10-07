"""The Array protocol, and role aliases for op parameters.

A role is the meaning of an array: an image, labels, boxes. An op parameter
states its role and its array type in one annotation, for example
``ImageOf[np.ndarray]`` or ``LabelsOf[Array]``.
"""

from __future__ import annotations

from typing import Annotated, Any, Protocol, TypeVar, runtime_checkable

from .op import Role

# The Array protocol, and one alias per role.
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
    """Anything array-shaped, for an op that does not need numpy.

    A subset of the Python array API standard. numpy, cupy, dask, zarr and
    xarray all satisfy it.

    We purposely don't support ``__array__``, because converting an Array to
    numpy is the runner's job.

    This is what an op takes. A host may keep its own protocol, as napari
    does with ``LayerDataProtocol``.

    Todo: consider reusing a community array protocol.
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


# The aliases take the array type as a parameter, so an op can say which
# array types it accepts:
#
#     ImageOf[np.ndarray]    numpy only
#     ImageOf[cp.ndarray]    cupy only
#     ImageOf[Array]         any array that satisfies the Array protocol
A = TypeVar("A")

#: An intensity image.
ImageOf = Annotated[A, Role.image]

#: A label image: each object is one integer value, and 0 is background.
LabelsOf = Annotated[A, Role.labels]

#: A stack of binary masks, (N, Y, X), one object per plane. Unlike labels,
#: masks may overlap.
MasksOf = Annotated[A, Role.masks]

#: Axis-aligned bounding boxes, (N, 4): [min_y, min_x, max_y, max_x].
BoxesOf = Annotated[A, Role.boxes]

#: Point coordinates, (N, D), in the image's axis order.
PointsOf = Annotated[A, Role.points]

#: Freeform shapes: polygons, lines, paths.
ShapesOf = Annotated[A, Role.shapes]

#: A mesh: vertices, faces and values.
SurfaceOf = Annotated[A, Role.surface]

#: Trajectories, (N, D + 2): track ID, time, then coordinates.
TracksOf = Annotated[A, Role.tracks]

#: Displacements, (N, 2, D): a start position and a direction.
VectorsOf = Annotated[A, Role.vectors]
