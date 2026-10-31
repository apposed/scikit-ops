"""Roles, the Array protocol, and role aliases for op parameters.

A role is the meaning of an array: an image, labels, boxes. An op parameter
states its role and its array type in one annotation, for example
``ImageOf[np.ndarray]`` or ``LabelsOf[Array]``.

Standard library only, because every worker environment imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Protocol, TypeVar, runtime_checkable

# The public names of this module, defined below: the roles, the Array
# protocol, one alias per role for op parameters (e.g. ImageOf[np.ndarray]
# or LabelsOf[Array]), dtypes, and Out/Mut for buffers.
__all__ = [
    "DTYPE_NAMES",
    "Array",
    "BoxesOf",
    "Direction",
    "Dtype",
    "ImageOf",
    "LabelsOf",
    "MasksOf",
    "Mut",
    "Out",
    "PointsOf",
    "Role",
    "ShapesOf",
    "SurfaceOf",
    "TracksOf",
    "VectorsOf",
]


class Role(Enum):
    """What a value *means* to a user or application.

    For example, images and labels are both arrays, but a host renders them
    differently and a user interprets them differently.

    Attach with ``Annotated[T, Role.<name>]``, or use the aliases below::

        @op
        def threshold(
            image: Annotated[np.ndarray, Role.image],
        ) -> Annotated[np.ndarray, Role.labels]: ...

    A host reads ``Role.labels`` and shows the result as a segmentation, not
    a grey image.
    """

    boxes = "boxes"
    image = "image"
    labels = "labels"
    masks = "masks"
    points = "points"
    shapes = "shapes"
    surface = "surface"
    tracks = "tracks"
    vectors = "vectors"


# The Array protocol (could be replaced with a community protocol if one exists).
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

# generic type placeholder for the array type, e.g. np.ndarray, cp.ndarray, or Array
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


# -- dtypes: what kind of number an array holds ---------------------------

# numpy's names, listed here so opspec doesn't have to import numpy.
DTYPE_NAMES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }
)


@dataclass(frozen=True, init=False)
class Dtype:
    """The dtypes an op takes for an array, by numpy name::

        Dtype("uint8", "uint16")  # either of these
        Dtype("float32")          # only float32

    Left out, any dtype. opspec only declares; converting another dtype is
    the runner's job. Ignored when the op is called directly.
    """

    names: tuple[str, ...]

    def __init__(self, *names: str) -> None:
        if not names:
            raise ValueError("Dtype needs at least one dtype, as Dtype('uint8')")
        unknown = [name for name in names if name not in DTYPE_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown dtype {unknown[0]!r}; use numpy's names, such as "
                "'uint8', 'uint16' or 'float32'"
            )
        object.__setattr__(self, "names", tuple(names))

    def __repr__(self) -> str:
        return f"Dtype({', '.join(repr(name) for name in self.names)})"


# -- directions: who allocates the arrays --------------------------------


class Direction(Enum):
    """Who allocates a parameter's array: for ``OUT`` the caller allocates it
    and the op fills it; ``MUT`` is an input the op changes in place."""

    OUT = "Out"
    MUT = "Mut"


def _annotate(item: Any, marker: Direction) -> Any:
    """``Out[X, extra]`` as ``Annotated[X, marker, extra]``."""
    parts = item if isinstance(item, tuple) else (item,)
    return Annotated[(parts[0], marker, *parts[1:])]


class Out:
    """Mark a parameter as an output buffer the caller allocates.

    ``labels: Out[np.ndarray]`` declares a computer-form op. A host doesn't
    ask the user for the buffer; it allocates it and reads it back.
    """

    def __class_getitem__(cls, item: Any) -> Any:
        return _annotate(item, Direction.OUT)


class Mut:
    """Mark a parameter as mutated in place, declaring an inplace-form op."""

    def __class_getitem__(cls, item: Any) -> Any:
        return _annotate(item, Direction.MUT)
