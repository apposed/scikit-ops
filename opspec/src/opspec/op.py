"""Declare ops, and read their signatures into specs.

Standard library only, because every worker environment imports it.
"""

from __future__ import annotations

import inspect
import math
import sys
import types as _types
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePath
from typing import (
    Annotated,
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


class Role(Enum):
    """What a value *means* to a user or application.

    For example, images and labels are both arrays, but a host renders them
    differently and a user interprets them differently.

    Attach with ``Annotated[T, Role.<name>]``::

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


def _unwrap_optional(annotation: Any) -> Any:
    """``X | None`` is X, for reading what X declares."""
    if _is_union(get_origin(annotation)):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def role_of(annotation: Any) -> Role | None:
    """Read the role off an annotation, or ``None`` if it declares no role.

    Read through ``X | None`` too: an output that may be missing still says
    what it is when it is there.
    """
    annotation = _unwrap_optional(annotation)
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Role):
                return meta
    return None


#: Axis names a viewer knows how to display. Any other string is still a
#: valid axis label.
CANONICAL = ("x", "y", "z", "c", "t")

#: Synonyms for the canonical names, from scikit-image, ImageJ, Bio-Formats,
#: CZI and OME-NGFF: ``row`` is ``y``. Other labels (``lifetime``,
#: ``batch``) pass through unchanged.
ALIASES = {
    "col": "x",
    "cols": "x",
    "column": "x",
    "columns": "x",
    "row": "y",
    "rows": "y",
    "pln": "z",
    "plane": "z",
    "planes": "z",
    "slice": "z",
    "slices": "z",
    "ch": "c",
    "chan": "c",
    "channel": "c",
    "channels": "c",
    "frame": "t",
    "frames": "t",
    "time": "t",
    "timepoint": "t",
    "timepoints": "t",
}


def canonical(label: str) -> str:
    """An axis label, lowercased and resolved: ``ROW`` -> ``y``.

    An unknown label is returned lowercased: ``"Lifetime"`` -> ``"lifetime"``.
    """
    folded = label.strip().casefold()
    return ALIASES.get(folded, folded)


WILDCARD = "*"


@dataclass(frozen=True)
class Slot:
    """One axis an op takes. ``name`` is a hint, not a requirement; None is
    a wildcard, any axis.
    """

    name: str | None
    optional: bool = False

    def __str__(self) -> str:
        return (self.name or WILDCARD) + ("?" if self.optional else "")


@dataclass(frozen=True, init=False)
class Axes:
    """How many axes an op takes, and what it calls them::

        Axes("y", "x")        # two axes, named y and x
        Axes(list("zyx"))     # three
        Axes("y", "x", "c?")  # two, plus a channel axis if there is one
        Axes("*", "*")        # two axes, no opinion which
        Axes(variadic=True)   # any number

    - Names are hints: a mismatch is a warning in the plan, not an error.
    - The number of axes is a rule: it is what the op's indexing needs.
    - ``variadic`` means the op handles extra axes itself, so they are not
      looped over.
    - Ignored when the op is called directly.
    """

    slots: tuple[Slot, ...]
    variadic: bool

    def __init__(self, *names: Any, variadic: bool = False) -> None:
        # frozen blocks normal assignment, so set fields as dataclass does.
        # Slots are stored parsed, so Axes("z", "y", "x") equals
        # Axes("pln", "row", "col").
        if len(names) == 1 and not isinstance(names[0], str):
            # A lone non-string is the sequence itself: Axes(list("zyx")).
            names = tuple(names[0])
        object.__setattr__(self, "slots", _parse_slots(names))
        object.__setattr__(self, "variadic", variadic)

    @property
    def names(self) -> tuple[str, ...]:
        """Each slot's name; ``"*"`` for a wildcard."""
        return tuple(str(slot).removesuffix("?") for slot in self.slots)

    @property
    def optional(self) -> frozenset[str]:
        """Names of the slots that need not be filled."""
        return frozenset(
            slot.name for slot in self.slots if slot.optional and slot.name
        )

    @property
    def core(self) -> tuple[str, ...]:
        """Names of the slots that must be filled."""
        return tuple(str(slot) for slot in self.slots if not slot.optional)

    def __repr__(self) -> str:
        shown = ", ".join(repr(str(slot)) for slot in self.slots)
        if self.variadic:
            shown = f"{shown}, variadic=True" if shown else "variadic=True"
        return f"Axes({shown})"


def _parse_slots(names: tuple[Any, ...]) -> tuple[Slot, ...]:
    """Check slot labels, resolve their names and split off the '?'."""
    slots: list[Slot] = []
    seen: set[str] = set()
    for label in names:
        if label == "?":
            raise ValueError(
                "A lone '?' is not an axis. Mark the axis it belongs to, as 'c?'."
            )
        if not isinstance(label, str) or not label.strip("?"):
            raise ValueError(f"Axis label {label!r} is not a non-empty string")
        if any(char.isspace() or char == "," for char in label.strip()):
            raise ValueError(
                f"Axis label {label!r} has a separator in it; pass one label "
                "per argument, as Axes('z', 'y', 'x')."
            )
        optional = label.endswith("?")
        text = label.removesuffix("?")
        if text == WILDCARD:
            if optional:
                # An optional slot is filled only by name, and a wildcard has
                # none, so '*?' could never be filled.
                raise ValueError(
                    "'*?' is not a usable slot: a wildcard has no name to match "
                    "on, and an optional slot is filled only by name. Use '*' "
                    "for an axis the op always takes, or variadic=True for a "
                    "tail of axes it may or may not be given."
                )
            # Wildcards may repeat: Axes('*', '*') is two axes of any kind.
            slots.append(Slot(None, False))
            continue
        name = canonical(text)
        if name in seen:
            raise ValueError(f"Repeated axis {name!r} in {names}")
        seen.add(name)
        slots.append(Slot(name, optional))
    return tuple(slots)


def axes_of(annotation: Any) -> Axes | None:
    """Read the ``Axes`` off an annotation, or ``None``.

    Reads through ``X | None``, as ``role_of`` does.
    """
    annotation = _unwrap_optional(annotation)
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Axes):
                return meta
    return None


# -- directions: who allocates the arrays --------------------------------


class _Direction:
    """Marks a parameter as an output buffer (``Out``) or mutated (``Mut``)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<opspec.{self.name}>"


OUT = _Direction("Out")
MUT = _Direction("Mut")

# Op computation forms, in the SciJava Ops sense.
FUNCTION = "function"  # returns a new output
COMPUTER = "computer"  # caller supplies the output buffer, op fills it
INPLACE = "inplace"  # op mutates one of its inputs


def _annotate(item: Any, marker: _Direction) -> Any:
    parts = item if isinstance(item, tuple) else (item,)
    return Annotated[(parts[0], marker, *parts[1:])]


class Out:
    """Mark a parameter as an output buffer the caller allocates.

    ``labels: Out[np.ndarray]`` declares a computer-form op. A host doesn't
    ask the user for the buffer; it allocates it and reads it back.
    """

    def __class_getitem__(cls, item: Any) -> Any:
        return _annotate(item, OUT)


class Mut:
    """Mark a parameter as mutated in place, declaring an inplace-form op."""

    def __class_getitem__(cls, item: Any) -> Any:
        return _annotate(item, MUT)


def direction_of(annotation: Any) -> _Direction | None:
    """Read the direction off an annotation, or ``None`` for an input."""
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, _Direction):
                return meta
    return None


# -- workflows: parameters that hold other ops --------------------------


@dataclass(frozen=True, init=False)
class Choices:
    """A list of the ops a workflow parameter may be filled with.

    Attached to a ``Callable`` parameter, so a GUI can offer a combo box
    instead of asking for an import path::

        psf_op: Annotated[Callable, Choices(gaussian=gaussian_psf,
                                            gibson_lanni=gibson_lanni)]

    The keywords are the menu labels: "gpu" is easier to read than
    ``richardson_lucy_cupy``.

    **The list limits the GUI, not the function**: a script may pass any op.
    That is how the list grows: someone tries an untested op in a script, it
    works, and it is added here, where the change is reviewed. The list is
    written by hand, not discovered, because it means "these are tested", not
    "these are installed".
    """

    options: tuple[tuple[str, Callable], ...]

    def __init__(self, **options: Callable) -> None:
        object.__setattr__(self, "options", tuple(options.items()))

    @property
    def labels(self) -> tuple[str, ...]:
        """The menu labels."""
        return tuple(label for label, _ in self.options)

    def op(self, label: str) -> Callable:
        """The op a label names."""
        return dict(self.options)[label]

    def label(self, fn: Callable) -> str | None:
        """The label for *fn*, or None if it isn't listed."""
        return next((label for label, op in self.options if op is fn), None)

    @property
    def ids(self) -> tuple[tuple[str, str], ...]:
        """``(label, "module:function")`` pairs: the form sent over the wire.

        A front end in another process, Fiji say, needs the menu but can't
        import the functions. A ``Choices`` rebuilt from the wire holds these
        IDs instead of functions, and ``ids`` returns them unchanged;
        ``op()`` and ``label()`` don't work on it.
        """
        return tuple(
            (label, op if isinstance(op, str) else f"{op.__module__}:{op.__name__}")
            for label, op in self.options
        )


@dataclass(frozen=True, init=False)
class ParamsFor:
    """Marks a dict parameter as holding the arguments of a chosen op::

        decon_op: Annotated[Callable, Choices(cpu=..., gpu=...)] = richardson_lucy
        decon_args: Annotated[dict, ParamsFor("decon_op",
                                              binds=("image", "psf"))] = None

    ``binds`` names the chosen op's parameters that the workflow fills
    itself, from its own inputs or an earlier step's output. A GUI shows only
    the chosen op's *other* parameters, so two steps that both take an image
    don't both ask for it.

    ``binds`` is declared, not inferred from names. Matching names would
    catch the image, but not that a mask op's ``boxes`` come from the
    detector, and half a rule is harder to explain than none.
    """

    chooser: str
    binds: tuple[str, ...]

    def __init__(self, chooser: str, *, binds: Any = ()) -> None:
        # One name as a string is common; don't bind it letter by letter.
        if isinstance(binds, str):
            binds = (binds,)
        object.__setattr__(self, "chooser", chooser)
        object.__setattr__(self, "binds", tuple(binds))


def choices_of(annotation: Any) -> Choices | None:
    """Read a ``Choices`` off an annotation, or ``None``."""
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Choices):
                return meta
    return None


def params_for_of(annotation: Any) -> ParamsFor | None:
    """Read a ``ParamsFor`` off an annotation, or ``None``."""
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, ParamsFor):
                return meta
    return None


# -- tiling: hints for cutting one call into several -----------------------
#
# All optional, and hints, not rules: a runner uses them for its defaults,
# and a caller can override any of them. See scikit-ops'
# docs/design/0017-memory-and-tiled-processing/tiling.md.

#: Bytes per element, by dtype name. opspec cannot ask numpy.
_ITEMSIZE = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def _dtype_name(dtype: Any) -> str:
    """``"float32"``, from a name, a numpy dtype or a numpy scalar type."""
    name = getattr(dtype, "name", None) if not isinstance(dtype, str) else dtype
    if not isinstance(name, str):
        name = getattr(dtype, "__name__", str(dtype))
    if name not in _ITEMSIZE:
        raise ValueError(f"Unknown dtype for memory accounting: {dtype!r}")
    return name


@dataclass(frozen=True)
class PeakMemory:
    """An op's peak memory, as a multiple of its input's size::

        PeakMemory(scale=2, dtype="float32")   # two float32 copies at once
        PeakMemory(10, "float64", pad=Overlap(param="psf", of="shape", scale=0.5))

    - ``dtype``: what the op's buffers hold; None means the input's dtype.
    - ``fixed``: bytes that don't grow with the input, such as a model's
      weights.
    - ``pad``: how far the op pads its input on each side, as an
      ``Overlap``. The multiple counts the padded size. An FFT op pads by
      part of its kernel, as in the second example.
    - ``device``: ``"cpu"`` (RAM) or ``"gpu"`` (GPU memory). The budget is
      taken from that device.
    """

    scale: float
    dtype: str | None = None
    fixed: int = 0
    pad: Overlap | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.dtype is not None:
            object.__setattr__(self, "dtype", _dtype_name(self.dtype))
        if self.device not in ("cpu", "gpu"):
            raise ValueError(f"PeakMemory device={self.device!r}: 'cpu' or 'gpu'")

    def bytes_for(self, n_elements: int, input_dtype: Any = "uint8") -> int:
        """Peak bytes for an input of *n_elements* elements."""
        itemsize = _ITEMSIZE[self.dtype or _dtype_name(input_dtype)]
        return math.ceil(self.scale * n_elements * itemsize) + self.fixed

    def to_dict(self) -> dict:
        data = {"scale": self.scale, "dtype": self.dtype, "fixed": self.fixed}
        if self.pad is not None:
            data["pad"] = self.pad.to_dict()
        if self.device != "cpu":
            data["device"] = self.device
        return data

    @classmethod
    def from_dict(cls, data: dict) -> PeakMemory:
        pad = Overlap.from_dict(data["pad"]) if data.get("pad") else None
        return cls(
            data["scale"],
            data.get("dtype"),
            data.get("fixed", 0),
            pad,
            data.get("device", "cpu"),
        )


@dataclass(frozen=True)
class Overlap:
    """How far a tile reads past its core on each side, in pixels::

        Overlap(10)                                   # always 10 pixels
        Overlap(param="sigma", scale=4)               # 4 x sigma
        Overlap(param="psf", of="shape", scale=0.5)   # half the PSF, per axis
        Overlap(param="psf", of="shape", scale=0.5,
                only_if="noncirc", otherwise=10)      # 10 when noncirc is off

    A formula is data, not code, so any front end can evaluate it.
    ``pixels`` is added to it. With ``only_if``, the formula applies when that
    parameter is true, and ``otherwise`` pixels when it is false.

    With ``of="shape"`` the parameter is an array, and the overlap is per
    axis, from its shape, rounded down: half a kernel of 31 is 15, exactly
    how far it reaches. Otherwise the parameter is a number, or one per axis
    (the largest is used), and the overlap is rounded up.
    """

    pixels: int = 0
    param: str | None = None
    scale: float = 1.0
    of: str | None = None
    only_if: str | None = None
    otherwise: int = 0

    def resolve(self, values: dict) -> int | tuple[int, ...]:
        """The overlap for a call with these argument *values*.

        One number, the same on every axis, or one per axis for ``of="shape"``.
        """
        if self.only_if is not None and not values.get(self.only_if):
            return self.otherwise
        if self.param is None:
            return self.pixels
        value = values.get(self.param)
        if value is None:
            raise ValueError(f"Overlap needs {self.param}, and the call has none")
        if self.of == "shape":
            return tuple(self.pixels + math.floor(self.scale * n) for n in value.shape)
        if isinstance(value, (list, tuple)):
            value = max(value)
        return self.pixels + math.ceil(self.scale * float(value))

    def to_dict(self) -> dict:
        data = {"pixels": self.pixels, "param": self.param, "scale": self.scale}
        if self.of is not None:
            data["of"] = self.of
        if self.only_if is not None:
            data["only_if"] = self.only_if
            data["otherwise"] = self.otherwise
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Overlap:
        return cls(
            data.get("pixels", 0),
            data.get("param"),
            data.get("scale", 1.0),
            data.get("of"),
            data.get("only_if"),
            data.get("otherwise", 0),
        )

    def __post_init__(self) -> None:
        if self.of not in (None, "shape"):
            raise ValueError(f"Overlap of={self.of!r}: only 'shape' is known")


@dataclass(frozen=True)
class _OpConfig:
    env: str | None
    main_thread: bool = False
    exclusive: bool = False
    tile: tuple[str, ...] = ()
    overlap: Overlap | None = None
    peak_memory: PeakMemory | None = None
    merge: str | None = None
    split: tuple[str, ...] = ()


def op(
    fn: Callable | None = None,
    *,
    env: str | None = None,
    main_thread: bool = False,
    exclusive: bool = False,
    tile: str | tuple[str, ...] | None = None,
    overlap: Overlap | int | None = None,
    peak_memory: PeakMemory | None = None,
    merge: str | None = None,
    split: tuple[str, ...] | None = None,
) -> Callable:
    """Declare a function as an op.

    Sets an attribute on the function and returns it unchanged, so calling
    it directly still works. Use it bare or with arguments, like
    ``@dataclass``::

        @op
        def smooth(image: ImageOf[np.ndarray]) -> ImageOf[np.ndarray]: ...

        @op(env="cupy")
        def deconvolve(image: ImageOf[cp.ndarray]) -> ImageOf[cp.ndarray]: ...

    Args:
        env: The environment the op runs in. Without one, the op runs
            wherever the caller is; a workflow (an op that calls other ops)
            is like this.
        main_thread: Whether the op must run on its worker's main thread.
        exclusive: Whether the op needs a worker to itself, instead of
            sharing one with other ops in the same environment.
        tile: The parameter or parameters a runner may cut into tiles to
            fit memory. All are tiled the same way.
        overlap: How far a tile reads past its core: an ``Overlap``, or a
            number of pixels.
        peak_memory: The op's peak memory, as a ``PeakMemory``. A runner
            sizes tiles from it.
        merge: How tiles are put back together. ``"crop"`` keeps each
            tile's core and drops the overlap. ``"blend"`` fades each tile
            out across the overlap, so neighbours mix where they meet.
        split: The axes a tile may be cut along, named as in the tiled
            input's ``Axes``. ``("y", "x")`` keeps z whole, as decon wants
            with a PSF long in z. Left out, any axis may be cut.
    """

    def decorate(f: Callable) -> Callable:
        f.__opspec__ = _OpConfig(
            env=env,
            main_thread=main_thread,
            exclusive=exclusive,
            tile=(tile,) if isinstance(tile, str) else tuple(tile or ()),
            overlap=Overlap(overlap) if isinstance(overlap, int) else overlap,
            peak_memory=peak_memory,
            merge=merge,
            split=tuple(split or ()),
        )
        return f

    return decorate(fn) if fn is not None else decorate


def is_op(obj: Any) -> bool:
    """Whether ``obj`` carries the ``@op`` decorator."""
    return callable(obj) and isinstance(getattr(obj, "__opspec__", None), _OpConfig)


def _ui_hints(annotation: Any) -> dict:
    """Widget hints for a host: the dicts in an annotation's metadata::

        sigma: Annotated[float, {"min": 0.1, "max": 10.0}] = 2.0

    Several dicts merge, left to right. opspec doesn't interpret the keys; a
    host uses the ones it knows.
    """
    hints: dict = {}
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, dict):
                hints.update(meta)
    return hints


def _strip(annotation: Any) -> Any:
    """The underlying type of a possibly-``Annotated`` annotation."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


# -- the wire vocabulary ------------------------------------------------
#
# Types sent to another process are names, not Python objects: a Java front
# end can't receive ``<class 'numpy.ndarray'>``. These are the names a
# generated dialog can render, plus UNKNOWN.

INT = "int"
FLOAT = "float"
STR = "str"
BOOL = "bool"
NDARRAY = "ndarray"
PATH = "path"
ENUM = "enum"
UNKNOWN = "unknown"

WIRE_TYPES = (INT, FLOAT, STR, BOOL, NDARRAY, PATH, ENUM, UNKNOWN)


@dataclass(frozen=True)
class Choice:
    """One member of an enum parameter: what to show, what to send."""

    name: str
    value: Any

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> Choice:
        return cls(name=data["name"], value=data["value"])


@dataclass(frozen=True)
class TypeSpec:
    """A parameter's type, as a wire name a front end understands.

    ``UNKNOWN`` is not an error: a front end that can't show a parameter
    leaves it at its default and says why, from ``detail``.
    """

    name: str
    choices: tuple[Choice, ...] = ()
    nullable: bool = False
    detail: str | None = None

    def to_dict(self) -> dict:
        data: dict = {"name": self.name}
        if self.choices:
            data["choices"] = [c.to_dict() for c in self.choices]
        if self.nullable:
            data["nullable"] = True
        if self.detail is not None:
            data["detail"] = self.detail
        return data

    @classmethod
    def from_dict(cls, data: dict) -> TypeSpec:
        return cls(
            name=data["name"],
            choices=tuple(Choice.from_dict(c) for c in data.get("choices", ())),
            nullable=bool(data.get("nullable", False)),
            detail=data.get("detail"),
        )


def _is_ndarray(annotation: Any) -> bool:
    """Recognize ``numpy.ndarray`` without importing numpy."""
    return (
        isinstance(annotation, type)
        and annotation.__name__ == "ndarray"
        and annotation.__module__.split(".")[0] == "numpy"
    )


def _spelling(annotation: Any) -> str:
    """An annotation's name, for a message."""
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _is_union(origin: Any) -> bool:
    """Whether an origin is a union: ``Union[X, Y]`` or ``X | Y``."""
    if origin is Union:
        return True
    union_type = getattr(_types, "UnionType", None)  # 3.10+: X | Y
    return union_type is not None and origin is union_type


def type_spec(annotation: Any) -> TypeSpec:
    """Classify a type annotation into the wire vocabulary."""
    if isinstance(annotation, TypeSpec):
        # Already classified: came off the wire, not off a live function.
        return annotation
    annotation = _strip(annotation)

    origin = get_origin(annotation)
    if origin is not None and _is_union(origin):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            inner = type_spec(args[0])  # Optional[X] is X, nullable
            return TypeSpec(inner.name, inner.choices, True, inner.detail)
        return TypeSpec(UNKNOWN, detail=_spelling(annotation))

    if annotation in (None, type(None), inspect.Parameter.empty):
        return TypeSpec(UNKNOWN, detail="unannotated")
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return TypeSpec(
            ENUM,
            choices=tuple(Choice(m.name, m.value) for m in annotation),
            detail=annotation.__name__,
        )
    # bool before int: bool subclasses int, and a checkbox is not a number.
    if annotation is bool:
        return TypeSpec(BOOL)
    if annotation is int:
        return TypeSpec(INT)
    if annotation is float:
        return TypeSpec(FLOAT)
    if annotation is str:
        return TypeSpec(STR)
    if isinstance(annotation, type) and issubclass(annotation, PurePath):
        return TypeSpec(PATH)
    if _is_ndarray(annotation):
        return TypeSpec(NDARRAY)
    return TypeSpec(UNKNOWN, detail=_spelling(annotation))


def _wire_default(value: Any) -> Any:
    """A default value JSON can carry; None if it can't."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_wire_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _wire_default(item) for key, item in value.items()}
    return None


def _axes_dict(axes: Axes) -> dict:
    return {
        "slots": [{"name": s.name, "optional": s.optional} for s in axes.slots],
        "variadic": axes.variadic,
    }


def _axes_from_dict(data: dict) -> Axes:
    result = Axes(variadic=bool(data.get("variadic", False)))
    slots = tuple(
        Slot(s.get("name"), bool(s.get("optional", False)))
        for s in data.get("slots", ())
    )
    object.__setattr__(result, "slots", slots)
    return result


@dataclass(frozen=True)
class ParamSpec:
    """One parameter of an op, as declared."""

    name: str
    type: Any
    default: Any
    role: Role | None = None
    axes: Axes | None = None
    ui: dict = field(default_factory=dict)
    #: None for an input, OUT for a caller-allocated buffer, MUT for one the
    #: op modifies in place.
    direction: _Direction | None = None
    #: The ops this parameter may be filled with, if it is a chooser.
    choices: Choices | None = None
    #: Which chooser's arguments this parameter carries, if any.
    params_for: ParamsFor | None = None

    @property
    def required(self) -> bool:
        return self.default is inspect.Parameter.empty

    def to_dict(self) -> dict:
        data: dict = {
            "name": self.name,
            "type": type_spec(self.type).to_dict(),
            "required": self.required,
        }
        if not self.required:
            data["default"] = _wire_default(self.default)
        if self.role is not None:
            data["role"] = self.role.value
        if self.axes is not None:
            data["axes"] = _axes_dict(self.axes)
        if self.ui:
            data["ui"] = dict(self.ui)
        if self.direction is not None:
            data["direction"] = self.direction.name
        if self.choices is not None:
            # IDs, not functions: another process can't import the functions.
            data["choices"] = [
                {"label": label, "op": op_id} for label, op_id in self.choices.ids
            ]
        if self.params_for is not None:
            data["params_for"] = {
                "chooser": self.params_for.chooser,
                "binds": list(self.params_for.binds),
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ParamSpec:
        """Rebuild from the wire form.

        ``type`` comes back as a ``TypeSpec``, not the Python type, which may
        not exist in the reading process.
        """
        return cls(
            name=data["name"],
            type=TypeSpec.from_dict(data["type"]),
            default=(
                inspect.Parameter.empty
                if data.get("required", False)
                else data.get("default")
            ),
            role=Role(data["role"]) if data.get("role") else None,
            axes=_axes_from_dict(data["axes"]) if data.get("axes") else None,
            ui=dict(data.get("ui", {})),
            direction={"Out": OUT, "Mut": MUT}.get(data.get("direction", "")),
            choices=_choices_from_dict(data.get("choices")),
            params_for=(
                ParamsFor(
                    data["params_for"]["chooser"],
                    binds=tuple(data["params_for"].get("binds", ())),
                )
                if data.get("params_for")
                else None
            ),
        )


def _choices_from_dict(data: list | None) -> Choices | None:
    """Rebuild a Choices from the wire form.

    The options hold op IDs, not functions, since the reading process can't
    import them. Only ``Choices.ids`` works on the result; ``op()`` and
    ``label()`` need the functions.
    """
    if not data:
        return None
    result = Choices()
    object.__setattr__(
        result,
        "options",
        tuple((entry["label"], entry["op"]) for entry in data),
    )
    return result


@dataclass(frozen=True)
class OutputSpec:
    """One of an op's outputs: its name, type and role."""

    name: str
    type: Any
    role: Role | None = None

    def to_dict(self) -> dict:
        data: dict = {"name": self.name, "type": type_spec(self.type).to_dict()}
        if self.role is not None:
            data["role"] = self.role.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> OutputSpec:
        return cls(
            name=data["name"],
            type=TypeSpec.from_dict(data["type"]),
            role=Role(data["role"]) if data.get("role") else None,
        )


class _Unbound:
    """A function's annotations, without its defaults.

    Before Python 3.11, ``get_type_hints`` turns the annotation of a
    parameter defaulting to ``None`` into ``Optional[...]``. Building that
    Union hashes the ``Annotated`` metadata, and UI hints are dicts, which
    can't be hashed. So on 3.10, an op with a slider hint and a ``None``
    default couldn't be read.

    This object has no ``__code__``, so ``get_type_hints`` finds no defaults
    and rewrites nothing, on any Python version.
    """

    def __init__(self, fn: Callable) -> None:
        fn = inspect.unwrap(fn)
        self.__annotations__ = getattr(fn, "__annotations__", {})
        self.__globals__ = getattr(fn, "__globals__", {})


def _resolve_hints(fn: Callable) -> dict:
    """Evaluate a function's annotations, which may be strings.

    ``from __future__ import annotations`` turns every annotation into a
    string, and so does running a file with ``exec`` from a module that
    imports it. Uses ``get_type_hints``, since
    ``inspect.signature(eval_str=True)`` needs Python 3.10.
    """
    try:
        return get_type_hints(_Unbound(fn), include_extras=True)
    except Exception as exc:
        raise TypeError(
            f"Could not resolve the annotations of op {fn.__qualname__} "
            f"under Python {sys.version_info.major}.{sys.version_info.minor}: "
            f"{type(exc).__name__}: {exc}\n"
            "An op's annotations are evaluated in the environment it runs in, "
            "so they must be valid there -- note that 'X | Y' unions need "
            "Python 3.10."
        ) from exc


def _outputs_of(
    return_type: Any, return_role: Role | None, fn: Callable | None = None
) -> tuple[OutputSpec, ...]:
    """An op's outputs; a NamedTuple return is one output per field.

    Field annotations may be strings. They are resolved first in the
    NamedTuple's own module, which may not be the op's (several ops can share
    a result type), then in *fn*'s globals, for a file run with ``exec``. If
    both fail, the raw annotations are used, and the roles are lost.
    """
    if return_type in (None, type(None), inspect.Parameter.empty):
        return ()
    fields = getattr(return_type, "_fields", None)
    if fields is None:
        return (OutputSpec("result", return_type, return_role),)
    hints = None
    for namespace in (None, getattr(fn, "__globals__", None)):
        try:
            hints = get_type_hints(return_type, globalns=namespace, include_extras=True)
            break
        except Exception:  # noqa: BLE001, S112 - a failure here just costs a role.
            continue
    if hints is None:
        hints = dict(getattr(return_type, "__annotations__", {}))
    return tuple(
        OutputSpec(name, _strip(hints.get(name)), role_of(hints.get(name)))
        for name in fields
    )


@dataclass(frozen=True)
class OpSpec:
    """A class to convert an op signature into data::

        spec = OpSpec.from_op(threshold)
        spec.return_role        # <Role.labels: 'labels'>

    ``to_dict`` and ``from_dict`` carry it to another process or language.
    """

    name: str
    module: str
    function: str
    env: str | None
    params: tuple[ParamSpec, ...]
    return_type: Any
    return_role: Role | None
    doc: str | None
    #: How the op computes: FUNCTION, COMPUTER or INPLACE.
    form: str = FUNCTION
    #: Whether the op must run on its worker's main thread.
    main_thread: bool = False
    #: Whether the op needs a worker to itself.
    exclusive: bool = False
    #: The parameters a runner may cut into tiles, and its hints for doing so.
    tile: tuple[str, ...] = ()
    overlap: Overlap | None = None
    peak_memory: PeakMemory | None = None
    merge: str | None = None
    split: tuple[str, ...] = ()

    #: The outputs, as ``from_op`` worked them out or as read from the wire.
    #: Off the wire the return type is only a name, so they can't be derived.
    _outputs: tuple[OutputSpec, ...] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_workflow(self) -> bool:
        """Whether the op has no environment, and runs where the caller is.

        A workflow is one: the ops it calls bring their own environments.
        """
        return self.env is None

    @property
    def inputs(self) -> tuple[ParamSpec, ...]:
        """The parameters a caller supplies: all but output buffers."""
        return tuple(p for p in self.params if p.direction is not OUT)

    @property
    def outputs(self) -> tuple[OutputSpec, ...]:
        """The op's outputs.

        For a computer- or inplace-form op, its buffers. Otherwise one per
        field of a NamedTuple return, or one named "result".
        """
        if self._outputs is not None:
            return self._outputs
        for marker in (OUT, MUT):
            buffers = tuple(p for p in self.params if p.direction is marker)
            if buffers:
                return tuple(OutputSpec(p.name, p.type, p.role) for p in buffers)
        return _outputs_of(self.return_type, self.return_role)

    def to_dict(self) -> dict:
        """A JSON-safe dict, for sending to another process or language.

        ``outputs`` is included because deriving it needs the live return
        type, which isn't sent.
        """
        return {
            "name": self.name,
            "module": self.module,
            "function": self.function,
            "env": self.env,
            "main_thread": self.main_thread,
            "exclusive": self.exclusive,
            "form": self.form,
            "params": [p.to_dict() for p in self.params],
            "return_type": type_spec(self.return_type).to_dict(),
            "return_role": self.return_role.value if self.return_role else None,
            "outputs": [o.to_dict() for o in self.outputs],
            "doc": self.doc,
            # Tiling hints only when declared, so older readers see no change.
            **({"tile": list(self.tile)} if self.tile else {}),
            **({"overlap": self.overlap.to_dict()} if self.overlap else {}),
            **({"peak_memory": self.peak_memory.to_dict()} if self.peak_memory else {}),
            **({"merge": self.merge} if self.merge else {}),
            **({"split": list(self.split)} if self.split else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> OpSpec:
        """Rebuild from the wire form.

        Types come back as ``TypeSpec``s, not the Python types they were read
        off; see ``ParamSpec.from_dict``.
        """
        return cls(
            name=data["name"],
            module=data["module"],
            function=data["function"],
            env=data.get("env"),
            params=tuple(ParamSpec.from_dict(p) for p in data["params"]),
            return_type=TypeSpec.from_dict(data["return_type"]),
            return_role=(
                Role(data["return_role"]) if data.get("return_role") else None
            ),
            doc=data.get("doc"),
            form=data.get("form", FUNCTION),
            main_thread=bool(data.get("main_thread", False)),
            exclusive=bool(data.get("exclusive", False)),
            tile=tuple(data.get("tile", ())),
            overlap=Overlap.from_dict(data["overlap"]) if data.get("overlap") else None,
            peak_memory=(
                PeakMemory.from_dict(data["peak_memory"])
                if data.get("peak_memory")
                else None
            ),
            merge=data.get("merge"),
            split=tuple(data.get("split", ())),
            _outputs=tuple(OutputSpec.from_dict(o) for o in data.get("outputs", ())),
        )

    @classmethod
    def from_op(cls, fn: Callable) -> OpSpec:
        """Read the spec of a decorated op.

        Annotations are resolved here, not in ``@op``, so an op may use types
        defined later in its module. The spec is cached on the function.
        """
        cached = getattr(fn, "__opspec_spec__", None)
        if cached is not None:
            return cached

        config = getattr(fn, "__opspec__", None)
        if not isinstance(config, _OpConfig):
            raise TypeError(f"Not an op: {fn!r} (missing @op decorator)")

        signature = inspect.signature(fn)
        hints = _resolve_hints(fn)

        params = []
        for name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(
                    f"Op {fn.__qualname__} may not declare *args or **kwargs: {name}"
                )
            annotation = hints.get(name, param.annotation)
            params.append(
                ParamSpec(
                    name=name,
                    type=_strip(annotation),
                    default=param.default,
                    role=role_of(annotation),
                    axes=axes_of(annotation),
                    ui=_ui_hints(annotation),
                    direction=direction_of(annotation),
                    choices=choices_of(annotation),
                    params_for=params_for_of(annotation),
                )
            )

        names = {p.name for p in params}
        unknown = [name for name in config.tile if name not in names]
        pad = config.peak_memory.pad if config.peak_memory else None
        for formula in (config.overlap, pad):
            for name in (formula.param, formula.only_if) if formula else ():
                if name and name not in names:
                    unknown.append(name)
        if unknown:
            raise TypeError(
                f"Op {fn.__qualname__}'s tiling hints name no such parameter: "
                f"{', '.join(unknown)}"
            )
        if config.split and config.tile:
            tiled = next(p for p in params if p.name == config.tile[0])
            named = tiled.axes.names if tiled.axes and not tiled.axes.variadic else ()
            stray = [n for n in config.split if named and n not in named]
            if stray:
                raise TypeError(
                    f"Op {fn.__qualname__} splits along {', '.join(stray)}, which "
                    f"{tiled.name}'s Axes{tuple(named)} does not name"
                )

        directions = {p.direction for p in params}
        if OUT in directions and MUT in directions:
            raise TypeError(
                f"Op {fn.__qualname__} mixes Out and Mut params; pick one form"
            )
        form = (
            COMPUTER
            if OUT in directions
            else INPLACE
            if MUT in directions
            else FUNCTION
        )

        returns = hints.get("return", signature.return_annotation)
        # A computer- or inplace-form op names its outputs with its buffers;
        # everything else takes them from the return annotation.
        buffers = [p for p in params if p.direction is OUT] or [
            p for p in params if p.direction is MUT
        ]
        outputs = (
            tuple(OutputSpec(p.name, p.type, p.role) for p in buffers)
            if buffers
            else _outputs_of(_strip(returns), role_of(returns), fn)
        )
        # A function defined by exec has no module, so call it __script__.
        module = fn.__module__ or "__script__"
        result = cls(
            name=f"{module}:{fn.__name__}",
            module=module,
            function=fn.__name__,
            env=config.env,
            params=tuple(params),
            return_type=_strip(returns),
            return_role=role_of(returns),
            doc=inspect.getdoc(fn),
            form=form,
            main_thread=config.main_thread,
            exclusive=config.exclusive,
            tile=config.tile,
            overlap=config.overlap,
            peak_memory=config.peak_memory,
            merge=config.merge,
            split=config.split,
            _outputs=outputs,
        )
        fn.__opspec_spec__ = result
        return result
