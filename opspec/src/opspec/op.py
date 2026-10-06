"""Op declaration and signature introspection.

Standard library only: this module is imported inside every worker
environment, so it must depend on nothing heavier.
"""

from __future__ import annotations

import inspect
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
    """What a value *means*

    Attach with ``Annotated[T, Role.<name>]``::

        @op
        def threshold(
            image: Annotated[np.ndarray, Role.image],
        ) -> Annotated[np.ndarray, Role.labels]: ...

    A host reads ``Role.labels`` and shows the result as a segmentation
    rather than a grey picture.
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


def role_of(annotation: Any) -> Role | None:
    """Read the role off an annotation, or ``None`` if it declares no role."""
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Role):
                return meta
    return None


#: Axis names a viewer can map onto display semantics. Privileged, not
#: exclusive: any string is a valid axis label.
CANONICAL = ("x", "y", "z", "c", "t")

#: Synonyms, resolved by lookup rather than guessed: ``row`` *is* ``y``.
#: From scikit-image, ImageJ, Bio-Formats, CZI and OME-NGFF. Labels with
#: no canonical equivalent (``lifetime``, ``batch``) pass through.
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
    """One axis label, case ignored and resolved: ``ROW`` -> ``y``.

    Unrecognized labels pass through lowercased, so ``"lifetime"`` survives.
    """
    folded = label.strip().casefold()
    return ALIASES.get(folded, folded)


WILDCARD = "*"


@dataclass(frozen=True)
class Slot:
    """One axis an op consumes. ``name`` is a hint, not a requirement.

    ``name`` None is a wildcard: no preference at all.
    """

    name: str | None
    optional: bool = False

    def __str__(self) -> str:
        return (self.name or WILDCARD) + ("?" if self.optional else "")


@dataclass(frozen=True, init=False)
class Axes:
    """How many axes an op consumes, and what it likes to call them::

        Axes("y", "x")        # two axes, named y and x
        Axes(list("zyx"))     # three
        Axes("y", "x", "c?")  # two, plus a channel axis if there is one
        Axes("*", "*")        # two axes, no opinion which
        Axes(variadic=True)   # any number

    - Names are hints. A mismatch is reported in the plan, never refused.
    - Arity binds: how many axes the op consumes is what its indexing needs.
    - ``variadic`` means the op handles extra axes itself, so they need
      not be looped over.
    - Inert at runtime. Calling the op directly ignores all of this.
    """

    slots: tuple[Slot, ...]
    variadic: bool

    def __init__(self, *names: Any, variadic: bool = False) -> None:
        # Note: frozen blocks ordinary assignment, so a hand-written __init__
        # has to set fields the way dataclass itself does. Storing the parsed
        # slots rather than the raw text is what makes Axes("z", "y", "x") and
        # Axes("pln", "row", "col") compare equal, as they should.
        if len(names) == 1 and not isinstance(names[0], str):
            # A lone non-string is the sequence itself: Axes(list("zyx")).
            names = tuple(names[0])
        object.__setattr__(self, "slots", _parse_slots(names))
        object.__setattr__(self, "variadic", variadic)

    @property
    def names(self) -> tuple[str, ...]:
        """Each slot's preferred name, with ``"*"`` standing in for a wildcard."""
        return tuple(str(slot).removesuffix("?") for slot in self.slots)

    @property
    def optional(self) -> frozenset[str]:
        """Names of the slots that need not be filled."""
        return frozenset(
            slot.name for slot in self.slots if slot.optional and slot.name
        )

    @property
    def core(self) -> tuple[str, ...]:
        """Preferred names of the slots that must be filled."""
        return tuple(str(slot) for slot in self.slots if not slot.optional)

    def __repr__(self) -> str:
        shown = ", ".join(repr(str(slot)) for slot in self.slots)
        if self.variadic:
            shown = f"{shown}, variadic=True" if shown else "variadic=True"
        return f"Axes({shown})"


def _parse_slots(names: tuple[Any, ...]) -> tuple[Slot, ...]:
    """Validate slot spellings, resolving names and splitting off the '?'."""
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
                # A wildcard has no name, and an optional slot is filled only
                # by a name match, so '*?' could never be filled by anything.
                raise ValueError(
                    "'*?' is not a usable slot: a wildcard has no name to match "
                    "on, and an optional slot is filled only by name. Use '*' "
                    "for an axis the op always takes, or variadic=True for a "
                    "tail of axes it may or may not be given."
                )
            # Wildcards are exempt from the repeat check: Axes('*', '*') is
            # two axes the op has no opinion about, which is the whole point.
            slots.append(Slot(None, False))
            continue
        name = canonical(text)
        if name in seen:
            raise ValueError(f"Repeated axis {name!r} in {names}")
        seen.add(name)
        slots.append(Slot(name, optional))
    return tuple(slots)


def axes_of(annotation: Any) -> Axes | None:
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Axes):
                return meta
    return None


# -- directions: who allocates the arrays --------------------------------


class _Direction:
    """Marker distinguishing input, output-buffer and mutated-buffer params."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<opspec.{self.name}>"


OUT = _Direction("Out")
MUT = _Direction("Mut")

# Op computation forms, in the SciJava Ops sense.
FUNCTION = "function"  # inputs in, freshly allocated output out
COMPUTER = "computer"  # caller supplies the output buffer, op fills it
INPLACE = "inplace"  # op mutates one of its inputs


def _annotate(item: Any, marker: _Direction) -> Any:
    parts = item if isinstance(item, tuple) else (item,)
    return Annotated[(parts[0], marker, *parts[1:])]


class Out:
    """Mark a parameter as a caller-allocated output buffer.

    ``labels: Out[np.ndarray]`` declares a computer-form op. A host does not
    ask a user for an output buffer; it allocates one and reads it back.
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
    """A curated list of the ops a parameter may be filled with.

    Attached to a ``Callable`` parameter of a workflow, so that a front end can
    offer a combo box rather than asking someone to type an import path::

        psf_op: Annotated[Callable, Choices(gaussian=gaussian_psf,
                                            gibson_lanni=gibson_lanni)]

    The keyword names are the menu labels: "gpu" is a better thing to show a
    researcher than ``richardson_lucy_cupy``.

    **The list constrains the GUI, not the function.** Passing an op that is
    not in it stays legal, because that is how the list grows -- someone tries
    an untested solver in a script, it works, and it gets added here where the
    change can be reviewed. Curated rather than discovered for the same reason:
    a list means "I have tested these", where an inventory means only "these
    are installed".
    """

    options: tuple[tuple[str, Callable], ...]

    def __init__(self, **options: Callable) -> None:
        object.__setattr__(self, "options", tuple(options.items()))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.options)

    def op(self, label: str) -> Callable:
        """The op a label names."""
        return dict(self.options)[label]

    def label(self, fn: Callable) -> str | None:
        """What this list calls *fn*, if it lists it at all."""
        return next((label for label, op in self.options if op is fn), None)

    @property
    def ids(self) -> tuple[tuple[str, str], ...]:
        """``(label, "module:function")`` pairs.

        The view that survives going over a wire: a Fiji front end needs the
        menu without needing the Python objects behind it. So it also has to
        survive coming *back* -- a ``Choices`` rebuilt from its wire form holds
        the IDs themselves, since the functions they name are not importable in
        the process that read them, and an ID passed through unchanged is what
        makes this property the fixed point it claims to be. ``op()`` and
        ``label()`` want live objects and are unavailable on such a one.
        """
        return tuple(
            (label, op if isinstance(op, str) else f"{op.__module__}:{op.__name__}")
            for label, op in self.options
        )


@dataclass(frozen=True, init=False)
class ParamsFor:
    """Marks a parameter as holding the arguments of a chosen op.

    A chooser needs somewhere to put the chosen op's own settings, and a plain
    dict is that somewhere::

        decon_op: Annotated[Callable, Choices(cpu=..., gpu=...)] = richardson_lucy
        decon_args: Annotated[dict, ParamsFor("decon_op",
                                              binds=("image", "psf"))] = None

    ``binds`` names the sub-op parameters the workflow supplies itself, from
    its own inputs or from an earlier stage's output. A front end renders every
    *other* parameter of the chosen op and leaves these alone -- which is what
    stops two stages that both take an image from asking for it twice.

    It is declared rather than inferred. Matching on name would hide the image
    for free but still not know that a mask generator's ``boxes`` come from the
    detector, and a rule that covers half the cases is harder to explain than
    no rule at all.
    """

    chooser: str
    binds: tuple[str, ...]

    def __init__(self, chooser: str, *, binds: Any = ()) -> None:
        # A lone string is the common case and iterating it would bind one
        # parameter per letter, so take it as the single name it obviously is.
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


@dataclass(frozen=True)
class _OpConfig:
    env: str | None
    main_thread: bool = False
    exclusive: bool = False


def op(
    fn: Callable | None = None,
    *,
    env: str | None = None,
    main_thread: bool = False,
    exclusive: bool = False,
) -> Callable:
    """Declare a function as an op.

    Sets an attribute on the function and returns the same function, so
    calling it directly is unaffected. Bare or called, like ``@dataclass``::

        @op
        def smooth(image: ImageOf[np.ndarray]) -> ImageOf[np.ndarray]: ...

        @op(env="cupy")
        def deconvolve(image: ImageOf[cp.ndarray]) -> ImageOf[cp.ndarray]: ...

    Args:
        env: Environment the op runs in. Omit it and the op has none: it
            runs wherever the caller is. A workflow, an op that calls other
            ops, is one of these.
        main_thread: Whether the op must run on its worker's main thread.
        exclusive: Whether the op needs a worker to itself, rather than
            sharing one with other ops in the same environment.
    """

    def decorate(f: Callable) -> Callable:
        f.__opspec__ = _OpConfig(env=env, main_thread=main_thread, exclusive=exclusive)
        return f

    return decorate(fn) if fn is not None else decorate


def is_op(obj: Any) -> bool:
    """Whether ``obj`` carries the ``@op`` decorator."""
    return callable(obj) and isinstance(getattr(obj, "__opspec__", None), _OpConfig)


def _ui_hints(annotation: Any) -> dict:
    """Collect dict metadata off an annotation: widget hints for a host::

        sigma: Annotated[float, {"min": 0.1, "max": 10.0}] = 2.0

    Several dicts merge, left to right. opspec does not interpret the keys;
    a host reads the ones it knows.
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
# Out of process, a type cannot be a Python object: a Java front end has no
# way to receive ``<class 'numpy.ndarray'>``, only a name for it. These are
# the names -- the set a generated dialog can render, plus UNKNOWN.

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
    """A type in the vocabulary a front end can act on.

    ``UNKNOWN`` is not a failure: a front end that cannot render one
    parameter leaves it at its default and says why, using ``detail``.
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
    """How an annotation is best named in a message to a human."""
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _is_union(origin: Any) -> bool:
    """Whether an origin is a union, spelled either way."""
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
            inner = type_spec(args[0])  # Optional[X] is X, and may be empty
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
    """A default value in a form JSON can carry."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_wire_default(item) for item in value]
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
            # The ids view, not the options: a front end in another language
            # needs the menu without needing the Python objects behind it.
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

        ``type`` comes back as a ``TypeSpec``, not the live Python type: that
        type does not exist in the process doing the reading, which is the
        whole reason for the wire vocabulary.
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
    """Rebuild a Choices from its wire form.

    The options come back as op *IDs* rather than functions, because the
    functions are not importable in the process doing the reading -- that is
    the whole reason the wire form exists. ``Choices.ids`` is therefore the
    only view that survives the round trip, and it is the one a front end
    uses; ``op()`` and ``label()`` want live objects and do not.
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
    """One of an op's outputs, as a front end needs to see it."""

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
    """A function's annotations, detached from its defaults.

    Before Python 3.11, ``get_type_hints`` rewrites the annotation of any
    parameter defaulting to ``None`` as ``Optional[<annotation>]``. Building
    that Union deduplicates its members through a set, which hashes the
    ``Annotated`` alias, which hashes its metadata -- and UI hints are dicts,
    which are unhashable. So an op with a slider hint and a ``None`` default
    could not be read at all on 3.10.

    Passing the annotations through an object with no ``__code__`` denies
    ``get_type_hints`` any defaults to find, which suppresses the rewrite and
    gives every Python version the 3.11+ reading of the signature.
    """

    def __init__(self, fn: Callable) -> None:
        fn = inspect.unwrap(fn)
        self.__annotations__ = getattr(fn, "__annotations__", {})
        self.__globals__ = getattr(fn, "__globals__", {})


def _resolve_hints(fn: Callable) -> dict:
    """Evaluate a function's annotations, which may be strings.

    ``from __future__ import annotations`` turns every annotation into a
    string, and so does running a file with ``exec`` from a module that
    imports it. Resolution happens through ``get_type_hints`` rather than
    ``inspect.signature(eval_str=True)``, which needs Python 3.10.
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
    """An op's outputs. A NamedTuple return is one output per field.

    Annotations are resolved against *fn*'s globals, because a file run with
    ``exec`` inherits the caller's ``from __future__ import annotations`` --
    which turns every annotation in it into a string.
    """
    if return_type in (None, type(None), inspect.Parameter.empty):
        return ()
    fields = getattr(return_type, "_fields", None)
    if fields is None:
        return (OutputSpec("result", return_type, return_role),)
    namespace = getattr(fn, "__globals__", None)
    try:
        hints = get_type_hints(return_type, globalns=namespace, include_extras=True)
    except Exception:  # noqa: BLE001 - any failure here just costs a role.
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

    #: Set only when rebuilt from the wire, where the return type is a name
    #: rather than the live type the property below derives outputs from.
    _outputs: tuple[OutputSpec, ...] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_workflow(self) -> bool:
        """Whether this op has no environment of its own.

        It runs where the caller is. A workflow, an op that calls other ops,
        is one: the ops it calls each bring their own environment.
        """
        return self.env is None

    @property
    def inputs(self) -> tuple[ParamSpec, ...]:
        """The parameters a caller supplies. Output buffers are not asked for."""
        return tuple(p for p in self.params if p.direction is not OUT)

    @property
    def outputs(self) -> tuple[OutputSpec, ...]:
        """This op's outputs, named.

        A computer- or inplace-form op names its buffers. Otherwise a
        NamedTuple return is one output each, and anything else is "result".
        """
        if self._outputs is not None:
            return self._outputs
        for marker in (OUT, MUT):
            buffers = tuple(p for p in self.params if p.direction is marker)
            if buffers:
                return tuple(OutputSpec(p.name, p.type, p.role) for p in buffers)
        return _outputs_of(self.return_type, self.return_role)

    def to_dict(self) -> dict:
        """A JSON-safe form, for the trip to another process or language.

        ``outputs`` is written out rather than left to be derived, because
        deriving it needs the live return type -- a NamedTuple's fields --
        which does not cross the boundary.
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
            _outputs=tuple(OutputSpec.from_dict(o) for o in data.get("outputs", ())),
        )

    @classmethod
    def from_op(cls, fn: Callable) -> OpSpec:
        """Read the spec off a decorated op.

        Annotations are resolved here rather than at decoration time, so an
        op may refer to types defined later in its own module. The result is
        kept on the function, so a spec is read once.
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
        # A function defined by exec has no module; name it for what it is.
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
            _outputs=outputs,
        )
        fn.__opspec_spec__ = result
        return result
