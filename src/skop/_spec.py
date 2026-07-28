"""Op declaration and signature introspection.

This module is part of skop's austere core: it is imported inside every
worker environment, so it must depend on nothing beyond the standard library.
"""

from __future__ import annotations

import inspect
import sys
import types
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
    """What a value *means*, beyond what its Python type says.

    Every op here passes ``np.ndarray`` around, but a front end needs to know
    whether a given array is a picture, a label image or a set of coordinates
    to display it correctly. The type cannot say so; this can.

    The vocabulary deliberately mirrors napari's layer types, so that mapping
    a role onto a napari type is a lookup rather than a judgement call. It is
    not napari-specific: a magicgui-only front end maps roles onto widgets,
    and Fiji maps them onto ImgLib2 types.

    ``masks`` is the one member that does not name a layer type, and it is a
    deliberate stretch rather than an oversight. A stack of possibly
    overlapping masks is not something a viewer displays directly, but it maps
    onto something that is -- a front end projects it to a label image with
    ``skop.masks``. The role still answers "which layer does this become",
    it just answers with a conversion rather than an identity. See
    docs/design/0008-mask-detector-ops.md.

    Roles are attached through ``skop.types``, or directly with
    ``Annotated[T, Role.<name>]`` for a role having no alias.
    """

    image = "image"
    labels = "labels"
    masks = "masks"
    points = "points"
    shapes = "shapes"
    surface = "surface"
    tracks = "tracks"
    vectors = "vectors"


#: Axis names that a viewer can map onto display semantics: the intersection
#: of OME-NGFF, bioimage.io and ImageJ2's AxisType. Privileged, not exclusive
#: -- an axis label is any string, because there is no agreed letter for a
#: lifetime bin, a stage position, or a polarization angle, and n-D means n-D.
CANONICAL = ("x", "y", "z", "c", "t")

#: Synonyms for the canonical names, gathered from the stacks that a caller's
#: labels are likely to have come from: scikit-image's ``(pln, row, col)``,
#: ImageJ's slices and frames, Bio-Formats and CZI's ``XYZCT``, OME-NGFF and
#: CF's long names. Resolving these is a lookup, not a guess -- ``row`` *is*
#: ``y`` -- so it happens in skop rather than separately in every front end.
#:
#: Only true synonyms belong here. ``slice`` earns its place because within
#: the hyperstack model this vocabulary shares, ImageJ's slice *is* z; the
#: ambiguous case is a plain stack, which has no axis labels to resolve in the
#: first place. Notably absent is bioio's ``s`` (RGB samples): bioio
#: distinguishes it from ``c`` deliberately, so folding the two would merge
#: what the source vocabulary kept apart. Labels with no
#: canonical equivalent -- bioimage.io's ``b`` for batch, SCIFIO's
#: ``lifetime``, ``polarization``, ``spectra`` -- are left exactly as they
#: are, and adapt like any other axis an op does not consume.
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
    """Resolve one axis label to the spelling skop matches on.

    Case is folded, so Bio-Formats' ``XYZCT`` and NGFF's ``xyzct`` are the
    same axes, and a known synonym resolves to its canonical name. Anything
    unrecognized is returned folded but otherwise untouched: an open
    vocabulary means ``"lifetime"`` has to survive this unchanged.
    """
    folded = label.strip().casefold()
    return ALIASES.get(folded, folded)


#: The spelling for a slot with no name preference at all.
WILDCARD = "*"


@dataclass(frozen=True)
class Slot:
    """One axis an image parameter consumes.

    ``name`` is a *hint*, not a requirement: it biases which of the caller's
    axes lands here, and a mismatch is reported rather than refused. A wildcard
    slot (``name`` is None) has no preference at all.
    """

    name: str | None
    optional: bool = False

    def __str__(self) -> str:
        return (self.name or WILDCARD) + ("?" if self.optional else "")


@dataclass(frozen=True, init=False)
class Axes:
    """How many axes an image parameter consumes, and what it likes to call them.

    One argument is one slot, or a single iterable gives them all. A trailing
    ``?`` marks a slot optional, and ``"*"`` is a slot with no name preference::

        Axes("y", "x")        # two axes; prefers to call them y and x
        Axes(list("zyx"))     # three
        Axes("y", "x", "c?")  # two, plus a channel axis if one is there
        Axes("*", "*")        # two axes, no opinion which
        Axes(variadic=True)   # any number of axes, whatever they are

    **Names are hints, never requirements.** A 2-D triangulation works on
    ``y x``, ``z x`` and ``z y`` alike, so a name only biases the default
    mapping; handing an op axes it did not name is reported through
    ``AdaptationPlan.warnings``, never refused. What binds is the *arity* --
    how many axes the op consumes -- because that is the part the op's own
    indexing depends on. Names resolve through ``canonical``, so a slot named
    ``"y"`` prefers an array axis labelled ``"row"``.

    ``variadic`` says the op copes with any number of further axes on its own:
    global thresholding is happy with 1-D, a plane or a whole volume, so a
    caller's leftover axes may be folded into the call rather than looped over.

    Attached through ``Annotated``, like a role, and read back off
    ``ParamSpec.axes``. It is inert at runtime: an op annotated this way is
    still called with whatever the caller passes when called directly.
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


class _Direction:
    """Marker distinguishing input, output-buffer and mutated-buffer params."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<skop.{self.name}>"


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

    ``labels: Out[np.ndarray]`` declares a computer-form op. Out params are
    hidden from generated GUIs -- a user is never asked for an output buffer.
    """

    def __class_getitem__(cls, item: Any) -> Any:
        return _annotate(item, OUT)


class Mut:
    """Mark a parameter as mutated in place, declaring an inplace-form op."""

    def __class_getitem__(cls, item: Any) -> Any:
        return _annotate(item, MUT)


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
        menu without needing the Python objects behind it.
        """
        return tuple(
            (label, f"{op.__module__}:{op.__name__}") for label, op in self.options
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


def direction_of(annotation: Any) -> _Direction | None:
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if meta is OUT or meta is MUT:
                return meta
    return None


def choices_of(annotation: Any) -> Choices | None:
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Choices):
                return meta
    return None


def params_for_of(annotation: Any) -> ParamsFor | None:
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, ParamsFor):
                return meta
    return None


def role_of(annotation: Any) -> Role | None:
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Role):
                return meta
    return None


def axes_of(annotation: Any) -> Axes | None:
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Axes):
                return meta
    return None


def _strip(annotation: Any) -> Any:
    """Return the underlying type of a possibly-Annotated annotation."""
    return (
        get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
    )


def _ui_hints(annotation: Any) -> dict:
    """Collect magicgui-style dict metadata out of an Annotated annotation."""
    hints: dict = {}
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, dict):
                hints.update(meta)
    return hints


# -- the wire vocabulary for a parameter's type -------------------------
#
# In-process, ``ParamSpec.type`` is a live Python type object and a front end
# reads it directly. Out of process it cannot be: a Java front end has no way
# to receive ``<class 'numpy.ndarray'>``, only a name for it. These are the
# names, and they are deliberately few -- the set a generated dialog can
# actually render a widget for, plus UNKNOWN for everything else.

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
    """One member of an enum parameter: what to show, and what to send."""

    name: str
    value: Any

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> Choice:
        return cls(name=data["name"], value=data["value"])


@dataclass(frozen=True)
class TypeSpec:
    """A parameter or output type, in the vocabulary a front end can act on.

    ``UNKNOWN`` is load-bearing rather than a failure: a front end that cannot
    render one parameter leaves it at its default and says so, or -- if the
    parameter is required -- disables the run and says why. One awkward
    parameter must not cost the whole op, so ``detail`` carries the original
    annotation's spelling for the message that explains it.
    """

    name: str
    choices: tuple[Choice, ...] = ()
    nullable: bool = False
    detail: str | None = None

    def to_dict(self) -> dict:
        data: dict = {"name": self.name}
        if self.choices:
            data["choices"] = [choice.to_dict() for choice in self.choices]
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
    """Recognize ``numpy.ndarray`` without importing numpy.

    This module is imported in every worker before numpy necessarily is, and
    is meant to hold to the standard library alone. A structural check costs
    a little precision -- some other library's ``ndarray`` would match -- and
    buys keeping that promise.
    """
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


def type_spec(annotation: Any) -> TypeSpec:
    """Classify a type annotation into the wire vocabulary."""
    if isinstance(annotation, TypeSpec):
        # Already classified: this spec came off the wire rather than off a
        # live function, and re-serializing it must not lose what it says.
        return annotation
    annotation = _strip(annotation)

    origin = get_origin(annotation)
    if origin is not None and _is_union(origin):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            # Optional[X] is X, with the front end told it may be left empty.
            inner = type_spec(args[0])
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
    # NB: bool before int, since bool is a subclass of int and a checkbox is
    # emphatically not a number field.
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


def _is_union(origin: Any) -> bool:
    """Whether an annotation's origin is a union, spelled either way."""
    if origin is Union:
        return True
    union_type = getattr(types, "UnionType", None)  # Python 3.10+: X | Y
    return union_type is not None and origin is union_type


#: Stands in for a default that a parameter does not have. JSON has no way to
#: say "no default" other than by saying nothing, and ``required`` already
#: does that, so a serialized ParamSpec simply omits the key.
_NO_DEFAULT = inspect.Parameter.empty


def _wire_default(value: Any) -> Any:
    """A default value in a form JSON can carry.

    Only the values a wire type can actually have need survive this. Anything
    else -- a tuple default on an UNKNOWN parameter, say -- becomes None,
    which the front end reads together with UNKNOWN as "leave this alone".
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_wire_default(item) for item in value]
    return None


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: Any
    default: Any
    direction: _Direction | None
    ui: dict = field(default_factory=dict)
    role: Role | None = None
    axes: Axes | None = None
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
        if self.direction is not None:
            data["direction"] = self.direction.name
        if self.ui:
            data["ui"] = dict(self.ui)
        if self.role is not None:
            data["role"] = self.role.value
        if self.axes is not None:
            data["axes"] = _axes_dict(self.axes)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ParamSpec:
        """Rebuild a ParamSpec from its wire form.

        ``type`` comes back as a ``TypeSpec`` rather than the live Python type
        it was built from, because the type object does not exist in the
        process doing the reading -- that is the whole reason for the wire
        vocabulary. So this round-trips the *dict*, not the object.
        """
        direction = data.get("direction")
        return cls(
            name=data["name"],
            type=TypeSpec.from_dict(data["type"]),
            default=(
                _NO_DEFAULT if data.get("required", False) else data.get("default")
            ),
            direction={"Out": OUT, "Mut": MUT}.get(direction) if direction else None,
            ui=dict(data.get("ui", {})),
            role=Role(data["role"]) if data.get("role") else None,
            axes=_axes_from_dict(data["axes"]) if data.get("axes") else None,
        )


def _axes_dict(axes: Axes) -> dict:
    return {
        "slots": [
            {"name": slot.name, "optional": slot.optional} for slot in axes.slots
        ],
        "variadic": axes.variadic,
    }


def _axes_from_dict(data: dict) -> Axes:
    result = Axes(variadic=bool(data.get("variadic", False)))
    slots = tuple(
        Slot(slot["name"], bool(slot.get("optional", False)))
        for slot in data.get("slots", ())
    )
    object.__setattr__(result, "slots", slots)
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


@dataclass(frozen=True)
class OpSpec:
    name: str
    module: str
    function: str
    env: str | None
    main_thread: bool
    exclusive: bool
    form: str
    params: tuple[ParamSpec, ...]
    return_type: Any
    doc: str | None
    return_role: Role | None = None

    # Set only when an OpSpec was rebuilt from its wire form, where the return
    # type is a name rather than the live type the properties below derive
    # outputs from. See from_dict.
    _outputs: tuple[str, ...] | None = field(default=None, repr=False, compare=False)
    _output_specs: tuple[OutputSpec, ...] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_workflow(self) -> bool:
        """Whether this op runs on the host and calls other ops.

        The absence of an environment is the statement. ``env`` is what pins an
        op to a worker; a workflow has nothing to pin, because the ops it calls
        each bring their own -- and it could not build them from inside a
        worker anyway.
        """
        return self.env is None

    @property
    def inputs(self) -> tuple[ParamSpec, ...]:
        return tuple(p for p in self.params if p.direction is not OUT)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Names of this op's outputs, in declaration order.

        This is the wire view, used on both sides of the Appose boundary to
        label task outputs. Front ends want ``output_specs`` instead.
        """
        if self._outputs is not None:
            return self._outputs
        out_params = tuple(p.name for p in self.params if p.direction is OUT)
        if out_params:
            return out_params
        mut_params = tuple(p.name for p in self.params if p.direction is MUT)
        if mut_params:
            return mut_params
        fields = getattr(self.return_type, "_fields", None)
        if fields is not None:
            # NamedTuple return: one output per field.
            return tuple(fields)
        return () if self.return_type in (None, type(None)) else ("result",)

    @property
    def output_specs(self) -> tuple[OutputSpec, ...]:
        """This op's outputs, with their types and roles."""
        if self._output_specs is not None:
            return self._output_specs
        names = self.outputs
        if not names:
            return ()
        if self.form != FUNCTION:
            # Results land in parameters the caller supplied.
            by_name = {p.name: p for p in self.params}
            return tuple(
                OutputSpec(name, by_name[name].type, by_name[name].role)
                for name in names
            )
        if names == ("result",):
            return (OutputSpec("result", self.return_type, self.return_role),)
        # NamedTuple return: each field carries its own annotation.
        hints = _field_hints(self.return_type)
        return tuple(
            OutputSpec(name, _strip(hints.get(name)), role_of(hints.get(name)))
            for name in names
        )

    def to_dict(self) -> dict:
        """This op, as JSON a front end in another language can read.

        ``outputs`` and ``output_specs`` are written out rather than left to
        be derived, because deriving them needs the live return type -- a
        NamedTuple's ``_fields`` -- which does not cross the boundary.
        """
        return {
            "name": self.name,
            "module": self.module,
            "function": self.function,
            "env": self.env,
            "main_thread": self.main_thread,
            "exclusive": self.exclusive,
            "form": self.form,
            "doc": self.doc,
            "params": [param.to_dict() for param in self.params],
            "return_type": type_spec(self.return_type).to_dict(),
            "return_role": self.return_role.value if self.return_role else None,
            "outputs": list(self.outputs),
            "output_specs": [output.to_dict() for output in self.output_specs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> OpSpec:
        """Rebuild an OpSpec from its wire form.

        Types come back as ``TypeSpec``s, not as the Python types they were
        read off; see ``ParamSpec.from_dict``.
        """
        return cls(
            name=data["name"],
            module=data["module"],
            function=data["function"],
            env=data["env"],
            main_thread=bool(data.get("main_thread", False)),
            exclusive=bool(data.get("exclusive", False)),
            form=data["form"],
            params=tuple(ParamSpec.from_dict(p) for p in data.get("params", ())),
            return_type=TypeSpec.from_dict(data["return_type"]),
            doc=data.get("doc"),
            return_role=(
                Role(data["return_role"]) if data.get("return_role") else None
            ),
            _outputs=tuple(data.get("outputs", ())),
            _output_specs=tuple(
                OutputSpec.from_dict(o) for o in data.get("output_specs", ())
            ),
        )


@dataclass(frozen=True)
class _OpConfig:
    env: str | None
    main_thread: bool
    exclusive: bool


def op(
    *,
    env: str | None = None,
    main_thread: bool = False,
    exclusive: bool = False,
) -> Callable[[Callable], Callable]:
    """Declare a function as an op.

    The decorator is a transparent passthrough: it attaches metadata and
    returns the function unchanged, so an op remains an ordinary Python
    function for direct callers.

    Args:
        env: ID of the environment this op runs in, naming ``envs/<id>/``.
            Omit it to declare a **workflow**: an op with nothing to pin,
            which runs on the host and calls other ops through the runner.
        main_thread: Whether the op must run on the worker's main thread.
        exclusive: Whether the op needs a worker to itself, rather than
            sharing one with other ops assigned to the same environment.
    """

    def decorate(fn: Callable) -> Callable:
        fn.__skop__ = _OpConfig(env=env, main_thread=main_thread, exclusive=exclusive)
        return fn

    return decorate


def is_op(obj: Any) -> bool:
    return callable(obj) and isinstance(getattr(obj, "__skop__", None), _OpConfig)


def spec(fn: Callable) -> OpSpec:
    """Build (and cache) the OpSpec for a decorated op function.

    Annotations are resolved here rather than at decoration time, so that an
    op may refer to types defined later in its own module.
    """
    cached = getattr(fn, "__skop_spec__", None)
    if cached is not None:
        return cached

    config = getattr(fn, "__skop__", None)
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
                direction=direction_of(annotation),
                ui=_ui_hints(annotation),
                role=role_of(annotation),
                axes=axes_of(annotation),
                choices=choices_of(annotation),
                params_for=params_for_of(annotation),
            )
        )

    directions = {p.direction for p in params}
    if OUT in directions and MUT in directions:
        raise TypeError(f"Op {fn.__qualname__} mixes Out and Mut params; pick one form")
    form = COMPUTER if OUT in directions else INPLACE if MUT in directions else FUNCTION

    return_annotation = hints.get("return", signature.return_annotation)
    result = OpSpec(
        name=f"{fn.__module__}:{fn.__name__}",
        module=fn.__module__,
        function=fn.__name__,
        env=config.env,
        main_thread=config.main_thread,
        exclusive=config.exclusive,
        form=form,
        params=tuple(params),
        return_type=_strip(return_annotation),
        doc=inspect.getdoc(fn),
        return_role=role_of(return_annotation),
    )
    fn.__skop_spec__ = result
    return result


def _field_hints(return_type: Any) -> dict:
    """Resolve a NamedTuple's field annotations, for their roles.

    Unlike an op's own annotations, these are optional: an unresolvable field
    annotation costs a role, not a usable op, so it must not raise.
    """
    try:
        return get_type_hints(return_type, include_extras=True)
    except Exception:  # noqa: BLE001 - any failure here just costs a role.
        return {}


class _Unbound:
    """A function's annotations, detached from its defaults.

    Before Python 3.11, ``get_type_hints`` rewrites the annotation of any
    parameter defaulting to ``None`` as ``Optional[<annotation>]``. That
    implicit rewrite is not merely obsolete, it is fatal here: building the
    Union deduplicates its members through a set, which hashes the
    ``Annotated`` alias, which hashes its metadata -- and skop's UI hints are
    dicts, which are unhashable. So an op with a ``FloatSlider`` hint and a
    ``None`` default cannot be introspected in, say, the 3.10 environment
    StarDist pins, however valid its annotations are.

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
    string, and op modules use it. Resolution happens through
    ``get_type_hints`` rather than ``inspect.signature(eval_str=True)``
    because the latter needs Python 3.10, while an environment is free to
    pin an older one -- UNSEG's pins it to 3.9.
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
