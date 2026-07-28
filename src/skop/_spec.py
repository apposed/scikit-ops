"""Op declaration and signature introspection.

This module is part of skop's austere core: it is imported inside every
worker environment, so it must depend on nothing beyond the standard library.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, get_args, get_origin, get_type_hints


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


def direction_of(annotation: Any) -> _Direction | None:
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if meta is OUT or meta is MUT:
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


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: Any
    default: Any
    direction: _Direction | None
    ui: dict = field(default_factory=dict)
    role: Role | None = None
    axes: Axes | None = None

    @property
    def required(self) -> bool:
        return self.default is inspect.Parameter.empty


@dataclass(frozen=True)
class OutputSpec:
    """One of an op's outputs, as a front end needs to see it."""

    name: str
    type: Any
    role: Role | None = None


@dataclass(frozen=True)
class OpSpec:
    name: str
    module: str
    function: str
    env: str
    main_thread: bool
    exclusive: bool
    form: str
    params: tuple[ParamSpec, ...]
    return_type: Any
    doc: str | None
    return_role: Role | None = None

    @property
    def inputs(self) -> tuple[ParamSpec, ...]:
        return tuple(p for p in self.params if p.direction is not OUT)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Names of this op's outputs, in declaration order.

        This is the wire view, used on both sides of the Appose boundary to
        label task outputs. Front ends want ``output_specs`` instead.
        """
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


@dataclass(frozen=True)
class _OpConfig:
    env: str
    main_thread: bool
    exclusive: bool


def op(
    *,
    env: str,
    main_thread: bool = False,
    exclusive: bool = False,
) -> Callable[[Callable], Callable]:
    """Declare a function as an op.

    The decorator is a transparent passthrough: it attaches metadata and
    returns the function unchanged, so an op remains an ordinary Python
    function for direct callers.

    Args:
        env: ID of the environment this op runs in, naming ``envs/<id>/``.
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
