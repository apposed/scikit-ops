"""Op declaration and signature introspection.

This module is part of opkit's austere core: it is imported inside every
worker environment, so it must depend on nothing beyond the standard library.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args, get_origin, get_type_hints


class _Direction:
    """Marker distinguishing input, output-buffer and mutated-buffer params."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<opkit.{self.name}>"


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

    @property
    def required(self) -> bool:
        return self.default is inspect.Parameter.empty


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

    @property
    def inputs(self) -> tuple[ParamSpec, ...]:
        return tuple(p for p in self.params if p.direction is not OUT)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Names of this op's outputs, in declaration order."""
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
        fn.__opkit__ = _OpConfig(env=env, main_thread=main_thread, exclusive=exclusive)
        return fn

    return decorate


def is_op(obj: Any) -> bool:
    return callable(obj) and isinstance(getattr(obj, "__opkit__", None), _OpConfig)


def spec(fn: Callable) -> OpSpec:
    """Build (and cache) the OpSpec for a decorated op function.

    Annotations are resolved here rather than at decoration time, so that an
    op may refer to types defined later in its own module.
    """
    cached = getattr(fn, "__opkit_spec__", None)
    if cached is not None:
        return cached

    config = getattr(fn, "__opkit__", None)
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
            )
        )

    directions = {p.direction for p in params}
    if OUT in directions and MUT in directions:
        raise TypeError(f"Op {fn.__qualname__} mixes Out and Mut params; pick one form")
    form = COMPUTER if OUT in directions else INPLACE if MUT in directions else FUNCTION

    result = OpSpec(
        name=f"{fn.__module__}:{fn.__name__}",
        module=fn.__module__,
        function=fn.__name__,
        env=config.env,
        main_thread=config.main_thread,
        exclusive=config.exclusive,
        form=form,
        params=tuple(params),
        return_type=_strip(hints.get("return", signature.return_annotation)),
        doc=inspect.getdoc(fn),
    )
    fn.__opkit_spec__ = result
    return result


def _resolve_hints(fn: Callable) -> dict:
    """Evaluate a function's annotations, which may be strings.

    ``from __future__ import annotations`` turns every annotation into a
    string, and op modules use it. Resolution happens through
    ``get_type_hints`` rather than ``inspect.signature(eval_str=True)``
    because the latter needs Python 3.10, while an environment is free to
    pin an older one -- UNSEG's pins it to 3.9.
    """
    try:
        return get_type_hints(fn, include_extras=True)
    except Exception as exc:
        raise TypeError(
            f"Could not resolve the annotations of op {fn.__qualname__} "
            f"under Python {sys.version_info.major}.{sys.version_info.minor}: "
            f"{type(exc).__name__}: {exc}\n"
            "An op's annotations are evaluated in the environment it runs in, "
            "so they must be valid there -- note that 'X | Y' unions need "
            "Python 3.10."
        ) from exc
