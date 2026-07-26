"""Fitting the array a caller has to the shape an op declared.

An op says what dimensional pattern it wants, through ``Axes`` in its
annotations. A caller says what its array actually is, by naming the array's
axes. Neither of those is a decision; this module makes the decision, as an
``AdaptationPlan`` -- an explicit, inspectable value that a front end can show
a user, or offer several of, before anything runs.

Planning is pure arithmetic over axis names and shapes, so it happens on the
host. Execution happens in the worker, so that iterating forty planes is one
task rather than forty round trips.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _progress, _spec


@dataclass(frozen=True)
class AdaptationPlan:
    """How to get from the array a caller has to the one an op accepts.

    Applied in this order: index the ``select`` axes down to one position
    each, transpose what remains by ``transpose``, then call the op once per
    position of the leading ``iterate`` axes and reassemble the results.
    """

    param: str
    input_axes: tuple[str, ...]
    select: tuple[tuple[str, int], ...]
    iterate: tuple[str, ...]
    transpose: tuple[int, ...]
    output_axes: tuple[str, ...]
    calls: int
    lossless: bool
    summary: str

    def to_dict(self) -> dict:
        """A JSON-safe form, for the trip across the Appose boundary."""
        return {
            "param": self.param,
            "input_axes": list(self.input_axes),
            "select": [[axis, index] for axis, index in self.select],
            "iterate": list(self.iterate),
            "transpose": list(self.transpose),
            "output_axes": list(self.output_axes),
            "calls": self.calls,
            "lossless": self.lossless,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AdaptationPlan:
        return cls(
            param=data["param"],
            input_axes=tuple(data["input_axes"]),
            select=tuple((axis, index) for axis, index in data["select"]),
            iterate=tuple(data["iterate"]),
            transpose=tuple(data["transpose"]),
            output_axes=tuple(data["output_axes"]),
            calls=data["calls"],
            lossless=data["lossless"],
            summary=data["summary"],
        )


def normalize_axes(axes: str | Sequence[str]) -> tuple[str, ...]:
    """Read a caller's axis labels, under the one-string-one-label rule.

    ``("lifetime", "y", "x")`` is three axes and ``"c"`` is one; ``list("zyx")``
    is the compact spelling. A lone canonical-looking string like ``"zyx"`` is
    refused rather than guessed at, the same way ``Axes`` refuses it. Labels
    resolve through ``canonical``, so ``("pln", "row", "col")`` is ``zyx``.
    """
    if isinstance(axes, str):
        if len(axes) > 1 and all(char in _spec.CANONICAL for char in axes.casefold()):
            raise ValueError(
                f"{axes!r} is one axis label. If you meant {len(axes)} axes, "
                f"write list({axes!r})."
            )
        return (_spec.canonical(axes),)
    return tuple(_spec.canonical(str(label)) for label in axes)


def _show(axes: Sequence[str]) -> str:
    """Axis labels as an error message wants them: readable, multi-character."""
    return ", ".join(axes)


def plans(
    fn: Any,
    param: str,
    array: Any,
    axes: str | Sequence[str],
    position: dict[str, int] | None = None,
) -> list[AdaptationPlan]:
    """Enumerate the ways *array* could be fed to op *fn*'s *param*.

    This is the front-end-facing entry point: given what an op wants and what
    a user actually selected, it returns every workable plan so that a GUI can
    offer the choice -- "this Z slice" against "all 41 of them" -- rather than
    make it. Lossless candidates come first, so a caller wanting a single
    answer can take the head, or pass the list to ``choose``.

    Args:
        fn: The op function.
        param: Name of the parameter the array is destined for.
        array: The array itself, or anything with a ``shape``, or a shape.
        axes: What that array actually is, e.g. ``"zyx"``. A front end has to
            work this out for itself; skop does not guess.
        position: Current position along each axis, for the candidate that
            indexes down to a single plane. A viewer's slider positions.
            Axes absent from it are indexed at 0.

    Returns:
        Every plan that would work, most faithful first. Never empty: a
        parameter that cannot be satisfied at all raises instead.
    """
    spec = _spec.spec(fn)
    found = next((p for p in spec.params if p.name == param), None)
    if found is None:
        known = ", ".join(p.name for p in spec.params)
        raise ValueError(f"Op {spec.name} has no parameter {param!r}. Has: {known}")
    shape = getattr(array, "shape", array)
    return _candidates(found, axes, tuple(shape), position)


def _candidates(
    param: _spec.ParamSpec,
    axes: str | Sequence[str],
    shape: Sequence[int],
    position: dict[str, int] | None = None,
) -> list[AdaptationPlan]:
    declared = param.axes
    if declared is None:
        raise ValueError(f"Parameter {param.name!r} declares no Axes")

    actual = normalize_axes(axes)
    _check_actual(param, actual, shape)

    sizes = dict(zip(actual, shape))
    present = tuple(axis for axis in declared.names if axis in actual)
    extra = tuple(axis for axis in actual if axis not in present)

    if not extra:
        return [_plan(param, actual, (), (), present, 1, True, "as is")]

    candidates = []
    extent = ", ".join(f"{sizes[axis]} {axis}" for axis in extra)

    if declared.extra is _spec.Extra.passthrough:
        candidates.append(
            _plan(
                param,
                actual,
                (),
                (),
                extra + present,
                1,
                True,
                f"pass {extent} through to the op",
            )
        )
    elif declared.extra is _spec.Extra.iterate:
        calls = math.prod(sizes[axis] for axis in extra)
        candidates.append(
            _plan(
                param,
                actual,
                (),
                extra,
                extra + present,
                calls,
                True,
                f"run {calls} times, once per {'/'.join(extra)} position",
            )
        )

    at = position or {}
    chosen = tuple((axis, int(at.get(axis, 0))) for axis in extra)
    where = ", ".join(f"{axis}={index}" for axis, index in chosen)
    candidates.append(
        _plan(
            param,
            actual,
            chosen,
            (),
            present,
            1,
            False,
            f"run once at {where}, discarding the rest of {extent}",
        )
    )
    return candidates


def _plan(
    param: _spec.ParamSpec,
    actual: tuple[str, ...],
    select: tuple[tuple[str, int], ...],
    iterate: tuple[str, ...],
    target: tuple[str, ...],
    calls: int,
    lossless: bool,
    summary: str,
) -> AdaptationPlan:
    selected = {axis for axis, _ in select}
    remaining = [axis for axis in actual if axis not in selected]
    return AdaptationPlan(
        param=param.name,
        input_axes=actual,
        select=select,
        iterate=iterate,
        transpose=tuple(remaining.index(axis) for axis in target),
        # Derived, not declared: an op is assumed to return its core axes,
        # with any iterated axes prepended. See the spec's open questions.
        output_axes=iterate + (param.axes.core if param.axes else ()),
        calls=calls,
        lossless=lossless,
        summary=summary,
    )


def _check_actual(
    param: _spec.ParamSpec, actual: tuple[str, ...], shape: Sequence[int]
) -> None:
    declared = param.axes
    assert declared is not None
    if len(actual) != len(shape):
        raise ValueError(
            f"Parameter {param.name!r}: {len(actual)} axis label(s) "
            f"({_show(actual)}) for a {len(shape)}-dimensional array {tuple(shape)}"
        )
    if len(set(actual)) != len(actual):
        raise ValueError(f"Parameter {param.name!r}: repeated axis in {_show(actual)}")
    missing = [axis for axis in declared.core if axis not in actual]
    if missing:
        raise ValueError(
            f"Parameter {param.name!r} needs axes {_show(declared.names)} but was "
            f"given {_show(actual)}, which has no {', '.join(missing)} axis. "
            "No adaptation can invent one."
        )


def choose(candidates: list[AdaptationPlan]) -> AdaptationPlan:
    """Pick a plan without asking anyone, or refuse to.

    Automatic selection stays timid on purpose: skop takes a plan only when
    the plan keeps all the data. Anything that discards some is a decision
    for whoever owns the data.
    """
    best = candidates[0]
    if best.lossless:
        return best
    options = "\n".join(f"  - {plan.summary}" for plan in candidates)
    raise ValueError(
        f"Parameter {best.param!r} was given axes {_show(best.input_axes)}, "
        "which the op does not accept, and every way of adapting it would "
        f"discard data. Choose one explicitly:\n{options}"
    )


# -- execution, in the worker -------------------------------------------


def apply(plan: AdaptationPlan, array: np.ndarray) -> np.ndarray:
    """Index and transpose *array* into the layout the op declared."""
    selected = dict(plan.select)
    if selected:
        array = array[
            tuple(
                selected[axis] if axis in selected else slice(None)
                for axis in plan.input_axes
            )
        ]
    return array.transpose(plan.transpose)


def execute(
    spec: _spec.OpSpec,
    fn: Any,
    args: dict,
    plans: dict[str, AdaptationPlan],
) -> Any:
    """Call *fn*, adapting its arguments and reassembling its results."""
    if not plans:
        return fn(**args)

    iterated = [plan for plan in plans.values() if plan.iterate]
    if len(iterated) > 1:
        names = ", ".join(sorted(plan.param for plan in iterated))
        raise ValueError(
            f"Op {spec.name}: only one parameter may be iterated at a time, "
            f"but plans iterate over {names}"
        )

    for name, plan in plans.items():
        args[name] = apply(plan, np.asarray(args[name]))

    if not iterated:
        return fn(**args)

    plan = iterated[0]
    if spec.form is not _spec.FUNCTION:
        raise ValueError(
            f"Op {spec.name} is {spec.form} form; iteration is implemented for "
            "function-form ops only, since the caller's buffer has the shape "
            "of the whole input rather than of one slice."
        )

    stack = args[plan.param]
    span = tuple(stack.shape[: len(plan.iterate)])
    gathered: list[list] = [[] for _ in spec.outputs]

    for step, index in enumerate(np.ndindex(*span)):
        if _progress.cancel_requested():
            raise RuntimeError(
                f"Op {spec.name} was cancelled after {step} of {plan.calls} slices"
            )
        where = ", ".join(f"{axis}={i}" for axis, i in zip(plan.iterate, index))
        _progress.progress(
            f"Slice {step + 1} of {plan.calls} ({where})", step, plan.calls
        )
        args[plan.param] = stack[index]
        for slot, value in enumerate(_split(spec, fn(**args))):
            gathered[slot].append(value)

    results = tuple(
        _reassemble(spec.name, name, values, span, output.role)
        for name, output, values in zip(spec.outputs, spec.output_specs, gathered)
    )
    if len(results) == 1:
        return results[0]
    factory = spec.return_type
    if getattr(factory, "_fields", None) == tuple(spec.outputs):
        return factory(*results)
    return results


def _split(spec: _spec.OpSpec, result: Any) -> tuple:
    """One call's return value, as one value per declared output."""
    names = spec.outputs
    if len(names) == 1:
        return (result,)
    if result is None or len(result) != len(names):
        got = "None" if result is None else str(len(result))
        raise TypeError(
            f"Op {spec.name} declares {len(names)} outputs {names} but one "
            f"iteration returned {got} value(s)"
        )
    return tuple(result)


def _reassemble(
    op_name: str,
    output: str,
    values: list,
    span: tuple[int, ...],
    role: _spec.Role | None,
) -> Any:
    """Stack one output slot's per-slice values back into a whole."""
    if all(isinstance(value, np.ndarray) for value in values):
        shapes = {value.shape for value in values}
        if len(shapes) == 1:
            if role is _spec.Role.labels:
                values = _renumber(values)
            return np.stack(values).reshape(span + shapes.pop())
        raise TypeError(
            f"Op {op_name}: output {output!r} cannot be stacked across slices, "
            f"because its shape varies between them ({len(shapes)} distinct "
            "shapes). Adapt this op with a single-slice plan instead."
        )
    if all(isinstance(value, (int, float, bool, np.number)) for value in values):
        return np.array(values).reshape(span)
    kinds = ", ".join(sorted({type(value).__name__ for value in values}))
    raise TypeError(
        f"Op {op_name}: output {output!r} cannot be stacked across slices; "
        f"iteration reassembles arrays and numbers, not {kinds}."
    )


def _renumber(values: list[np.ndarray]) -> list[np.ndarray]:
    """Make label IDs unique across slices.

    Every slice numbers its own objects from 1, so stacking as-is would claim
    that object 1 in slice 0 and object 1 in slice 1 are the same cell. They
    are not: the op author's ``Extra.iterate`` said the slices were treated
    independently.
    """
    shifted = []
    offset = 0
    for value in values:
        top = int(value.max()) if value.size else 0
        shifted.append(np.where(value > 0, value + offset, 0) if offset else value)
        offset += top
    return shifted
