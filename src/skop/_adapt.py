"""Fitting the array a caller has to the axes an op consumes.

An op says how many axes it consumes, and what it likes to call them, through
``Axes`` in its annotations. A caller says what its array actually is, by
naming the array's axes. Neither of those is a decision; this module makes one,
as an ``AdaptationPlan`` -- an explicit, inspectable value a front end can show
a user, and hand back edited.

The plan is built best-effort and then *overridden*, rather than chosen from an
enumeration. Which of the caller's axes fills which slot, and what becomes of
the ones left over, is a per-axis decision belonging to whoever owns the data:
whether a stack should be processed plane by plane or as a volume is a property
of the experiment, not of the op. So skop picks a sensible default, says where
that default looks doubtful (``AdaptationPlan.warnings``), and forbids nothing.

Planning is pure arithmetic over axis names and shapes, so it happens on the
host. Execution happens in the worker, so that iterating forty planes is one
task rather than forty round trips.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Planning is opspec's: pure arithmetic over axis names and shapes, so it
# needs no numpy. What follows it here is execution, which touches pixels.
from opspec.plan import (
    DISPOSITIONS,
    ITERATE,
    PASS,
    SELECT,
    AdaptationPlan,
    _name,
    build,
    normalize_axes,
    plan,
)

from . import _progress, _spec

__all__ = [
    "DISPOSITIONS",
    "ITERATE",
    "PASS",
    "SELECT",
    "AdaptationPlan",
    "apply",
    "build",
    "execute",
    "normalize_axes",
    "plan",
]


# -- execution, in the worker -------------------------------------------


def apply(plan: AdaptationPlan, array: np.ndarray) -> np.ndarray:
    """Index and transpose *array* into the layout the op declared."""
    selected = dict(plan.select)
    if selected:
        array = array[
            tuple(
                selected[i] if i in selected else slice(None)
                for i in range(len(plan.input_axes))
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

    for name, param_plan in plans.items():
        args[name] = apply(param_plan, np.asarray(args[name]))

    if not iterated:
        return fn(**args)

    plan = iterated[0]
    if spec.form != _spec.FUNCTION:
        raise ValueError(
            f"Op {spec.name} is {spec.form} form; iteration is implemented for "
            "function-form ops only, since the caller's buffer has the shape "
            "of the whole input rather than of one slice."
        )

    stack = args[plan.param]
    span = tuple(stack.shape[: len(plan.iterate)])
    labels = [_name(plan.input_axes, i) for i in plan.iterate]
    gathered: list[list] = [[] for _ in spec.outputs]

    for step, index in enumerate(np.ndindex(*span)):
        if _progress.cancel_requested():
            raise RuntimeError(
                f"Op {spec.name} was cancelled after {step} of {plan.calls} slices"
            )
        where = ", ".join(f"{label}={i}" for label, i in zip(labels, index))
        _progress.progress(
            f"Slice {step + 1} of {plan.calls} ({where})", step, plan.calls
        )
        args[plan.param] = stack[index]
        for slot, value in enumerate(_split(spec, fn(**args))):
            gathered[slot].append(value)

    results = tuple(
        _reassemble(spec.name, output.name, values, span, output.role)
        for output, values in zip(spec.outputs, gathered)
    )
    if len(results) == 1:
        return results[0]
    factory = spec.return_type
    if getattr(factory, "_fields", None) == tuple(o.name for o in spec.outputs):
        return factory(*results)
    return results


def _split(spec: _spec.OpSpec, result: Any) -> tuple:
    """One call's return value, as one value per declared output."""
    names = tuple(o.name for o in spec.outputs)
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
    are not: iterating an axis is precisely the statement that the slices were
    processed independently.
    """
    shifted = []
    offset = 0
    for value in values:
        top = int(value.max()) if value.size else 0
        shifted.append(np.where(value > 0, value + offset, 0) if offset else value)
        offset += top
    return shifted
