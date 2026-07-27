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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _progress, _spec

#: What becomes of an input axis that no slot consumed.
ITERATE = "iterate"  # call the op once per position, and stack the results
SELECT = "select"  # index down to one position, discarding the rest
PASS = "pass"  # hand it to the op as an axis, for a variadic op to deal with

DISPOSITIONS = (ITERATE, SELECT, PASS)


@dataclass(frozen=True)
class AdaptationPlan:
    """How to get from the array a caller has to the one an op accepts.

    Applied in this order: index the ``select`` axes down to one position
    each, transpose what remains by ``transpose``, then call the op once per
    position of the leading ``iterate`` axes and reassemble the results.

    ``mapping`` is the editable half: one entry per declared slot, giving the
    index of the input axis that fills it, or None for an optional slot left
    empty. Everything else follows from it.
    """

    param: str
    input_axes: tuple[str, ...]
    mapping: tuple[int | None, ...]
    select: tuple[tuple[int, int], ...]
    iterate: tuple[int, ...]
    passed: tuple[int, ...]
    transpose: tuple[int, ...]
    output_axes: tuple[str, ...]
    calls: int
    lossless: bool
    warnings: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict:
        """A JSON-safe form, for the trip across the Appose boundary."""
        return {
            "param": self.param,
            "input_axes": list(self.input_axes),
            "mapping": list(self.mapping),
            "select": [[axis, index] for axis, index in self.select],
            "iterate": list(self.iterate),
            "passed": list(self.passed),
            "transpose": list(self.transpose),
            "output_axes": list(self.output_axes),
            "calls": self.calls,
            "lossless": self.lossless,
            "warnings": list(self.warnings),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AdaptationPlan:
        return cls(
            param=data["param"],
            input_axes=tuple(data["input_axes"]),
            mapping=tuple(data["mapping"]),
            select=tuple((axis, index) for axis, index in data["select"]),
            iterate=tuple(data["iterate"]),
            passed=tuple(data["passed"]),
            transpose=tuple(data["transpose"]),
            output_axes=tuple(data["output_axes"]),
            calls=data["calls"],
            lossless=data["lossless"],
            warnings=tuple(data["warnings"]),
            summary=data["summary"],
        )


def normalize_axes(axes: str | Sequence[str | None]) -> tuple[str, ...]:
    """Read a caller's axis labels, under the one-string-one-label rule.

    ``("lifetime", "y", "x")`` is three axes and ``"c"`` is one; ``list("zyx")``
    is the compact spelling. A lone canonical-looking string like ``"zyx"`` is
    refused rather than guessed at. Labels resolve through ``canonical``, so
    ``("pln", "row", "col")`` is ``zyx``.

    ``None`` is a deliberately *unnamed* axis, which a plain ndarray has every
    right to be. It matches no slot by name, so it can only be filled
    positionally -- and never draws a warning for it.
    """
    if isinstance(axes, str):
        if len(axes) > 1 and all(char in _spec.CANONICAL for char in axes.casefold()):
            raise ValueError(
                f"{axes!r} is one axis label. If you meant {len(axes)} axes, "
                f"write list({axes!r})."
            )
        return (_spec.canonical(axes),)
    return tuple("" if label is None else _spec.canonical(str(label)) for label in axes)


def _show(axes: Sequence[str]) -> str:
    """Axis labels as an error message wants them: readable, multi-character."""
    return ", ".join(_name(axes, i) for i in range(len(axes)))


def _name(axes: Sequence[str], index: int) -> str:
    """One axis, named if it has a name and numbered if it does not."""
    return axes[index] or f"axis {index}"


def plan(
    fn: Any,
    param: str,
    array: Any,
    axes: str | Sequence[str | None],
    position: dict[str, int] | None = None,
    mapping: Sequence[int | None] | None = None,
    dispositions: dict[int, str] | None = None,
) -> AdaptationPlan:
    """Work out how *array* should be fed to op *fn*'s *param*.

    This is the front-end-facing entry point. Called with nothing but the
    axes, it returns skop's best effort. Called with *mapping* or
    *dispositions*, it returns what the caller asked for instead -- which is
    how a GUI offers per-axis control without skop having to enumerate a
    combinatorial space of possibilities.

    Args:
        fn: The op function.
        param: Name of the parameter the array is destined for.
        array: The array itself, or anything with a ``shape``, or a shape.
        axes: What that array actually is, e.g. ``list("zyx")``. A front end
            has to work this out for itself; skop does not guess. ``None`` in
            place of a label means that axis has no name.
        position: Current position along each named axis, for any axis being
            indexed down to a single plane. A viewer's slider positions.
            Axes absent from it are indexed at 0.
        mapping: One input-axis index (or None) per declared slot, overriding
            the best-effort assignment.
        dispositions: What to do with each leftover input axis, by index: one
            of ``"iterate"``, ``"select"`` or ``"pass"``. Anything unnamed
            keeps its default.

    Returns:
        One plan. Raises only when no mapping could work at all -- when the
        array has fewer axes than the op requires, or the labels do not
        describe the array.
    """
    spec = _spec.spec(fn)
    found = next((p for p in spec.params if p.name == param), None)
    if found is None:
        known = ", ".join(p.name for p in spec.params)
        raise ValueError(f"Op {spec.name} has no parameter {param!r}. Has: {known}")
    shape = getattr(array, "shape", array)
    return build(found, axes, tuple(shape), position, mapping, dispositions)


def build(
    param: _spec.ParamSpec,
    axes: str | Sequence[str | None],
    shape: Sequence[int],
    position: dict[str, int] | None = None,
    mapping: Sequence[int | None] | None = None,
    dispositions: dict[int, str] | None = None,
) -> AdaptationPlan:
    """Assemble one plan, from a ParamSpec rather than a function."""
    declared = param.axes
    if declared is None:
        raise ValueError(f"Parameter {param.name!r} declares no Axes")

    actual = normalize_axes(axes)
    _check_actual(param, actual, shape)
    slots = declared.slots

    filled = (
        _default_mapping(slots, actual, param)
        if mapping is None
        else _checked_mapping(mapping, slots, actual, param)
    )

    used = {index for index in filled if index is not None}
    leftover = [i for i in range(len(actual)) if i not in used]
    chosen = _dispositions(leftover, dispositions, declared, param)

    at = position or {}
    select = tuple(
        (i, int(at.get(actual[i], 0))) for i in leftover if chosen[i] == SELECT
    )
    iterate = tuple(i for i in leftover if chosen[i] == ITERATE)
    passed = tuple(i for i in leftover if chosen[i] == PASS)

    sizes = dict(enumerate(shape))
    calls = math.prod(sizes[i] for i in iterate) if iterate else 1

    # Iterated axes lead, so the worker can walk them with one np.ndindex.
    # Passed-through axes sit outside the op's own, matching how an op that
    # copes with extra dimensions expects to find them.
    target = list(iterate) + list(passed) + [i for i in filled if i is not None]
    dropped = {i for i, _ in select}
    remaining = [i for i in range(len(actual)) if i not in dropped]

    return AdaptationPlan(
        param=param.name,
        input_axes=actual,
        mapping=filled,
        select=select,
        iterate=iterate,
        passed=passed,
        transpose=tuple(remaining.index(i) for i in target),
        # Derived, not declared, and in the caller's own vocabulary: a remapped
        # axis keeps the name its owner gave it. See the spec's open questions.
        output_axes=tuple(actual[i] for i in target),
        calls=calls,
        lossless=not select,
        warnings=_warnings(slots, filled, actual),
        summary=_summarize(actual, select, iterate, passed, sizes, calls),
    )


def _check_actual(
    param: _spec.ParamSpec, actual: tuple[str, ...], shape: Sequence[int]
) -> None:
    """Reject labels that do not describe the array they claim to."""
    if len(actual) != len(shape):
        raise ValueError(
            f"Parameter {param.name!r}: {len(actual)} axis label(s) "
            f"({_show(actual)}) for a {len(shape)}-dimensional array {tuple(shape)}"
        )
    named = [label for label in actual if label]
    if len(set(named)) != len(named):
        raise ValueError(f"Parameter {param.name!r}: repeated axis in {_show(actual)}")


def _default_mapping(
    slots: tuple[_spec.Slot, ...], actual: tuple[str, ...], param: _spec.ParamSpec
) -> tuple[int | None, ...]:
    """Decide which input axis fills which slot, best effort.

    Names come first, because a name match is evidence. What is left over is
    assigned by position, *right-aligned*, which is why a plain unlabelled 3-D
    array feeds a ``("z", "y", "x")`` op as 0, 1, 2 -- the innermost axes are
    the ones an imaging op means when it says ``y x``.

    Optional slots are filled by name and by nothing else. Handing a ``(z, y,
    x)`` stack to ``Axes("y", "x", "c?")`` must never drop ``z`` into the
    channel slot, where an op would average across it rather than iterate.
    """
    mapping: list[int | None] = [None] * len(slots)
    taken: set[int] = set()

    for s, slot in enumerate(slots):
        if slot.name is None:
            continue
        for i, label in enumerate(actual):
            if i not in taken and label and label == slot.name:
                mapping[s] = i
                taken.add(i)
                break

    empty = [
        s for s, slot in enumerate(slots) if mapping[s] is None and not slot.optional
    ]
    free = [i for i in range(len(actual)) if i not in taken]
    if len(free) < len(empty):
        required = len([slot for slot in slots if not slot.optional])
        raise ValueError(
            f"Parameter {param.name!r} consumes {required} axes but was given "
            f"{len(actual)} ({_show(actual)}). No adaptation can invent one."
        )
    for s, i in zip(empty, free[len(free) - len(empty) :]):
        mapping[s] = i

    return tuple(mapping)


def _checked_mapping(
    mapping: Sequence[int | None],
    slots: tuple[_spec.Slot, ...],
    actual: tuple[str, ...],
    param: _spec.ParamSpec,
) -> tuple[int | None, ...]:
    """Validate a mapping someone else chose."""
    if len(mapping) != len(slots):
        raise ValueError(
            f"Parameter {param.name!r} has {len(slots)} axis slot(s), but the "
            f"mapping gives {len(mapping)}"
        )
    seen: set[int] = set()
    for slot, index in zip(slots, mapping):
        if index is None:
            if not slot.optional:
                raise ValueError(
                    f"Parameter {param.name!r}: slot {str(slot)!r} is required, "
                    "so it cannot be left unmapped"
                )
            continue
        if not 0 <= index < len(actual):
            raise ValueError(
                f"Parameter {param.name!r}: slot {str(slot)!r} maps to axis "
                f"{index}, which the array does not have"
            )
        if index in seen:
            raise ValueError(
                f"Parameter {param.name!r}: {_name(actual, index)} is mapped to "
                "more than one slot"
            )
        seen.add(index)
    return tuple(mapping)


def _dispositions(
    leftover: list[int],
    given: dict[int, str] | None,
    declared: _spec.Axes,
    param: _spec.ParamSpec,
) -> dict[int, str]:
    """What happens to each axis no slot took.

    The default keeps everything: a variadic op is handed its leftovers whole,
    since it said it copes with them, and anything else is iterated. Neither
    discards data, so the automatic answer never silently loses any -- that
    only happens when a caller asks for it.
    """
    default = PASS if declared.variadic else ITERATE
    chosen = {i: default for i in leftover}
    for key, value in (given or {}).items():
        index = int(key)
        if index not in chosen:
            raise ValueError(
                f"Parameter {param.name!r}: axis {index} is consumed by a slot, "
                "so it has no disposition"
            )
        if value not in DISPOSITIONS:
            raise ValueError(
                f"Parameter {param.name!r}: {value!r} is not one of "
                f"{', '.join(DISPOSITIONS)}"
            )
        if value == PASS and not declared.variadic:
            raise ValueError(
                f"Parameter {param.name!r} is not variadic, so it cannot be "
                "handed extra axes. Iterate over them, or select one position."
            )
        chosen[index] = value
    return chosen


def _warnings(
    slots: tuple[_spec.Slot, ...],
    mapping: tuple[int | None, ...],
    actual: tuple[str, ...],
) -> tuple[str, ...]:
    """Flag slots being fed an axis that is not what they asked for.

    A name is a hint, so a mismatch is worth saying out loud and not worth
    refusing: it is exactly the case where the op will run happily and may not
    be computing what the user meant. An unnamed axis says nothing either way,
    and a wildcard slot asked for nothing, so neither warns.
    """
    notes = []
    for slot, index in zip(slots, mapping):
        if slot.name is None or index is None:
            continue
        label = actual[index]
        if label and label != slot.name:
            notes.append(f"{slot.name} is being fed the {label} axis")
    return tuple(notes)


def _summarize(
    actual: tuple[str, ...],
    select: tuple[tuple[int, int], ...],
    iterate: tuple[int, ...],
    passed: tuple[int, ...],
    sizes: dict[int, int],
    calls: int,
) -> str:
    """One line saying what will happen, for a front end to show."""
    parts = []
    if iterate:
        where = "/".join(_name(actual, i) for i in iterate)
        parts.append(f"run {calls} times, once per {where} position")
    if passed:
        extent = ", ".join(f"{sizes[i]} {_name(actual, i)}" for i in passed)
        parts.append(f"pass {extent} through to the op")
    if select:
        where = ", ".join(f"{_name(actual, i)}={at}" for i, at in select)
        parts.append(f"run at {where}, discarding the rest")
    return "; ".join(parts) if parts else "as is"


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
