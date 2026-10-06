"""Fitting the array a caller passes to the axes an op consumes.

The op declares ``Axes``. The caller names its array's axes. This works out
which axis fills which slot, what to loop over, and how many calls that is,
as an ``AdaptationPlan``.

Arithmetic over names and shapes only: no array is touched, so no numpy.
Carrying a plan out belongs to a runner.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .op import CANONICAL, Axes, OpSpec, ParamSpec, Slot, canonical

#: What becomes of an input axis that no slot consumed.
ITERATE = "iterate"  # call the op once per position, and stack the results
SELECT = "select"  # index down to one position, discarding the rest
PASS = "pass"  # hand it to the op as an axis, for a variadic op to deal with

DISPOSITIONS = (ITERATE, SELECT, PASS)


@dataclass(frozen=True)
class AdaptationPlan:
    """How to get from the array a caller has to the one an op accepts.

    Applied in order: ``select`` down to one position each, ``transpose``
    what remains, then one call per position of the leading ``iterate``
    axes.

    ``mapping`` is the editable half -- one input axis per declared slot,
    None for an optional slot left empty. The rest follows from it.
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
    uses_all_data: bool
    warnings: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict:
        """A JSON-safe form, for the trip to another process."""
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
            "uses_all_data": self.uses_all_data,
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
            uses_all_data=data["uses_all_data"],
            warnings=tuple(data["warnings"]),
            summary=data["summary"],
        )


def normalize_axes(axes: str | Sequence[str | None]) -> tuple[str, ...]:
    """Axis labels, one string per axis: ``list("zyx")``, not ``"zyx"``.

    - Resolved through ``canonical``, so ``("pln", "row", "col")`` is zyx.
    - ``"zyx"`` as one string is refused, not guessed at.
    - ``None`` is an unnamed axis: filled positionally, never warned about.
    """
    if isinstance(axes, str):
        if len(axes) > 1 and all(char in CANONICAL for char in axes.casefold()):
            raise ValueError(
                f"{axes!r} is one axis label. If you meant {len(axes)} axes, "
                f"write list({axes!r})."
            )
        return (canonical(axes),)
    return tuple("" if label is None else canonical(str(label)) for label in axes)


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
    position: dict[str | int, int] | None = None,
    mapping: Sequence[int | None] | None = None,
    dispositions: dict[int, str] | None = None,
) -> AdaptationPlan:
    """Work out how *array* should be fed to op *fn*'s *param*.

    With axes alone, returns the best guess. With *mapping* or
    *dispositions*, returns what the caller asked for -- which is how a GUI
    offers per-axis control.

    Args:
        fn: The op function.
        param: Which parameter the array is for.
        array: The array, anything with ``shape``, or a shape.
        axes: What the array is, e.g. ``list("zyx")``. ``None`` for an
            unnamed axis. Not guessed.
        position: Where to sit on each selected axis. A viewer's sliders,
            keyed by name or index. Missing axes sit at 0.
        mapping: One input axis (or None) per slot, overriding the guess.
        dispositions: Per leftover axis, by index: ``"iterate"``,
            ``"select"`` or ``"pass"``.

    Raises:
        ValueError: when no mapping could work -- too few axes, or labels
            that do not describe the array.
    """
    spec = OpSpec.from_op(fn)
    found = next((p for p in spec.params if p.name == param), None)
    if found is None:
        known = ", ".join(p.name for p in spec.params)
        raise ValueError(f"Op {spec.name} has no parameter {param!r}. Has: {known}")
    shape = getattr(array, "shape", array)
    return build(found, axes, tuple(shape), position, mapping, dispositions)


def build(
    param: ParamSpec,
    axes: str | Sequence[str | None],
    shape: Sequence[int],
    position: dict[str | int, int] | None = None,
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
    # By index first: an index names exactly one axis, where a name may name
    # none of them (an unnamed axis) or, before _check_actual, several.
    select = tuple(
        (i, int(at[i] if i in at else at.get(actual[i], 0)))
        for i in leftover
        if chosen[i] == SELECT
    )
    iterate = tuple(i for i in leftover if chosen[i] == ITERATE)
    passed = tuple(i for i in leftover if chosen[i] == PASS)

    sizes = dict(enumerate(shape))
    calls = math.prod(sizes[i] for i in iterate) if iterate else 1

    # Iterated axes lead, so a runner can walk them with one np.ndindex.
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
        uses_all_data=not select,
        warnings=_warnings(slots, filled, actual),
        summary=_summarize(actual, select, iterate, passed, sizes, calls),
    )


def _check_actual(
    param: ParamSpec, actual: tuple[str, ...], shape: Sequence[int]
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
    slots: tuple[Slot, ...], actual: tuple[str, ...], param: ParamSpec
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
    slots: tuple[Slot, ...],
    actual: tuple[str, ...],
    param: ParamSpec,
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
    declared: Axes,
    param: ParamSpec,
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
    slots: tuple[Slot, ...],
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
