"""Fit an nd input array for the axes an op supports.

The op declares ``Axes``, and the caller names its array's axes. ``plan``
works out which axis fills which slot, which axes to loop over, and how many
calls that takes, and returns an ``AdaptationPlan``.

Only names and shapes are used, never the array itself, so no numpy. A
runner carries the plan out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .op import CANONICAL, Axes, OpSpec, ParamSpec, Slot, canonical

#: What happens to an input axis that no slot takes.
ITERATE = "iterate"  # call the op once per position and stack the results
SELECT = "select"  # keep one position and drop the rest
PASS = "pass"  # give the axis to the op; only a variadic op accepts this

DISPOSITIONS = (ITERATE, SELECT, PASS)


@dataclass(frozen=True)
class AdaptationPlan:
    """How to turn the caller's array into the one the op accepts.

    In order: ``select`` one position on each selected axis, ``transpose``
    what is left, then call the op once per position of the leading
    ``iterate`` axes.

    A caller can choose ``mapping`` and what happens to the leftover axes
    (see ``plan``). Everything else follows from those.
    """

    param: str  # the parameter this plan is for
    input_axes: tuple[str, ...]  # the caller's axis names; "" if unnamed
    mapping: tuple[int | None, ...]  # per slot, the input axis; None if empty
    select: tuple[tuple[int, int], ...]  # (axis, position) pairs to keep
    iterate: tuple[int, ...]  # axes to loop over
    passed: tuple[int, ...]  # axes given to a variadic op as they are
    transpose: tuple[int, ...]  # axis order after select
    output_axes: tuple[str, ...]  # axis names after select and transpose
    calls: int  # how many times the op is called
    uses_all_data: bool  # False if select drops data
    warnings: tuple[str, ...]  # slots given an axis with a different name
    summary: str  # one line for a GUI

    def to_dict(self) -> dict:
        """A JSON-safe dict, for sending to another process."""
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

    - Synonyms go through ``canonical``: ``("pln", "row", "col")`` is z, y, x.
    - ``"zyx"`` as one string is an error, not a guess.
    - ``None`` is an unnamed axis, returned as ``""``. It is filled by
      position and never warned about.
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
    """Axis labels for an error message; an unnamed axis shows as "axis N"."""
    return ", ".join(_name(axes, i) for i in range(len(axes)))


def _name(axes: Sequence[str], index: int) -> str:
    """An axis's name, or "axis N" if it has none."""
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
    """Work out how to pass *array* to parameter *param* of op *fn*.

    With only *axes*, returns the best guess. *mapping* and *dispositions*
    override the guess; a GUI uses them to give per-axis control.

    Args:
        fn: The op function.
        param: The parameter the array is for.
        array: The array, anything with a ``shape``, or a shape.
        axes: The array's axis names, e.g. ``list("zyx")``; ``None`` for an
            unnamed axis. Never guessed.
        position: The position on each selected axis, e.g. a viewer's
            sliders, keyed by name or index. Missing axes use 0.
        mapping: For each slot, the input axis that fills it, or None.
            Replaces the guess.
        dispositions: For leftover axes, by index: ``"iterate"``,
            ``"select"`` or ``"pass"``.

    Raises:
        ValueError: If no mapping can work: too few axes, or labels that
            don't match the array.
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
    """``plan``, from a ParamSpec instead of a function."""
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
    # Look up by index first: an index is always one axis, while a name may
    # match none (an unnamed axis).
    select = tuple(
        (i, int(at[i] if i in at else at.get(actual[i], 0)))
        for i in leftover
        if chosen[i] == SELECT
    )
    iterate = tuple(i for i in leftover if chosen[i] == ITERATE)
    passed = tuple(i for i in leftover if chosen[i] == PASS)

    sizes = dict(enumerate(shape))
    calls = math.prod(sizes[i] for i in iterate) if iterate else 1

    # Iterated axes go first, so a runner can loop over them with one
    # np.ndindex. Passed axes go next, in front of the op's own axes, which is
    # where a variadic op expects extra axes.
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
        # In the caller's names: an axis keeps its name even when it fills a
        # slot with a different one.
        output_axes=tuple(actual[i] for i in target),
        calls=calls,
        uses_all_data=not select,
        warnings=_warnings(slots, filled, actual),
        summary=_summarize(actual, select, iterate, passed, sizes, calls),
    )


def _check_actual(
    param: ParamSpec, actual: tuple[str, ...], shape: Sequence[int]
) -> None:
    """Refuse labels that don't match the array: wrong count, or a repeat."""
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
    """Guess which input axis fills which slot.

    Names first, since a matching name is evidence. Required slots still
    empty are filled by position from the right, so an unlabelled 3-D array
    feeds a ``("z", "y", "x")`` op as 0, 1, 2: by ``y x`` an imaging op means
    the innermost axes.

    Optional slots are filled by name only. Given a ``(z, y, x)`` stack,
    ``Axes("y", "x", "c?")`` must not take ``z`` as its channel, or the op
    would average over z instead of looping over it.
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
    """Check a mapping the caller chose."""
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

    By default nothing is dropped: a variadic op gets the leftover axes,
    since it says it can handle them, and any other op is looped over them.
    Data is dropped only when the caller asks for ``select``.
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
    """Warn when a slot gets an axis with a different name.

    A name is a hint, so a mismatch is a warning, not an error: the op runs,
    but may not compute what the user meant. An unnamed axis or a wildcard
    slot never warns.
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
    """One line saying what will happen, for a GUI to show."""
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
