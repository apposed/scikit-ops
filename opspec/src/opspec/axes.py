"""Axes

an annotation that is used by parameters to express what axes is expected.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        # Names are kept as written; matching them to data axes is planning's
        # job (plan.canonical), as is checking them (plan.check_axes).
        if len(names) == 1 and not isinstance(names[0], str):
            # A lone non-string is the sequence itself: Axes(list("zyx")).
            names = tuple(names[0])
        object.__setattr__(self, "slots", tuple(_slot(label) for label in names))
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


def _slot(label: Any) -> Slot:
    """Read one label as written: a trailing '?' is optional, '*' is any axis."""
    if not isinstance(label, str):
        raise TypeError(f"Axis label {label!r} is not a string")
    text = label.strip()
    optional = text.endswith("?")
    text = text.removesuffix("?")
    return Slot(None if text == WILDCARD else text, optional)
