"""skop: write an op once, run it anywhere.

An op is an ordinary Python function carrying an ``@op`` decorator that names
the environment it needs. The same function can be called directly, run in
its own environment through Appose, or wrapped in a GUI.

The top level of this package is deliberately austere -- standard library,
numpy and appose only -- because it is imported inside every op environment.
Anything heavier lives in a submodule that is imported lazily.
"""

from __future__ import annotations

from typing import Any

from ._adapt import ITERATE, PASS, SELECT, AdaptationPlan, plan
from ._progress import cancel_requested, progress
from ._spec import (
    BOOL,
    COMPUTER,
    ENUM,
    FLOAT,
    FUNCTION,
    INPLACE,
    INT,
    NDARRAY,
    PATH,
    STR,
    UNKNOWN,
    WIRE_TYPES,
    Axes,
    Choice,
    Choices,
    Mut,
    OpSpec,
    Out,
    OutputSpec,
    ParamsFor,
    ParamSpec,
    Role,
    Slot,
    TypeSpec,
    is_op,
    op,
    spec,
    type_spec,
)

__all__ = [
    "BOOL",
    "COMPUTER",
    "ENUM",
    "FLOAT",
    "FUNCTION",
    "INPLACE",
    "INT",
    "ITERATE",
    "NDARRAY",
    "PASS",
    "PATH",
    "SELECT",
    "STR",
    "UNKNOWN",
    "WIRE_TYPES",
    "AdaptationPlan",
    "Axes",
    "Choice",
    "Choices",
    "Mut",
    "OpSpec",
    "Out",
    "OutputSpec",
    "ParamSpec",
    "ParamsFor",
    "Role",
    "Runner",
    "Slot",
    "TypeSpec",
    "cancel_requested",
    "discover",
    "is_op",
    "op",
    "plan",
    "progress",
    "run",
    "spec",
    "type_spec",
]

_LAZY = {"Runner": "runner", "run": "runner", "discover": "discovery"}


def __getattr__(name: str) -> Any:
    # Host-side machinery is imported on demand, so that merely importing
    # skop inside a worker stays cheap.
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module_name}", __name__), name)
