"""Progress reporting from inside an op.

An op calls ``skop.progress(...)`` without knowing whether it is running
under Appose or being called directly. Outside a worker, it is a no-op.
"""

from __future__ import annotations

import contextvars
from typing import Any

_current: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "skop_task", default=None
)


def progress(
    message: str | None = None,
    current: int | None = None,
    maximum: int | None = None,
) -> None:
    """Report progress to whoever is running this op, if anyone is listening."""
    task = _current.get()
    if task is None:
        return
    task.update(message=message, current=current, maximum=maximum)


def cancel_requested() -> bool:
    """Whether the caller has asked this op to stop early."""
    task = _current.get()
    return bool(task is not None and task.cancel_requested)


def _bind(task: Any):
    return _current.set(task)


def _unbind(token) -> None:
    _current.reset(token)
