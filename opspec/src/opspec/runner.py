"""Interfaces for running ops. No implementation.

Two jobs, two Protocols:

- ``Builder`` provides environments. Slow, rare, needs progress.
- ``Runner`` calls ops. One per call.

Structural, so an existing class satisfies them by having the methods.
It need not import opspec or inherit anything.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

__all__ = ["Builder", "Runner"]

#: Importable now.
READY = "ready"
#: Not here, but this builder can provide it.
MISSING = "missing"
#: Provided, but this process must restart to see it.
RESTART_REQUIRED = "restart_required"
#: This builder cannot provide it.
UNAVAILABLE = "unavailable"

STATUSES = (READY, MISSING, RESTART_REQUIRED, UNAVAILABLE)


@runtime_checkable
class Builder(Protocol):
    """Provides the environments ops ask for with ``env=``.

    An out-of-process builder solves and installs one. An in-process
    builder may install into the interpreter it is running in, which is
    what ``restart_required`` is for.
    """

    def environment_status(self, env_id: str) -> str:
        """One of ``STATUSES``."""
        ...

    def ensure_environment(
        self, env_id: str, on_progress: Callable[[str, int, int], None] | None = None
    ) -> None:
        """Make the environment usable, or raise.

        ``on_progress(title, current, maximum)`` reports a long build.
        """
        ...


@runtime_checkable
class Runner(Protocol):
    """Calls ops.

    ``run`` blocks and returns the op's result. A caller wanting to cancel
    takes the task through ``on_start``.
    """

    def run(
        self,
        fn: Any,
        args: dict | None = None,
        *,
        axes: dict[str, Any] | None = None,
        plans: dict[str, Any] | None = None,
        position: dict[str, int] | None = None,
        on_progress: Callable[[Any], None] | None = None,
        on_start: Callable[[Any], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run ``fn`` and return its result.

        Args:
            fn: The op, as decorated with ``@op``.
            args: Op arguments, merged with ``**kwargs``.
            axes: What each array argument is, ``{"image": list("zyx")}``.
            plans: Ready-made plans, by parameter name, instead of ``axes``.
            position: Where to sit on each selected axis.
            on_progress: Called with progress events.
            on_start: Called with the task, for cancelling.
        """
        ...

    def close(self) -> None:
        """Release whatever this runner holds."""
        ...
