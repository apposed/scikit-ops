"""Record what a Runner ran (optional).

``Runner(recorder=...)`` hands every finished run to a ``Recorder``. With
none, nothing is recorded (the default, as before).

Recording happens here in the host, never in the op's environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Record:
    """One recorded run. The same shape for every recorder."""

    id: str
    message: str
    #: name, op, params and inputs
    metadata: dict


class Recorder(Protocol):
    def identify(self, value) -> str | None:
        """Where an input came from, or None if untracked."""

    def save(self, name: str, op: str, params: dict, inputs: dict, result) -> Any:
        """Record one run; return the result (stored, or as-is)."""

    def log(self, name: str) -> list[Record]:
        """Past runs saved as ``name``, newest first."""
