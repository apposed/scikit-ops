"""Record which ops were run and parameters (optional).

``Runner(recorder=...)``  sets a recorder that is called after the op finishes:

- none: nothing is recorded (the default)

Implemented so far

- ``TextRecorder``: one JSON line per run, standard library only
- ``IcechunkRecorder`` (``skop.icechunk_recorder``): the recipe plus the
  result image, as an Icechunk commit

Recording happens here in the host, never in the op's environment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Record:
    id: str
    message: str
    #: name, op, params and inputs
    metadata: dict


class Recorder(Protocol):
    def identify(self, value) -> str | None:
        """the source (name, path, etc) of an input (for numpy nothing, for Zarr the array path)"""

    def save(self, name: str, op: str, params: dict, inputs: dict, result) -> Any:
        """Record the actions; return the result (stored, or as-is)."""

    def log(self, name: str) -> list[Record]:
        """Past runs saved as ``name``, newest first."""


class TextRecorder:
    """One JSON line per run. Writes text description of the op and parameters"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def identify(self, value) -> str | None:
        return None  # plain arrays have no address

    def save(self, name, op, params, inputs, result):
        metadata = {"name": name, "op": op, "params": params, "inputs": inputs}
        line = {
            "id": str(time.time_ns()),
            "message": f"{op} {params}",
            "metadata": metadata,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        return result

    def log(self, name):
        if not self.path.exists():
            return []
        lines = [json.loads(s) for s in self.path.read_text().splitlines()]
        return [Record(**r) for r in reversed(lines) if r["metadata"]["name"] == name]
