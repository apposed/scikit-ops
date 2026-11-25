"""micro_sam's SAM1 code, wherever the installed micro_sam keeps it.

micro_sam 2 moved it under ``micro_sam.v1``; 1.x has it at the top. Call
only inside op bodies, so discovery never imports micro_sam.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def sam1(module: str) -> ModuleType:
    """``micro_sam.v1.<module>`` on micro_sam 2, ``micro_sam.<module>`` on 1.x."""
    try:
        return importlib.import_module(f"micro_sam.v1.{module}")
    except ModuleNotFoundError as error:
        # Only a missing micro_sam.v1 means 1.x; anything else is a real failure.
        if error.name not in ("micro_sam.v1", f"micro_sam.v1.{module}"):
            raise
        return importlib.import_module(f"micro_sam.{module}")
