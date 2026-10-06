"""A result type shared by ops in other modules, as skop's detectors share one."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from opspec.types import BoxesOf


class Boxes(NamedTuple):
    boxes: BoxesOf[np.ndarray]
