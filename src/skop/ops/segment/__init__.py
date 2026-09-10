"""Segmentation ops: label the objects in an image.

Each op here gets a module of its own, because each brings its own
environment, its own pretrained models and its own supporting types. The
re-exports below are the namespace's public surface -- written out rather
than generated, so that an IDE can offer them.
"""

from __future__ import annotations

from .cellpose4 import cellpose4
from .cellpose3 import cellpose3
from .stardist2d import stardist2d_fluo, stardist2d_he
from .starfun3d import segment_nuclei
from .unseg import unseg

__all__ = [
    "cellpose4",
    "cellpose3",
    "segment_nuclei",
    "stardist2d_fluo",
    "stardist2d_he",
    "unseg",
]
