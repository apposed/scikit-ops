"""Segmentation ops: label the objects in an image.

Each op here gets a module of its own, because each brings its own
environment, its own pretrained models and its own supporting types. The
re-exports below are the namespace's public surface -- written out rather
than generated, so that an IDE can offer them.
"""

from __future__ import annotations

from .cellpose import cellpose
from .stardist2d import stardist2d
from .starfun3d import segment_nuclei
from .unseg import unseg

__all__ = ["cellpose", "segment_nuclei", "stardist2d", "unseg"]
