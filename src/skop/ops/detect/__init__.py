"""Object detection: where are the objects, without saying what they are.

Both ops here answer the same question and return the same thing, so a
caller can swap one for the other -- the first stage of a detect-then-segment
workflow, feeding boxes to a mask detector.

They are class-agnostic on purpose. A COCO-pretrained detector has no
category for a cell or a coin and reports nothing; these have a single class,
called object, and report everything.

One op per environment, since ``@op(env=...)`` is fixed per function:
``fastsam`` in the shared 'pytorch' environment, ``object_aware_yolo`` in one
of its own, because segment-everything vendors an ultralytics fork that must
not meet the real one.
"""

from __future__ import annotations

from ._result import Boxes
from .fastsam import fastsam
from .object_aware_yolo import object_aware_yolo

__all__ = ["Boxes", "fastsam", "object_aware_yolo"]
