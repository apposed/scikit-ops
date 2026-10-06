"""Object detection, class-agnostic or with a user's trained YOLO model.

FastSAM and object-aware YOLO return the same thing, so a
caller can swap one for the other -- the first stage of a detect-then-segment
workflow, feeding boxes to a mask detector.

Those two are class-agnostic on purpose. A COCO-pretrained detector has no
category for a cell or a coin and reports nothing; these have a single class,
called object, and report everything.

One op per environment, since ``@op(env=...)`` is fixed per function:
``fastsam`` in the shared 'pytorch' environment, ``object_aware_yolo`` in one
of its own, because segment-everything vendors an ultralytics fork that must
not meet the real one.

``yolo`` runs a user's detection checkpoint on overlapping tiles, returning
boxes, confidences and class IDs. It shares the 'pytorch' environment.
"""

from __future__ import annotations

from ._result import Boxes, Detections
from .fastsam import fastsam
from .jdll_yolo import jdll_yolo
from .object_aware_yolo import object_aware_yolo
from .yolo import yolo

__all__ = ["Boxes", "Detections", "fastsam", "jdll_yolo", "object_aware_yolo", "yolo"]
