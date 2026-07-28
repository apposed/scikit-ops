"""Mask detection: given boxes, draw the object inside each one.

Stage two of a detect-then-segment workflow, fed by ``skop.ops.detect``. The
distinction from ``skop.ops.segment`` is what decides where the objects are:
Cellpose and StarDist decide for themselves and return a label image, while a
mask detector is *told*, by a prompt, and returns one mask per prompt. That
separation is what lets a workflow pair any detector with any mask detector.

They return masks rather than labels because SAM-family masks overlap -- a
cell and the debris on it, a coin behind a coin -- and a label image cannot
say so. ``skop.masks`` holds the projections onto something a viewer shows.

Every op here shares a signature and returns ``Masks``, so they are
substitutable. One op per environment, since ``@op(env=...)`` is fixed per
function: ``microsam_masks`` in the shared 'pytorch' environment, because
micro_sam is a conda package that coexists with every other torch library
there, and ``mobilesam_masks`` in segment-everything's own, alongside the
object-aware detector it was trained to pair with and inside the vendored
tree it cannot leave.
"""

from __future__ import annotations

from ._result import Masks
from .microsam import microsam_masks
from .mobilesam import mobilesam_masks

__all__ = ["Masks", "microsam_masks", "mobilesam_masks"]
