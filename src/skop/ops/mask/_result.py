"""What every mask detector returns.

One class, shared by every mask detector op, so that "these ops are
substitutable" is a fact rather than a convention -- a workflow choosing
between them gets the same fields and the same output names either way. The
same reasoning as ``skop.ops.detect._result``, one stage downstream.

Two fields rather than one, because a mask detector does not always answer
every prompt. SAM returns an empty prediction for a box it cannot make sense
of, both implementations drop those, and so N out is not N in. Without the
surviving prompts there is no way back to which detection each mask came
from.

The masks are a stack rather than a label image because SAM masks overlap,
and rather than the list of annotation dicts the SAM ecosystem passes around
because that cannot cross the Appose boundary: ``skop._codec`` rejects
``bool`` outright, and N dicts of arrays would become N shared-memory blocks
where one stack is one. See docs/design/0008-mask-detector-ops.md.

No area, no stability score, no predicted IoU. Area is
``masks.sum(axis=(1, 2))`` and does not need carrying; the other two are
MobileSAM's and have no micro_sam counterpart, so they cannot live in a shared
return type until docs/design/0009-per-object-features.md says how per-object
values travel. The same reason ``Boxes`` carries no scores.
"""

from __future__ import annotations

from typing import NamedTuple

from skop.types import BoxesData, MasksData


class Masks(NamedTuple):
    #: (N, Y, X) uint8, 0 or 1, one object per plane. These may overlap.
    masks: MasksData
    #: (N, 4) as [min_y, min_x, max_y, max_x] -- the prompt each mask answers.
    boxes: BoxesData
