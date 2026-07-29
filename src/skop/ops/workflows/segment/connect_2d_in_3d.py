"""Segment a volume plane by plane, then join the planes back into objects.

The workflow ``skop.ops.normalize`` was written for. A 2-D segmenter handed a
volume is looped over its planes, and every one of these segmenters normalizes
whatever array it is given -- so each plane gets stretched to its own range,
and a top or bottom plane holding nothing but out-of-focus blur is stretched
until that blur fills the range and reads as nuclei. The segmenter then finds
objects there that are not there.

Three ops in a row fix it, and the order is the whole point:

1. normalize over the **whole volume**, so an empty plane stays dim relative
   to the planes with signal in them.
2. the segmenter, **plane by plane and with its own normalization off**, so
   nothing puts the stretch back.
3. connect, which links objects that overlap between adjacent planes, so one
   cell stops being a stack of unrelated labels.

Step 2 is why ``normalize`` is bound rather than offered. Leaving a checkbox
there would let someone switch back on precisely the behaviour this exists to
avoid, and get a result that looks like a failure of the idea rather than of
the setting. Running a segmenter with its own normalization on is still a
perfectly good thing to do -- from the Ops panel, where that is what you asked
for.

**All three stages are choosers, including the two with one choice each.**
There is only one normalizer and one connector today and there will be more,
and a stage that starts as a chooser gains its second option by having a name
added to a list rather than by being rewritten. It also keeps the shape
uniform: every stage's settings come from the chosen op's own signature, so
nothing here restates a parameter that lives somewhere else and can drift from
it.

**Three dimensions, and the dimensions are held fixed.** The connector needs
an axis to walk along, so there is no 2-D version of this. The workflow
declares ``Axes("z", "y", "x")`` on its own input, which is what lets a front
end ask which of *your* axes is z -- but it does not pass that question down.
Each stage's axis handling is pinned here, because getting it wrong is the bug
the workflow is for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, NamedTuple

from skop import Axes, Choices, ParamsFor, op, progress, run
from skop.ops.labels import connect
from skop.ops.normalize import percentile
from skop.ops.segment.cellpose import cellpose
from skop.ops.segment.cellpose3 import cellpose3
from skop.ops.segment.stardist2d import stardist2d_fluo
from skop.types import ImageData, LabelsData

#: What every stage is handed, and what the workflow refuses to make adjustable.
_ZYX = list("zyx")


class Connected(NamedTuple):
    """What each stage produced, in the order the stages ran.

    Stage order rather than importance: a front end adds these to a viewer in
    declaration order, so the pipeline reads down the layer list, and the last
    stage -- the answer -- ends up on top where a new layer belongs.
    """

    #: The normalized volume the segmenter actually saw. Returned because it
    #: is the evidence: if the top plane is still dim here, the normalization
    #: did its job, and a spurious object there is the segmenter's doing.
    normalized: ImageData
    #: One label per object, consistent through the stack.
    labels: LabelsData


@op()
def connect_2d_in_3d(
    image: Annotated[ImageData, Axes("z", "y", "x")],
    normalizer: Annotated[Callable, Choices(percentile=percentile)] = percentile,
    normalizer_args: Annotated[
        dict | None, ParamsFor("normalizer", binds="image")
    ] = None,
    segmenter: Annotated[
        Callable,
        Choices(stardist=stardist2d_fluo, cellpose=cellpose, cellpose3=cellpose3),
    ] = stardist2d_fluo,
    segmenter_args: Annotated[
        dict | None, ParamsFor("segmenter", binds=("image", "normalize"))
    ] = None,
    connector: Annotated[Callable, Choices(connect=connect)] = connect,
    connector_args: Annotated[
        dict | None, ParamsFor("connector", binds="labels")
    ] = None,
) -> Connected:
    """Segment a volume plane by plane and join the planes into objects.

    Args:
        image: The volume, as z, y, x. Normalized once, whole, before any
            plane is segmented.
        normalizer: Which op scales the volume. Whichever it is, it is run on
            the whole volume rather than plane by plane -- that is the step
            this workflow exists to get right.
        normalizer_args: Settings for the chosen normalizer, minus the volume.
        segmenter: Which 2-D segmenter to run on each plane. Each brings its
            own environment, and picking one for the first time builds it.
        segmenter_args: Settings for the chosen segmenter, minus the plane and
            its normalization, both of which this workflow supplies.
        connector: Which op links objects between adjacent planes.
        connector_args: Settings for the chosen connector, minus the labels.

    Returns:
        normalized: The volume as the segmenter saw it.
        labels: The connected label volume, numbered from 1.

    Note: the three stages live in three environments, so a run crosses shared
    memory three times. That is the cost of composing on the host, and it is
    paid per call rather than per plane -- skop loops the segmenter over z
    inside its own worker.
    """
    progress("Normalizing the whole volume")
    # Naming the axes hands a variadic op the volume entire rather than a
    # plane at a time. This is the step the whole workflow exists for.
    normalized = run(
        normalizer, image=image, axes={"image": _ZYX}, **(normalizer_args or {})
    )

    progress("Segmenting plane by plane")
    # The segmenter declares two axes and is given three, so skop loops it over
    # z. normalize=False is what keeps it from undoing the step above.
    labels = run(
        segmenter,
        image=normalized,
        normalize=False,
        axes={"image": _ZYX},
        **(segmenter_args or {}),
    )

    progress("Connecting objects between planes")
    joined = run(
        connector, labels=labels, axes={"labels": _ZYX}, **(connector_args or {})
    )

    return Connected(normalized, joined)
