"""Find the objects, then draw each one: a detector paired with a masker.

The two-stage segmenter from docs/spec/workflow-ops.md, and the reason the
detectors and mask detectors were built to share signatures in the first
place. Any of the two detectors pairs with either of the two mask detectors,
which is four segmenters from four ops -- and the point of the pairing is that
they are genuinely different: the object-aware YOLO finds objects a
photograph-trained SAM misses, and micro_sam draws cells that MobileSAM
smears.

Both stages take the image. Only one asks for it: ``binds`` on each stage's
arguments names what this workflow supplies, so a front end renders the rest
and nothing gets typed twice. The masker's ``boxes`` are bound for the same
reason, though they come from the detector rather than from the panel.

Note that a pairing may cross environments -- ``fastsam`` in 'pytorch' with
``mobilesam_masks`` in 'segment-everything' means two workers, and the image
crosses shared memory twice. That is the cost of composing on the host, and
it is why this is a workflow rather than an op.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, NamedTuple

from skop import Choices, ParamsFor, op, progress, run
from skop.ops.detect.fastsam import fastsam
from skop.ops.detect.object_aware_yolo import object_aware_yolo
from skop.ops.mask.microsam import microsam_masks
from skop.ops.mask.mobilesam import mobilesam_masks
from skop.types import BoxesData, ImageData, MasksData


class Detected(NamedTuple):
    """What each stage produced, in the order the stages ran.

    Stage order rather than importance: a front end adds these to a viewer in
    declaration order, so the pipeline reads down the layer list, and the last
    stage -- the answer -- ends up on top where a new layer belongs. This is
    why the fields sit the other way round from ``skop.ops.mask.Masks``, which
    every mask detector returns. There, the masks are the point and the boxes
    record which prompts survived; here they are stage one and stage two.
    """

    #: The prompts that produced the masks: the detector's boxes, minus any
    #: the mask detector answered with nothing.
    boxes: BoxesData
    #: (N, Y, X) uint8, one object per plane, possibly overlapping.
    masks: MasksData


@op()
def detect_then_mask(
    image: ImageData,
    detector: Annotated[
        Callable,
        Choices(object_aware_yolo=object_aware_yolo, fastsam=fastsam),
    ] = object_aware_yolo,
    detector_args: Annotated[dict | None, ParamsFor("detector", binds="image")] = None,
    masker: Annotated[
        Callable,
        Choices(mobilesam=mobilesam_masks, microsam=microsam_masks),
    ] = mobilesam_masks,
    masker_args: Annotated[
        dict | None, ParamsFor("masker", binds=("image", "boxes"))
    ] = None,
) -> Detected:
    """Detect objects and segment each one.

    Args:
        image: The image to segment.
        detector: Which op finds the bounding boxes. ``object_aware_yolo`` was
            trained on microscopy; ``fastsam`` is a general detector.
        detector_args: Settings for the chosen detector, minus the image.
        masker: Which op draws a mask inside each box. ``microsam`` is
            finetuned for microscopy and slower; ``mobilesam`` is the one the
            object-aware detector was trained to pair with.
        masker_args: Settings for the chosen mask detector, minus the image
            and the boxes, which this workflow supplies.

    Returns:
        boxes: the box each mask answers.
        masks: (N, Y, X) uint8, one object per plane, possibly overlapping.

    A detector that finds nothing is an ordinary outcome, not a failure: the
    mask detector is skipped and the result is empty.
    """
    progress("Detecting objects")
    boxes = run(detector, image=image, **(detector_args or {})).boxes

    progress(f"Segmenting {len(boxes)} objects")
    found = run(masker, image=image, boxes=boxes, **(masker_args or {}))
    return Detected(found.boxes, found.masks)
