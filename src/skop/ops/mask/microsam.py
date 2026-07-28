"""Box-prompted segmentation with micro_sam.

micro_sam is Segment Anything finetuned for microscopy, by the Computational
Cell Analytics group at Göttingen. The finetuning is the point: plain SAM was
trained on photographs and is unremarkable on a field of nuclei, where
``vit_b_lm`` was trained on light microscopy and is not.

This op prompts it with boxes -- stage two of the detect-then-segment workflow
in docs/spec/workflow-ops.md, fed by an op from ``skop.ops.detect``.

Ported rather than wrapped, unlike its MobileSAM sibling: micro_sam is a
normal conda package, and the code between it and here is a loop. The original
is segment_everything/mask_detectors/microsam.py, which drove micro_sam
through ``sam_annotator._state.AnnotatorState`` -- a singleton belonging to
micro_sam's napari plugin. The public path below does the same work without
reaching into another package's private module.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from skop import boxes as _boxes
from skop import masks as _masks
from skop import op, progress
from skop.types import BoxesData, ImageData

from ._result import Masks


class PretrainedModel(Enum):
    """micro_sam's finetuned checkpoints, by the imaging they were trained on.

    All ViT-B. Larger encoders (``vit_l``, ``vit_h``) exist for some of these
    and are better and slower; they are not offered here until someone wants
    one.
    """

    light_microscopy = "vit_b_lm"
    electron_microscopy = "vit_b_em_organelles"
    histopathology = "vit_b_histopathology"
    #: Plain Segment Anything, unfinetuned -- the baseline to compare against.
    natural_image = "vit_b"


@op(env="pytorch")
def microsam_masks(
    image: ImageData,
    boxes: BoxesData,
    model: PretrainedModel = PretrainedModel.light_microscopy,
) -> Masks:
    """Segment one object per bounding box.

    Args:
        image: Image to segment. 2-D or RGB. micro_sam normalizes it to 8-bit
            itself, by min and max -- so a single hot pixel costs contrast
            everywhere else, and clipping such an image first is worth it.
        boxes: (N, 4) as [min_y, min_x, max_y, max_x], from a detector op, a
            Shapes layer, or ``skop.boxes.from_labels``.
        model: Which finetuned checkpoint to prompt.

    Returns:
        masks: (N, Y, X) uint8, one object per plane. These may overlap, which
            is why they are not a label image; project them with
            ``skop.masks``.
        boxes: the prompts that produced them.

    Masks come back in prompt order, not sorted -- ``masks[i]`` answers
    ``boxes[i]``. Boxes SAM finds nothing in are dropped from both, so this may
    be shorter than what went in.

    Note: the weights are ~375 MB, downloaded into micro_sam's own cache on
    first use, which MICROSAM_CACHEDIR relocates.
    """
    # Checked before the imports: a detector that found nothing is an ordinary
    # workflow outcome, and there is no reason to load a 375 MB model to
    # answer no prompts.
    prompts = _boxes.as_boxes(boxes)
    if len(prompts) == 0:
        return Masks(_masks.empty(image.shape[:2]), _boxes.EMPTY.copy())

    from micro_sam.prompt_based_segmentation import segment_from_box
    from micro_sam.util import get_sam_model, precompute_image_embeddings

    progress(f"Loading {model.value}")
    predictor = get_sam_model(model_type=model.value)

    # The expensive half, and it depends only on the image -- which is why
    # this op takes every box at once rather than one per call (workflow-ops).
    progress("Computing image embeddings")
    embeddings = precompute_image_embeddings(predictor, image, ndim=2, verbose=False)

    # micro_sam's own box prompts are [min_y, min_x, max_y, max_x]: its
    # _process_box does the swap to SAM's xyxy internally. That is skop's
    # canonical order, so nothing is converted here.
    found = []
    kept = []
    for i, box in enumerate(prompts):
        progress(f"Segmenting object {i + 1} of {len(prompts)}", i, len(prompts))
        mask = np.squeeze(segment_from_box(predictor, box, image_embeddings=embeddings))
        # SAM answers some prompts with nothing at all. Dropping those is why
        # the surviving boxes are returned alongside the masks.
        if not mask.any():
            continue
        found.append(mask.astype(np.uint8))
        kept.append(i)

    progress(f"Segmented {len(found)} of {len(prompts)} objects")
    if not found:
        return Masks(_masks.empty(image.shape[:2]), _boxes.EMPTY.copy())
    return Masks(np.stack(found), prompts[kept])
