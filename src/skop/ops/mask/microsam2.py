"""Box-prompted segmentation with micro_sam 2 (SAM2).

The SAM2 twin of ``skop.ops.mask.microsam``: same signature, same ``Masks``,
so the detect-then-mask workflow can swap one for the other. micro_sam 2
offers both its finetuned ``hvit_*_cells`` models and plain SAM2 (``hvit_b``
...). Plain SAM2 has no automatic-segmentation decoder, but a box prompt
needs none -- every SAM2 model is promptable.

Driven through ``micro_sam.v2.util``'s predictor rather than
``prompt_based_segmentation.promptable_segmentation_2d``, which paints its
masks into one label image and so loses the overlaps this op exists to keep.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

import ophelper.boxes
import ophelper.masks
from opspec.op import op
from opspec.runtime import progress
from opspec.types import BoxesOf, ImageOf

from ._result import Masks


class Sam2Model(Enum):
    """micro_sam 2 checkpoints. ``*_cells`` are finetuned on light microscopy;
    the plain ones are Meta's SAM2.1, trained on photographs and video."""

    cells_base = "hvit_b_cells"
    cells_tiny = "hvit_t_cells"
    cells_small = "hvit_s_cells"
    cells_large = "hvit_l_cells"
    sam2_base = "hvit_b"
    sam2_tiny = "hvit_t"
    sam2_small = "hvit_s"
    sam2_large = "hvit_l"


#: Boxes per predictor call. Each answer is a full-image mask, so this bounds
#: the memory one call holds.
_BATCH = 32

#: One predictor, for the model last asked for.
_loaded: dict = {}


def _predictor(model_type: str):
    if model_type not in _loaded:
        from micro_sam.v2.util import get_sam2_image_predictor, get_sam2_model

        _loaded.clear()
        progress(f"Loading micro_sam 2 {model_type}")
        _loaded[model_type] = get_sam2_image_predictor(get_sam2_model(model_type=model_type))
    return _loaded[model_type]


@op(env="pytorch")
def microsam2_masks(
    image: ImageOf[np.ndarray],
    boxes: BoxesOf[np.ndarray],
    model: Sam2Model = Sam2Model.cells_base,
) -> Masks:
    """Segment one object per bounding box with micro_sam 2.

    Args:
        image: Image to segment. 2-D or RGB.
        boxes: (N, 4) as [min_y, min_x, max_y, max_x], from a detector op, a
            Shapes layer, or ``ophelper.boxes.from_labels``.
        model: ``cells_*`` are finetuned for microscopy; ``sam2_*`` are plain
            SAM2, which outlines whatever is in the box, unstained or not.

    Returns:
        masks: (N, Y, X) uint8, one object per plane. These may overlap.
        boxes: the prompts that produced them.

    Masks come back in prompt order -- ``masks[i]`` answers ``boxes[i]``.
    Boxes SAM2 finds nothing in are dropped from both.
    """
    prompts = ophelper.boxes.as_boxes(boxes)
    if len(prompts) == 0:
        return Masks(ophelper.masks.empty(image.shape[:2]), ophelper.boxes.EMPTY.copy())

    from micro_sam.v2.normalization import to_image
    from micro_sam.v2.util import autocast, encode_image

    predictor = _predictor(model.value)

    progress("Computing image embeddings")
    encode_image(predictor, to_image(np.asarray(image)))

    # skop boxes are [min_y, min_x, max_y, max_x]; SAM2 takes xyxy.
    height, width = image.shape[:2]
    xyxy = np.stack(
        [
            np.clip(prompts[:, 1], 0, width),
            np.clip(prompts[:, 0], 0, height),
            np.clip(prompts[:, 3], 0, width),
            np.clip(prompts[:, 2], 0, height),
        ],
        axis=1,
    ).astype(np.float32)

    found = []
    kept = []
    for start in range(0, len(xyxy), _BATCH):
        progress(f"Segmenting objects {start + 1}-{min(start + _BATCH, len(xyxy))} of {len(xyxy)}",
                 start, len(xyxy))
        with autocast(predictor.device):
            masks, _, _ = predictor.predict(box=xyxy[start:start + _BATCH], multimask_output=False)
        masks = np.asarray(masks).reshape(-1, height, width)
        for i, mask in enumerate(masks, start=start):
            # SAM2 answers some prompts with nothing; drop them and their box.
            if not (mask > 0).any():
                continue
            found.append((mask > 0).astype(np.uint8))
            kept.append(i)

    progress(f"Segmented {len(found)} of {len(prompts)} objects")
    if not found:
        return Masks(ophelper.masks.empty(image.shape[:2]), ophelper.boxes.EMPTY.copy())
    return Masks(np.stack(found), prompts[kept])
