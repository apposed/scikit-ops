"""Box-prompted segmentation with MobileSAMv2's prompt-guided decoder.

The other half of the model ``skop.ops.detect.object_aware_yolo`` wraps, and
the reason both live in the same environment: the object-aware YOLO was
trained to produce box prompts for exactly this decoder, and both come out of
the vendored MobileSAMv2 tree that segment-everything carries.

The model is assembled from three pieces, which is what
``weights_helper.create_mobile_sam_model`` does:

- an empty ``vit_h`` Sam, for the postprocessing and the thresholds,
- EfficientViT-L2 as its image encoder (``l2.pt``, 234 MB),
- MobileSAMv2's prompt encoder and mask decoder, which are one 15 MB file.

Assembled here rather than by calling ``create_mobile_sam_model``, for the
reason ``object_aware_yolo`` gives: ``segment_everything.weights_helper``
imports toolz and ``napari.utils.progress`` at module scope, and
segment-everything declares neither. Importing it means installing napari into
a worker to download a file. Both weight files come from the HuggingFace
mirror already listed in its own ``WEIGHTS_URLS``, rather than through gdown
from Google Drive.

Its sibling ``microsam_masks`` is a port; this is a wrapper. That asymmetry is
the whole argument of docs/design/0007: micro_sam is a conda package, and this
decoder only exists inside a 28,000-line vendored tree.
"""

from __future__ import annotations

import numpy as np

from skop import boxes as _boxes
from skop import masks as _masks
from skop import op, progress
from skop.assets import file_from_url
from skop.types import BoxesData, ImageData

from .._util import to_rgb
from ._result import Masks

# EfficientViT-L2, the image encoder. The same file weights_helper fetches
# through gdown as "efficientvit_l2"; this is its HuggingFace mirror.
_ENCODER = "https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/l2.pt"


@op(env="segment-everything")
def mobilesam_masks(
    image: ImageData,
    boxes: BoxesData,
    batch_size: int = 100,
) -> Masks:
    """Segment one object per bounding box.

    Args:
        image: Image to segment. 2-D or RGB; anything else is stretched to
            8-bit RGB first, which is what the model was trained on.
        boxes: (N, 4) as [min_y, min_x, max_y, max_x], from a detector op, a
            Shapes layer, or ``skop.boxes.from_labels``.
        batch_size: How many boxes to decode at once. The image is encoded
            once regardless; this only trades GPU memory against the number
            of decoder calls.

    Returns:
        masks: (N, Y, X) uint8, one object per plane. These may overlap, which
            is why they are not a label image; project them with
            ``skop.masks``.
        boxes: the prompts that produced them.

    Masks come back in prompt order, not sorted -- ``masks[i]`` answers
    ``boxes[i]``. Boxes the decoder finds nothing in are dropped from both, so
    this may be shorter than what went in.

    Note: the image encoder is 234 MB, downloaded into skop's asset cache on
    first use. The decoder ships inside segment-everything itself.
    """
    from pathlib import Path

    import segment_everything
    import torch
    from segment_everything.vendored.mobilesamv2 import (
        SamPredictor,
        sam_model_registry,
    )

    prompts = _boxes.as_boxes(boxes)
    if len(prompts) == 0:
        return Masks(_masks.empty(image.shape[:2]), _boxes.EMPTY.copy())

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    progress("Loading MobileSAMv2")
    encoder_weights = file_from_url(_ENCODER, "mobilesam/l2.pt")
    # Shipped inside the installed package rather than downloaded -- this is
    # the one entry in WEIGHTS_URLS marked "local:". Same file as the
    # Prompt_guided_Mask_Decoder.pt on the mirror above, if it ever moves.
    decoder_weights = (
        Path(segment_everything.__file__).parent
        / "vendored/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt"
    )

    prompt_guided = sam_model_registry["PromptGuidedDecoder"](str(decoder_weights))
    model = sam_model_registry["vit_h"]()
    model.prompt_encoder = prompt_guided["PromtEncoder"]  # sic, upstream spelling
    model.mask_decoder = prompt_guided["MaskDecoder"]
    model.image_encoder = sam_model_registry["efficientvit_l2"](str(encoder_weights))
    model.to(device)

    predictor = SamPredictor(model)

    # The expensive half, and it depends only on the image -- which is why
    # this op takes every box at once rather than one per call (0005).
    progress("Encoding the image")
    predictor.set_image(to_rgb(image))

    # SamPredictor's transform resizes prompts alongside the image it just
    # encoded, and it reads them as xyxy -- the one conversion this op needs.
    scaled = predictor.transform.apply_boxes(
        _boxes.to_xyxy(prompts), predictor.original_size
    )
    scaled = torch.as_tensor(scaled, dtype=torch.float32, device=device)

    # One encoding, broadcast across each batch of prompts. Sliced from the
    # full repeat rather than reassigned in place: the original narrowed these
    # tensors on every iteration, which only survived because the short batch
    # is always the last one.
    image_embedding = torch.repeat_interleave(predictor.features, batch_size, dim=0)
    prompt_embedding = torch.repeat_interleave(
        model.prompt_encoder.get_dense_pe(), batch_size, dim=0
    )

    decoded = []
    for start in range(0, len(scaled), batch_size):
        batch = scaled[start : start + batch_size]
        progress(
            f"Segmenting objects {start + 1}-{start + len(batch)} of {len(scaled)}",
            start,
            len(scaled),
        )
        with torch.no_grad():
            sparse, dense = model.prompt_encoder(points=None, boxes=batch, masks=None)
            low_res, _predicted_iou = model.mask_decoder(
                image_embeddings=image_embedding[: len(batch)],
                image_pe=prompt_embedding[: len(batch)],
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                simple_type=True,
            )
            # _predicted_iou is a confidence per mask, dropped for the same
            # reason Boxes drops its scores -- see _result.py and 0009. The
            # original also computed a stability score here, which cost a
            # second thresholding pass for a value nothing consumed.
            full = predictor.model.postprocess_masks(
                low_res, predictor.input_size, predictor.original_size
            )
            decoded.append((full > model.mask_threshold).squeeze(1).to(torch.uint8))

    stacked = torch.cat(decoded).cpu().numpy()
    del decoded
    if device == "cuda":
        torch.cuda.empty_cache()

    # The decoder answers some prompts with nothing at all. Dropping those is
    # why the surviving boxes are returned alongside the masks.
    kept = stacked.any(axis=(1, 2))
    progress(f"Segmented {int(kept.sum())} of {len(prompts)} objects")
    if not kept.any():
        return Masks(_masks.empty(image.shape[:2]), _boxes.EMPTY.copy())
    return Masks(np.ascontiguousarray(stacked[kept]), prompts[kept])
